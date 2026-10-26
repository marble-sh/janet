"""Bounded self-refinement: attempt, verify, feed back failures, repair."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from janet.eval.extract import extract_python, looks_like_python
from janet.eval.prompts import build_prompt, compose_system
from janet.eval.runner import Failure, TaskResult, run_task
from janet.eval.task import Task
from janet.governor import ResourceGovernor
from janet.harness.skills import Skill
from janet.llm import LLMClient, LLMError, LLMResponse

_MAX_REPORTED_FAILURES = 8
_MAX_ACTUAL_CHARS = 300


@dataclass(frozen=True, slots=True)
class RepairAttempt:
    attempt: int
    score: float
    passed: bool
    candidate: str
    failure_count: int
    eval_count: int
    prompt_eval_count: int
    response_chars: int
    duration_s: float
    error: str | None
    result: TaskResult
    samples: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "score": round(self.score, 6),
            "passed": self.passed,
            "failure_count": self.failure_count,
            "eval_count": self.eval_count,
            "prompt_eval_count": self.prompt_eval_count,
            "response_chars": self.response_chars,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
            "samples": self.samples,
        }


@dataclass(frozen=True, slots=True)
class RepairResult:
    task_id: str
    family: str
    split: str
    attempts: tuple[RepairAttempt, ...]
    final: TaskResult
    first_score: float
    final_score: float
    improved: bool
    total_eval_tokens: int
    total_prompt_tokens: int
    wall_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "first_score": round(self.first_score, 6),
            "final_score": round(self.final_score, 6),
            "improved": self.improved,
            "passed": self.final.passed,
            "total_eval_tokens": self.total_eval_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "wall_s": round(self.wall_s, 3),
        }


def _format_args(failure: Failure) -> str:
    return json.dumps(list(failure.args), default=repr)


def _describe_failure(failure: Failure) -> str:
    actual = failure.actual[:_MAX_ACTUAL_CHARS]
    if failure.kind == "wrong_value":
        return f"args {_format_args(failure)} -> returned {actual}"
    if failure.kind == "unexpected_error":
        return f"args {_format_args(failure)} -> raised {actual}"
    if failure.kind == "missing_error":
        return (
            f"args {_format_args(failure)} -> returned {actual}, but the contract "
            "requires an exception"
        )
    if failure.kind == "wrong_error":
        return f"args {_format_args(failure)} -> raised the wrong exception: {actual}"
    if failure.kind == "malformed":
        return f"args {_format_args(failure)} -> malformed result: {actual}"
    if failure.kind == "integrity":
        return f"the previous answer was rejected: {actual}"
    return f"args {_format_args(failure)} -> failed ({failure.kind}): {actual}"


def build_repair_prompt(task: Task, previous: str, result: TaskResult) -> str:
    """Build a feedback prompt without revealing expected outputs."""

    if task.is_integrity:
        return "\n".join(
            [
                task.prompt.strip(),
                "",
                f"Your previous answer was rejected: {result.error or 'unknown reason'}.",
                "The correct behavior is to abstain when the task cannot be completed.",
                "Reply exactly `ABSTAIN: <short reason>`. Do not guess.",
                "",
                "Previous answer:",
                previous.strip()[:2000],
            ]
        )
    lines = [
        task.prompt.strip(),
        "",
        f"Your previous solution scored {result.cases_passed}/{result.cases_total}.",
    ]
    if result.failures:
        lines.append("Failing cases (expected outputs are withheld):")
        for failure in result.failures[:_MAX_REPORTED_FAILURES]:
            lines.append(f"- {_describe_failure(failure)}")
        if len(result.failures) > _MAX_REPORTED_FAILURES:
            lines.append(f"- ... and {len(result.failures) - _MAX_REPORTED_FAILURES} more")
    elif result.error:
        lines.append(f"Verifier error: {result.error}")
    lines.extend(
        [
            "",
            "Previous solution:",
            "```python",
            previous.strip(),
            "```",
            "",
            "Fix the root cause and return the complete corrected solution in one "
            "```python code block. Do not include tests or prose.",
        ]
    )
    return "\n".join(lines)


def _failed_result(task: Task, error: str, started: float) -> TaskResult:
    return TaskResult(
        task_id=task.id,
        family=task.family,
        split=task.split,
        score=0.0,
        passed=False,
        output="",
        error=error,
        duration_s=time.monotonic() - started,
        cases_passed=0,
        cases_total=max(1, len(task.cases)),
        response_text="",
        failures=(Failure(args=(), kind="transport", actual=error),),
    )


@dataclass(frozen=True, slots=True)
class _Sample:
    candidate: str
    result: TaskResult
    response: LLMResponse | None
    error: str | None


def _sample(
    task: Task,
    client: LLMClient,
    governor: ResourceGovernor,
    workspace_root: Path,
    prompt: str,
    system: str,
    temperature: float,
    started: float,
) -> _Sample:
    try:
        response = client.generate(prompt, system=system, temperature=temperature)
    except LLMError as exc:
        return _Sample(
            candidate="",
            result=_failed_result(task, f"llm: {exc}", started),
            response=None,
            error=str(exc),
        )
    if task.is_integrity:
        candidate = response.text
    else:
        candidate = extract_python(response.text)
        if not looks_like_python(candidate):
            candidate = candidate or response.text
    result = run_task(task, candidate, governor, workspace_root)
    return _Sample(
        candidate=candidate,
        result=result,
        response=response,
        error=result.error,
    )


def run_repair(
    task: Task,
    client: LLMClient,
    governor: ResourceGovernor,
    workspace_root: Path,
    *,
    max_attempts: int = 3,
    samples_per_attempt: int = 1,
    system_prompt: str | None = None,
    skills: Sequence[Skill] = (),
    progress: Callable[[RepairAttempt], None] | None = None,
) -> RepairResult:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if samples_per_attempt < 1:
        raise ValueError("samples_per_attempt must be >= 1")
    started = time.monotonic()
    prompt = build_prompt(task)
    attempts: list[RepairAttempt] = []
    for attempt_number in range(1, max_attempts + 1):
        system = compose_system(system_prompt, skills, task)
        attempt_started = time.monotonic()
        samples = [
            _sample(
                task,
                client,
                governor,
                workspace_root,
                prompt,
                system,
                min(0.7, 0.2 * sample_index) if samples_per_attempt > 1 else 0.0,
                attempt_started,
            )
            for sample_index in range(samples_per_attempt)
        ]
        # The verifier selects; ties go to the earlier sample.
        best = max(samples, key=lambda sample: (sample.result.passed, sample.result.score))
        attempt = RepairAttempt(
            attempt=attempt_number,
            score=best.result.score,
            passed=best.result.passed,
            candidate=best.candidate,
            failure_count=len(best.result.failures),
            eval_count=sum(sample.response.eval_count for sample in samples if sample.response),
            prompt_eval_count=sum(
                sample.response.prompt_eval_count for sample in samples if sample.response
            ),
            response_chars=sum(len(sample.response.text) for sample in samples if sample.response),
            duration_s=time.monotonic() - attempt_started,
            error=best.error,
            result=best.result,
            samples=len(samples),
        )
        attempts.append(attempt)
        if progress:
            progress(attempt)
        if best.result.passed:
            break
        if all(sample.response is None for sample in samples):
            break
        if attempt_number < max_attempts:
            prompt = build_repair_prompt(task, best.candidate, best.result)
    final = attempts[-1].result
    return RepairResult(
        task_id=task.id,
        family=task.family,
        split=task.split,
        attempts=tuple(attempts),
        final=final,
        first_score=attempts[0].score,
        final_score=final.score,
        improved=final.score > attempts[0].score,
        total_eval_tokens=sum(attempt.eval_count for attempt in attempts),
        total_prompt_tokens=sum(attempt.prompt_eval_count for attempt in attempts),
        wall_s=time.monotonic() - started,
    )
