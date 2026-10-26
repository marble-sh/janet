"""One-shot baseline: ask the frozen local model to fix each task."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from janet.config import Config
from janet.eval.extract import extract_python, looks_like_python
from janet.eval.prompts import build_prompt, compose_system
from janet.eval.runner import TaskResult, run_task
from janet.eval.task import Task
from janet.governor import ResourceGovernor
from janet.harness.skills import Skill
from janet.llm import LLMClient, LLMError, LLMResponse
from janet.preempt import StopCheck
from janet.preempt import check as check_stop


@dataclass(frozen=True, slots=True)
class BaselineRow:
    task_id: str
    family: str
    split: str
    score: float
    passed: bool
    error: str | None
    duration_s: float
    eval_count: int
    prompt_eval_count: int
    response_chars: int
    extracted: bool
    artifact: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "score": round(self.score, 6),
            "passed": self.passed,
            "error": self.error,
            "duration_s": round(self.duration_s, 3),
            "eval_count": self.eval_count,
            "prompt_eval_count": self.prompt_eval_count,
            "response_chars": self.response_chars,
            "extracted": self.extracted,
            "artifact": self.artifact,
        }


@dataclass(frozen=True, slots=True)
class BaselineSummary:
    tasks: int
    passed: int
    mean_score: float
    total_eval_tokens: int
    total_prompt_tokens: int
    total_wall_s: float
    mean_tokens_per_second: float

    def to_dict(self) -> dict[str, object]:
        return {
            "tasks": self.tasks,
            "passed": self.passed,
            "pass_rate": round(self.passed / self.tasks, 6) if self.tasks else 0.0,
            "mean_score": round(self.mean_score, 6),
            "total_eval_tokens": self.total_eval_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_wall_s": round(self.total_wall_s, 3),
            "mean_tokens_per_second": round(self.mean_tokens_per_second, 3),
        }


@dataclass(frozen=True, slots=True)
class BaselineReport:
    run_id: str
    model: str
    split: str
    rows: tuple[BaselineRow, ...]
    summary: BaselineSummary
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "split": self.split,
            "created_at": self.created_at,
            "summary": self.summary.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
        }


def _summarize(rows: list[BaselineRow]) -> BaselineSummary:
    if not rows:
        return BaselineSummary(0, 0, 0.0, 0, 0, 0.0, 0.0)
    total_eval = sum(row.eval_count for row in rows)
    total_prompt = sum(row.prompt_eval_count for row in rows)
    total_time = sum(row.duration_s for row in rows)
    rates = [
        row.eval_count / row.duration_s for row in rows if row.eval_count > 0 and row.duration_s > 0
    ]
    return BaselineSummary(
        tasks=len(rows),
        passed=sum(1 for row in rows if row.passed),
        mean_score=sum(row.score for row in rows) / len(rows),
        total_eval_tokens=total_eval,
        total_prompt_tokens=total_prompt,
        total_wall_s=total_time,
        mean_tokens_per_second=(sum(rates) / len(rates)) if rates else 0.0,
    )


def render_markdown(report: BaselineReport) -> str:
    summary = report.summary
    lines = [
        f"# Baseline — {report.run_id}",
        "",
        f"- Model: `{report.model}`",
        f"- Split: `{report.split}`",
        f"- Tasks: {summary.tasks}",
        f"- Passed: {summary.passed}",
        f"- Mean score: {summary.mean_score:.3f}",
        f"- Eval tokens: {summary.total_eval_tokens}",
        f"- Wall time: {summary.total_wall_s:.1f}s",
        "",
        "| Task | Family | Score | Passed | Error |",
        "|---|---|---:|---|---|",
    ]
    for row in report.rows:
        error = (row.error or "").replace("|", "/")[:80]
        lines.append(
            f"| `{row.task_id}` | {row.family} | {row.score:.3f} | "
            f"{'yes' if row.passed else 'no'} | {error} |"
        )
    return "\n".join(lines) + "\n"


def run_baseline(
    tasks: tuple[Task, ...],
    config: Config,
    governor: ResourceGovernor,
    run_dir: Path,
    *,
    client: LLMClient | None = None,
    limit: int | None = None,
    only: set[str] | None = None,
    system_prompt: str | None = None,
    skills: Sequence[Skill] = (),
    stop: StopCheck | None = None,
    progress: Callable[[BaselineRow], None] | None = None,
) -> BaselineReport:
    run_dir = run_dir.expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "work").mkdir(exist_ok=True)
    (run_dir / "candidates").mkdir(exist_ok=True)
    client = client or LLMClient(config.llm)
    selected = [task for task in tasks if only is None or task.id in only]
    if limit is not None:
        selected = selected[: max(0, limit)]
    rows: list[BaselineRow] = []
    started_all = time.monotonic()
    for task in selected:
        check_stop(stop)
        if governor.should_yield():
            time.sleep(5.0)
        prompt = build_prompt(task)
        system = compose_system(system_prompt, skills, task)
        task_started = time.monotonic()
        try:
            response: LLMResponse = client.generate(prompt, system=system)
        except LLMError as exc:
            row = BaselineRow(
                task_id=task.id,
                family=task.family,
                split=task.split,
                score=0.0,
                passed=False,
                error=f"llm: {exc}",
                duration_s=time.monotonic() - task_started,
                eval_count=0,
                prompt_eval_count=0,
                response_chars=0,
                extracted=False,
            )
            rows.append(row)
            if progress:
                progress(row)
            continue
        if task.is_integrity:
            result: TaskResult = run_task(task, response.text, governor, run_dir / "work")
            extracted = True
            artifact_path = run_dir / "candidates" / f"{task.id}.txt"
            artifact_path.write_text(response.text, encoding="utf-8")
        else:
            candidate = extract_python(response.text)
            extracted = looks_like_python(candidate)
            result = run_task(task, candidate, governor, run_dir / "work")
            artifact_path = run_dir / "candidates" / f"{task.id}.py"
            artifact_path.write_text(candidate, encoding="utf-8")
        row = BaselineRow(
            task_id=task.id,
            family=task.family,
            split=task.split,
            score=result.score,
            passed=result.passed,
            error=result.error,
            duration_s=time.monotonic() - task_started,
            eval_count=response.eval_count,
            prompt_eval_count=response.prompt_eval_count,
            response_chars=len(response.text),
            extracted=extracted,
            artifact=str(artifact_path.relative_to(run_dir)),
        )
        rows.append(row)
        if progress:
            progress(row)
    elapsed = time.monotonic() - started_all
    summary = _summarize(rows)
    # Replace wall time with the end-to-end time, which includes sandbox work.
    summary = BaselineSummary(
        tasks=summary.tasks,
        passed=summary.passed,
        mean_score=summary.mean_score,
        total_eval_tokens=summary.total_eval_tokens,
        total_prompt_tokens=summary.total_prompt_tokens,
        total_wall_s=elapsed,
        mean_tokens_per_second=summary.mean_tokens_per_second,
    )
    report = BaselineReport(
        run_id=run_dir.name,
        model=config.llm.policy_model,
        split=selected[0].split if selected else "none",
        rows=tuple(rows),
        summary=summary,
    )
    (run_dir / "baseline.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "baseline.md").write_text(render_markdown(report), encoding="utf-8")
    return report
