from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from janet.bench.materialize import materialize
from janet.eval.repair import (
    RepairAttempt,
    _describe_failure,
    build_repair_prompt,
    run_repair,
)
from janet.eval.runner import Failure, TaskResult
from janet.eval.task import Task, load_suite
from janet.governor import ResourceGovernor
from janet.llm import LLMClient, LLMError, LLMResponse
from janet.probe import Budget


class SequenceClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, **_kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        if not self.responses:
            raise LLMError("no responses left")
        return LLMResponse(
            text=self.responses.pop(0),
            model="fake",
            eval_count=10,
            prompt_eval_count=5,
            eval_duration_ns=1_000_000_000,
            total_duration_ns=2_000_000_000,
        )


class ErrorClient:
    def generate(self, _prompt: str, **_kwargs: Any) -> LLMResponse:
        raise LLMError("server down")


def find(tasks: tuple[Task, ...], task_id: str) -> Task:
    return next(task for task in tasks if task.id == task_id)


def make_result(task: Task, failures: tuple[Failure, ...] = ()) -> TaskResult:
    return TaskResult(
        task_id=task.id,
        family=task.family,
        split=task.split,
        score=0.5,
        passed=False,
        output="",
        error=None,
        duration_s=0.1,
        cases_passed=1,
        cases_total=2,
        response_text="",
        failures=failures,
    )


def test_build_repair_prompt_hides_expected_values(tmp_path: Path) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "flatten_nested")
    failure = Failure(args=(["ab", ["c"]],), kind="wrong_value", actual="['c']")
    prompt = build_repair_prompt(task, task.solution, make_result(task, (failure,)))
    assert "Failing cases" in prompt
    assert "expected outputs are withheld" in prompt
    assert "['c']" in prompt
    assert "args" in prompt
    assert "Previous solution" in prompt
    assert "def flatten" in prompt


def test_build_repair_prompt_for_integrity(tmp_path: Path) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "missing_revenue")
    result = make_result(
        task, (Failure(args=(), kind="integrity", actual="forbidden answer pattern"),)
    )
    prompt = build_repair_prompt(task, "Acme earned $5 million.", result)
    assert "rejected" in prompt
    assert "ABSTAIN" in prompt
    assert "Acme earned" in prompt


def test_describe_failure_kinds() -> None:
    kinds = [
        "wrong_value",
        "unexpected_error",
        "missing_error",
        "wrong_error",
        "malformed",
        "integrity",
        "transport",
        "other",
    ]
    for kind in kinds:
        text = _describe_failure(Failure(args=(1,), kind=kind, actual="detail"))
        assert "detail" in text


def test_run_repair_improves_a_failing_task(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "flatten_nested")
    reference = (task.source_dir / "reference.py").read_text(encoding="utf-8")
    client = SequenceClient([task.solution, reference])
    result = run_repair(
        task,
        cast(LLMClient, client),
        ResourceGovernor(budget),
        tmp_path / "work",
        max_attempts=3,
    )
    assert len(result.attempts) == 2
    assert result.first_score < 1.0
    assert result.final_score == 1.0
    assert result.final.passed
    assert result.improved
    assert "Failing cases" in client.prompts[1]


def test_run_repair_stops_after_one_attempt(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "flatten_nested")
    client = SequenceClient([task.solution])
    result = run_repair(
        task,
        cast(LLMClient, client),
        ResourceGovernor(budget),
        tmp_path / "work",
        max_attempts=1,
    )
    assert len(result.attempts) == 1
    assert not result.final.passed
    assert not result.improved


def test_run_repair_recovers_integrity(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "missing_revenue")
    client = SequenceClient(
        ["Acme's revenue was $5 million.", "ABSTAIN: no financial data was provided."]
    )
    result = run_repair(
        task,
        cast(LLMClient, client),
        ResourceGovernor(budget),
        tmp_path / "work",
        max_attempts=2,
    )
    assert len(result.attempts) == 2
    assert result.final.passed
    assert result.improved
    assert "rejected" in client.prompts[1]


def test_run_repair_handles_a_transport_error(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "clamp_value")
    result = run_repair(
        task,
        cast(LLMClient, ErrorClient()),
        ResourceGovernor(budget),
        tmp_path / "work",
        max_attempts=2,
    )
    assert len(result.attempts) == 1
    assert result.final.error is not None
    assert result.final.error.startswith("llm:")


def test_run_repair_rejects_zero_attempts(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    with pytest.raises(ValueError, match="max_attempts"):
        run_repair(
            find(tasks, "clamp_value"),
            cast(LLMClient, SequenceClient([])),
            ResourceGovernor(budget),
            tmp_path / "work",
            max_attempts=0,
        )


def test_best_of_n_selects_the_verifier_winner(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "flatten_nested")
    reference = (task.source_dir / "reference.py").read_text(encoding="utf-8")
    client = SequenceClient([task.solution, reference])
    result = run_repair(
        task,
        cast(LLMClient, client),
        ResourceGovernor(budget),
        tmp_path / "work",
        max_attempts=1,
        samples_per_attempt=2,
    )
    assert result.final.passed
    assert result.attempts[0].samples == 2
    assert result.attempts[0].score == 1.0


def test_run_repair_rejects_zero_samples(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    with pytest.raises(ValueError, match="samples_per_attempt"):
        run_repair(
            find(tasks, "clamp_value"),
            cast(LLMClient, SequenceClient([])),
            ResourceGovernor(budget),
            tmp_path / "work",
            samples_per_attempt=0,
        )


def test_repair_attempt_serialization(tmp_path: Path) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "clamp_value")
    result = make_result(task)
    attempt = RepairAttempt(
        attempt=1,
        score=0.5,
        passed=False,
        candidate="def clamp(): pass",
        failure_count=1,
        eval_count=10,
        prompt_eval_count=5,
        response_chars=20,
        duration_s=0.1,
        error=None,
        result=result,
    )
    payload = attempt.to_dict()
    assert payload["attempt"] == 1
    assert payload["score"] == 0.5
