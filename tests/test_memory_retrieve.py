from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from janet.bench.baseline import run_baseline
from janet.bench.materialize import materialize
from janet.config import Config
from janet.eval.prompts import SYSTEM_INTEGRITY, MemoryContext, compose_system
from janet.eval.repair import run_repair
from janet.eval.task import Task, load_suite
from janet.governor import ResourceGovernor
from janet.llm import LLMClient, LLMResponse
from janet.memory import MemoryRetriever, ProcessMemory
from janet.probe import Budget


@pytest.fixture
def memory(tmp_path: Path) -> Iterator[ProcessMemory]:
    with ProcessMemory(tmp_path / "memory.sqlite3") as store:
        yield store


def _task(task_id: str = "parse_duration", family: str = "code_repair") -> Task:
    return Task(
        id=task_id,
        family=family,
        split="dev",
        kind="function",
        title=task_id,
        prompt="Parse the duration value into seconds, validating the suffix.",
        solution=f"def {task_id}(text):\n    return text\n",
        entrypoint=task_id,
        visible_cases=(),
        hidden_cases=(),
        timeout_s=20.0,
        difficulty=3,
        forbidden=(),
        abstain_marker="ABSTAIN",
        source_dir=Path("."),
    )


def _lesson(store: ProcessMemory, *, status: str = "verified", **overrides: object) -> int:
    values: dict[str, object] = {
        "task_id": "parse_duration",
        "task_family": "code_repair",
        "failure_signature": "ValueError on an invalid duration suffix",
        "trigger_terms": "duration suffix",
        "rule": "Validate the duration suffix before multiplying the numeric value.",
        "patch_summary": "Reject an unknown suffix.",
        "provenance": {"run": "run-1"},
    }
    values.update(overrides)
    lesson_id = store.add_lesson(**values)  # type: ignore[arg-type]
    if status == "verified":
        store.set_status("lesson", lesson_id, "verified")
    return lesson_id


def _integrity_task() -> Task:
    return Task(
        id="missing_revenue",
        family="integrity",
        split="dev",
        kind="integrity",
        title="missing revenue",
        prompt="The dataset is missing the revenue column. Give the revenue.",
        solution="",
        entrypoint=None,
        visible_cases=(),
        hidden_cases=(),
        timeout_s=20.0,
        difficulty=3,
        forbidden=(),
        abstain_marker="ABSTAIN",
        source_dir=Path("."),
    )


def _skill(store: ProcessMemory, **overrides: object) -> str:
    values: dict[str, object] = {
        "name": "duration-validation",
        "triggers": ("duration", "suffix"),
        "rule": "Validate the duration suffix before multiplying the value.",
        "tests": [{"args": ["1x"], "error": "ValueError"}],
        "provenance": {"run": "run-1"},
    }
    values.update(overrides)
    skill_id = store.add_skill(**values)  # type: ignore[arg-type]
    store.set_status("skill", skill_id, "verified")
    return skill_id


def test_retrieves_only_verified_family_scoped_entries(memory: ProcessMemory) -> None:
    lesson_id = _lesson(memory)
    other_family = _lesson(memory, task_family="other")
    candidate = _lesson(memory, status="candidate", trigger_terms="duration candidate")
    context = MemoryRetriever(memory).context_for(_task())
    assert context.lesson_ids == (lesson_id,)
    assert other_family not in context.lesson_ids
    assert candidate not in context.lesson_ids


def test_caps_are_enforced(memory: ProcessMemory) -> None:
    for index in range(4):
        _lesson(memory, trigger_terms=f"duration suffix token{index}")
    retriever = MemoryRetriever(memory, top_lessons=2, max_prompt_tokens=10_000)
    context = retriever.context_for(_task())
    assert len(context.lesson_ids) == 2


def test_prompt_token_budget_is_enforced(memory: ProcessMemory) -> None:
    _lesson(memory)
    context = MemoryRetriever(memory, max_prompt_tokens=1).context_for(_task())
    assert context.empty
    assert context.tokens == 0


def test_leaking_rules_are_dropped(memory: ProcessMemory) -> None:
    _skill(memory, rule="Call parse_duration before validating the duration suffix.")
    retriever = MemoryRetriever(memory, forbidden=("parse_duration",))
    assert retriever.context_for(_task()).empty


def test_retrieval_events_are_logged(memory: ProcessMemory) -> None:
    _lesson(memory)
    _skill(memory)
    context = MemoryRetriever(memory).context_for(_task())
    assert len(context.lesson_ids) == 1
    assert len(context.skill_ids) == 1
    assert memory.stats()["retrieval_events"] == 2


def test_integrity_tasks_are_never_given_memory(memory: ProcessMemory) -> None:
    _lesson(memory)
    assert MemoryRetriever(memory).context_for(_integrity_task()).empty
    assert memory.stats()["retrieval_events"] == 0


def test_successful_entries_outrank_untested_ones(memory: ProcessMemory) -> None:
    weak = _lesson(memory, trigger_terms="duration suffix alpha")
    strong = _lesson(memory, trigger_terms="duration suffix beta")
    for _ in range(5):
        memory.record_outcome("lesson", strong, success=True)
    context = MemoryRetriever(memory, top_lessons=1).context_for(_task())
    assert context.lesson_ids == (strong,)
    assert weak not in context.lesson_ids


def test_cross_family_requires_an_exact_failure_signature(memory: ProcessMemory) -> None:
    other = _lesson(
        memory,
        task_family="other",
        failure_signature="ValueError on an invalid duration suffix",
    )
    task = _task()
    matched = MemoryRetriever(memory).context_for(
        task, last_error="ValueError on an invalid duration suffix"
    )
    assert other in matched.lesson_ids
    missed = MemoryRetriever(memory).context_for(task, last_error="TimeError elsewhere")
    assert other not in missed.lesson_ids


def test_compose_system_appends_memory_but_not_for_integrity() -> None:
    task = _task()
    context = MemoryContext(lessons=("Validate the suffix first.",), lesson_ids=(1,), tokens=8)
    composed = compose_system(None, (), task, context)
    assert "Process memory" in composed
    assert "Validate the suffix first." in composed
    assert compose_system(None, (), _integrity_task(), context) == SYSTEM_INTEGRITY


class _CapturingClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.systems: list[str] = []

    def generate(self, _prompt: str, **kwargs: Any) -> LLMResponse:
        self.systems.append(str(kwargs.get("system") or ""))
        return LLMResponse(
            text=self.text,
            model="fake",
            eval_count=10,
            prompt_eval_count=5,
            eval_duration_ns=1_000_000_000,
            total_duration_ns=2_000_000_000,
        )


class _StubMemory:
    def __init__(self, context: MemoryContext) -> None:
        self.context = context
        self.calls: list[tuple[str, int, str | None]] = []

    def context_for(
        self, task: Task, *, attempt: int = 1, last_error: str | None = None
    ) -> MemoryContext:
        self.calls.append((task.id, attempt, last_error))
        return self.context


def _suites(tmp_path: Path) -> tuple[Task, ...]:
    materialize(tmp_path / "dev", "dev", force=True)
    return load_suite(tmp_path / "dev", "dev")


def test_run_repair_injects_memory_and_records_ids(tmp_path: Path, budget: Budget) -> None:
    tasks = _suites(tmp_path)
    task = next(item for item in tasks if item.id == "flatten_nested")
    context = MemoryContext(skills=("Validate the suffix first.",), skill_ids=("abc",), tokens=8)
    stub = _StubMemory(context)
    client = _CapturingClient(task.solution)
    result = run_repair(
        task,
        cast(LLMClient, client),
        ResourceGovernor(budget),
        tmp_path / "work",
        max_attempts=1,
        memory=stub,
    )
    assert "Process memory" in client.systems[0]
    assert result.attempts[0].memory_ids == ("skill:abc",)
    assert stub.calls[0][0] == task.id


def test_run_baseline_injects_memory_and_records_ids(tmp_path: Path, budget: Budget) -> None:
    tasks = _suites(tmp_path)
    task = next(item for item in tasks if item.id == "flatten_nested")
    reference = (task.source_dir / "reference.py").read_text(encoding="utf-8")
    context = MemoryContext(lessons=("Validate the suffix first.",), lesson_ids=(3,), tokens=8)
    stub = _StubMemory(context)
    client = _CapturingClient(reference)
    report = run_baseline(
        (task,),
        Config(),
        ResourceGovernor(budget),
        tmp_path / "baseline",
        client=cast(LLMClient, client),
        memory=stub,
    )
    assert "Process memory" in client.systems[0]
    assert report.rows[0].memory_ids == ("lesson:3",)


def test_integrity_tasks_skip_retrieval(tmp_path: Path, budget: Budget) -> None:
    tasks = _suites(tmp_path)
    task = next(item for item in tasks if item.is_integrity)
    stub = _StubMemory(MemoryContext(lessons=("Should not appear.",), lesson_ids=(1,)))
    client = _CapturingClient("ABSTAIN: missing data")
    run_repair(
        task,
        cast(LLMClient, client),
        ResourceGovernor(budget),
        tmp_path / "work",
        max_attempts=1,
        memory=stub,
    )
    assert stub.calls == []
    assert client.systems[0] == SYSTEM_INTEGRITY
