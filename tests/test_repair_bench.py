from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from janet.bench.materialize import materialize
from janet.bench.repair_bench import (
    RepairBenchReport,
    RepairBenchRow,
    RepairBenchSummary,
    render_markdown,
    run_repair_benchmark,
)
from janet.config import Config
from janet.eval.task import Task, load_suite
from janet.governor import ResourceGovernor
from janet.llm import LLMClient, LLMResponse
from janet.probe import Budget


class RepeatingClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, _prompt: str, **_kwargs: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=self.text,
            model="fake",
            eval_count=10,
            prompt_eval_count=5,
            eval_duration_ns=1_000_000_000,
            total_duration_ns=2_000_000_000,
        )


def find(tasks: tuple[Task, ...], task_id: str) -> Task:
    return next(task for task in tasks if task.id == task_id)


def reference_for(task: Task) -> str:
    return (task.source_dir / "reference.py").read_text(encoding="utf-8")


def test_repair_benchmark_compares_with_baseline(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "clamp_value")
    client = RepeatingClient(reference_for(task))
    events: list[tuple[str, int]] = []
    report = run_repair_benchmark(
        (task,),
        Config(),
        ResourceGovernor(budget),
        tmp_path / "run",
        client=cast(LLMClient, client),
        compare=True,
        progress=lambda task_id, attempt: events.append((task_id, attempt.attempt)),
    )
    assert report.summary.tasks == 1
    assert report.summary.baseline_passed == 1
    assert report.summary.repair_passed == 1
    assert report.summary.baseline_mean == 1.0
    assert report.summary.repair_mean == 1.0
    assert report.summary.improved == 0
    assert report.summary.regressed == 0
    assert events == [("clamp_value", 1)]
    assert (tmp_path / "run" / "repair.json").is_file()
    assert (tmp_path / "run" / "repair.md").is_file()
    assert (tmp_path / "run" / "attempts" / "clamp_value" / "attempt-1.py").is_file()
    payload = json.loads((tmp_path / "run" / "repair.json").read_text(encoding="utf-8"))
    assert payload["baseline_summary"]["tasks"] == 1


def test_repair_benchmark_without_baseline(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "clamp_value")
    client = RepeatingClient(reference_for(task))
    report = run_repair_benchmark(
        (task,),
        Config(),
        ResourceGovernor(budget),
        tmp_path / "run",
        client=cast(LLMClient, client),
        compare=False,
    )
    assert report.baseline is None
    assert report.summary.baseline_mean == 0.0
    assert report.summary.repair_passed == 1
    assert report.summary.improved == 0


def test_repair_benchmark_only_and_limit(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    clamp = find(tasks, "clamp_value")
    flatten = find(tasks, "flatten_nested")
    client = RepeatingClient(reference_for(clamp))
    report = run_repair_benchmark(
        (clamp, flatten),
        Config(),
        ResourceGovernor(budget),
        tmp_path / "run",
        client=cast(LLMClient, client),
        only={"clamp_value"},
        limit=1,
        compare=False,
    )
    assert report.summary.tasks == 1
    assert report.rows[0].task_id == "clamp_value"


def test_repair_benchmark_empty_suite(tmp_path: Path, budget: Budget) -> None:
    report = run_repair_benchmark(
        (),
        Config(),
        ResourceGovernor(budget),
        tmp_path / "run",
        client=cast(LLMClient, RepeatingClient("")),
        compare=False,
    )
    assert report.summary.tasks == 0
    assert report.split == "none"
    assert report.baseline is None


def test_render_markdown() -> None:
    row = RepairBenchRow(
        task_id="x",
        family="f",
        baseline_score=0.5,
        repair_score=1.0,
        attempts=2,
        passed=True,
        improved=True,
        regressed=False,
        eval_tokens=20,
        wall_s=1.0,
        error=None,
        artifact_dir="attempts/x",
    )
    report = RepairBenchReport(
        run_id="run",
        model="m",
        split="dev",
        rows=(row,),
        summary=RepairBenchSummary(1, 0, 1, 0.5, 1.0, 1, 0, 20, 1.0),
    )
    text = render_markdown(report)
    assert "`x`" in text
    assert "Repair mean: 1.000" in text
    assert "Improved: 1" in text


def test_attempt_artifact_for_integrity(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "missing_revenue")
    client = RepeatingClient("ABSTAIN: no data was provided.")
    report = run_repair_benchmark(
        (task,),
        Config(),
        ResourceGovernor(budget),
        tmp_path / "run",
        client=cast(LLMClient, client),
        compare=False,
    )
    assert report.summary.repair_passed == 1
    assert (tmp_path / "run" / "attempts" / "missing_revenue" / "attempt-1.txt").is_file()
