from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from janet.bench.baseline import (
    BaselineRow,
    render_markdown,
    run_baseline,
)
from janet.bench.materialize import materialize
from janet.config import Config
from janet.eval.prompts import build_prompt
from janet.eval.task import Task, load_suite
from janet.governor import ResourceGovernor
from janet.llm import LLMClient, LLMError, LLMResponse
from janet.probe import Budget


class FakeClient:
    def __init__(self, text: str, *, error: LLMError | None = None) -> None:
        self.text = text
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str, **_kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return LLMResponse(
            text=self.text,
            model="fake",
            eval_count=10,
            prompt_eval_count=5,
            eval_duration_ns=1_000_000_000,
            total_duration_ns=2_000_000_000,
        )

    def list_models(self) -> list[str]:
        return ["fake"]


def find(tasks: tuple[Task, ...], task_id: str) -> Task:
    return next(task for task in tasks if task.id == task_id)


def test_build_prompt_contains_the_buggy_code(tmp_path: Path) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "clamp_value")
    prompt = build_prompt(task)
    assert "Current implementation" in prompt
    assert "def clamp" in prompt
    assert "Visible examples" in prompt
    assert "raises ValueError" not in prompt  # clamp's visible cases are valid


def test_build_prompt_for_integrity_is_the_task_prompt(tmp_path: Path) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "missing_revenue")
    assert build_prompt(task) == task.prompt


def test_baseline_records_a_pass(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "clamp_value")
    reference = (task.source_dir / "reference.py").read_text(encoding="utf-8")
    fake = FakeClient(f"```python\n{reference}\n```")
    report = run_baseline(
        (task,),
        Config(),
        ResourceGovernor(budget),
        tmp_path / "run",
        client=cast(LLMClient, fake),
    )
    assert report.summary.tasks == 1
    assert report.summary.passed == 1
    assert report.summary.mean_score == 1.0
    assert (tmp_path / "run" / "candidates" / "clamp_value.py").is_file()
    assert (tmp_path / "run" / "baseline.json").is_file()
    assert (tmp_path / "run" / "baseline.md").is_file()
    assert fake.prompts and "def clamp" in fake.prompts[0]


def test_baseline_handles_llm_errors(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "clamp_value")
    fake = FakeClient("", error=LLMError("server down"))
    report = run_baseline(
        (task,),
        Config(),
        ResourceGovernor(budget),
        tmp_path / "run",
        client=cast(LLMClient, fake),
    )
    assert report.summary.tasks == 1
    assert report.summary.passed == 0
    assert report.rows[0].error is not None
    assert report.rows[0].error.startswith("llm:")


def test_baseline_only_filters_tasks(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    first = find(tasks, "clamp_value")
    second = find(tasks, "flatten_nested")
    reference = (first.source_dir / "reference.py").read_text(encoding="utf-8")
    fake = FakeClient(reference)
    report = run_baseline(
        (first, second),
        Config(),
        ResourceGovernor(budget),
        tmp_path / "run",
        client=cast(LLMClient, fake),
        only={"clamp_value"},
    )
    assert report.summary.tasks == 1
    assert report.rows[0].task_id == "clamp_value"


def test_baseline_empty_suite_writes_a_report(tmp_path: Path, budget: Budget) -> None:
    report = run_baseline(
        (),
        Config(),
        ResourceGovernor(budget),
        tmp_path / "run",
        client=cast(LLMClient, FakeClient("")),
    )
    assert report.summary.tasks == 0
    assert report.split == "none"
    payload = json.loads((tmp_path / "run" / "baseline.json").read_text(encoding="utf-8"))
    assert payload["summary"]["tasks"] == 0


def test_render_markdown() -> None:
    row = BaselineRow(
        task_id="x",
        family="f",
        split="dev",
        score=1.0,
        passed=True,
        error=None,
        duration_s=1.0,
        eval_count=10,
        prompt_eval_count=5,
        response_chars=20,
        extracted=True,
        artifact="candidates/x.py",
    )
    from janet.bench.baseline import BaselineReport, BaselineSummary

    report = BaselineReport(
        run_id="run",
        model="m",
        split="dev",
        rows=(row,),
        summary=BaselineSummary(1, 1, 1.0, 10, 5, 1.0, 10.0),
    )
    text = render_markdown(report)
    assert "`x`" in text
    assert "Mean score: 1.000" in text
