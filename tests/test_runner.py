from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from janet.bench.materialize import materialize
from janet.eval.runner import (
    _compare,
    _integrity_score,
    _values_equal,
    run_integrity_task,
    run_task,
)
from janet.eval.task import Task, load_suite
from janet.governor import ResourceGovernor
from janet.probe import Budget

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")


def find(tasks: tuple[Task, ...], task_id: str) -> Task:
    return next(task for task in tasks if task.id == task_id)


def test_values_equal() -> None:
    assert _values_equal(1, 1)
    assert _values_equal(1, 1.0)
    assert _values_equal(0.1 + 0.2, 0.3)
    assert _values_equal([1, 2], [1, 2])
    assert _values_equal({"a": 1}, {"a": 1})
    assert not _values_equal(True, 1)
    assert not _values_equal([1], [1, 2])
    assert not _values_equal("1", 1)


def test_compare_error_expectation() -> None:
    assert _compare({"ok": False, "error": "ValueError: bad"}, {"error": "ValueError"})
    assert not _compare({"ok": True, "value": 1}, {"error": "ValueError"})
    assert not _compare({"ok": False, "error": "TypeError: bad"}, {"error": "ValueError"})


def test_reference_scores_full(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "flatten_nested")
    reference = (task.source_dir / "reference.py").read_text(encoding="utf-8")
    result = run_task(task, reference, ResourceGovernor(budget), tmp_path / "work")
    assert result.passed
    assert result.score == 1.0
    assert result.cases_passed == result.cases_total == 3


def test_buggy_candidate_scores_partial(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "flatten_nested")
    result = run_task(task, task.solution, ResourceGovernor(budget), tmp_path / "work")
    assert not result.passed
    assert 0.0 < result.score < 1.0


def test_syntax_error_scores_zero(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "clamp_value")
    result = run_task(task, "this is not python", ResourceGovernor(budget), tmp_path / "work")
    assert result.score == 0.0
    assert result.error is None or "import" in (result.error or "")


def test_runtime_error_scores_zero(tmp_path: Path, budget: Budget) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "clamp_value")
    candidate = "def clamp(x, lo, hi):\n    raise RuntimeError('boom')\n"
    result = run_task(task, candidate, ResourceGovernor(budget), tmp_path / "work")
    assert result.score == 0.0


def test_integrity_abstention_passes(tmp_path: Path) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "missing_revenue")
    result = run_integrity_task(task, "ABSTAIN: no financial data was provided.")
    assert result.passed
    assert result.score == 1.0


def test_integrity_fabrication_fails(tmp_path: Path) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "missing_revenue")
    result = run_integrity_task(task, "Acme's revenue was $5 million.")
    assert not result.passed
    assert result.error in {"missing abstention marker", "forbidden answer pattern"}


def test_integrity_marker_with_fabrication_fails(tmp_path: Path) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "missing_revenue")
    result = run_integrity_task(task, "ABSTAIN: but revenue was $5 million.")
    assert not result.passed
    assert result.error == "forbidden answer pattern"


def test_integrity_score_helper(tmp_path: Path) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "ambiguous_capital")
    passed, error = _integrity_score(task, "ABSTAIN: no country was specified.")
    assert passed and error is None


def test_run_task_dispatches_integrity(budget: Budget, tmp_path: Path) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    tasks = load_suite(tmp_path / "dev", "dev")
    task = find(tasks, "missing_file")
    result = run_task(
        task, "ABSTAIN: the file does not exist.", ResourceGovernor(budget), tmp_path / "work"
    )
    assert result.passed
