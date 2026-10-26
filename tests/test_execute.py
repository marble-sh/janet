from __future__ import annotations

from pathlib import Path

import pytest

from janet.eval.execute import execute_cases
from janet.governor import ResourceGovernor
from janet.probe import Budget

_ADD = "def add(a, b):\n    return a + b\n"


def test_execute_cases_returns_one_record_per_input(tmp_path: Path, budget: Budget) -> None:
    workspace = tmp_path / "work"
    results = execute_cases(_ADD, "add", [[1, 2], [3, 4]], ResourceGovernor(budget), workspace)
    assert results == [
        {"ok": True, "value": 3},
        {"ok": True, "value": 7},
    ]
    assert (workspace / "solution.py").is_file()
    assert (workspace / "inputs.json").is_file()


def test_execute_cases_records_function_errors(tmp_path: Path, budget: Budget) -> None:
    source = "def f(x):\n    raise ValueError('bad')\n"
    results = execute_cases(source, "f", [[1]], ResourceGovernor(budget), tmp_path / "work")
    assert results is not None
    assert results[0]["ok"] is False
    assert "ValueError" in results[0]["error"]


def test_execute_cases_returns_none_on_import_failure(tmp_path: Path, budget: Budget) -> None:
    results = execute_cases(
        "this is not python", "f", [[1]], ResourceGovernor(budget), tmp_path / "work"
    )
    assert results is None


def test_execute_cases_returns_none_on_timeout(tmp_path: Path, budget: Budget) -> None:
    source = "def f(x):\n    while True:\n        pass\n"
    results = execute_cases(
        source,
        "f",
        [[1]],
        ResourceGovernor(budget),
        tmp_path / "work",
        timeout_s=0.5,
    )
    assert results is None


def test_execute_cases_replaces_a_stale_workspace(tmp_path: Path, budget: Budget) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "stale.txt").write_text("old", encoding="utf-8")
    results = execute_cases(_ADD, "add", [[1, 1]], ResourceGovernor(budget), workspace)
    assert results == [{"ok": True, "value": 2}]
    assert not (workspace / "stale.txt").exists()


def test_execute_cases_handles_unserialisable_values(tmp_path: Path, budget: Budget) -> None:
    source = "def f(x):\n    return {1, 2}\n"
    results = execute_cases(source, "f", [[1]], ResourceGovernor(budget), tmp_path / "work")
    assert results is not None
    assert results[0]["ok"] is True
    assert isinstance(results[0]["value"], str)


@pytest.mark.parametrize("entrypoint", ["missing", "value"])
def test_execute_cases_returns_none_for_bad_entrypoints(
    entrypoint: str, tmp_path: Path, budget: Budget
) -> None:
    source = "value = 3\n"
    results = execute_cases(source, entrypoint, [[1]], ResourceGovernor(budget), tmp_path / "work")
    assert results is None
