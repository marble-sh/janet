from __future__ import annotations

import json
from pathlib import Path

import pytest

from janet.eval import worker


def _solution(tmp_path: Path, code: str) -> Path:
    path = tmp_path / "solution.py"
    path.write_text(code, encoding="utf-8")
    return path


def _inputs(tmp_path: Path, payload: str) -> Path:
    path = tmp_path / "inputs.json"
    path.write_text(payload, encoding="utf-8")
    return path


def test_worker_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    solution = _solution(tmp_path, "def add(a, b):\n    return a + b\n")
    inputs = _inputs(tmp_path, "[[1, 2], [3, 4]]")
    assert worker.main([str(solution), str(inputs), "add"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"] == [
        {"ok": True, "value": 3},
        {"ok": True, "value": 7},
    ]


def test_worker_requires_three_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert worker.main([]) == 2
    assert "usage" in capsys.readouterr().err


def test_worker_rejects_bad_inputs_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    solution = _solution(tmp_path, "def f():\n    return 1\n")
    inputs = _inputs(tmp_path, "not json")
    assert worker.main([str(solution), str(inputs), "f"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"].startswith("inputs:")


def test_worker_rejects_non_list_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    solution = _solution(tmp_path, "def f():\n    return 1\n")
    inputs = _inputs(tmp_path, "{}")
    assert worker.main([str(solution), str(inputs), "f"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "inputs must be a list"


def test_worker_reports_import_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    solution = _solution(tmp_path, "this is not python")
    inputs = _inputs(tmp_path, "[[]]")
    assert worker.main([str(solution), str(inputs), "f"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"].startswith("import:")


def test_worker_reports_a_non_callable_entrypoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    solution = _solution(tmp_path, "value = 3\n")
    inputs = _inputs(tmp_path, "[[]]")
    assert worker.main([str(solution), str(inputs), "value"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "not callable" in payload["error"]


def test_worker_reports_case_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    solution = _solution(tmp_path, "def f(x):\n    raise RuntimeError('boom')\n")
    inputs = _inputs(tmp_path, "[[1], 2]")
    assert worker.main([str(solution), str(inputs), "f"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["ok"] is False
    assert "RuntimeError" in payload["results"][0]["error"]
    assert payload["results"][1] == {"ok": False, "error": "case args must be a list"}
