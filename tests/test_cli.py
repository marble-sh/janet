from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from janet.bench.seeds import SEEDS
from janet.cli import main
from janet.llm import LLMResponse


def parse_json_output(output: str) -> dict[str, Any]:
    payload = json.loads(output)
    assert isinstance(payload, dict)
    return payload


def test_probe_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["probe", "--json"]) == 0
    payload = parse_json_output(capsys.readouterr().out)
    assert "machine" in payload
    assert "budget" in payload
    budget = payload["budget"]
    assert isinstance(budget, dict)
    assert budget["cpu_cores"] >= 1


def test_dry_run_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--dry-run", "--config", "configs/default.toml"]) == 0
    payload = parse_json_output(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["generations"] == 24
    assert "config" in payload


def test_run_requires_a_materialized_anchor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[paths]\n"
        f"runs = {json.dumps(str(tmp_path / 'runs'))}\n"
        f"anchor = {json.dumps(str(tmp_path / 'missing-anchor'))}\n",
        encoding="utf-8",
    )
    assert main(["run", "--config", str(config)]) == 2
    captured = capsys.readouterr()
    assert "anchor manifest check failed" in captured.err


def test_doctor_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["doctor", "--json"])
    payload = parse_json_output(capsys.readouterr().out)
    assert code in (0, 1)
    assert isinstance(payload["checks"], list)
    names = {check["name"] for check in payload["checks"] if isinstance(check, dict)}
    assert {"python", "bwrap", "sandbox", "models"} <= names


def test_missing_config_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["probe", "--config", "/nonexistent/janet.toml"]) == 2
    assert "config file not found" in capsys.readouterr().err


def test_version_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "janet" in capsys.readouterr().out


def test_no_command_exits_with_usage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_default_config_path_is_relative_to_repo() -> None:
    assert Path("configs/default.toml").is_file()


def test_bench_tasks_lists_seeds(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "tasks"]) == 0
    out = capsys.readouterr().out
    assert "clamp_value" in out
    assert "missing_revenue" in out


def test_bench_materialize_writes_and_verifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "suite"
    assert (
        main(
            [
                "bench",
                "materialize",
                "--root",
                str(root),
                "--split",
                "dev",
                "--force",
            ]
        )
        == 0
    )
    payload = parse_json_output(capsys.readouterr().out)
    assert payload["manifest_ok"] is True
    assert payload["task_count"] == 20
    assert (root / "manifest.json").is_file()


class _FakeModel:
    def __init__(self, _config: object) -> None:
        self._reference = next(seed for seed in SEEDS if seed.id == "binary_search_index").reference

    def list_models(self) -> list[str]:
        return ["qwen2.5-coder:7b-instruct-q4_K_M"]

    def generate(self, _prompt: str, **_kwargs: Any) -> LLMResponse:
        return LLMResponse(
            text=self._reference,
            model="fake",
            eval_count=1,
            prompt_eval_count=1,
            eval_duration_ns=1,
            total_duration_ns=1,
        )


def test_bench_baseline_with_a_fake_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import shutil

    run_id = "test-bench-cli"
    run_dir = Path("runs") / run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    monkeypatch.setattr("janet.cli.LLMClient", _FakeModel)
    try:
        assert (
            main(["bench", "baseline", "--limit", "1", "--run-id", run_id, "--min-difficulty", "1"])
            == 0
        )
        payload = parse_json_output(capsys.readouterr().out)
        assert payload["tasks"] == 1
        assert payload["passed"] == 1
        assert (run_dir / "baseline.json").is_file()
        assert (run_dir / "candidates" / "binary_search_index.py").is_file()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
