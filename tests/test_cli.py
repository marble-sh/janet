from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from janet.bench.seeds import SEEDS
from janet.cli import _memory_retriever, main
from janet.config import Config, MemoryConfig, PathsConfig
from janet.llm import LLMResponse
from janet.memory import ProcessMemory


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
    assert payload["task_count"] == 26
    assert (root / "manifest.json").is_file()


def test_bench_headroom_reports_hard_tasks(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "headroom", "--json"]) == 0
    payload = parse_json_output(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dev_hard"] >= 2
    assert payload["anchor_hard"] >= 4
    assert payload["fresh_hard"] >= 2
    assert "headroom" in payload["hard_families"]


def test_memory_retriever_respects_the_override(tmp_path: Path) -> None:
    config = Config(
        paths=PathsConfig(memory=tmp_path / "memory.sqlite3"),
        memory=MemoryConfig(enabled=False),
    )
    assert _memory_retriever(config, (), (), None) == (None, None)
    assert _memory_retriever(config, (), (), "off") == (None, None)
    retriever, store = _memory_retriever(config, (), (), "on")
    assert retriever is not None
    assert store is not None
    store.close()


def test_memory_consolidate_merges_and_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _memory_config(tmp_path)
    with ProcessMemory(tmp_path / "memory.sqlite3") as store:
        for triggers in ("duration suffix", "duration suffix seconds"):
            lesson_id = store.add_lesson(
                task_id="parse_duration",
                task_family="code_repair",
                failure_signature="ValueError on an invalid suffix",
                trigger_terms=triggers,
                rule="Validate the duration suffix before multiplying the numeric value.",
                patch_summary="Reject the suffix.",
                provenance={"run": "run-1"},
            )
            store.set_status("lesson", lesson_id, "verified")
    assert main(["memory", "consolidate", "--config", str(config), "--json"]) == 0
    payload = parse_json_output(capsys.readouterr().out)
    assert payload["merged_lessons"] == 1
    assert payload["kept_lessons"] == 1


def _memory_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.toml"
    config.write_text(
        f"[paths]\nmemory = {json.dumps(str(tmp_path / 'memory.sqlite3'))}\n",
        encoding="utf-8",
    )
    return config


def _seed_memory(path: Path) -> None:
    with ProcessMemory(path) as store:
        lesson_id = store.add_lesson(
            task_id="parse_duration",
            task_family="code_repair",
            failure_signature="timeout on invalid suffix",
            trigger_terms="duration timeout suffix",
            rule="Validate the suffix before multiplying the numeric value.",
            patch_summary="Reject an unknown duration suffix.",
            provenance={"run": "run-1"},
        )
        store.set_status("lesson", lesson_id, "verified")


def test_memory_stats_and_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _memory_config(tmp_path)
    _seed_memory(tmp_path / "memory.sqlite3")
    assert main(["memory", "stats", "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "verified=1" in out
    assert "evidence: 0" in out
    assert main(["memory", "verify", "--config", str(config), "--json"]) == 0
    payload = parse_json_output(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["integrity"] == "ok"


def test_memory_list_and_search(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _memory_config(tmp_path)
    _seed_memory(tmp_path / "memory.sqlite3")
    assert main(["memory", "list", "--config", str(config), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["task_id"] for row in rows] == ["parse_duration"]
    assert main(["memory", "search", "timeout", "--config", str(config), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert main(["memory", "list", "--config", str(config), "--limit", "0"]) == 2
    assert "--limit" in capsys.readouterr().err


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
