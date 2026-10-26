from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from helpers import make_sign_task

from janet.cli import main
from janet.config import Config, PathsConfig
from janet.eval.judge import Calibration, Sample
from janet.eval.task import Task
from janet.harness.bandit import BanditState
from janet.loop.evolution import EvolutionSummary
from janet.loop.passive import SessionReport
from janet.preempt import Preempted
from janet.probe import Budget
from janet.suites import DevSuite, Suites


def _config(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            runs=tmp_path / "runs",
            archive=tmp_path / "archive",
            events=tmp_path / "events.jsonl",
            anchor=tmp_path / "anchor",
            fresh=tmp_path / "fresh",
            evolved=tmp_path / "evolved",
            daemon=tmp_path / "daemon",
            bandit=tmp_path / "runs" / "bandit.json",
        )
    )


@pytest.fixture
def task(tmp_path: Path) -> Task:
    return make_sign_task(tmp_path / "tasks")


def test_evolve_command_prints_a_summary(
    tmp_path: Path, task: Task, monkeypatch: pytest.MonkeyPatch, budget: Budget
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("janet.cli._load", lambda _path: config)
    monkeypatch.setattr("janet.cli._budget", lambda _config: (object(), budget))
    monkeypatch.setattr("janet.cli._local_client", lambda _config, **_kw: object())
    monkeypatch.setattr(
        "janet.cli.load_dev_suite",
        lambda _config, **_kw: DevSuite(tasks=(task,), raw=(task,), fingerprint="", root=tmp_path),
    )
    summary = EvolutionSummary(
        tasks=1, accepted=1, mutants=4, killed_before=1, killed_after=2, results=()
    )
    called: dict[str, Any] = {}

    def fake_evolve(*_args: Any, **kwargs: Any) -> EvolutionSummary:
        called.update(kwargs)
        return summary

    monkeypatch.setattr("janet.cli.evolve_dev_suite", fake_evolve)
    assert main(["evolve", "--max-new", "2"]) == 0
    assert called["max_new_per_task"] == 2
    assert list(config.paths.runs.glob("evolve-*"))


def test_evolve_rejects_a_bad_limit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("janet.cli._load", lambda _path: Config())
    assert main(["evolve", "--limit", "0"]) == 2
    assert "limit" in capsys.readouterr().err


def test_evolve_rejects_an_empty_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("janet.cli._load", lambda _path: _config(tmp_path))
    monkeypatch.setattr(
        "janet.cli.load_dev_suite",
        lambda _config, **_kw: DevSuite(tasks=(), raw=(), fingerprint="", root=tmp_path),
    )
    assert main(["evolve"]) == 2
    assert "no dev tasks" in capsys.readouterr().err


def test_calibrate_writes_a_report(
    tmp_path: Path, task: Task, monkeypatch: pytest.MonkeyPatch, budget: Budget
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("janet.cli._load", lambda _path: config)
    monkeypatch.setattr("janet.cli._budget", lambda _config: (object(), budget))
    monkeypatch.setattr("janet.cli._local_client", lambda _config, **_kw: object())
    monkeypatch.setattr(
        "janet.cli.load_dev_suite",
        lambda _config, **_kw: DevSuite(tasks=(task,), raw=(task,), fingerprint="", root=tmp_path),
    )
    monkeypatch.setattr(
        "janet.cli.build_samples",
        lambda *_args, **_kw: [Sample(task.id, "source", True, "reference")],
    )
    monkeypatch.setattr(
        "janet.cli.calibrate",
        lambda *_args, **_kw: Calibration(2, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1),
    )
    assert main(["calibrate"]) == 0
    reports = list((config.paths.runs).glob("calibrate-*/calibration.json"))
    assert reports
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["calibration"]["trustworthy"] is True


def test_calibrate_rejects_a_bad_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("janet.cli._load", lambda _path: Config())
    assert main(["calibrate", "--limit", "0"]) == 2


def test_run_uses_the_evolved_suite(
    tmp_path: Path,
    task: Task,
    monkeypatch: pytest.MonkeyPatch,
    budget: Budget,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("janet.cli._load", lambda _path: config)
    monkeypatch.setattr("janet.cli._budget", lambda _config: (object(), budget))
    monkeypatch.setattr("janet.cli._local_client", lambda _config, **_kw: object())
    suites = Suites(dev=(task,), anchor=(task,), dev_suite="fp", anchor_root=tmp_path / "anchor")
    monkeypatch.setattr("janet.cli.load_suites", lambda _config: suites)
    seen: dict[str, Any] = {}

    def fake_loop(*_args: Any, **kwargs: Any) -> tuple[Any, ...]:
        seen.update(kwargs)
        return ()

    monkeypatch.setattr("janet.cli.run_loop", fake_loop)
    assert main(["run", "--generations", "1"]) == 0
    assert seen["dev_suite"] == "fp"
    assert seen["anchor_root"] == tmp_path / "anchor"
    assert json.loads(capsys.readouterr().out) == []


def test_bandit_command_reports_the_learned_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("janet.cli._load", lambda _path: config)
    state = BanditState.fresh()
    state.record("skill", 0.5)
    state.save(config.paths.bandit)
    assert main(["bandit", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_pulls"] == 1
    assert main(["bandit"]) == 0
    assert "skill" in capsys.readouterr().out


def test_daemon_status_returns_three_when_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("janet.cli._load", lambda _path: _config(tmp_path))
    monkeypatch.setattr(
        "janet.cli.daemon_status",
        lambda _config: {"running": False, "state": {"jobs": {}}},
    )
    assert main(["daemon", "status"]) == 3
    assert main(["daemon", "status", "--json"]) == 3


def test_daemon_start_stop_and_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("janet.cli._load", lambda _path: _config(tmp_path))
    monkeypatch.setattr("janet.cli.daemon_start", lambda _config, _path: {"started": True})
    monkeypatch.setattr("janet.cli.daemon_stop", lambda _config: {"stopped": True})
    monkeypatch.setattr("janet.cli.run_foreground", lambda *_a, **_kw: 2)
    assert main(["daemon", "start"]) == 0
    assert main(["daemon", "stop"]) == 0
    assert main(["daemon", "run"]) == 0


def test_daemon_once_prints_its_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("janet.cli._load", lambda _path: _config(tmp_path))
    monkeypatch.setattr(
        "janet.cli.run_once", lambda *_a, **_kw: SessionReport(jobs=(), preempted=False, seconds=1)
    )
    assert main(["daemon", "once"]) == 0


def test_preempted_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def preempted(_args: object) -> int:
        raise Preempted

    monkeypatch.setattr("janet.cli._cmd_probe", preempted)
    assert main(["probe"]) == 0
    assert "preempted" in capsys.readouterr().err
