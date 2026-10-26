from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from janet.config import Config, DaemonConfig, PathsConfig
from janet.daemon import (
    IdlePolicy,
    IdleStatus,
    _daemon_command,
    _is_our_daemon,
    _pid_alive,
    _read_cmdlines,
    _read_pid,
    daemon_start,
    daemon_status,
    daemon_stop,
    format_status,
    idle_policy,
    idle_status,
    is_foreground_janet,
    other_janet_commands,
    pid_path,
    run_foreground,
    state_path,
    wait_interruptible,
)
from janet.loop.passive import PassiveState
from janet.probe import GIB, Budget

_POLICY = IdlePolicy(
    max_load=6.0,
    min_free_mem_bytes=4 * GIB,
    min_free_disk_bytes=3 * GIB,
)


def _config(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            runs=tmp_path / "runs",
            archive=tmp_path / "archive",
            events=tmp_path / "events.jsonl",
            evolved=tmp_path / "evolved",
            daemon=tmp_path / "daemon",
        ),
        daemon=DaemonConfig(idle_seconds=0.0, check_seconds=1.0),
    )


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["janet", "run"], True),
        (["/home/me/.venv/bin/janet", "bench", "baseline"], True),
        (["python", "-m", "janet", "evolve"], True),
        (["uv", "run", "janet", "calibrate"], True),
        (["janet", "daemon", "run"], False),
        (["janet", "probe"], False),
        (["python", "worker.py"], False),
        (["janet"], False),
    ],
)
def test_is_foreground_janet(argv: list[str], expected: bool) -> None:
    assert is_foreground_janet(argv) is expected


def test_other_janet_commands_excludes_self_and_daemon() -> None:
    commands = [
        (10, ["janet", "run"]),
        (11, ["janet", "daemon", "run"]),
        (12, ["python", "-m", "janet", "bench"]),
        (13, ["pytest"]),
    ]
    found = other_janet_commands(commands, exclude_pid=12)
    assert found == ["janet run"]


def test_read_cmdlines_from_a_fake_proc(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "10").mkdir(parents=True)
    (proc / "10" / "cmdline").write_bytes(b"janet\0run\0")
    (proc / "11").mkdir()
    (proc / "11" / "cmdline").write_bytes(b"")
    (proc / "not-a-pid").mkdir()
    commands = _read_cmdlines(proc)
    assert commands == [(10, ["janet", "run"])]


def test_idle_status_reports_the_first_blocker() -> None:
    assert idle_status(_POLICY, load1=0.1, mem_available=8 * GIB, disk_free=8 * GIB, busy=[]).idle
    assert not idle_status(
        _POLICY, load1=0.1, mem_available=8 * GIB, disk_free=8 * GIB, busy=["janet run"]
    ).idle
    assert not idle_status(
        _POLICY, load1=9.0, mem_available=8 * GIB, disk_free=8 * GIB, busy=[]
    ).idle
    assert not idle_status(
        _POLICY, load1=0.1, mem_available=1 * GIB, disk_free=8 * GIB, busy=[]
    ).idle
    assert not idle_status(
        _POLICY, load1=0.1, mem_available=8 * GIB, disk_free=1 * GIB, busy=[]
    ).idle


def test_idle_status_payload_and_policy(tmp_path: Path, budget: Budget) -> None:
    config = _config(tmp_path)
    policy = idle_policy(config, budget)
    assert policy.max_load == budget.cpu_cores * config.daemon.max_load_fraction
    status = idle_status(policy, load1=0.0, mem_available=99 * GIB, disk_free=99 * GIB, busy=[])
    assert status.idle
    assert status.to_dict()["idle"] is True


def test_wait_interruptible_stops_immediately() -> None:
    assert wait_interruptible(10.0, lambda: True) is False
    assert wait_interruptible(0.0, None) is True


def test_pid_helpers(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert _read_pid(config) is None
    assert not _pid_alive(None)
    assert not _pid_alive(999_999)
    path = pid_path(config)
    path.parent.mkdir(parents=True)
    path.write_text("not a pid\n", encoding="utf-8")
    assert _read_pid(config) is None


def test_daemon_status_reads_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    state_path(config).parent.mkdir(parents=True)
    PassiveState(sessions=2, jobs={"evolve": 2}).save(state_path(config))
    monkeypatch.setattr("janet.daemon._unit_active", lambda: False)
    monkeypatch.setattr("janet.daemon._pid_alive", lambda _pid: False)
    payload = daemon_status(config)
    assert payload["running"] is False
    state = payload["state"]
    assert isinstance(state, dict)
    assert state["sessions"] == 2
    assert "evolve" in format_status(payload)


def test_daemon_start_refuses_a_running_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("janet.daemon._unit_active", lambda: True)
    payload = daemon_start(_config(tmp_path), None)
    assert payload["started"] is False


def test_daemon_start_uses_systemd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("janet.daemon._unit_active", lambda: False)
    monkeypatch.setattr("janet.daemon._pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        "janet.daemon.shutil.which",
        lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else "/usr/bin/janet",
    )
    captured: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.append(argv)
        return subprocess.CompletedProcess(argv, 0, "Running as unit", "")

    monkeypatch.setattr("janet.daemon.subprocess.run", fake_run)
    payload = daemon_start(config, None)
    assert payload["started"] is True
    assert payload["mode"] == "systemd"
    assert captured
    joined = " ".join(captured[0])
    assert "CPUQuota=300%" in joined
    assert "MemoryMax=10G" in joined
    assert "/usr/bin/janet daemon run" in joined


def test_daemon_start_falls_back_to_a_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("janet.daemon._unit_active", lambda: False)
    monkeypatch.setattr("janet.daemon._pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        "janet.daemon.shutil.which",
        lambda name: "/usr/bin/janet" if name == "janet" else None,
    )
    monkeypatch.setattr("janet.daemon.time.sleep", lambda _seconds: None)

    class FakeProcess:
        pid = 4242

    monkeypatch.setattr("janet.daemon.subprocess.Popen", lambda *_args, **_kwargs: FakeProcess())
    payload = daemon_start(config, None)
    assert payload["started"] is True
    assert payload["mode"] == "process"
    assert payload["pid"] == 4242


def test_daemon_stop_stops_the_unit_and_removes_the_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    pid_path(config).parent.mkdir(parents=True)
    pid_path(config).write_text("123\n", encoding="utf-8")
    monkeypatch.setattr("janet.daemon._unit_active", lambda: True)
    monkeypatch.setattr(
        "janet.daemon._systemctl",
        lambda *_args: subprocess.CompletedProcess([], 0, "", ""),
    )
    monkeypatch.setattr("janet.daemon._pid_alive", lambda _pid: False)
    payload = daemon_stop(config)
    assert payload["stopped"] is True
    assert payload["unit_stopped"] is True
    assert not pid_path(config).exists()


def test_daemon_stop_kills_a_detached_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    pid_path(config).parent.mkdir(parents=True)
    pid_path(config).write_text("123\n", encoding="utf-8")
    monkeypatch.setattr("janet.daemon._unit_active", lambda: False)
    calls = iter([True, False])
    monkeypatch.setattr("janet.daemon._pid_alive", lambda _pid: next(calls, False))
    monkeypatch.setattr("janet.daemon._is_our_daemon", lambda _pid: True)
    monkeypatch.setattr("janet.daemon.time.sleep", lambda _seconds: None)
    signals: list[int] = []
    monkeypatch.setattr("janet.daemon.os.kill", lambda _pid, sig: signals.append(sig))
    payload = daemon_stop(config)
    assert payload["stopped"] is True
    assert signals


def test_daemon_stop_ignores_a_recycled_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    pid_path(config).parent.mkdir(parents=True)
    pid_path(config).write_text("123\n", encoding="utf-8")
    monkeypatch.setattr("janet.daemon._unit_active", lambda: False)
    monkeypatch.setattr("janet.daemon._pid_alive", lambda _pid: True)
    monkeypatch.setattr("janet.daemon._is_our_daemon", lambda _pid: False)
    signals: list[int] = []
    monkeypatch.setattr("janet.daemon.os.kill", lambda _pid, sig: signals.append(sig))
    payload = daemon_stop(config)
    assert payload["stopped"] is False
    assert signals == []
    assert not pid_path(config).exists()


def test_is_our_daemon_checks_the_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "janet.daemon._read_cmdlines",
        lambda: [(7, ["janet", "daemon", "run"]), (8, ["pytest"])],
    )
    assert _is_our_daemon(7) is True
    assert _is_our_daemon(8) is False
    assert _is_our_daemon(None) is False


def test_daemon_command_without_a_config_is_module_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("janet.daemon.shutil.which", lambda _name: None)
    command = _daemon_command(None)
    assert command[:3] == [sys.executable, "-m", "janet"]
    assert command[3:] == ["daemon", "run"]


def test_run_foreground_disabled_returns_zero(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = Config(paths=config.paths, daemon=DaemonConfig(enabled=False))
    assert run_foreground(config) == 0


def test_run_foreground_honours_an_immediate_stop(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert run_foreground(config, stop=lambda: True) == 0
    assert not pid_path(config).exists()


def test_idle_status_dataclass_is_immutable() -> None:
    status = IdleStatus(True, "idle", 0.0, 1, 1)
    with pytest.raises(AttributeError):
        status.idle = False  # type: ignore[misc]
