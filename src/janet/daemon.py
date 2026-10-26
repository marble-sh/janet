"""Idle-only background daemon for Janet.

The daemon is an ordinary Janet process with three properties:

* it starts a work session only after the machine has been idle for a while and
  no other Janet work is running;
* every job is bounded by the session deadline and yields at the next task
  boundary as soon as foreground work appears;
* ``janet daemon start`` wraps it in a transient systemd user unit with CPU,
  memory, and nice caps, so even a bug cannot exceed the resource contract.

It is not a second scheduler and it does not bypass the governor. It is a polite
tenant of the idle time the governor would otherwise leave unused.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from janet.config import Config, ConfigError
from janet.eval.task import TaskError
from janet.events import EventLog
from janet.governor import ResourceGovernor
from janet.llm import LLMClient, LLMError
from janet.loop.passive import PassiveState, SessionContext, SessionReport, run_session
from janet.preempt import StopCheck, StopFlag
from janet.probe import GIB, Budget, compute_budget, parse_meminfo, probe_machine
from janet.suites import load_dev_suite, load_fresh, load_suites

logger = logging.getLogger(__name__)

UNIT = "janet-daemon"
_WORK_SUBCOMMANDS = frozenset({"run", "bench", "archive", "evolve", "calibrate"})


class DaemonError(RuntimeError):
    """Raised when the daemon cannot start, stop, or take its lock."""


@dataclass(frozen=True, slots=True)
class IdlePolicy:
    max_load: float
    min_free_mem_bytes: int
    min_free_disk_bytes: int


@dataclass(frozen=True, slots=True)
class IdleStatus:
    idle: bool
    reason: str
    load: float
    free_mem_bytes: int
    free_disk_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "idle": self.idle,
            "reason": self.reason,
            "load": round(self.load, 3),
            "free_mem_gib": round(self.free_mem_bytes / GIB, 2),
            "free_disk_gib": round(self.free_disk_bytes / GIB, 2),
        }


def daemon_dir(config: Config) -> Path:
    return config.paths.daemon.expanduser()


def state_path(config: Config) -> Path:
    return daemon_dir(config) / "state.json"


def pid_path(config: Config) -> Path:
    return daemon_dir(config) / "daemon.pid"


def _read_cmdlines(proc_root: Path = Path("/proc")) -> list[tuple[int, list[str]]]:
    commands: list[tuple[int, list[str]]] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return commands
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if argv:
            commands.append((int(entry.name), argv))
    return commands


def is_foreground_janet(argv: Sequence[str]) -> bool:
    tokens = [Path(token).name for token in argv]
    try:
        index = tokens.index("janet")
    except ValueError:
        return False
    rest = tokens[index + 1 :]
    if not rest or rest[0] == "daemon":
        return False
    return rest[0] in _WORK_SUBCOMMANDS


def other_janet_commands(
    commands: Sequence[tuple[int, Sequence[str]]] | None = None,
    *,
    exclude_pid: int | None = None,
) -> list[str]:
    """Command lines of other Janet work, excluding the daemon itself."""

    if commands is None:
        commands = _read_cmdlines()
    found: list[str] = []
    for pid, argv in commands:
        if exclude_pid is not None and pid == exclude_pid:
            continue
        if is_foreground_janet(argv):
            found.append(" ".join(argv))
    return found


def _available_mem() -> int:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return 0
    return parse_meminfo(text).available_bytes


def _free_disk() -> int:
    try:
        return shutil.disk_usage(Path.cwd()).free
    except OSError:
        return 0


def idle_policy(config: Config, budget: Budget) -> IdlePolicy:
    return IdlePolicy(
        max_load=budget.cpu_cores * config.daemon.max_load_fraction,
        min_free_mem_bytes=int(config.daemon.min_free_mem_gib * GIB),
        min_free_disk_bytes=int(config.daemon.min_free_disk_gib * GIB),
    )


def idle_status(
    policy: IdlePolicy,
    *,
    load1: float | None = None,
    mem_available: int | None = None,
    disk_free: int | None = None,
    busy: Sequence[str] | None = None,
) -> IdleStatus:
    """Whether the machine is idle enough to start background work."""

    busy = other_janet_commands(exclude_pid=os.getpid()) if busy is None else list(busy)
    load = os.getloadavg()[0] if load1 is None else load1
    free_mem = _available_mem() if mem_available is None else mem_available
    free_disk = _free_disk() if disk_free is None else disk_free
    if busy:
        return IdleStatus(False, f"foreground janet: {busy[0]}", load, free_mem, free_disk)
    if load > policy.max_load:
        return IdleStatus(
            False,
            f"load {load:.2f} > {policy.max_load:.2f}",
            load,
            free_mem,
            free_disk,
        )
    if free_mem < policy.min_free_mem_bytes:
        return IdleStatus(
            False,
            f"free memory {free_mem / GIB:.2f} GiB < {policy.min_free_mem_bytes / GIB:.2f} GiB",
            load,
            free_mem,
            free_disk,
        )
    if free_disk < policy.min_free_disk_bytes:
        return IdleStatus(
            False,
            f"free disk {free_disk / GIB:.2f} GiB < {policy.min_free_disk_bytes / GIB:.2f} GiB",
            load,
            free_mem,
            free_disk,
        )
    return IdleStatus(True, "idle", load, free_mem, free_disk)


def _write_pid(config: Config, pid: int) -> None:
    path = pid_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="utf-8")


def _read_pid(config: Config) -> int | None:
    path = pid_path(config)
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_our_daemon(pid: int | None) -> bool:
    """Guard against killing a recycled PID from a stale pid file."""

    if pid is None:
        return False
    for candidate, argv in _read_cmdlines():
        if candidate != pid:
            continue
        tokens = [Path(token).name for token in argv]
        return "janet" in tokens and "daemon" in tokens
    return False


@contextmanager
def _lock(config: Config) -> Iterator[None]:
    path = daemon_dir(config) / "daemon.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DaemonError("another janet daemon already holds the lock") from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def wait_interruptible(seconds: float, stop: StopCheck | None) -> bool:
    """Sleep, returning False if a stop was requested."""

    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        if stop is not None and stop():
            return False
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    return stop is None or not stop()


def _session_context(config: Config) -> SessionContext:
    suites = load_suites(config)
    raw = load_dev_suite(config, include_evolved=False).raw
    try:
        fresh = load_fresh(config)
    except ConfigError:
        fresh = ()
    return SessionContext(
        dev=suites.dev,
        raw_dev=raw,
        anchor=suites.anchor,
        fresh=fresh,
        dev_suite=suites.dev_suite,
    )


def run_once(
    config: Config,
    *,
    session_seconds: float | None = None,
    stop: StopCheck | None = None,
) -> SessionReport:
    """Run one ladder session now, regardless of idle state (the ``once`` command)."""

    machine = probe_machine()
    budget = compute_budget(machine, config.budget)
    governor = ResourceGovernor(budget)
    client = LLMClient(config.llm)
    path = state_path(config)
    state = PassiveState.load(path)
    log = EventLog(config.paths.events.expanduser(), run_id="daemon")
    context = _session_context(config)
    report = run_session(
        config,
        client,
        governor,
        context,
        state,
        path,
        daemon_dir(config) / "runs",
        stop=stop,
        session_seconds=session_seconds,
        log=log,
    )
    state.sessions += 1
    state.save(path)
    return report


def run_foreground(
    config: Config,
    *,
    session_seconds: float | None = None,
    max_sessions: int | None = None,
    stop: StopCheck | None = None,
    install_signals: bool = True,
) -> int:
    """The daemon's main loop; returns the number of sessions started."""

    if not config.daemon.enabled:
        logger.info("daemon is disabled in configuration")
        return 0
    machine = probe_machine()
    budget = compute_budget(machine, config.budget)
    policy = idle_policy(config, budget)
    governor = ResourceGovernor(budget)
    client = LLMClient(config.llm)
    directory = daemon_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    path = state_path(config)
    state = PassiveState.load(path)
    log = EventLog(config.paths.events.expanduser(), run_id="daemon")
    flag = StopFlag()

    def stopped() -> bool:
        return flag() or (stop is not None and stop())

    def session_stop() -> bool:
        if stopped():
            return True
        # Yield to foreground Janet work at the next task boundary.
        if other_janet_commands(exclude_pid=os.getpid()):
            return True
        try:
            # The daemon's own work must not trip this; only an overloaded machine
            # (external work plus the daemon) does.
            return os.getloadavg()[0] > budget.cpu_cores_spike
        except OSError:
            return False

    previous: dict[signal.Signals, Any] = {}
    if install_signals and threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_args: flag.request())
    sessions = 0
    idle_since: float | None = None
    wrote_pid = False
    try:
        with _lock(config):
            _write_pid(config, os.getpid())
            wrote_pid = True
            while not stopped():
                status = idle_status(policy, busy=other_janet_commands(exclude_pid=os.getpid()))
                if not status.idle:
                    idle_since = None
                    log.emit("daemon.idle_check", **status.to_dict())
                    if not wait_interruptible(config.daemon.check_seconds, stopped):
                        break
                    continue
                if idle_since is None:
                    idle_since = time.monotonic()
                waited = time.monotonic() - idle_since
                if waited < config.daemon.idle_seconds:
                    log.emit("daemon.waiting", seconds=round(waited, 1), **status.to_dict())
                    if not wait_interruptible(
                        min(config.daemon.check_seconds, config.daemon.idle_seconds - waited),
                        stopped,
                    ):
                        break
                    continue
                try:
                    client.list_models()
                except LLMError as exc:
                    idle_since = None
                    log.emit("daemon.model_unavailable", error=str(exc))
                    if not wait_interruptible(max(60.0, config.daemon.check_seconds), stopped):
                        break
                    continue
                try:
                    context = _session_context(config)
                except (ConfigError, TaskError) as exc:
                    idle_since = None
                    log.emit("daemon.suite_error", error=str(exc))
                    if not wait_interruptible(300.0, stopped):
                        break
                    continue
                log.emit("daemon.session_start", sessions=state.sessions + 1)
                report = run_session(
                    config,
                    client,
                    governor,
                    context,
                    state,
                    path,
                    directory / "runs",
                    stop=session_stop,
                    session_seconds=session_seconds,
                    log=log,
                )
                state.sessions += 1
                state.save(path)
                sessions += 1
                log.emit(
                    "daemon.session_finished",
                    sessions=state.sessions,
                    **report.to_dict(),
                )
                idle_since = None
                if max_sessions is not None and sessions >= max_sessions:
                    break
                if not wait_interruptible(config.daemon.cooldown_seconds, stopped):
                    break
    finally:
        if wrote_pid and _read_pid(config) == os.getpid():
            pid_path(config).unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return sessions


def _systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _unit_active() -> bool:
    return _systemctl("is-active", "--quiet", UNIT).returncode == 0


def _daemon_command(config_path: str | None) -> list[str]:
    executable = shutil.which("janet")
    command = [executable] if executable else [sys.executable, "-m", "janet"]
    command += ["daemon", "run"]
    if config_path is not None:
        command += ["--config", str(Path(config_path).resolve())]
    return command


def daemon_start(config: Config, config_path: str | None) -> dict[str, object]:
    """Start the daemon in a transient systemd unit, falling back to a detached process."""

    directory = daemon_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    if _unit_active() or _pid_alive(_read_pid(config)):
        return {"started": False, "reason": "already running", "unit": UNIT}
    command = _daemon_command(config_path)
    runner = shutil.which("systemd-run")
    if runner is not None:
        completed = subprocess.run(
            [
                runner,
                "--user",
                f"--unit={UNIT}",
                "--collect",
                f"--property=MemoryMax={config.daemon.memory_max_gib:g}G",
                f"--property=MemoryHigh={config.daemon.memory_high_gib:g}G",
                f"--property=CPUQuota={config.daemon.cpu_quota_percent}%",
                f"--property=Nice={config.daemon.nice}",
                "--property=TimeoutStopSec=180",
                f"--property=WorkingDirectory={Path.cwd()}",
                "--",
                *command,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            return {
                "started": True,
                "mode": "systemd",
                "unit": UNIT,
                "detail": completed.stdout.strip() or completed.stderr.strip(),
            }
        logger.warning("systemd-run failed: %s", completed.stderr.strip())
    log_file = directory / "daemon.log"
    handle = log_file.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        handle.close()
    time.sleep(1.0)
    return {"started": True, "mode": "process", "pid": process.pid, "log": str(log_file)}


def daemon_stop(config: Config) -> dict[str, object]:
    """Stop the daemon, whether it is a unit or a detached process."""

    stopped_unit = False
    if _unit_active():
        stopped_unit = _systemctl("stop", UNIT).returncode == 0
    pid = _read_pid(config)
    killed = False
    if _pid_alive(pid) and _is_our_daemon(pid):
        assert pid is not None
        try:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
        except OSError:
            logger.exception("failed to signal the daemon")
        for _ in range(20):
            if not _pid_alive(pid):
                break
            time.sleep(0.25)
        if _pid_alive(pid):
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        killed = True
    pid_path(config).unlink(missing_ok=True)
    return {"stopped": stopped_unit or killed, "unit_stopped": stopped_unit, "pid": pid}


def daemon_status(config: Config) -> dict[str, object]:
    pid = _read_pid(config)
    state = PassiveState.load(state_path(config))
    active = _unit_active()
    running = active or _pid_alive(pid)
    return {
        "running": running,
        "unit": UNIT,
        "unit_active": active,
        "pid": pid,
        "pid_alive": _pid_alive(pid),
        "state": {
            "sessions": state.sessions,
            "generation": state.generation,
            "cursor": state.cursor,
            "current_hash": state.current_hash,
            "jobs": state.jobs,
            "last_confirm_hash": state.last_confirm_hash,
            "last_confirm_at": state.last_confirm_at,
            "last_calibrate_at": state.last_calibrate_at,
            "last_error": state.last_error,
        },
    }


def format_status(payload: dict[str, object]) -> str:
    """A compact human-readable status line."""

    state = payload.get("state")
    jobs = state.get("jobs", {}) if isinstance(state, dict) else {}
    return json.dumps(
        {
            "running": payload.get("running"),
            "unit_active": payload.get("unit_active"),
            "pid": payload.get("pid"),
            "jobs": jobs,
        },
        sort_keys=True,
    )
