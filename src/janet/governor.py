"""Resource governance: rlimits, CPU affinity, process-tree monitoring, spikes.

The contract is a soft cap (a fraction of total hardware) with a rolling
allowance of time above it. Children get a hard rlimit at the spike ceiling so
a runaway cannot take the machine down; the monitor accounts for time above
the soft cap and kills only when the allowance is exhausted.
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

from janet.probe import GIB, MIB, Budget, MemInfo, parse_meminfo

_MAX_CAPTURE_BYTES = 1 * MIB


class GovernorError(RuntimeError):
    """Raised when the governor cannot enforce the configured contract."""


@dataclass(slots=True)
class SpikeBudget:
    """A rolling allowance of seconds spent above the soft cap."""

    allowance_seconds: float
    window_seconds: float = 3600.0
    _events: deque[tuple[float, float]] = field(default_factory=deque, repr=False)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def used(self, now: float) -> float:
        self._prune(now)
        return sum(duration for _, duration in self._events)

    def remaining(self, now: float) -> float:
        return max(0.0, self.allowance_seconds - self.used(now))

    def record(self, duration: float, now: float) -> bool:
        """Record time above the soft cap. Returns False if exhausted."""

        if duration <= 0.0:
            return self.remaining(now) > 0.0
        self._prune(now)
        self._events.append((now, duration))
        return self.used(now) <= self.allowance_seconds

    def exhausted(self, now: float) -> bool:
        return self.used(now) >= self.allowance_seconds

    def reset(self) -> None:
        self._events.clear()


@dataclass(frozen=True, slots=True)
class RunResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    killed_reason: str | None
    peak_rss_bytes: int
    cpu_seconds: float
    spike_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.killed_reason is None


@dataclass(frozen=True, slots=True)
class TreeStats:
    rss_bytes: int
    cpu_seconds: float


def _read_proc_status(pid: int) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0])
        except ValueError:
            continue
    return values


def _read_proc_stat(pid: int) -> tuple[int, float]:
    """Return (ppid, cpu_seconds) from /proc/<pid>/stat, or (-1, 0.0)."""

    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return (-1, 0.0)
    # The comm field can contain spaces and parentheses; split after the last ')'.
    close = text.rfind(")")
    if close == -1:
        return (-1, 0.0)
    fields = text[close + 2 :].split()
    if len(fields) < 13:
        return (-1, 0.0)
    try:
        ppid = int(fields[1])
        utime = int(fields[11])
        stime = int(fields[12])
    except (IndexError, ValueError):
        return (-1, 0.0)
    ticks = os.sysconf("SC_CLK_TCK")
    return (ppid, (utime + stime) / ticks)


def tree_stats(pid: int) -> TreeStats:
    """Sum RSS and CPU time over a process and all its descendants."""

    parents: dict[int, int] = {}
    cpu: dict[int, float] = {}
    rss: dict[int, int] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return TreeStats(0, 0.0)
    for entry in entries:
        if not entry.isdigit():
            continue
        child = int(entry)
        ppid, child_cpu = _read_proc_stat(child)
        if ppid == -1:
            continue
        parents[child] = ppid
        cpu[child] = child_cpu
        rss[child] = _read_proc_status(child).get("VmRSS", 0) * 1024

    descendants = {pid}
    changed = True
    while changed:
        changed = False
        for child, parent in parents.items():
            if parent in descendants and child not in descendants:
                descendants.add(child)
                changed = True
    return TreeStats(
        rss_bytes=sum(rss.get(child, 0) for child in descendants),
        cpu_seconds=sum(cpu.get(child, 0.0) for child in descendants),
    )


def _truncate(text: str) -> str:
    if len(text) <= _MAX_CAPTURE_BYTES:
        return text
    return text[:_MAX_CAPTURE_BYTES] + "\n...[truncated]"


def _clean_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": "/tmp",
    }
    if extra:
        env.update(extra)
    return env


class ResourceGovernor:
    """Runs child processes under the resource contract."""

    def __init__(
        self,
        budget: Budget,
        *,
        poll_interval_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.budget = budget
        self.poll_interval_s = poll_interval_s
        self._clock = clock
        self._sleep = sleep
        self.spikes = SpikeBudget(budget.spike_seconds_per_hour)
        self._lock = threading.Lock()

    def apply_to_self(self) -> None:
        """Constrain the orchestrator itself and set thread-pool defaults.

        Call this before importing numerical libraries: thread-pool environment
        variables are read at import time.
        """

        try:
            allowed = sorted(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - non-Linux
            allowed = list(range(os.cpu_count() or 1))
        chosen = allowed[: self.budget.cpu_cores]
        if chosen:
            try:
                os.sched_setaffinity(0, set(chosen))
            except OSError as exc:  # pragma: no cover - platform dependent
                raise GovernorError(f"cannot set CPU affinity: {exc}") from exc
        with contextlib.suppress(OSError):
            os.nice(10)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(name, str(self.budget.cpu_cores))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def should_yield(self) -> bool:
        """True when the machine is under enough pressure that we should pause."""

        try:
            mem: MemInfo = parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        reserve = int(self.budget.mem_bytes * 0.1)
        if mem.available_bytes < reserve:
            return True
        try:
            load = os.getloadavg()[0]
        except OSError:
            return False
        return load > 0.9 * self.budget.cpu_cores

    def _kill_tree(self, pid: int, reason: str) -> str:
        with self._lock:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                return reason
        deadline = self._clock() + 3.0
        while self._clock() < deadline:
            if not _process_alive(pid):
                return reason
            self._sleep(0.1)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        return reason

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout_s: float | None = None,
        mem_bytes: int | None = None,
        cpu_cores: int | None = None,
        stdin_text: str | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> RunResult:
        """Run a command under the contract and capture its output."""

        if not argv:
            raise GovernorError("empty command")
        timeout = min(
            self.budget.max_wall_seconds if timeout_s is None else timeout_s,
            self.budget.max_wall_seconds,
        )
        if timeout <= 0:
            raise GovernorError("timeout must be > 0")
        mem_hard = self.budget.mem_bytes_spike if mem_bytes is None else mem_bytes
        mem_hard = max(mem_hard, 64 * MIB)
        cores = self.budget.cpu_cores_spike if cpu_cores is None else cpu_cores
        try:
            allowed = sorted(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - non-Linux
            allowed = list(range(os.cpu_count() or 1))
        chosen = allowed[: max(1, min(cores, len(allowed)))]
        limit_argv = [
            sys.executable,
            "-m",
            "janet._limit",
            "--mem-bytes",
            str(mem_hard),
            "--cpu-seconds",
            str(math.ceil(timeout) + 5),
            "--cpus",
            ",".join(str(cpu) for cpu in chosen),
            "--",
            *argv,
        ]

        started = self._clock()
        try:
            process = subprocess.Popen(
                limit_argv,
                cwd=cwd,
                env=_clean_env(extra_env),
                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise GovernorError(f"cannot start {argv[0]!r}: {exc}") from exc

        peak_rss = 0
        peak_cpu = 0.0
        spike_seconds = 0.0
        killed_reason: str | None = None
        stop = threading.Event()
        last_sample = self._clock()
        last_cpu = 0.0

        def monitor() -> None:
            nonlocal peak_rss, peak_cpu, spike_seconds, killed_reason, last_sample, last_cpu
            while not stop.wait(self.poll_interval_s):
                if process.poll() is not None:
                    return
                stats = tree_stats(process.pid)
                peak_rss = max(peak_rss, stats.rss_bytes)
                peak_cpu = max(peak_cpu, stats.cpu_seconds)
                now = self._clock()
                elapsed = max(now - last_sample, 1e-6)
                cores_used = max(0.0, (stats.cpu_seconds - last_cpu) / elapsed)
                last_sample = now
                last_cpu = stats.cpu_seconds
                above_soft = (
                    stats.rss_bytes > self.budget.mem_bytes or cores_used > self.budget.cpu_cores
                )
                if above_soft:
                    spike_seconds += elapsed
                    if not self.spikes.record(elapsed, now):
                        killed_reason = self._kill_tree(process.pid, "spike_budget_exhausted")
                        return
                if stats.rss_bytes > mem_hard:
                    killed_reason = self._kill_tree(process.pid, "memory_hard_limit")
                    return

        thread = threading.Thread(target=monitor, name="janet-governor", daemon=True)
        thread.start()
        timed_out = False
        try:
            stdout, stderr = process.communicate(input=stdin_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            killed_reason = self._kill_tree(process.pid, "timeout")
            stdout, stderr = process.communicate()
        finally:
            stop.set()
            thread.join(timeout=2.0)
            if process.poll() is None:
                killed_reason = self._kill_tree(process.pid, "orphaned")
                try:
                    stdout, stderr = process.communicate()
                except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                    stdout, stderr = "", ""

        duration = self._clock() - started
        return RunResult(
            argv=tuple(argv),
            returncode=process.returncode if process.returncode is not None else -1,
            stdout=_truncate(stdout),
            stderr=_truncate(stderr),
            duration_s=duration,
            timed_out=timed_out,
            killed_reason=killed_reason,
            peak_rss_bytes=peak_rss,
            cpu_seconds=peak_cpu,
            spike_seconds=spike_seconds,
        )

    def __enter__(self) -> ResourceGovernor:
        self.apply_to_self()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def format_bytes(value: int) -> str:
    return f"{value / GIB:.2f} GiB"
