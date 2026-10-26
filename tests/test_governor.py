from __future__ import annotations

import sys

import pytest

from janet.governor import (
    GovernorError,
    ResourceGovernor,
    SpikeBudget,
    tree_stats,
)
from janet.probe import GIB, MIB, Budget


def test_spike_budget_accumulates_and_expires() -> None:
    spikes = SpikeBudget(allowance_seconds=300.0, window_seconds=3600.0)
    assert spikes.remaining(0.0) == 300.0
    assert spikes.record(100.0, 0.0) is True
    assert spikes.remaining(10.0) == 200.0
    assert spikes.record(250.0, 10.0) is False
    assert spikes.exhausted(10.0) is True
    assert spikes.remaining(3700.0) == 300.0


def test_spike_budget_ignores_zero_duration() -> None:
    spikes = SpikeBudget(allowance_seconds=10.0)
    assert spikes.record(0.0, 0.0) is True
    assert spikes.used(0.0) == 0.0


def test_governor_captures_output(budget: Budget) -> None:
    governor = ResourceGovernor(budget)
    result = governor.run(
        [sys.executable, "-c", "print('hello')"],
        timeout_s=10.0,
    )
    assert result.ok
    assert result.stdout.strip() == "hello"
    assert result.returncode == 0


def test_governor_clears_the_host_environment(budget: Budget) -> None:
    governor = ResourceGovernor(budget)
    result = governor.run(
        [sys.executable, "-c", "import os; print(os.environ.get('JANET_SECRET', 'missing'))"],
        timeout_s=10.0,
        extra_env={"JANET_SECRET": "visible"},
    )
    assert result.ok
    assert result.stdout.strip() == "visible"
    result = governor.run(
        [sys.executable, "-c", "import os; print(os.environ.get('JANET_SECRET', 'missing'))"],
        timeout_s=10.0,
    )
    assert result.ok
    assert result.stdout.strip() == "missing"


def test_governor_enforces_timeout(budget: Budget) -> None:
    governor = ResourceGovernor(budget)
    result = governor.run(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout_s=0.5,
    )
    assert result.timed_out
    assert result.killed_reason == "timeout"
    assert result.returncode != 0
    assert result.duration_s < 5.0


def test_governor_enforces_memory(budget: Budget) -> None:
    governor = ResourceGovernor(budget)
    result = governor.run(
        [sys.executable, "-c", "x = bytearray(300 * 1024 * 1024); print(len(x))"],
        timeout_s=10.0,
        mem_bytes=64 * MIB,
    )
    assert not result.ok
    assert result.peak_rss_bytes < 300 * MIB


def test_governor_rejects_empty_command(budget: Budget) -> None:
    governor = ResourceGovernor(budget)
    with pytest.raises(GovernorError, match="empty"):
        governor.run([], timeout_s=1.0)


def test_governor_rejects_zero_timeout(budget: Budget) -> None:
    governor = ResourceGovernor(budget)
    with pytest.raises(GovernorError, match="timeout"):
        governor.run([sys.executable, "-c", "pass"], timeout_s=0.0)


def test_tree_stats_sees_the_current_process() -> None:
    import os

    stats = tree_stats(os.getpid())
    assert stats.rss_bytes > 0
    assert stats.cpu_seconds >= 0.0


def test_governor_does_not_exceed_its_wall_budget() -> None:
    budget = Budget(
        cpu_cores=1,
        cpu_cores_spike=1,
        mem_bytes=1 * GIB,
        mem_bytes_spike=1 * GIB,
        disk_bytes=1 * GIB,
        max_wall_seconds=1.0,
        spike_seconds_per_hour=0.0,
    )
    governor = ResourceGovernor(budget)
    result = governor.run(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout_s=60.0,
    )
    assert result.timed_out
    assert result.duration_s < 5.0
