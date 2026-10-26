from __future__ import annotations

from pathlib import Path

import pytest

from janet.governor import ResourceGovernor
from janet.probe import GIB, Budget
from janet.sandbox import Sandbox


@pytest.fixture
def budget() -> Budget:
    return Budget(
        cpu_cores=2,
        cpu_cores_spike=4,
        mem_bytes=4 * GIB,
        mem_bytes_spike=6 * GIB,
        disk_bytes=10 * GIB,
        max_wall_seconds=30.0,
        spike_seconds_per_hour=300.0,
    )


@pytest.fixture
def sandbox(tmp_path: Path, budget: Budget) -> Sandbox:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return Sandbox(workspace, ResourceGovernor(budget))
