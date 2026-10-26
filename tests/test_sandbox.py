from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from janet.governor import ResourceGovernor
from janet.probe import GIB, Budget
from janet.sandbox import Sandbox, SandboxError, SandboxLimits

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")


@pytest.fixture
def sandbox(tmp_path: Path, budget: Budget) -> Sandbox:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return Sandbox(workspace, ResourceGovernor(budget))


def test_preflight_denies_network(sandbox: Sandbox) -> None:
    result = sandbox.preflight()
    assert result.ok
    assert "network-denied" in result.stdout


def test_runs_python_in_the_jail(sandbox: Sandbox) -> None:
    result = sandbox.run(
        [
            "/usr/bin/python3",
            "-c",
            "import os; print(os.getcwd()); print(os.path.exists('/tmp/workspace'))",
        ]
    )
    assert result.ok
    assert "/tmp/workspace" in result.stdout
    assert "True" in result.stdout


def test_workspace_is_writable(sandbox: Sandbox) -> None:
    result = sandbox.run(
        [
            "/usr/bin/python3",
            "-c",
            "from pathlib import Path; Path('out.txt').write_text('written')",
        ]
    )
    assert result.ok
    assert (sandbox.workspace / "out.txt").read_text(encoding="utf-8") == "written"


def test_host_home_is_hidden(sandbox: Sandbox) -> None:
    # The repository is under /home; the sandbox masks /home with an empty tmpfs.
    repo_file = str(Path("pyproject.toml").resolve())
    result = sandbox.run(
        [
            "/usr/bin/python3",
            "-c",
            f"from pathlib import Path; print(Path({repo_file!r}).exists())",
        ]
    )
    assert result.ok
    assert result.stdout.strip() == "False"


def test_timeout_is_enforced(sandbox: Sandbox) -> None:
    result = sandbox.run(
        ["/usr/bin/python3", "-c", "import time; time.sleep(10)"],
        limits=SandboxLimits(timeout_s=0.5, mem_bytes=256 * 1024 * 1024, cpu_cores=1),
    )
    assert result.timed_out
    assert result.killed_reason == "timeout"


def test_missing_workspace_is_rejected(tmp_path: Path, budget: Budget) -> None:
    with pytest.raises(SandboxError, match="not a directory"):
        Sandbox(tmp_path / "missing", ResourceGovernor(budget))


def test_empty_command_is_rejected(sandbox: Sandbox) -> None:
    with pytest.raises(SandboxError, match="empty"):
        sandbox.run([])


def test_missing_bwrap_is_reported(
    tmp_path: Path, budget: Budget, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(SandboxError, match="bubblewrap"):
        Sandbox(workspace, ResourceGovernor(budget))


def test_limits_have_sane_defaults() -> None:
    limits = SandboxLimits()
    assert limits.timeout_s > 0
    assert limits.mem_bytes >= 256 * 1024 * 1024
    assert limits.cpu_cores >= 1


def test_sandbox_budget_is_respected(sandbox: Sandbox, budget: Budget) -> None:
    result = sandbox.run(["/usr/bin/python3", "-c", "print('ok')"])
    assert result.ok
    assert result.peak_rss_bytes <= budget.mem_bytes_spike + GIB
