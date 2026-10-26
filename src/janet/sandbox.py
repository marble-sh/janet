"""A bubblewrap sandbox for candidate code.

The candidate sees a read-only root, an empty ``/home``, an empty ``/tmp``, no
network, and a single writable directory mounted at ``/workspace``. The anchor
suite lives under the real home directory and is therefore invisible.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from janet.governor import ResourceGovernor, RunResult
from janet.probe import MIB

_DEFAULT_TIMEOUT_S = 60.0
_DEFAULT_MEM_BYTES = 1024 * MIB
_DEFAULT_CORES = 2
_MASKED_DIRS = ("/home", "/root", "/mnt", "/media", "/srv")


class SandboxError(RuntimeError):
    """Raised when the sandbox cannot be established or fails to start."""


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    timeout_s: float = _DEFAULT_TIMEOUT_S
    mem_bytes: int = _DEFAULT_MEM_BYTES
    cpu_cores: int = _DEFAULT_CORES


class Sandbox:
    """Runs commands in a bubblewrap jail."""

    def __init__(
        self,
        workspace: Path,
        governor: ResourceGovernor,
        *,
        executable: str | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        if not self.workspace.is_dir():
            raise SandboxError(f"workspace is not a directory: {self.workspace}")
        self.governor = governor
        self.executable = executable or shutil.which("bwrap")
        if self.executable is None:
            raise SandboxError(
                "bubblewrap (bwrap) is required; install it or run with "
                "JANET_ALLOW_UNSANDBOXED=1 only for trusted tests"
            )

    def _argv(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None,
    ) -> list[str]:
        argv = [
            self.executable or "bwrap",
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--ro-bind",
            "/",
            "/",
        ]
        for path in _MASKED_DIRS:
            if Path(path).exists():
                argv.extend(["--tmpfs", path])
        argv.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"])
        argv.extend(
            [
                "--bind",
                str(self.workspace),
                "/tmp/workspace",
                "--chdir",
                "/tmp/workspace",
                "--clearenv",
                "--setenv",
                "PATH",
                "/usr/bin:/bin",
                "--setenv",
                "HOME",
                "/tmp/workspace",
                "--setenv",
                "TMPDIR",
                "/tmp",
                "--setenv",
                "LANG",
                "C.UTF-8",
                "--setenv",
                "LC_ALL",
                "C.UTF-8",
                "--setenv",
                "PYTHONNOUSERSITE",
                "1",
                "--setenv",
                "PYTHONDONTWRITEBYTECODE",
                "1",
                "--cap-drop",
                "ALL",
                "--",
                *command,
            ]
        )
        if env:
            # Insert setenv pairs before the final "--".
            index = argv.index("--")
            pairs: list[str] = []
            for key, value in env.items():
                pairs.extend(["--setenv", key, value])
            argv[index:index] = pairs
        return argv

    def run(
        self,
        command: Sequence[str],
        *,
        limits: SandboxLimits | None = None,
        stdin_text: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        """Run a command inside the jail."""

        if not command:
            raise SandboxError("empty command")
        limits = limits or SandboxLimits()
        argv = self._argv(command, env=env)
        result = self.governor.run(
            argv,
            cwd=self.workspace,
            timeout_s=limits.timeout_s,
            mem_bytes=limits.mem_bytes,
            cpu_cores=limits.cpu_cores,
            stdin_text=stdin_text,
        )
        if result.returncode != 0 and result.stderr.startswith("bwrap:"):
            raise SandboxError(result.stderr.strip())
        return result

    def preflight(self) -> RunResult:
        """Verify that the jail can start and that network is denied."""

        limits = SandboxLimits(timeout_s=10.0, mem_bytes=256 * MIB, cpu_cores=1)
        return self.run(
            [
                "/usr/bin/python3",
                "-c",
                (
                    "import socket\n"
                    "try:\n"
                    "    socket.create_connection(('1.1.1.1', 80), 1)\n"
                    "except OSError:\n"
                    "    print('network-denied')\n"
                    "else:\n"
                    "    raise SystemExit('network-available')\n"
                ),
            ],
            limits=limits,
        )
