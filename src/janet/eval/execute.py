"""Run a function over many argument lists in the sandbox, without expected values.

Used for trusted references (to derive expected outputs for new inputs) and for
mutants (to see which inputs expose them). Candidates under evaluation use
``runner.run_function_task`` instead, which also audits and scores.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from janet.governor import ResourceGovernor
from janet.probe import MIB
from janet.sandbox import Sandbox, SandboxError, SandboxLimits

_WORKER = Path(__file__).with_name("worker.py")


def execute_cases(
    source: str,
    entrypoint: str,
    inputs: Sequence[Sequence[Any]],
    governor: ResourceGovernor,
    workspace: Path,
    *,
    timeout_s: float = 10.0,
) -> list[dict[str, Any]] | None:
    """Per-input ``{"ok": bool, "value"|"error": ...}`` records, or None on a hard failure.

    A hard failure (timeout, kill, unparsable output, import error) returns None so
    callers can treat the whole program as broken rather than guess per case.
    """

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text(source, encoding="utf-8")
    (workspace / "worker.py").write_text(_WORKER.read_text(encoding="utf-8"), encoding="utf-8")
    (workspace / "inputs.json").write_text(
        json.dumps([list(args) for args in inputs], default=repr), encoding="utf-8"
    )
    try:
        run = Sandbox(workspace, governor).run(
            ["/usr/bin/python3", "worker.py", "solution.py", "inputs.json", entrypoint],
            limits=SandboxLimits(timeout_s=timeout_s, mem_bytes=512 * MIB, cpu_cores=1),
        )
    except SandboxError:
        return None
    if not run.ok:
        return None
    try:
        payload = json.loads(run.stdout)
    except json.JSONDecodeError:
        return None
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or len(results) != len(inputs):
        return None
    return [r if isinstance(r, dict) else {"ok": False, "error": "malformed"} for r in results]
