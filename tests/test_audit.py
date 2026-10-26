from __future__ import annotations

from pathlib import Path

import pytest

from janet.bench.materialize import materialize
from janet.eval.audit import audit_source
from janet.eval.runner import run_task
from janet.eval.task import load_suite
from janet.governor import ResourceGovernor
from janet.probe import Budget


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import socket\n", "import socket"),
        ("from subprocess import run\n", "from subprocess import ..."),
        ("import os\nos.system('ls')\n", "os.system"),
        ("x = eval('1')\n", "call eval()"),
        ("open('/etc/passwd')\n", "open('/etc/passwd')"),
        ("open('../x')\n", "open('../x')"),
        ("open('out.txt', 'w')\n", "open() for writing"),
        ("import os\nos.walk('.')\n", "os.walk"),
        ("().__class__.__subclasses__()\n", "attribute __subclasses__"),
        ("import importlib\n", "import importlib"),
    ],
)
def test_audit_flags_risky_code(source: str, expected: str) -> None:
    assert expected in audit_source(source)


@pytest.mark.parametrize(
    "source",
    [
        "def f(x):\n    return sorted(x)\n",
        "import re\nimport heapq\nfrom collections import Counter\n",
        "def f(s):\n    return open_count(s)\n",
        "this is not python at all (",
    ],
)
def test_audit_passes_benign_code(source: str) -> None:
    assert audit_source(source) == ()


def test_audit_deduplicates() -> None:
    assert audit_source("import socket\nimport socket\n") == ("import socket",)


def test_flagged_candidate_is_scored_zero_and_never_executed(
    tmp_path: Path, budget: Budget
) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    task = next(t for t in load_suite(tmp_path / "dev", "dev") if t.id == "clamp_value")
    marker = tmp_path / "ran"
    source = (
        "import socket\n"
        f"open({str(marker)!r}, 'w').write('x')\n"
        "def clamp(x, lo, hi):\n    return x\n"
    )
    result = run_task(task, source, ResourceGovernor(budget), tmp_path / "work")
    assert result.score == 0.0
    assert result.error is not None and result.error.startswith("audit:")
    assert result.failures[0].kind == "audit"
    assert not marker.exists()


def test_anchor_is_invisible_inside_the_sandbox(tmp_path: Path, budget: Budget) -> None:
    from janet.sandbox import Sandbox

    workspace = tmp_path / "ws"
    workspace.mkdir()
    sandbox = Sandbox(workspace, ResourceGovernor(budget))
    anchor = Path.home() / ".local/share/janet/anchor/manifest.json"
    result = sandbox.run(
        [
            "/usr/bin/python3",
            "-c",
            f"import os; print(os.path.exists({str(anchor)!r}))",
        ]
    )
    assert result.ok
    assert result.stdout.strip() == "False"
