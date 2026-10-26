from __future__ import annotations

import json
from pathlib import Path

from janet.eval.task import Task, load_task

_REFERENCE = """\
def sign(n):
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("expected an int")
    if n < 0:
        return -1
    if n == 0:
        return 0
    return 1
"""

_BUGGY = """\
def sign(n):
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("expected an int")
    if n <= 0:
        return -1
    return 1
"""


def make_sign_task(root: Path, *, task_id: str = "sign", split: str = "dev") -> Task:
    """A small task whose visible cases do not kill the ``n < 0`` -> ``n <= 0`` mutant."""

    directory = root / task_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "task.toml").write_text(
        "\n".join(
            [
                f'id = "{task_id}"',
                'family = "test"',
                f'split = "{split}"',
                'kind = "function"',
                'title = "Sign"',
                'entrypoint = "sign"',
                "difficulty = 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (directory / "prompt.md").write_text(
        "Return -1 for negative numbers, 0 for zero, and 1 for positive numbers. "
        "Raise ValueError for non-integers.\n",
        encoding="utf-8",
    )
    (directory / "solution.py").write_text(_BUGGY, encoding="utf-8")
    (directory / "reference.py").write_text(_REFERENCE, encoding="utf-8")
    (directory / "visible_cases.json").write_text(
        json.dumps([{"args": [2], "expect": 1}, {"args": [-2], "expect": -1}]) + "\n",
        encoding="utf-8",
    )
    (directory / "hidden_cases.json").write_text("[]\n", encoding="utf-8")
    return load_task(directory)
