"""Task schema and suite loading.

A task is a directory with ``task.toml``, ``prompt.md``, ``solution.py``, and
JSON case files. The candidate sees only the prompt and the buggy solution; the
evaluator holds the expected outputs.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_VALID_SPLITS = frozenset({"dev", "anchor", "fresh", "integrity"})
_VALID_KINDS = frozenset({"function", "integrity"})


class TaskError(ValueError):
    """Raised when a task or suite is malformed."""


@dataclass(frozen=True, slots=True)
class Case:
    args: tuple[Any, ...]
    expect: Any


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    family: str
    split: str
    kind: str
    title: str
    prompt: str
    solution: str
    entrypoint: str | None
    visible_cases: tuple[Case, ...]
    hidden_cases: tuple[Case, ...]
    timeout_s: float
    difficulty: int
    forbidden: tuple[str, ...]
    abstain_marker: str
    source_dir: Path

    @property
    def cases(self) -> tuple[Case, ...]:
        return self.visible_cases + self.hidden_cases

    @property
    def is_integrity(self) -> bool:
        return self.kind == "integrity"


def load_cases(path: Path) -> tuple[Case, ...]:
    if not path.is_file():
        return ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaskError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise TaskError(f"{path} must contain a JSON list")
    cases: list[Case] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise TaskError(f"{path}[{index}] must be an object")
        args = entry.get("args")
        if not isinstance(args, list):
            raise TaskError(f"{path}[{index}] is missing a list 'args'")
        if "error" in entry and "expect" in entry:
            raise TaskError(f"{path}[{index}] cannot have both 'error' and 'expect'")
        if "error" in entry:
            error = entry["error"]
            if not isinstance(error, str) or not error:
                raise TaskError(f"{path}[{index}].error must be a non-empty string")
            expect: Any = {"error": error}
        elif "expect" in entry:
            expect = entry["expect"]
        else:
            raise TaskError(f"{path}[{index}] is missing 'expect' or 'error'")
        cases.append(Case(args=tuple(args), expect=expect))
    return tuple(cases)


def _required_string(table: dict[str, Any], key: str, path: Path) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise TaskError(f"{path}: '{key}' must be a non-empty string")
    return value


def load_task(directory: Path) -> Task:
    manifest = directory / "task.toml"
    if not manifest.is_file():
        raise TaskError(f"missing task.toml in {directory}")
    try:
        table = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise TaskError(f"invalid TOML in {manifest}: {exc}") from exc

    task_id = _required_string(table, "id", manifest)
    family = _required_string(table, "family", manifest)
    split = _required_string(table, "split", manifest)
    if split not in _VALID_SPLITS:
        raise TaskError(f"{manifest}: split must be one of {sorted(_VALID_SPLITS)}")
    kind = str(table.get("kind", "function"))
    if kind not in _VALID_KINDS:
        raise TaskError(f"{manifest}: kind must be one of {sorted(_VALID_KINDS)}")
    title = _required_string(table, "title", manifest)

    prompt_path = directory / "prompt.md"
    if not prompt_path.is_file():
        raise TaskError(f"missing prompt.md in {directory}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    solution_path = directory / "solution.py"
    solution = solution_path.read_text(encoding="utf-8") if solution_path.is_file() else ""

    entrypoint: str | None = None
    if kind == "function":
        entrypoint = _required_string(table, "entrypoint", manifest)

    forbidden_raw = table.get("forbidden", [])
    if not isinstance(forbidden_raw, list) or not all(
        isinstance(pattern, str) for pattern in forbidden_raw
    ):
        raise TaskError(f"{manifest}: 'forbidden' must be a list of strings")
    for pattern in forbidden_raw:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise TaskError(f"{manifest}: invalid forbidden regex {pattern!r}: {exc}") from exc

    timeout_s = float(table.get("timeout_s", 30.0))
    if timeout_s <= 0:
        raise TaskError(f"{manifest}: timeout_s must be > 0")
    difficulty = int(table.get("difficulty", 1))
    if difficulty < 1:
        raise TaskError(f"{manifest}: difficulty must be >= 1")
    abstain_marker = str(table.get("abstain_marker", "ABSTAIN"))
    if not abstain_marker:
        raise TaskError(f"{manifest}: abstain_marker must be non-empty")

    visible = load_cases(directory / "visible_cases.json")
    hidden = load_cases(directory / "hidden_cases.json")
    if kind == "function" and not visible and not hidden:
        raise TaskError(f"{manifest}: function task has no cases")

    return Task(
        id=task_id,
        family=family,
        split=split,
        kind=kind,
        title=title,
        prompt=prompt,
        solution=solution,
        entrypoint=entrypoint,
        visible_cases=visible,
        hidden_cases=hidden,
        timeout_s=timeout_s,
        difficulty=difficulty,
        forbidden=tuple(forbidden_raw),
        abstain_marker=abstain_marker,
        source_dir=directory,
    )


def load_suite(root: Path, split: str | None = None) -> tuple[Task, ...]:
    if not root.is_dir():
        raise TaskError(f"suite root is not a directory: {root}")
    if split is not None and split not in _VALID_SPLITS:
        raise TaskError(f"unknown split: {split}")
    tasks: list[Task] = []
    seen: set[str] = set()
    for manifest in sorted(root.rglob("task.toml")):
        task = load_task(manifest.parent)
        if task.id in seen:
            raise TaskError(f"duplicate task id: {task.id}")
        seen.add(task.id)
        if split is None or task.split == split:
            tasks.append(task)
    if not tasks:
        raise TaskError(f"no tasks found under {root}" + (f" for split {split}" if split else ""))
    return tuple(sorted(tasks, key=lambda task: (task.family, task.id)))


def select_tasks(tasks: tuple[Task, ...], min_difficulty: int) -> tuple[Task, ...]:
    """Keep tasks at or above a difficulty; integrity tasks are always kept.

    Saturated easy tasks add no selection signal but integrity tasks guard honesty.
    """

    return tuple(t for t in tasks if t.is_integrity or t.difficulty >= min_difficulty)
