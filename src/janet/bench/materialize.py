"""Materialize seed tasks into dev and hidden anchor suites."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from janet.bench.catalog import ALL_SEEDS
from janet.bench.seeds import CaseSeed, TaskSeed


class MaterializeError(RuntimeError):
    """Raised when a suite cannot be written safely."""


@dataclass(frozen=True, slots=True)
class Materialized:
    root: Path
    split: str
    task_count: int
    manifest: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "split": self.split,
            "task_count": self.task_count,
            "manifest": str(self.manifest),
        }


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"cannot serialize {type(value).__name__} as TOML")


def _compute_expected(reference: str, entrypoint: str, args: tuple[Any, ...]) -> Any:
    namespace: dict[str, Any] = {}
    exec(compile(reference, "<reference>", "exec"), namespace)
    function = namespace[entrypoint]
    return function(*args)


def _case_entries(seed: TaskSeed, cases: tuple[CaseSeed, ...]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for case in cases:
        entry: dict[str, Any] = {"args": list(case.args)}
        if case.error is not None:
            entry["error"] = case.error
        else:
            if seed.entrypoint is None:
                raise MaterializeError(f"{seed.id}: case without an entrypoint")
            entry["expect"] = _compute_expected(seed.reference, seed.entrypoint, case.args)
        entries.append(entry)
    return entries


def _task_toml(seed: TaskSeed, split: str) -> str:
    lines = [
        f"id = {_toml_value(seed.id)}",
        f"family = {_toml_value(seed.family)}",
        f"split = {_toml_value(split)}",
        f"kind = {_toml_value(seed.kind)}",
        f"title = {_toml_value(seed.title)}",
    ]
    if seed.entrypoint is not None:
        lines.append(f"entrypoint = {_toml_value(seed.entrypoint)}")
    lines.extend(
        [
            f"timeout_s = {_toml_value(float(seed.timeout_s))}",
            f"difficulty = {_toml_value(seed.difficulty)}",
            'abstain_marker = "ABSTAIN"',
            f"forbidden = {_toml_value(list(seed.forbidden))}",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_task(directory: Path, seed: TaskSeed, split: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "task.toml").write_text(_task_toml(seed, split), encoding="utf-8")
    (directory / "prompt.md").write_text(seed.prompt.strip() + "\n", encoding="utf-8")
    if seed.kind == "function":
        (directory / "solution.py").write_text(seed.buggy.lstrip(), encoding="utf-8")
        (directory / "reference.py").write_text(seed.reference.lstrip(), encoding="utf-8")
        if split == "dev":
            visible = _case_entries(seed, seed.visible)
            hidden: list[dict[str, Any]] = []
        else:
            visible = []
            hidden = _case_entries(seed, seed.anchor)
        (directory / "visible_cases.json").write_text(
            json.dumps(visible, indent=2) + "\n", encoding="utf-8"
        )
        (directory / "hidden_cases.json").write_text(
            json.dumps(hidden, indent=2) + "\n", encoding="utf-8"
        )
    else:
        (directory / "solution.py").write_text("", encoding="utf-8")
        (directory / "visible_cases.json").write_text("[]\n", encoding="utf-8")
        (directory / "hidden_cases.json").write_text("[]\n", encoding="utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(
    root: Path,
    split: str,
    *,
    seeds: tuple[TaskSeed, ...] = ALL_SEEDS,
    force: bool = False,
) -> Materialized:
    if split not in {"dev", "anchor", "fresh"}:
        raise MaterializeError("split must be 'dev', 'anchor', or 'fresh'")
    root = root.expanduser()
    if root.exists():
        if not force:
            raise MaterializeError(f"{root} already exists; pass force=True to replace it")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    if split == "fresh":
        selected = tuple(seed for seed in seeds if seed.fresh_only)
    elif split == "anchor":
        selected = tuple(seed for seed in seeds if not seed.fresh_only)
    else:
        selected = tuple(seed for seed in seeds if not seed.anchor_only and not seed.fresh_only)
    for seed in selected:
        _write_task(root / seed.family / seed.id, seed, split)
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(root))] = _hash_file(path)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"split": split, "files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return Materialized(
        root=root,
        split=split,
        task_count=len(selected),
        manifest=manifest,
    )


def verify_manifest(root: Path) -> tuple[bool, str]:
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return (False, f"missing manifest: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (False, f"invalid manifest: {exc}")
    files = payload.get("files")
    if not isinstance(files, dict):
        return (False, "manifest has no files map")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            return (False, "manifest entries must be strings")
        path = root / relative
        if not path.is_file():
            return (False, f"missing file: {relative}")
        if _hash_file(path) != expected:
            return (False, f"hash mismatch: {relative}")
    return (True, f"{len(files)} files verified")
