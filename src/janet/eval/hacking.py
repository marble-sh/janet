"""Cheap reward-hacking detectors over candidate trajectories (W4).

The verifier still decides; these flags are evidence for the report and for a
human review. They catch two cheap failure modes:

* the same solution submitted for different tasks (memorization or a degenerate
  answer), normalised through the AST so whitespace, comments, docstrings, and a
  renamed entrypoint do not hide it;
* a solution that names another benchmark task's identifiers.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from janet.harness.guard import find_text_leaks


def _canonicalize(tree: ast.Module) -> ast.Module:
    """Strip the entrypoint name so a renamed copy still fingerprints alike."""

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.name = "_entry"
        elif isinstance(node, ast.ClassDef):
            node.name = "_entry_class"
    return tree


def solution_fingerprint(source: str) -> str:
    """Whitespace/comment-insensitive fingerprint of a candidate solution."""

    try:
        tree = _canonicalize(ast.parse(source))
    except SyntaxError:
        normalized = " ".join(source.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    dumped = ast.dump(tree, annotate_fields=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def duplicate_solutions(
    samples: Sequence[tuple[str, str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Groups of different tasks that received the same solution."""

    groups: dict[str, list[str]] = {}
    for task_id, source in samples:
        if not source.strip():
            continue
        groups.setdefault(solution_fingerprint(source), []).append(task_id)
    found: list[tuple[str, tuple[str, ...]]] = []
    for task_ids in groups.values():
        unique = sorted(set(task_ids))
        if len(unique) >= 2:
            found.append((unique[0], tuple(unique)))
    return tuple(sorted(found))


def _docstring_ids(tree: ast.AST) -> set[int]:
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", [])
        if not body or not isinstance(body[0], ast.Expr):
            continue
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found.add(id(value))
    return found


def _identifier_text(tree: ast.AST) -> str:
    """Identifiers and non-docstring literals, excluding comments and prose."""

    docstrings = _docstring_ids(tree)
    parts: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.Attribute):
            parts.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parts.append(node.name)
        elif isinstance(node, ast.arg):
            parts.append(node.arg)
        elif isinstance(node, ast.alias):
            parts.append(node.name)
            if node.asname:
                parts.append(node.asname)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            parts.extend(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            parts.append(node.name)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            parts.append(node.value)
    return "\n".join(parts)


def hardcoding_flags(source: str, terms: Iterable[str]) -> tuple[str, ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return find_text_leaks(source, terms)
    return find_text_leaks(_identifier_text(tree), terms)


@dataclass(frozen=True, slots=True)
class HackingReport:
    samples: int
    duplicate_groups: tuple[tuple[str, tuple[str, ...]], ...]
    hardcoded: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def flagged(self) -> int:
        return len(self.duplicate_groups) + len(self.hardcoded)

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "duplicate_groups": [
                {"first_task": first, "tasks": list(tasks)}
                for first, tasks in self.duplicate_groups
            ],
            "hardcoded": [{"task": task, "terms": list(terms)} for task, terms in self.hardcoded],
            "flagged": self.flagged,
        }


def audit_trajectories(
    samples: Sequence[tuple[str, str]],
    *,
    forbidden: Iterable[str] = (),
    forbidden_by_task: Mapping[str, Iterable[str]] | None = None,
) -> HackingReport:
    """Audit final candidates.

    ``forbidden_by_task`` holds each task's *own* forbidden terms. A candidate
    must define its own entrypoint, so the caller should exclude that task's id
    and entrypoint; only other benchmark identifiers are suspicious.
    """

    duplicates = duplicate_solutions(samples)
    audited = [(task_id, source) for task_id, source in samples if source.strip()]
    shared = tuple(forbidden)
    hardcoded: list[tuple[str, tuple[str, ...]]] = []
    for task_id, source in audited:
        terms = tuple(forbidden_by_task.get(task_id, shared)) if forbidden_by_task else shared
        leaks = hardcoding_flags(source, terms)
        if leaks:
            hardcoded.append((task_id, leaks))
    return HackingReport(
        samples=len(audited),
        duplicate_groups=duplicates,
        hardcoded=tuple(hardcoded),
    )


__all__ = [
    "HackingReport",
    "audit_trajectories",
    "duplicate_solutions",
    "hardcoding_flags",
    "solution_fingerprint",
]
