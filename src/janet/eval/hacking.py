"""Cheap reward-hacking detectors over candidate trajectories (W4).

The verifier still decides; these flags are evidence for the report and for a
human review. They catch two cheap failure modes:

* the same solution submitted for different tasks (memorization or a degenerate
  answer), normalised through the AST so whitespace and comments do not hide it;
* a solution that names a benchmark task, entrypoint, or other forbidden term.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from janet.harness.guard import find_text_leaks


def solution_fingerprint(source: str) -> str:
    """Whitespace/comment-insensitive fingerprint of a candidate solution."""

    try:
        tree = ast.parse(source)
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


def hardcoding_flags(source: str, terms: Iterable[str]) -> tuple[str, ...]:
    return find_text_leaks(source, terms)


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
    shared = tuple(forbidden)
    hardcoded: list[tuple[str, tuple[str, ...]]] = []
    for task_id, source in samples:
        terms = tuple(forbidden_by_task.get(task_id, shared)) if forbidden_by_task else shared
        leaks = hardcoding_flags(source, terms)
        if leaks:
            hardcoded.append((task_id, leaks))
    return HackingReport(
        samples=len(samples),
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
