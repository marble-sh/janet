"""Structural headroom report for the materialized benchmark suites."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from janet.eval.task import Task


class HeadroomError(RuntimeError):
    """Raised when a suite set cannot support a headroom claim."""


@dataclass(frozen=True, slots=True)
class HeadroomReport:
    dev_tasks: int
    anchor_tasks: int
    fresh_tasks: int
    dev_hard: int
    anchor_hard: int
    fresh_hard: int
    min_difficulty: int
    hard_families: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dev_tasks": self.dev_tasks,
            "anchor_tasks": self.anchor_tasks,
            "fresh_tasks": self.fresh_tasks,
            "dev_hard": self.dev_hard,
            "anchor_hard": self.anchor_hard,
            "fresh_hard": self.fresh_hard,
            "min_difficulty": self.min_difficulty,
            "hard_families": list(self.hard_families),
        }


def _hard(tasks: Iterable[Task], min_difficulty: int) -> int:
    return sum(1 for task in tasks if not task.is_integrity and task.difficulty >= min_difficulty)


def headroom_report(
    dev: tuple[Task, ...],
    anchor: tuple[Task, ...],
    fresh: tuple[Task, ...],
    *,
    min_difficulty: int = 3,
) -> HeadroomReport:
    if min_difficulty < 1:
        raise HeadroomError("min_difficulty must be >= 1")
    families = sorted(
        {
            task.family
            for task in (*dev, *anchor, *fresh)
            if not task.is_integrity and task.difficulty >= min_difficulty
        }
    )
    return HeadroomReport(
        dev_tasks=len(dev),
        anchor_tasks=len(anchor),
        fresh_tasks=len(fresh),
        dev_hard=_hard(dev, min_difficulty),
        anchor_hard=_hard(anchor, min_difficulty),
        fresh_hard=_hard(fresh, min_difficulty),
        min_difficulty=min_difficulty,
        hard_families=tuple(families),
    )


def assert_headroom(
    report: HeadroomReport,
    *,
    min_dev: int = 2,
    min_anchor: int = 4,
    min_fresh: int = 2,
) -> None:
    problems: list[str] = []
    if report.dev_hard < min_dev:
        problems.append(f"dev has {report.dev_hard} hard tasks, need {min_dev}")
    if report.anchor_hard < min_anchor:
        problems.append(f"anchor has {report.anchor_hard} hard tasks, need {min_anchor}")
    if report.fresh_hard < min_fresh:
        problems.append(f"fresh has {report.fresh_hard} hard tasks, need {min_fresh}")
    if len(report.hard_families) < 2:
        problems.append("hard tasks must span at least two families")
    if problems:
        raise HeadroomError("; ".join(problems))
