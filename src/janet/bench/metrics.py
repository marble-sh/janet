"""Stratified benchmark metrics.

Capability is the mean over measured non-integrity tasks and is the only
capability claim. Headroom is the non-saturated, difficulty-gated subset used
to steer the loop; it is not a capability estimate. Integrity abstention is
reported separately and never blended into either mean.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from janet.eval.task import Task


@dataclass(frozen=True, slots=True)
class SplitMetrics:
    min_difficulty: int
    capability_n: int
    capability_mean: float | None
    headroom_n: int
    headroom_mean: float | None
    integrity_n: int
    integrity_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "min_difficulty": self.min_difficulty,
            "capability_n": self.capability_n,
            "capability_mean": (
                round(self.capability_mean, 6) if self.capability_mean is not None else None
            ),
            "headroom_n": self.headroom_n,
            "headroom_mean": (
                round(self.headroom_mean, 6) if self.headroom_mean is not None else None
            ),
            "integrity_n": self.integrity_n,
            "integrity_rate": (
                round(self.integrity_rate, 6) if self.integrity_rate is not None else None
            ),
        }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stratified_metrics(
    tasks: Sequence[Task],
    scores: Mapping[str, float],
    *,
    passed: Mapping[str, bool] | None = None,
    min_difficulty: int = 3,
) -> SplitMetrics:
    """Compute the three reporting strata from measured task scores.

    ``tasks`` is the set the caller made available for this run; ``scores``
    contains only the tasks that were actually measured. This keeps the
    denominators honest when ``--limit`` or ``--only`` is used.
    """

    if min_difficulty < 1:
        raise ValueError("min_difficulty must be >= 1")
    capability: list[float] = []
    headroom: list[float] = []
    integrity: list[float] = []
    for task in tasks:
        score = scores.get(task.id)
        if score is None:
            continue
        if task.is_integrity:
            if passed is not None:
                integrity.append(1.0 if passed.get(task.id, False) else 0.0)
            else:
                integrity.append(1.0 if score >= 1.0 else 0.0)
            continue
        capability.append(score)
        if not task.saturated and task.difficulty >= min_difficulty:
            headroom.append(score)
    return SplitMetrics(
        min_difficulty=min_difficulty,
        capability_n=len(capability),
        capability_mean=_mean(capability),
        headroom_n=len(headroom),
        headroom_mean=_mean(headroom),
        integrity_n=len(integrity),
        integrity_rate=_mean(integrity),
    )


__all__ = ["SplitMetrics", "stratified_metrics"]
