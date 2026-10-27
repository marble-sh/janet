"""Proxy-decoupling detector (the A-Evolve-Training observation, made routine).

The dev score is a proxy: the loop optimizes it and, with an evolving evaluator,
even redefines it. The held-out score (anchor tasks the dev split never contained)
is the external check. When dev keeps rising while held-out does not follow, the
proxy has stopped tracking the target; the loop must then demand held-out evidence
for every acceptance instead of trusting dev.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


class DecouplingError(ValueError):
    """Raised when a detector is configured with invalid parameters."""


@dataclass(frozen=True, slots=True)
class Decoupling:
    n: int
    dev_gain: float
    heldout_gain: float
    rank_corr: float | None
    flagged: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "dev_gain": round(self.dev_gain, 4),
            "heldout_gain": round(self.heldout_gain, 4),
            "rank_corr": None if self.rank_corr is None else round(self.rank_corr, 4),
            "flagged": self.flagged,
            "reason": self.reason,
        }


_TIE_EPSILON = 1e-9


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and (abs(values[order[j + 1]] - values[order[i]]) <= _TIE_EPSILON):
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation, or None when it is undefined (too short or constant)."""

    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return float(cov / (vx * vy) ** 0.5)


def _half_gain(values: Sequence[float]) -> float:
    half = len(values) // 2
    return sum(values[half:]) / len(values[half:]) - sum(values[:half]) / len(values[:half])


def detect(
    points: Sequence[tuple[float, float]],
    *,
    min_points: int = 4,
    gap: float = 0.08,
    corr_floor: float = -0.2,
) -> Decoupling:
    """``points`` are (dev, held-out) pairs in the order the variants were evaluated."""

    n = len(points)
    if n < min_points:
        return Decoupling(n, 0.0, 0.0, None, False, f"need {min_points} points, have {n}")
    dev = [p[0] for p in points]
    held = [p[1] for p in points]
    dev_gain, held_gain = _half_gain(dev), _half_gain(held)
    # Dev scores that differ only by float noise are a constant proxy, not a
    # rising one; ranking that noise invents a correlation that is not there.
    if abs(dev_gain) <= _TIE_EPSILON:
        dev_gain = 0.0
    corr = spearman(dev, held)
    if dev_gain - held_gain >= gap and dev_gain > _TIE_EPSILON:
        reason = f"dev rose {dev_gain:+.3f} while held-out moved {held_gain:+.3f}"
        return Decoupling(n, dev_gain, held_gain, corr, True, reason)
    if corr is not None and corr < corr_floor:
        reason = f"dev and held-out are anti-correlated (rho={corr:.2f})"
        return Decoupling(n, dev_gain, held_gain, corr, True, reason)
    return Decoupling(n, dev_gain, held_gain, corr, False, "dev tracks held-out")


@dataclass(frozen=True, slots=True)
class CusumResult:
    n: int
    statistic: float
    peak: float
    baseline: float
    drift: float
    threshold: float
    flagged: bool
    first_flag: int | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "statistic": round(self.statistic, 6),
            "peak": round(self.peak, 6),
            "baseline": round(self.baseline, 6),
            "drift": self.drift,
            "threshold": self.threshold,
            "flagged": self.flagged,
            "first_flag": self.first_flag,
            "reason": self.reason,
        }


def cusum(
    points: Sequence[tuple[float, float]],
    *,
    drift: float = 0.01,
    threshold: float = 0.05,
    baseline_points: int = 3,
    min_points: int = 5,
    baseline: float | None = None,
) -> CusumResult:
    """One-sided CUSUM on the dev-minus-heldout gap.

    The raw gap is normally positive (dev is easier than the hidden anchor), so
    the detector is referenced to a baseline level; it flags a *rise* in the gap
    relative to that level, not the level itself. The first ``baseline_points``
    observations estimate the baseline unless one is supplied.
    """

    if drift < 0.0:
        raise DecouplingError("drift must be >= 0")
    if threshold <= 0.0:
        raise DecouplingError("threshold must be > 0")
    if baseline_points < 1:
        raise DecouplingError("baseline_points must be >= 1")
    if min_points < baseline_points + 1:
        raise DecouplingError("min_points must be greater than baseline_points")
    gaps = [float(dev) - float(held) for dev, held in points]
    n = len(gaps)
    if n < min_points:
        return CusumResult(
            n, 0.0, 0.0, 0.0, drift, threshold, False, None, f"need {min_points} points, have {n}"
        )
    reference = baseline if baseline is not None else sum(gaps[:baseline_points]) / baseline_points
    statistic = 0.0
    peak = 0.0
    first_flag: int | None = None
    for index in range(baseline_points, n):
        statistic = max(0.0, statistic + (gaps[index] - reference) - drift)
        peak = max(peak, statistic)
        if statistic >= threshold and first_flag is None:
            first_flag = index + 1
    if first_flag is not None:
        reason = (
            f"gap rose above baseline {reference:.3f} at point {first_flag} "
            f"(cusum {peak:.3f} >= {threshold:.3f})"
        )
        return CusumResult(
            n, statistic, peak, reference, drift, threshold, True, first_flag, reason
        )
    return CusumResult(
        n, statistic, peak, reference, drift, threshold, False, None, "gap is stable"
    )


def assess(
    points: Sequence[tuple[float, float]],
    *,
    gap: float = 0.08,
    corr_floor: float = -0.2,
    drift: float = 0.01,
    threshold: float = 0.05,
) -> tuple[Decoupling, CusumResult]:
    """Run both detectors; either one turns on strict acceptance."""

    return detect(points, gap=gap, corr_floor=corr_floor), cusum(
        points, drift=drift, threshold=threshold
    )


@dataclass(frozen=True, slots=True)
class StaleProbe:
    pre_points: int
    post_points: int
    shift: float
    detection_latency: int | None
    detection_index: int | None
    reframed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "pre_points": self.pre_points,
            "post_points": self.post_points,
            "shift": self.shift,
            "detection_latency": self.detection_latency,
            "detection_index": self.detection_index,
            "reframed": self.reframed,
        }


def stale_objective_probe(
    *,
    pre_points: int = 8,
    post_points: int = 8,
    shift: float = 0.06,
    noise: float = 0.0,
    seed: int = 0,
    drift: float = 0.01,
    threshold: float = 0.05,
) -> StaleProbe:
    """Synthetic objective shift: measure CUSUM detection latency.

    The stream is a stable dev-minus-heldout gap followed by a step of ``shift``.
    ``reframed`` reports whether the loop would switch to strict acceptance
    within the post-shift window.
    """

    if pre_points < 2 or post_points < 1:
        raise DecouplingError("pre_points must be >= 2 and post_points >= 1")
    if noise < 0.0:
        raise DecouplingError("noise must be >= 0")
    rng = random.Random(seed)
    gaps = [rng.uniform(-noise, noise) for _ in range(pre_points)]
    gaps.extend(shift + rng.uniform(-noise, noise) for _ in range(post_points))
    points = [(0.5 + gap, 0.5) for gap in gaps]
    result = cusum(points, drift=drift, threshold=threshold, baseline=0.0)
    if result.first_flag is None or result.first_flag <= pre_points:
        return StaleProbe(pre_points, post_points, shift, None, None, False)
    latency = result.first_flag - pre_points
    return StaleProbe(
        pre_points, post_points, shift, latency, result.first_flag, latency <= post_points
    )
