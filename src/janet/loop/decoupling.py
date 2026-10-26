"""Proxy-decoupling detector (the A-Evolve-Training observation, made routine).

The dev score is a proxy: the loop optimizes it and, with an evolving evaluator,
even redefines it. The held-out score (anchor tasks the dev split never contained)
is the external check. When dev keeps rising while held-out does not follow, the
proxy has stopped tracking the target; the loop must then demand held-out evidence
for every acceptance instead of trusting dev.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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
