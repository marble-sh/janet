"""Paired statistical evidence for acceptance and release (P6.3).

Pure Python, no dependencies: a percentile bootstrap CI on the paired mean
difference, a one-sided exact sign test, an SGM-style e-value with a geometric
alpha-spending schedule, and a certificate that is only valid for the declared
target and verifier version.

The fresh split is never used for selection; `paired_test` is also used there
as a release wrapper, not as a tuning signal.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

_TIE_EPSILON = 1e-12


class StatsError(ValueError):
    """Raised when paired evidence cannot be computed as requested."""


def _percentile(values: Sequence[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = quantile * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def bootstrap_ci(
    deltas: Sequence[float], *, alpha: float, resamples: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference."""

    if not deltas:
        raise StatsError("deltas must not be empty")
    if not 0.0 < alpha < 1.0:
        raise StatsError("alpha must be in (0, 1)")
    if resamples < 1:
        raise StatsError("resamples must be >= 1")
    if len(deltas) == 1:
        return (deltas[0], deltas[0])
    rng = random.Random(seed)
    count = len(deltas)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(count):
            total += deltas[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    low = _percentile(means, alpha / 2.0)
    high = _percentile(means, 1.0 - alpha / 2.0)
    return (low, high)


def theil_sen(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Median of pairwise slopes; None when undefined."""

    if len(xs) != len(ys):
        raise StatsError("xs and ys must have the same length")
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs))
        for j in range(i + 1, len(xs))
        if abs(xs[j] - xs[i]) > _TIE_EPSILON
    ]
    return float(statistics.median(slopes)) if slopes else None


def bootstrap_slope_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for a Theil-Sen slope; None with <3 points."""

    if len(xs) != len(ys):
        raise StatsError("xs and ys must have the same length")
    if len(xs) < 3:
        return None
    if not 0.0 < alpha < 1.0:
        raise StatsError("alpha must be in (0, 1)")
    if resamples < 1:
        raise StatsError("resamples must be >= 1")
    rng = random.Random(seed)
    count = len(xs)
    slopes: list[float] = []
    for _ in range(resamples):
        indices = [rng.randrange(count) for _ in range(count)]
        slope = theil_sen([xs[i] for i in indices], [ys[i] for i in indices])
        if slope is not None:
            slopes.append(slope)
    if not slopes:
        return None
    slopes.sort()
    return (_percentile(slopes, alpha / 2.0), _percentile(slopes, 1.0 - alpha / 2.0))


def sign_test_p(wins: int, losses: int) -> float:
    """One-sided exact binomial p-value for wins under p = 0.5 (ties excluded)."""

    if wins < 0 or losses < 0:
        raise StatsError("wins and losses must be >= 0")
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(wins, trials + 1))
    return min(1.0, float(tail) / float(2**trials))


def e_value(wins: int, losses: int, *, alternative: float = 0.65) -> float:
    """Likelihood-ratio e-value for a Bernoulli sign test (ties contribute 1)."""

    if wins < 0 or losses < 0:
        raise StatsError("wins and losses must be >= 0")
    if not 0.5 < alternative < 1.0:
        raise StatsError("alternative must be in (0.5, 1)")
    if wins + losses == 0:
        return 1.0
    return ((2.0 * alternative) ** wins) * ((2.0 * (1.0 - alternative)) ** losses)


@dataclass(frozen=True, slots=True)
class AlphaSpending:
    """Geometric alpha spending over a declared generation horizon."""

    horizon: int = 24
    alpha_total: float = 0.05
    decay: float = 0.5

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise StatsError("horizon must be >= 1")
        if not 0.0 < self.alpha_total < 1.0:
            raise StatsError("alpha_total must be in (0, 1)")
        if not 0.0 < self.decay <= 1.0:
            raise StatsError("decay must be in (0, 1]")

    def weights(self) -> tuple[float, ...]:
        raw = tuple(self.decay**generation for generation in range(1, self.horizon + 1))
        total = sum(raw)
        return tuple(value / total for value in raw)

    def alpha(self, generation: int) -> float:
        if not 1 <= generation <= self.horizon:
            raise StatsError(f"generation must be in 1..{self.horizon}")
        return self.alpha_total * self.weights()[generation - 1]


@dataclass(frozen=True, slots=True)
class PairedTest:
    n: int
    wins: int
    losses: int
    ties: int
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    e_value: float
    alpha: float
    alternative: float
    decision: str

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"

    @property
    def ci_excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "mean_delta": round(self.mean_delta, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "p_value": round(self.p_value, 6),
            "e_value": round(self.e_value, 6),
            "alpha": round(self.alpha, 6),
            "alternative": self.alternative,
            "decision": self.decision,
        }


def paired_test(
    parent: Sequence[float],
    candidate: Sequence[float],
    *,
    alpha: float,
    alternative: float = 0.65,
    resamples: int = 2000,
    seed: int = 0,
) -> PairedTest:
    """Compare paired per-task scores; ties are non-evidence, not wins."""

    if len(parent) != len(candidate):
        raise StatsError("paired samples must have the same length")
    if not parent:
        raise StatsError("paired samples must not be empty")
    if not 0.0 < alpha < 1.0:
        raise StatsError("alpha must be in (0, 1)")
    deltas = tuple(
        float(right) - float(left) for left, right in zip(parent, candidate, strict=True)
    )
    wins = sum(1 for delta in deltas if delta > _TIE_EPSILON)
    losses = sum(1 for delta in deltas if delta < -_TIE_EPSILON)
    ties = len(deltas) - wins - losses
    p_value = sign_test_p(wins, losses)
    evidence = e_value(wins, losses, alternative=alternative)
    ci_low, ci_high = bootstrap_ci(deltas, alpha=alpha, resamples=resamples, seed=seed)
    decision = "accept" if p_value <= alpha or evidence >= (1.0 / alpha) else "reject"
    return PairedTest(
        n=len(deltas),
        wins=wins,
        losses=losses,
        ties=ties,
        mean_delta=sum(deltas) / len(deltas),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        e_value=evidence,
        alpha=alpha,
        alternative=alternative,
        decision=decision,
    )


@dataclass(frozen=True, slots=True)
class EvidenceCertificate:
    """A decision bound to one evaluation target and verifier version."""

    target: str
    verifier_version: str
    generation: int
    test: PairedTest
    created_at: str

    def valid_for(self, target: str, verifier_version: str) -> bool:
        return self.target == target and self.verifier_version == verifier_version

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "verifier_version": self.verifier_version,
            "generation": self.generation,
            "created_at": self.created_at,
            "test": self.test.to_dict(),
        }


def certify(
    test: PairedTest, *, target: str, verifier_version: str, generation: int
) -> EvidenceCertificate:
    return EvidenceCertificate(
        target=target,
        verifier_version=verifier_version,
        generation=generation,
        test=test,
        created_at=datetime.now(UTC).isoformat(),
    )


__all__ = [
    "AlphaSpending",
    "EvidenceCertificate",
    "PairedTest",
    "StatsError",
    "bootstrap_ci",
    "bootstrap_slope_ci",
    "certify",
    "e_value",
    "paired_test",
    "sign_test_p",
    "theil_sen",
]
