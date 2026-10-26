"""UCB1 operator bandit for harness mutations.

The P3/P4 proposer is a 7B model that repeatedly proposes the same hypothesis
("validate inputs"). The bandit does not ask the model to be more creative; it
chooses which *kind* of change to ask for, and learns from whether the paired
acceptance gate rewarded that kind. This is the P6 operator layer.

Reward is shaped from the gate, never from a model's self-report:

* accepted variant: +1.0
* anchor improved but the strict gate blocked it: +0.5
* anchor unchanged: 0.0
* anchor regressed: -0.5
* dev-floor rejection, so the anchor was not measured: -0.05 (no evidence)
* proposal error: -0.25

The state is a small JSON file so a foreground run and the idle daemon share the
same learned policy. Operators are fixed and inspectable; adding one starts it
with zero pulls. ``REWARD_VERSION`` is part of the file: changing the reward
shape starts a fresh policy rather than mixing incomparable means.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPERATORS: tuple[str, ...] = ("prompt", "skill", "budget", "prompt+skill")

# Bump when the reward definition changes; old state files are discarded.
REWARD_VERSION = 2
_HELDOUT_EPSILON = 0.01

OPERATOR_INSTRUCTIONS: dict[str, str] = {
    "prompt": (
        "Change only the system_prompt. Keep the skill library and the repair "
        "budget identical. Address a failure mode as a general rule."
    ),
    "skill": (
        "Change only the skills list: add or revise exactly one skill (name, "
        "triggers, rule). Keep the system_prompt and the repair budget identical."
    ),
    "budget": (
        "Change only max_attempts and/or samples_per_attempt. Keep the system "
        "prompt and the skills identical."
    ),
    "prompt+skill": (
        "Rewrite the system_prompt and add or revise exactly one skill. Keep the "
        "repair budget identical."
    ),
}

_TIE_EPSILON = 1e-12


class BanditError(ValueError):
    """Raised when the bandit state is invalid or an unknown arm is used."""


@dataclass(slots=True)
class ArmStats:
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "pulls": self.pulls,
            "total_reward": round(self.total_reward, 6),
            "mean_reward": round(self.mean, 6),
        }


def _fresh_arms() -> dict[str, ArmStats]:
    return {name: ArmStats() for name in OPERATORS}


@dataclass(slots=True)
class BanditState:
    arms: dict[str, ArmStats] = field(default_factory=_fresh_arms)
    updated_at: str = ""

    @classmethod
    def fresh(cls) -> BanditState:
        return cls()

    def _stats(self, arm: str) -> ArmStats:
        if arm not in self.arms:
            self.arms[arm] = ArmStats()
        return self.arms[arm]

    def choose(self, exploration: float = 0.7) -> str:
        """Pick an arm; untried arms first, then UCB1 with a deterministic tie-break."""

        if exploration < 0.0:
            raise BanditError("exploration must be >= 0")
        for name in OPERATORS:
            if self._stats(name).pulls == 0:
                return name
        total = sum(self._stats(name).pulls for name in OPERATORS)
        best_name = OPERATORS[0]
        best_score = -math.inf
        for name in OPERATORS:
            stats = self._stats(name)
            bonus = exploration * math.sqrt(math.log(total) / stats.pulls)
            score = stats.mean + bonus
            if score > best_score + _TIE_EPSILON:
                best_score = score
                best_name = name
        return best_name

    def record(self, arm: str, reward: float) -> None:
        if arm not in OPERATORS:
            raise BanditError(f"unknown operator arm: {arm}")
        stats = self._stats(arm)
        stats.pulls += 1
        stats.total_reward += reward
        self.updated_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, object]:
        return {
            "reward_version": REWARD_VERSION,
            "updated_at": self.updated_at,
            "total_pulls": sum(self._stats(name).pulls for name in OPERATORS),
            "arms": [{"name": name, **self._stats(name).to_dict()} for name in OPERATORS],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> BanditState:
        """Load the policy; a missing or corrupt file starts a fresh one."""

        state = cls.fresh()
        if not path.is_file():
            return state
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return state
        if not isinstance(raw, dict):
            return state
        if raw.get("reward_version") != REWARD_VERSION:
            # The means were learned under a different reward function.
            return state
        updated = raw.get("updated_at")
        if isinstance(updated, str):
            state.updated_at = updated
        entries = raw.get("arms")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not isinstance(name, str) or name not in OPERATORS:
                    continue
                state.arms[name] = _arm_from(entry)
        elif isinstance(entries, dict):
            for name, entry in entries.items():
                if not isinstance(name, str) or name not in OPERATORS:
                    continue
                if isinstance(entry, dict):
                    state.arms[name] = _arm_from(entry)
        return state


def _arm_from(entry: dict[str, Any]) -> ArmStats:
    pulls = entry.get("pulls")
    total = entry.get("total_reward")
    return ArmStats(
        pulls=pulls if isinstance(pulls, int) and not isinstance(pulls, bool) and pulls >= 0 else 0,
        total_reward=(
            float(total) if isinstance(total, (int, float)) and not isinstance(total, bool) else 0.0
        ),
    )


def reward_for(
    *,
    improved: bool,
    error: str | None,
    anchor_score: float | None,
    parent_anchor: float | None,
    heldout_score: float | None = None,
    parent_heldout: float | None = None,
) -> float:
    """Shape the paired gate's outcome into a bandit reward.

    The held-out anchor score (anchor tasks the dev split never contained) is
    the capability signal when both sides have it. A total-anchor gain that does
    not move the held-out score is only a quarter reward; a held-out regression
    is penalized even if the total anchor improved. The fresh split is never
    used here, so it stays an unbiased confirmation set.
    """

    if error is not None:
        return -0.25
    if anchor_score is None:
        # The dev-floor gate withholding the anchor is no evidence about the
        # operator; it only consumed compute.
        return -0.05
    if heldout_score is not None and parent_heldout is not None:
        delta = heldout_score - parent_heldout
        if delta > _HELDOUT_EPSILON:
            return 1.0 if improved else 0.5
        if delta < -_HELDOUT_EPSILON:
            return -0.25 if improved else -0.5
        return 0.25 if improved else 0.0
    # Fallback for an old variant without a held-out sample.
    if improved:
        return 1.0
    if parent_anchor is None:
        return 0.0
    delta = anchor_score - parent_anchor
    if delta > _TIE_EPSILON:
        return 0.5
    if delta < -_TIE_EPSILON:
        return -0.5
    return 0.0


def format_bandit(state: BanditState) -> str:
    """A compact, human-readable report."""

    lines = [f"updated: {state.updated_at or 'never'}"]
    for name in OPERATORS:
        stats = state.arms.get(name, ArmStats())
        lines.append(
            f"{name:12} pulls={stats.pulls:3d} "
            f"mean={stats.mean:+.3f} total={stats.total_reward:+.3f}"
        )
    return "\n".join(lines)
