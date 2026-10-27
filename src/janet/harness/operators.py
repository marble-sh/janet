"""Operator classification and change-scope validation (P6.1).

Each bandit arm permits exactly one kind of harness edit. The proposer sees the
arm's instruction, and this module enforces the boundary after the fact: a
candidate that changes more than its arm allows is rejected before any compute
is spent evaluating it.
"""

from __future__ import annotations

from dataclasses import dataclass

from janet.harness.spec import HarnessSpec


class OperatorError(ValueError):
    """Raised when a proposal exceeds the scope of its operator arm."""


@dataclass(frozen=True, slots=True)
class HarnessChange:
    prompt: bool
    skills_added: tuple[str, ...]
    skills_removed: tuple[str, ...]
    skills_revised: tuple[str, ...]
    budget: bool
    memory: bool

    @property
    def skills_changed(self) -> bool:
        return bool(self.skills_added or self.skills_removed or self.skills_revised)

    @property
    def count(self) -> int:
        return sum(
            (
                self.prompt,
                self.budget,
                self.memory,
                bool(self.skills_added),
                bool(self.skills_removed),
                bool(self.skills_revised),
            )
        )


def diff_harness(parent: HarnessSpec, candidate: HarnessSpec) -> HarnessChange:
    parent_skills = {skill.name: skill for skill in parent.skills}
    candidate_skills = {skill.name: skill for skill in candidate.skills}
    added = tuple(sorted(candidate_skills.keys() - parent_skills.keys()))
    removed = tuple(sorted(parent_skills.keys() - candidate_skills.keys()))
    revised = tuple(
        sorted(
            name
            for name in parent_skills.keys() & candidate_skills.keys()
            if parent_skills[name].to_dict() != candidate_skills[name].to_dict()
        )
    )
    return HarnessChange(
        prompt=candidate.system_prompt.strip() != parent.system_prompt.strip(),
        skills_added=added,
        skills_removed=removed,
        skills_revised=revised,
        budget=(candidate.max_attempts, candidate.samples_per_attempt)
        != (parent.max_attempts, parent.samples_per_attempt),
        memory=candidate.memory_policy != parent.memory_policy,
    )


def classify_operator(parent: HarnessSpec, candidate: HarnessSpec) -> str:
    """Name the arm a change belongs to, or ``invalid``/``none``."""

    change = diff_harness(parent, candidate)
    if change.count == 0:
        return "none"
    one_skill = (
        len(change.skills_added) + len(change.skills_revised) == 1 and not change.skills_removed
    )
    only_prompt = change.prompt and not (change.skills_changed or change.budget or change.memory)
    if only_prompt:
        return "prompt"
    if change.prompt and one_skill and not (change.budget or change.memory):
        return "prompt+skill"
    if change.skills_changed and not (change.prompt or change.budget or change.memory):
        if len(change.skills_added) == 1 and not change.skills_revised:
            return "skill_add"
        if len(change.skills_revised) == 1 and not change.skills_added:
            return "skill_revise"
        return "invalid"
    if change.budget and not (change.prompt or change.skills_changed or change.memory):
        return "budget"
    if change.memory and not (change.prompt or change.skills_changed or change.budget):
        return "memory_policy"
    return "invalid"


def validate_operator(parent: HarnessSpec, candidate: HarnessSpec, arm: str) -> None:
    """Reject a candidate that changes more than ``arm`` permits."""

    actual = classify_operator(parent, candidate)
    if actual != arm:
        raise OperatorError(
            f"the {arm} arm permits exactly one scoped change, but the proposal "
            f"classified as {actual!r}"
        )


__all__ = [
    "HarnessChange",
    "OperatorError",
    "classify_operator",
    "diff_harness",
    "validate_operator",
]
