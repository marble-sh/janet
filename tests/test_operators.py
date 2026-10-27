from __future__ import annotations

import json

import pytest

from janet.harness.operators import (
    OperatorError,
    classify_operator,
    diff_harness,
    validate_operator,
)
from janet.harness.skills import Skill
from janet.harness.spec import HarnessError, HarnessSpec, MemoryPolicy

RULE = "Validate inputs before computing and raise ValueError for invalid ranges."


def _skill(
    name: str = "validate", triggers: tuple[str, ...] = ("raise",), rule: str = RULE
) -> Skill:
    return Skill(name=name, triggers=triggers, rule=rule)


def _base() -> HarnessSpec:
    return HarnessSpec(name="p", system_prompt="A" * 50)


def test_arm_boundaries_are_enforced() -> None:
    parent = _base()
    candidate = HarnessSpec(name="c", system_prompt="B" * 50, max_attempts=2)
    assert classify_operator(parent, candidate) == "invalid"
    with pytest.raises(OperatorError, match="prompt"):
        validate_operator(parent, candidate, "prompt")


def test_skill_add_requires_exactly_one_new_skill() -> None:
    parent = _base()
    two = HarnessSpec(name="c", system_prompt="A" * 50, skills=(_skill("one"), _skill("two")))
    assert classify_operator(parent, two) == "invalid"
    with pytest.raises(OperatorError):
        validate_operator(parent, two, "skill_add")
    one = HarnessSpec(name="c", system_prompt="A" * 50, skills=(_skill(),))
    validate_operator(parent, one, "skill_add")


def test_skill_revise_keeps_the_name_and_count() -> None:
    parent = HarnessSpec(name="p", system_prompt="A" * 50, skills=(_skill(),))
    revised = HarnessSpec(
        name="c", system_prompt="A" * 50, skills=(_skill(rule=RULE + " Check bounds."),)
    )
    assert diff_harness(parent, revised).skills_revised == ("validate",)
    validate_operator(parent, revised, "skill_revise")
    with pytest.raises(OperatorError):
        validate_operator(parent, revised, "skill_add")


def test_memory_policy_is_its_own_arm() -> None:
    parent = _base()
    candidate = HarnessSpec(
        name="c", system_prompt="A" * 50, memory_policy=MemoryPolicy(enabled=True)
    )
    assert classify_operator(parent, candidate) == "memory_policy"
    validate_operator(parent, candidate, "memory_policy")
    with pytest.raises(OperatorError):
        validate_operator(parent, candidate, "prompt")


def test_default_policy_keeps_the_hash_stable() -> None:
    implicit = HarnessSpec(name="x", system_prompt="A" * 50)
    explicit = HarnessSpec(name="x", system_prompt="A" * 50, memory_policy=MemoryPolicy())
    assert implicit.hash == explicit.hash
    changed = HarnessSpec(
        name="x", system_prompt="A" * 50, memory_policy=MemoryPolicy(enabled=True, top_skills=1)
    )
    assert changed.hash != implicit.hash


def test_memory_policy_round_trips_through_json() -> None:
    spec = HarnessSpec(
        name="x",
        system_prompt="A" * 50,
        memory_policy=MemoryPolicy(enabled=True, top_lessons=2, min_success=0.5),
    )
    restored = HarnessSpec.from_dict(json.loads(spec.to_json()))
    assert restored.memory_policy == spec.memory_policy
    assert restored.hash == spec.hash


@pytest.mark.parametrize(
    "payload",
    [
        {"max_prompt_tokens": 5000},
        {"top_skills": -1},
        {"min_success": 2.0},
        {"enabled": True, "top_skills": 0, "top_lessons": 0},
        {"bogus": 1},
    ],
)
def test_invalid_memory_policies_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(HarnessError):
        MemoryPolicy.from_dict(payload)


def test_prompt_plus_skill_accepts_one_skill_change() -> None:
    parent = HarnessSpec(name="p", system_prompt="A" * 50, skills=(_skill(),))
    candidate = HarnessSpec(
        name="c",
        system_prompt="B" * 50,
        skills=(_skill(rule=RULE + " Check bounds."),),
    )
    assert classify_operator(parent, candidate) == "prompt+skill"
    validate_operator(parent, candidate, "prompt+skill")
