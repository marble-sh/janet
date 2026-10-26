"""Skills: small, retrievable rules attached to a harness.

A skill is admitted only through the same Pareto gate as any other harness
change, and only if its triggers match at least one dev task (an inert skill
cannot be evaluated). Skills may not name benchmark tasks; see ``guard``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

MAX_SKILLS = 6
_MAX_RULE = 400
_MIN_RULE = 20
_MAX_TRIGGERS = 5
_TRIGGER_RE = re.compile(r"^[a-z0-9_ ]{2,32}$")


class SkillError(ValueError):
    """Raised when a skill is malformed."""


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    triggers: tuple[str, ...]
    rule: str

    def validate(self) -> None:
        if not 1 <= len(self.name.strip()) <= 48:
            raise SkillError("skill name must be 1..48 characters")
        if not 1 <= len(self.triggers) <= _MAX_TRIGGERS:
            raise SkillError(f"skill triggers must have 1..{_MAX_TRIGGERS} entries")
        for trigger in self.triggers:
            if not _TRIGGER_RE.match(trigger):
                raise SkillError(f"invalid trigger {trigger!r}: use 2..32 chars of a-z0-9_ ")
        if not _MIN_RULE <= len(self.rule.strip()) <= _MAX_RULE:
            raise SkillError(f"skill rule must be {_MIN_RULE}..{_MAX_RULE} characters")

    def matches(self, haystack: str) -> bool:
        return any(trigger in haystack for trigger in self.triggers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.strip(),
            "triggers": sorted(self.triggers),
            "rule": self.rule.strip(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> Skill:
        if not isinstance(data, dict):
            raise SkillError("skill must be an object")
        unknown = set(data) - {"name", "triggers", "rule"}
        if unknown:
            raise SkillError(f"unknown skill fields: {sorted(unknown)}")
        name = data.get("name")
        triggers = data.get("triggers")
        rule = data.get("rule")
        if not isinstance(name, str) or not isinstance(rule, str):
            raise SkillError("skill name and rule must be strings")
        if not isinstance(triggers, list) or not all(isinstance(t, str) for t in triggers):
            raise SkillError("skill triggers must be a list of strings")
        skill = cls(
            name=name,
            triggers=tuple(t.strip().lower() for t in triggers),
            rule=rule,
        )
        skill.validate()
        return skill


def parse_skills(raw: Any) -> tuple[Skill, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise SkillError("skills must be a list")
    if len(raw) > MAX_SKILLS:
        raise SkillError(f"at most {MAX_SKILLS} skills are allowed")
    skills = tuple(Skill.from_dict(item) for item in raw)
    names = [skill.name.strip().lower() for skill in skills]
    if len(names) != len(set(names)):
        raise SkillError("skill names must be unique")
    return skills
