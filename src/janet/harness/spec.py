"""Declarative harness variants.

A variant is a system prompt, a small skill library, and the repair budget. The
content hash covers only behavior-affecting fields, so two paths to the same
behavior share a hash; lineage is stored separately by the archive. The
``skills`` key is hashed only when non-empty, so pre-skill hashes are stable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

from janet.eval.prompts import SYSTEM_FUNCTION
from janet.harness.skills import Skill, SkillError, parse_skills

_MAX_PROMPT_CHARS = 4000
_MIN_PROMPT_CHARS = 40
_MAX_ATTEMPTS = 5
_MAX_SAMPLES = 4


class HarnessError(ValueError):
    """Raised when a harness variant is malformed."""


@dataclass(frozen=True, slots=True)
class HarnessSpec:
    name: str
    system_prompt: str
    max_attempts: int = 3
    samples_per_attempt: int = 3
    mutation_note: str = ""
    parent: str | None = None
    skills: tuple[Skill, ...] = ()

    def validate(self) -> None:
        name = self.name.strip()
        if not name or len(name) > 64:
            raise HarnessError("name must be 1..64 characters")
        prompt = self.system_prompt.strip()
        if not _MIN_PROMPT_CHARS <= len(prompt) <= _MAX_PROMPT_CHARS:
            raise HarnessError(
                f"system_prompt must be {_MIN_PROMPT_CHARS}..{_MAX_PROMPT_CHARS} characters"
            )
        if not 1 <= self.max_attempts <= _MAX_ATTEMPTS:
            raise HarnessError(f"max_attempts must be 1..{_MAX_ATTEMPTS}")
        if not 1 <= self.samples_per_attempt <= _MAX_SAMPLES:
            raise HarnessError(f"samples_per_attempt must be 1..{_MAX_SAMPLES}")
        if len(self.mutation_note) > 500:
            raise HarnessError("mutation_note must be <= 500 characters")
        try:
            parse_skills([skill.to_dict() for skill in self.skills])
        except SkillError as exc:
            raise HarnessError(str(exc)) from exc

    def content_dict(self) -> dict[str, Any]:
        content: dict[str, Any] = {
            "system_prompt": self.system_prompt.strip(),
            "max_attempts": self.max_attempts,
            "samples_per_attempt": self.samples_per_attempt,
        }
        if self.skills:
            content["skills"] = sorted(
                (skill.to_dict() for skill in self.skills), key=lambda item: item["name"]
            )
        return content

    @property
    def hash(self) -> str:
        canonical = json.dumps(self.content_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def normalized(self) -> dict[str, Any]:
        return {
            "name": self.name.strip(),
            **self.content_dict(),
            "mutation_note": self.mutation_note.strip(),
            "parent": self.parent,
        }

    def to_json(self) -> str:
        return json.dumps(self.normalized(), indent=2, sort_keys=True) + "\n"

    def with_parent(self, parent: str) -> HarnessSpec:
        return replace(self, parent=parent)

    @classmethod
    def from_dict(cls, data: Any, *, parent: str | None = None) -> HarnessSpec:
        if not isinstance(data, dict):
            raise HarnessError("harness payload must be an object")
        allowed = {
            "name",
            "system_prompt",
            "max_attempts",
            "samples_per_attempt",
            "mutation_note",
            "parent",
            "skills",
        }
        unknown = set(data) - allowed
        if unknown:
            raise HarnessError(f"unknown harness fields: {sorted(unknown)}")
        name = data.get("name", "candidate")
        prompt = data.get("system_prompt")
        if not isinstance(name, str) or not isinstance(prompt, str):
            raise HarnessError("name and system_prompt must be strings")
        attempts = data.get("max_attempts", 3)
        samples = data.get("samples_per_attempt", 3)
        note = data.get("mutation_note", "")
        parent_value = parent if parent is not None else data.get("parent")
        if not isinstance(attempts, int) or not isinstance(samples, int):
            raise HarnessError("max_attempts and samples_per_attempt must be integers")
        if not isinstance(note, str):
            raise HarnessError("mutation_note must be a string")
        if parent_value is not None and not isinstance(parent_value, str):
            raise HarnessError("parent must be a string or null")
        try:
            skills = parse_skills(data.get("skills"))
        except SkillError as exc:
            raise HarnessError(str(exc)) from exc
        spec = cls(
            name=name,
            system_prompt=prompt,
            max_attempts=attempts,
            samples_per_attempt=samples,
            mutation_note=note,
            parent=parent_value,
            skills=skills,
        )
        spec.validate()
        return spec


SEED_HARNESS = HarnessSpec(
    name="seed",
    system_prompt=SYSTEM_FUNCTION,
    max_attempts=3,
    samples_per_attempt=3,
    mutation_note="Initial harness: verified repair loop, no domain hints.",
)
