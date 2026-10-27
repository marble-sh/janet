"""Bounded consolidation for process memory (P5.4).

Four deterministic passes over verified memory:

* merge near-duplicate entries (Jaccard on trigger tokens + rule tokens);
* promote a lesson to a *candidate* skill after independent successes in at
  least two task families and only when the lesson carries replayable tests;
* retire skills that are used often but fail, or that have gone stale;
* cap the store, keeping the strongest evidence and retiring the rest.

Every action leaves evidence, the FTS index is rebuilt, and the database is
vacuumed. Admission is still the only path to `verified`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from janet.memory.store import LessonRecord, ProcessMemory, SkillRecord

CONSOLIDATION_VERSION = "p5.4-1"

_ACTIVE = ("candidate", "verified", "quarantined")
_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}")
_TRIGGER_RE = re.compile(r"[a-z0-9_]{2,32}")
_WORD_RE = re.compile(r"[a-z0-9]+")
_MAX_SCAN = 10_000


@dataclass(frozen=True, slots=True)
class ConsolidationReport:
    run_id: str
    merged_lessons: int
    merged_skills: int
    promoted: int
    retired: int
    capped: int
    kept_lessons: int
    kept_skills: int
    db_bytes: int
    over_cap: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "merged_lessons": self.merged_lessons,
            "merged_skills": self.merged_skills,
            "promoted": self.promoted,
            "retired": self.retired,
            "capped": self.capped,
            "kept_lessons": self.kept_lessons,
            "kept_skills": self.kept_skills,
            "db_bytes": self.db_bytes,
            "over_cap": self.over_cap,
        }


def _tokens(*parts: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(" ".join(parts).lower()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _stamp(record: LessonRecord | SkillRecord) -> datetime:
    raw = record.last_used_at or record.created_at
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _active_lessons(memory: ProcessMemory) -> list[LessonRecord]:
    records: dict[int, LessonRecord] = {}
    for status in _ACTIVE:
        for record in memory.list_lessons(status=status, limit=_MAX_SCAN):
            records[record.id] = record
    return list(records.values())


def _active_skills(memory: ProcessMemory) -> list[SkillRecord]:
    records: dict[str, SkillRecord] = {}
    for status in _ACTIVE:
        for record in memory.list_skills(status=status, limit=_MAX_SCAN):
            records[record.id] = record
    return list(records.values())


def _strongest_first[Record: (LessonRecord, SkillRecord)](
    records: Sequence[Record],
) -> list[Record]:
    return sorted(
        records,
        key=lambda record: (-record.success_count, record.failure_count, record.created_at),
    )


def _merge_lessons(memory: ProcessMemory, run_id: str, similarity: float) -> tuple[int, int]:
    records = _strongest_first(_active_lessons(memory))
    retired = 0
    merged = 0
    used: set[int] = set()
    for index, winner in enumerate(records):
        if winner.id in used:
            continue
        winner_tokens = _tokens(winner.trigger_terms, winner.rule)
        losers = [
            loser
            for loser in records[index + 1 :]
            if loser.id not in used
            and loser.task_family == winner.task_family
            and _jaccard(winner_tokens, _tokens(loser.trigger_terms, loser.rule)) > similarity
        ]
        if not losers:
            continue
        for loser in losers:
            memory.set_status("lesson", loser.id, "retired")
            used.add(loser.id)
            retired += 1
        merged += 1
        memory.record_evidence(
            "lesson",
            winner.id,
            run_id=run_id,
            split="dev",
            verifier_version=CONSOLIDATION_VERSION,
            test_ids=(),
            raw={
                "action": "merge",
                "merged_from": [loser.id for loser in losers],
                "task_family": winner.task_family,
            },
        )
    return merged, retired


def _merge_skills(memory: ProcessMemory, run_id: str, similarity: float) -> tuple[int, int]:
    records = _strongest_first(_active_skills(memory))
    retired = 0
    merged = 0
    used: set[str] = set()
    for index, winner in enumerate(records):
        if winner.id in used:
            continue
        winner_tokens = _tokens(" ".join(winner.triggers), winner.rule)
        losers = [
            loser
            for loser in records[index + 1 :]
            if loser.id not in used
            and _jaccard(winner_tokens, _tokens(" ".join(loser.triggers), loser.rule)) > similarity
        ]
        if not losers:
            continue
        for loser in losers:
            memory.set_status("skill", loser.id, "retired")
            used.add(loser.id)
            retired += 1
        merged += 1
        memory.record_evidence(
            "skill",
            winner.id,
            run_id=run_id,
            split="dev",
            verifier_version=CONSOLIDATION_VERSION,
            test_ids=(),
            raw={"action": "merge", "merged_from": [loser.id for loser in losers]},
        )
    return merged, retired


def _triggers_from(trigger_terms: str) -> tuple[str, ...]:
    triggers: list[str] = []
    for token in _TRIGGER_RE.findall(trigger_terms.lower()):
        if token not in triggers:
            triggers.append(token)
        if len(triggers) == 3:
            break
    return tuple(triggers)


def _skill_name(rule: str, lesson_id: int) -> str:
    words = [word for word in _WORD_RE.findall(rule.lower()) if len(word) >= 3][:4]
    base = "-".join(words) or "lesson"
    return f"{base[:40]}-{lesson_id}"[:48]


def _already_promoted(memory: ProcessMemory, lesson_id: int) -> bool:
    return any(
        evidence.raw.get("action") == "promote"
        for evidence in memory.list_evidence("lesson", lesson_id)
    )


def _promote_lessons(
    memory: ProcessMemory,
    run_id: str,
    *,
    min_successes: int,
    min_families: int,
) -> int:
    promoted = 0
    for lesson in memory.list_lessons(status="verified", limit=_MAX_SCAN):
        if lesson.success_count < min_successes or _already_promoted(memory, lesson.id):
            continue
        outcomes = memory.family_outcomes(lesson.id)
        families = sorted(family for family, (successes, _) in outcomes.items() if successes > 0)
        if len(families) < min_families:
            continue
        tests = lesson.provenance.get("tests")
        if not isinstance(tests, list) or not tests:
            continue
        triggers = _triggers_from(lesson.trigger_terms)
        if not triggers:
            continue
        skill_id = memory.add_skill(
            name=_skill_name(lesson.rule, lesson.id),
            triggers=triggers,
            rule=lesson.rule,
            tests=tests,
            provenance={
                "promoted_from_lesson": lesson.id,
                "families": families,
                "rule": lesson.rule,
            },
        )
        record = memory.get_skill(skill_id)
        if record is None or record.status == "retired":
            continue
        memory.record_evidence(
            "lesson",
            lesson.id,
            run_id=run_id,
            split="dev",
            verifier_version=CONSOLIDATION_VERSION,
            test_ids=(),
            raw={"action": "promote", "skill_id": skill_id, "families": families},
        )
        promoted += 1
    return promoted


def _retire_skills(
    memory: ProcessMemory,
    *,
    min_uses: int,
    min_success_rate: float,
    max_idle_days: float,
    now: datetime,
) -> int:
    retired = 0
    for skill in _active_skills(memory):
        uses = skill.success_count + skill.failure_count
        rate = skill.success_count / uses if uses else 0.0
        idle_days = max(0.0, (now - _stamp(skill)).total_seconds() / 86400.0)
        failing = uses >= min_uses and rate < min_success_rate
        stale = idle_days >= max_idle_days
        if failing or stale:
            memory.set_status("skill", skill.id, "retired")
            retired += 1
    return retired


def _cap[Record: (LessonRecord, SkillRecord)](
    records: Sequence[Record],
    limit: int,
    retire: Callable[[Record], None],
) -> int:
    if limit < 0:
        raise ValueError("cap must be >= 0")
    if len(records) <= limit:
        return 0
    ranked = sorted(
        records,
        key=lambda record: (
            -(record.success_count - record.failure_count),
            -_stamp(record).timestamp(),
            str(record.id),
        ),
    )
    retired = 0
    for record in ranked[limit:]:
        retire(record)
        retired += 1
    return retired


def consolidate(
    memory: ProcessMemory,
    *,
    run_id: str | None = None,
    max_lessons: int = 200,
    max_skills: int = 24,
    max_db_bytes: int = 64 * 1024 * 1024,
    similarity: float = 0.80,
    promote_successes: int = 3,
    promote_families: int = 2,
    min_uses: int = 10,
    min_success_rate: float = 0.5,
    max_idle_days: float = 30.0,
    now: datetime | None = None,
) -> ConsolidationReport:
    if max_lessons < 0 or max_skills < 0 or max_db_bytes < 1:
        raise ValueError("caps must be non-negative and max_db_bytes must be >= 1")
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("similarity must be in [0, 1]")
    if promote_successes < 1 or promote_families < 1:
        raise ValueError("promotion thresholds must be >= 1")
    if min_uses < 0 or not 0.0 <= min_success_rate <= 1.0:
        raise ValueError("retirement thresholds are out of range")
    if max_idle_days < 0.0:
        raise ValueError("max_idle_days must be >= 0")

    clock = now or datetime.now(UTC)
    if clock.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    stamp = run_id or f"consolidate-{clock.strftime('%Y%m%dT%H%M%SZ')}"

    merged_lessons, merge_retired_lessons = _merge_lessons(memory, stamp, similarity)
    merged_skills, merge_retired_skills = _merge_skills(memory, stamp, similarity)
    promoted = _promote_lessons(
        memory, stamp, min_successes=promote_successes, min_families=promote_families
    )
    retired = _retire_skills(
        memory,
        min_uses=min_uses,
        min_success_rate=min_success_rate,
        max_idle_days=max_idle_days,
        now=clock,
    )
    capped = _cap(
        _active_lessons(memory),
        max_lessons,
        lambda item: memory.set_status("lesson", item.id, "retired"),
    )
    capped += _cap(
        _active_skills(memory),
        max_skills,
        lambda item: memory.set_status("skill", item.id, "retired"),
    )
    memory.rebuild_fts()
    memory.vacuum()
    kept_lessons = len(_active_lessons(memory))
    kept_skills = len(_active_skills(memory))
    db_bytes = memory.path.stat().st_size if memory.path.is_file() else 0
    return ConsolidationReport(
        run_id=stamp,
        merged_lessons=merged_lessons,
        merged_skills=merged_skills,
        promoted=promoted,
        retired=retired + merge_retired_lessons + merge_retired_skills,
        capped=capped,
        kept_lessons=kept_lessons,
        kept_skills=kept_skills,
        db_bytes=db_bytes,
        over_cap=db_bytes > max_db_bytes,
    )


__all__ = ["CONSOLIDATION_VERSION", "ConsolidationReport", "consolidate"]
