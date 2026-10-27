from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from janet.memory import ProcessMemory, consolidate

_GOOD_RULE = "Validate the duration suffix before multiplying the numeric value."


@pytest.fixture
def memory(tmp_path: Path) -> Iterator[ProcessMemory]:
    with ProcessMemory(tmp_path / "memory.sqlite3") as store:
        yield store


def _lesson(
    store: ProcessMemory,
    *,
    rule: str = _GOOD_RULE,
    triggers: str = "duration suffix",
    family: str = "code_repair",
    successes: tuple[str, ...] = (),
    provenance: dict[str, object] | None = None,
) -> int:
    lesson_id = store.add_lesson(
        task_id="parse_duration",
        task_family=family,
        failure_signature="ValueError on an invalid duration suffix",
        trigger_terms=triggers,
        rule=rule,
        patch_summary="Reject the unknown suffix.",
        provenance=provenance or {"run": "run-1"},
    )
    store.set_status("lesson", lesson_id, "verified")
    for outcome_family in successes:
        store.record_outcome("lesson", lesson_id, success=True, task_family=outcome_family)
    return lesson_id


def _skill(
    store: ProcessMemory,
    *,
    name: str = "duration-validation",
    triggers: tuple[str, ...] = ("duration", "suffix"),
    rule: str = _GOOD_RULE,
    failures: int = 0,
    successes: int = 0,
) -> str:
    skill_id = store.add_skill(
        name=name,
        triggers=triggers,
        rule=rule,
        tests=[{"args": ["1x"], "error": "ValueError"}],
        provenance={"run": "run-1"},
    )
    store.set_status("skill", skill_id, "verified")
    for _ in range(successes):
        store.record_outcome("skill", skill_id, success=True)
    for _ in range(failures):
        store.record_outcome("skill", skill_id, success=False)
    return skill_id


def test_merges_near_duplicate_lessons_and_links_provenance(memory: ProcessMemory) -> None:
    winner = _lesson(memory, successes=("code_repair", "code_repair", "code_repair"))
    loser = _lesson(memory, triggers="duration suffix seconds")
    report = consolidate(memory, run_id="consolidate-test")
    assert report.merged_lessons == 1
    assert report.retired == 1
    assert memory.get_lesson(loser) is not None
    assert memory.get_lesson(loser).status == "retired"  # type: ignore[union-attr]
    assert memory.get_lesson(winner).status == "verified"  # type: ignore[union-attr]
    evidence = memory.list_evidence("lesson", winner)
    assert evidence[0].raw["action"] == "merge"
    assert evidence[0].raw["merged_from"] == [loser]


def test_does_not_merge_across_families(memory: ProcessMemory) -> None:
    _lesson(memory)
    _lesson(memory, family="parsing")
    report = consolidate(memory)
    assert report.merged_lessons == 0
    assert report.kept_lessons == 2


def test_merge_is_idempotent(memory: ProcessMemory) -> None:
    _lesson(memory, successes=("code_repair",))
    _lesson(memory, triggers="duration suffix seconds")
    first = consolidate(memory)
    second = consolidate(memory)
    assert first.merged_lessons == 1
    assert second.merged_lessons == 0


def test_promotes_a_lesson_with_two_family_successes(memory: ProcessMemory) -> None:
    lesson_id = _lesson(
        memory,
        successes=("code_repair", "code_repair", "parsing"),
        provenance={"run": "run-1", "tests": [{"args": ["1x"], "error": "ValueError"}]},
    )
    report = consolidate(memory, run_id="consolidate-test")
    assert report.promoted == 1
    candidates = memory.list_skills(status="candidate")
    assert len(candidates) == 1
    assert candidates[0].provenance["promoted_from_lesson"] == lesson_id
    assert memory.list_evidence("lesson", lesson_id)[0].raw["action"] == "promote"
    again = consolidate(memory)
    assert again.promoted == 0


def test_promotion_requires_tests(memory: ProcessMemory) -> None:
    _lesson(memory, successes=("code_repair", "code_repair", "parsing"))
    assert consolidate(memory).promoted == 0


def test_promotion_requires_two_families(memory: ProcessMemory) -> None:
    _lesson(
        memory,
        successes=("code_repair", "code_repair", "code_repair"),
        provenance={"run": "run-1", "tests": [{"args": ["1x"], "error": "ValueError"}]},
    )
    assert consolidate(memory).promoted == 0


def test_retires_a_failing_skill(memory: ProcessMemory) -> None:
    skill_id = _skill(memory, failures=10)
    report = consolidate(memory)
    assert report.retired == 1
    assert memory.get_skill(skill_id).status == "retired"  # type: ignore[union-attr]


def test_retires_a_stale_skill(memory: ProcessMemory) -> None:
    skill_id = _skill(memory)
    later = datetime.now(UTC) + timedelta(days=60)
    report = consolidate(memory, now=later)
    assert report.retired == 1
    assert memory.get_skill(skill_id).status == "retired"  # type: ignore[union-attr]


def test_cap_keeps_the_strongest(memory: ProcessMemory) -> None:
    strong = _skill(memory, name="strong", rule="Always validate the parsed duration suffix.")
    _skill(memory, name="weak-one", rule="Check the parsed value against its range bound.")
    _skill(memory, name="weak-two", rule="Reject the unknown format before parsing it.")
    for _ in range(3):
        memory.record_outcome("skill", strong, success=True)
    report = consolidate(memory, max_skills=2, similarity=1.0)
    assert report.capped == 1
    assert report.kept_skills == 2
    assert memory.get_skill(strong).status == "verified"  # type: ignore[union-attr]


def test_consolidate_rebuilds_fts_and_vacuums(memory: ProcessMemory) -> None:
    _lesson(memory, successes=("code_repair",))
    _lesson(memory, triggers="duration suffix seconds")
    report = consolidate(memory)
    assert memory.health()["ok"] is True
    assert report.db_bytes > 0
    assert report.over_cap is False


def test_invalid_parameters_are_rejected(memory: ProcessMemory) -> None:
    with pytest.raises(ValueError, match="similarity"):
        consolidate(memory, similarity=1.5)
    with pytest.raises(ValueError, match="timezone-aware"):
        consolidate(memory, now=datetime(2026, 1, 1))
