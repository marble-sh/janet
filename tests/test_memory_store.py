from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from janet.memory import MemoryError, ProcessMemory


@pytest.fixture
def memory(tmp_path: Path) -> Iterator[ProcessMemory]:
    with ProcessMemory(tmp_path / "memory.sqlite3") as store:
        yield store


def _lesson(store: ProcessMemory, **overrides: object) -> int:
    values: dict[str, object] = {
        "task_id": "parse_duration",
        "task_family": "code_repair",
        "failure_signature": "timeout on invalid suffix",
        "trigger_terms": "duration timeout suffix",
        "rule": "Validate the suffix before multiplying the numeric value.",
        "patch_summary": "Reject an unknown duration suffix.",
        "provenance": {"variant": "abc", "run": "run-1"},
    }
    values.update(overrides)
    return store.add_lesson(**values)  # type: ignore[arg-type]


def _skill(store: ProcessMemory, **overrides: object) -> str:
    values: dict[str, object] = {
        "name": "duration-validation",
        "triggers": ("duration", "timeout"),
        "rule": "Always validate the suffix before parsing the numeric prefix.",
        "tests": [{"args": ["1x"], "error": "ValueError"}],
        "provenance": {"variant": "abc"},
    }
    values.update(overrides)
    return store.add_skill(**values)  # type: ignore[arg-type]


def test_add_and_get_lesson(memory: ProcessMemory) -> None:
    lesson_id = _lesson(memory)
    record = memory.get_lesson(lesson_id)
    assert record is not None
    assert record.task_id == "parse_duration"
    assert record.status == "candidate"
    assert record.provenance["variant"] == "abc"
    assert memory.stats()["lessons"] == {"candidate": 1}


def test_duplicate_lesson_is_idempotent(memory: ProcessMemory) -> None:
    first = _lesson(memory)
    second = _lesson(memory)
    assert first == second
    assert memory.stats()["lessons"] == {"candidate": 1}


def test_search_lessons_is_fts_and_status_filtered(memory: ProcessMemory) -> None:
    lesson_id = _lesson(memory)
    assert memory.search_lessons("timeout") == ()
    memory.set_status("lesson", lesson_id, "verified")
    hits = memory.search_lessons("timeout")
    assert [record.id for record in hits] == [lesson_id]
    assert memory.search_lessons("timeout", task_family="other") == ()


def test_quarantined_entries_are_not_retrieved(memory: ProcessMemory) -> None:
    lesson_id = _lesson(memory)
    memory.set_status("lesson", lesson_id, "quarantined")
    assert memory.search_lessons("timeout") == ()
    assert memory.list_lessons(status="quarantined")[0].id == lesson_id


def test_add_skill_and_search(memory: ProcessMemory) -> None:
    skill_id = _skill(memory)
    record = memory.get_skill(skill_id)
    assert record is not None
    assert record.triggers == ("duration", "timeout")
    assert record.tests[0]["error"] == "ValueError"
    memory.set_status("skill", skill_id, "verified")
    assert [hit.id for hit in memory.search_skills("duration")] == [skill_id]


def test_status_transitions(memory: ProcessMemory) -> None:
    lesson_id = _lesson(memory)
    memory.set_status("lesson", lesson_id, "verified")
    with pytest.raises(MemoryError, match="illegal lesson transition"):
        memory.set_status("lesson", lesson_id, "candidate")
    memory.set_status("lesson", lesson_id, "quarantined")
    memory.set_status("lesson", lesson_id, "candidate")
    memory.set_status("lesson", lesson_id, "retired")
    with pytest.raises(MemoryError, match="illegal lesson transition"):
        memory.set_status("lesson", lesson_id, "verified")


def test_record_outcome_and_evidence(memory: ProcessMemory) -> None:
    lesson_id = _lesson(memory)
    memory.record_outcome("lesson", lesson_id, success=True, cost_tokens=12)
    memory.record_outcome("lesson", lesson_id, success=False, cost_tokens=3)
    record = memory.get_lesson(lesson_id)
    assert record is not None
    assert (record.success_count, record.failure_count, record.cost_tokens) == (1, 1, 15)
    evidence_id = memory.record_evidence(
        "lesson",
        lesson_id,
        run_id="run-1",
        split="dev",
        verifier_version="v1",
        test_ids=["parse_duration:0"],
        raw={"passed": True},
        score_before=0.5,
        score_after=1.0,
    )
    assert evidence_id > 0
    assert memory.stats()["evidence"] == 1


def test_evidence_requires_a_known_subject(memory: ProcessMemory) -> None:
    with pytest.raises(MemoryError, match="unknown lesson"):
        memory.record_evidence(
            "lesson",
            999,
            run_id="run-1",
            split="dev",
            verifier_version="v1",
            test_ids=[],
            raw={},
        )


def test_record_retrieval(memory: ProcessMemory) -> None:
    lesson_id = _lesson(memory)
    memory.record_retrieval(
        "lesson",
        lesson_id,
        task_id="parse_duration",
        task_family="code_repair",
        query="timeout",
        score=0.4,
        rank=0,
    )
    assert memory.stats()["retrieval_events"] == 1


def test_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    with ProcessMemory(path) as store:
        lesson_id = _lesson(store)
        store.set_status("lesson", lesson_id, "verified")
    with ProcessMemory(path) as reopened:
        assert reopened.search_lessons("timeout")[0].id == lesson_id
        assert reopened.stats()["schema_version"] == 1


def test_schema_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    ProcessMemory(path).close()
    with ProcessMemory(path) as reopened:
        assert reopened.stats()["lessons"] == {}


def test_invalid_inputs_are_rejected(memory: ProcessMemory) -> None:
    with pytest.raises(MemoryError, match="status"):
        memory.list_lessons(status="bogus")
    with pytest.raises(MemoryError, match="trigger"):
        _skill(memory, triggers=("x",))
    with pytest.raises(MemoryError, match="skill rule"):
        _skill(memory, rule="too short")
    with pytest.raises(MemoryError, match="limit"):
        memory.search_lessons("x", limit=0)
    with pytest.raises(MemoryError, match="cost_tokens"):
        lesson_id = _lesson(memory)
        memory.record_outcome("lesson", lesson_id, success=True, cost_tokens=-1)


def test_rebuild_fts(memory: ProcessMemory) -> None:
    lesson_id = _lesson(memory)
    memory.set_status("lesson", lesson_id, "verified")
    memory.rebuild_fts()
    assert memory.search_lessons("timeout")[0].id == lesson_id


def test_health_reports_integrity_and_fts_parity(memory: ProcessMemory) -> None:
    _lesson(memory)
    _skill(memory)
    payload = memory.health()
    assert payload["ok"] is True
    assert payload["integrity"] == "ok"
    assert payload["schema_version"] == 1
    assert payload["indexes"]["lessons"] == {"content": 1, "index": 1, "ok": True}
    assert payload["indexes"]["skills"] == {"content": 1, "index": 1, "ok": True}
