from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from janet.bench.repair_bench import RepairBenchReport, RepairBenchRow, RepairBenchSummary
from janet.harness.archive import Archive, ArchiveError
from janet.harness.mutate import (
    MUTATOR_SYSTEM,
    ProposalError,
    build_feedback,
    propose_mutation,
)
from janet.harness.spec import SEED_HARNESS, HarnessError, HarnessSpec
from janet.llm import LLMClient, LLMResponse


def test_spec_round_trip() -> None:
    spec = HarnessSpec(
        name="test",
        system_prompt="A" * 50,
        max_attempts=2,
        samples_per_attempt=2,
        mutation_note="note",
        parent="parent",
    )
    restored = HarnessSpec.from_dict(json.loads(spec.to_json()))
    assert restored == spec


def test_hash_is_content_addressed() -> None:
    first = HarnessSpec(name="a", system_prompt="A" * 50)
    second = HarnessSpec(name="b", system_prompt="A" * 50, mutation_note="different")
    assert first.hash == second.hash
    third = HarnessSpec(name="a", system_prompt="B" * 50)
    assert first.hash != third.hash


def test_spec_validation() -> None:
    with pytest.raises(HarnessError, match="name"):
        HarnessSpec(name="", system_prompt="A" * 50).validate()
    with pytest.raises(HarnessError, match="system_prompt"):
        HarnessSpec(name="x", system_prompt="short").validate()
    with pytest.raises(HarnessError, match="max_attempts"):
        HarnessSpec(name="x", system_prompt="A" * 50, max_attempts=0).validate()
    with pytest.raises(HarnessError, match="samples_per_attempt"):
        HarnessSpec(name="x", system_prompt="A" * 50, samples_per_attempt=5).validate()
    with pytest.raises(HarnessError, match="mutation_note"):
        HarnessSpec(name="x", system_prompt="A" * 50, mutation_note="n" * 501).validate()


def test_from_dict_rejects_unknown_fields() -> None:
    with pytest.raises(HarnessError, match="unknown"):
        HarnessSpec.from_dict({"name": "x", "system_prompt": "A" * 50, "extra": 1})


def test_from_dict_sets_parent() -> None:
    spec = HarnessSpec.from_dict({"name": "x", "system_prompt": "A" * 50}, parent="parent-hash")
    assert spec.parent == "parent-hash"


def test_archive_adds_and_selects_the_best(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    try:
        low = HarnessSpec(name="low", system_prompt="A" * 50)
        high = HarnessSpec(name="high", system_prompt="B" * 50)
        archive.add(low, dev_score=0.4, anchor_score=0.3)
        archive.add(high, dev_score=0.8, anchor_score=0.7)
        assert archive.get(low.hash) is not None
        best = archive.best()
        assert best is not None
        assert best.hash == high.hash
        assert len(archive.list()) == 2
        assert (tmp_path / "archive" / "variants" / ".git").is_dir()
    finally:
        archive.close()


def test_archive_promote_and_rollback(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    try:
        spec = HarnessSpec(name="winner", system_prompt="A" * 50)
        archive.add(spec, dev_score=1.0, anchor_score=1.0)
        promoted = archive.promote(spec.hash)
        assert promoted.status == "promoted"
        rolled = archive.rollback()
        assert rolled is not None
        assert rolled.hash == spec.hash
    finally:
        archive.close()


def test_archive_current_root_accept_and_lineage(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    try:
        assert archive.seed() is None
        assert archive.current() is None
        seed = HarnessSpec(name="seed", system_prompt="A" * 50)
        child = HarnessSpec(name="child", system_prompt="B" * 50, parent=seed.hash)
        rejected = HarnessSpec(name="rej", system_prompt="C" * 50, parent=child.hash)
        archive.add(seed, dev_score=0.5, anchor_score=0.5)
        assert archive.current() is not None
        assert archive.current().hash == seed.hash  # type: ignore[union-attr]
        archive.add(child, dev_score=0.4, anchor_score=0.6)
        archive.add(rejected, dev_score=1.0, anchor_score=1.0)
        # A high dev score alone never makes a variant the parent.
        assert archive.current().hash == seed.hash  # type: ignore[union-attr]
        assert archive.accept(child.hash).status == "accepted"
        assert archive.accept(child.hash).status == "accepted"  # idempotent
        assert archive.current().hash == child.hash  # type: ignore[union-attr]
        assert [r.hash for r in archive.lineage(rejected.hash)] == [
            seed.hash,
            child.hash,
            rejected.hash,
        ]
        archive.promote(child.hash)
        assert archive.current().hash == child.hash  # type: ignore[union-attr]
        with pytest.raises(ArchiveError, match="unknown"):
            archive.accept("nope")
        assert archive.lineage("nope") == ()
    finally:
        archive.close()


def test_archive_promote_unknown(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    try:
        with pytest.raises(ArchiveError, match="unknown"):
            archive.promote("nope")
    finally:
        archive.close()


def test_archive_ensure_seed_is_idempotent(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    try:
        first = archive.ensure_seed(SEED_HARNESS, dev_score=0.5, anchor_score=0.5)
        second = archive.ensure_seed(SEED_HARNESS, dev_score=0.9, anchor_score=0.9)
        assert first.hash == second.hash
        assert len(archive.list()) == 1
    finally:
        archive.close()


def test_archive_upserts_scores(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    try:
        spec = HarnessSpec(name="x", system_prompt="A" * 50)
        archive.add(spec, dev_score=0.1, anchor_score=None)
        archive.add(spec, dev_score=0.9, anchor_score=0.8)
        record = archive.get(spec.hash)
        assert record is not None
        assert record.dev_score == 0.9
        assert record.anchor_score == 0.8
        assert len(archive.list()) == 1
    finally:
        archive.close()


def _report() -> RepairBenchReport:
    row = RepairBenchRow(
        task_id="parse_duration",
        family="code_repair",
        baseline_score=0.0,
        repair_score=0.33,
        attempts=3,
        passed=False,
        improved=True,
        regressed=False,
        eval_tokens=100,
        wall_s=1.0,
        error="wrong output",
        artifact_dir="attempts/parse_duration",
    )
    return RepairBenchReport(
        run_id="run",
        model="fake",
        split="dev",
        rows=(row,),
        summary=RepairBenchSummary(1, 0, 0, 0.0, 0.33, 1, 0, 100, 1.0),
    )


def test_build_feedback() -> None:
    text = build_feedback(_report())
    assert "parse_duration" in text
    assert "0.33" in text
    assert "Held-out anchor: not evaluated" in text


def test_build_feedback_includes_anchor_score() -> None:
    text = build_feedback(_report(), anchor_score=0.91)
    assert "Held-out anchor score: 0.9100" in text
    assert "inputs and expected outputs withheld" in text


class FakeMutationClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def generate(self, _prompt: str, *, system: str | None = None, **_kwargs: Any) -> LLMResponse:
        assert system == MUTATOR_SYSTEM
        self.calls += 1
        return LLMResponse(
            text=self.responses.pop(0),
            model="fake",
            eval_count=1,
            prompt_eval_count=1,
            eval_duration_ns=1,
            total_duration_ns=1,
        )


def _mutation_payload(prompt: str = "B" * 60) -> str:
    return json.dumps(
        {
            "name": "candidate",
            "system_prompt": prompt,
            "max_attempts": 2,
            "samples_per_attempt": 2,
            "mutation_note": "test change",
        }
    )


def test_propose_mutation() -> None:
    client = FakeMutationClient([_mutation_payload()])
    spec = propose_mutation(cast_client(client), SEED_HARNESS, "feedback")
    assert spec.parent == SEED_HARNESS.hash
    assert spec.hash != SEED_HARNESS.hash
    assert spec.max_attempts == 2


def test_propose_mutation_retries_after_invalid_json() -> None:
    client = FakeMutationClient(["not json", _mutation_payload()])
    spec = propose_mutation(cast_client(client), SEED_HARNESS, "feedback")
    assert spec.hash != SEED_HARNESS.hash
    assert client.calls == 2


def test_propose_mutation_rejects_no_change() -> None:
    unchanged = json.dumps(
        {
            "name": "same",
            "system_prompt": SEED_HARNESS.system_prompt,
            "max_attempts": SEED_HARNESS.max_attempts,
            "samples_per_attempt": SEED_HARNESS.samples_per_attempt,
            "mutation_note": "no change",
        }
    )
    client = FakeMutationClient([unchanged, unchanged, unchanged])
    with pytest.raises(ProposalError, match="no valid mutation"):
        propose_mutation(cast_client(client), SEED_HARNESS, "feedback")


def cast_client(client: FakeMutationClient) -> LLMClient:
    return client  # type: ignore[return-value]
