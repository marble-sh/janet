from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from janet.bench.materialize import materialize
from janet.eval.prompts import SYSTEM_FUNCTION, SYSTEM_INTEGRITY, compose_system
from janet.eval.task import Task, load_suite
from janet.harness.archive import Archive, ArchiveError
from janet.harness.guard import (
    find_leaks,
    forbidden_terms,
    inert_new_skills,
    trigger_vocabulary,
)
from janet.harness.mutate import ProposalError, propose_mutation
from janet.harness.skills import MAX_SKILLS, Skill, SkillError, parse_skills
from janet.harness.spec import SEED_HARNESS, HarnessError, HarnessSpec
from janet.llm import LLMClient, LLMResponse
from janet.loop.generation import classify_operator

RULE = "Validate inputs before computing and raise ValueError for invalid ranges."


def skill(name: str = "validate", triggers: tuple[str, ...] = ("raise",)) -> Skill:
    return Skill(name=name, triggers=triggers, rule=RULE)


def find(tasks: tuple[Task, ...], task_id: str) -> Task:
    return next(task for task in tasks if task.id == task_id)


@pytest.fixture
def tasks(tmp_path: Path) -> tuple[Task, ...]:
    materialize(tmp_path / "anchor", "anchor", force=True)
    return load_suite(tmp_path / "anchor", "anchor")


def test_skill_round_trip_and_matching() -> None:
    restored = Skill.from_dict(skill().to_dict())
    assert restored == skill()
    assert restored.matches("this must raise an error")
    assert not restored.matches("nothing relevant")


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "", "triggers": ["ab"], "rule": RULE},
        {"name": "x", "triggers": [], "rule": RULE},
        {"name": "x", "triggers": ["A!"], "rule": RULE},
        {"name": "x", "triggers": ["ab"], "rule": "short"},
        {"name": "x", "triggers": ["ab"], "rule": RULE, "extra": 1},
        {"name": "x", "triggers": "ab", "rule": RULE},
        {"name": 3, "triggers": ["ab"], "rule": RULE},
    ],
)
def test_invalid_skills_are_rejected(payload: dict[str, Any]) -> None:
    with pytest.raises(SkillError):
        Skill.from_dict(payload)


def test_parse_skills_limits() -> None:
    assert parse_skills(None) == ()
    with pytest.raises(SkillError, match="list"):
        parse_skills("nope")
    too_many = [skill(f"s{i}").to_dict() for i in range(MAX_SKILLS + 1)]
    with pytest.raises(SkillError, match="at most"):
        parse_skills(too_many)
    with pytest.raises(SkillError, match="unique"):
        parse_skills([skill().to_dict(), skill().to_dict()])


def test_pre_skill_hash_is_stable_and_skills_change_it() -> None:
    plain = HarnessSpec(name="a", system_prompt="A" * 50)
    assert "skills" not in plain.content_dict()
    with_skill = HarnessSpec(name="a", system_prompt="A" * 50, skills=(skill(),))
    assert with_skill.hash != plain.hash
    reordered = HarnessSpec(name="b", system_prompt="A" * 50, skills=(skill("b"), skill("a")))
    swapped = HarnessSpec(name="c", system_prompt="A" * 50, skills=(skill("a"), skill("b")))
    assert reordered.hash == swapped.hash


def test_spec_round_trip_with_skills() -> None:
    spec = HarnessSpec(name="a", system_prompt="A" * 50, skills=(skill(),))
    restored = HarnessSpec.from_dict(json.loads(spec.to_json()))
    assert restored.skills == spec.skills
    assert restored.hash == spec.hash
    with pytest.raises(HarnessError):
        HarnessSpec.from_dict({"name": "a", "system_prompt": "A" * 50, "skills": [{"name": 1}]})


def test_compose_system_injects_only_matching_rules(tasks: tuple[Task, ...]) -> None:
    clamp = find(tasks, "clamp_value")
    assert "raise" in clamp.prompt.lower() or "raises" in clamp.prompt.lower()
    composed = compose_system(None, (skill(),), clamp)
    assert composed.startswith(SYSTEM_FUNCTION)
    assert RULE in composed
    miss = compose_system(None, (skill("other", ("zzzz",)),), clamp)
    assert miss == SYSTEM_FUNCTION


def test_integrity_tasks_never_receive_skills(tasks: tuple[Task, ...]) -> None:
    integrity = find(tasks, "missing_revenue")
    always = Skill(name="all", triggers=("the",), rule=RULE)
    assert compose_system("custom prompt " * 5, (always,), integrity) == SYSTEM_INTEGRITY


def test_forbidden_terms_and_leaks(tasks: tuple[Task, ...]) -> None:
    terms = forbidden_terms(tasks)
    assert "flatten_nested" in terms
    assert "median" in terms
    leaky = HarnessSpec(
        name="x", system_prompt="Make sure flatten_nested handles empty lists well."
    )
    assert find_leaks(leaky, terms) == ("flatten_nested",)
    hidden_in_skill = HarnessSpec(
        name="x",
        system_prompt="A" * 50,
        skills=(Skill(name="k", triggers=("raise",), rule=f"Special case for MEDIAN: {RULE}"),),
    )
    assert "median" in find_leaks(hidden_in_skill, terms)
    clean = HarnessSpec(name="x", system_prompt="Always import every module you use.")
    assert find_leaks(clean, terms) == ()
    assert find_leaks(HarnessSpec(name="x", system_prompt="medians are fine " * 4), terms) == ()


def test_inert_skill_detection(tasks: tuple[Task, ...]) -> None:
    parent = HarnessSpec(name="p", system_prompt="A" * 50)
    inert = HarnessSpec(name="c", system_prompt="A" * 50, skills=(skill("dead", ("qqqq zzzz",)),))
    live = HarnessSpec(name="c", system_prompt="A" * 50, skills=(skill("live", ("raise",)),))
    assert inert_new_skills(inert, parent, tasks) == ("dead",)
    assert inert_new_skills(live, parent, tasks) == ()
    # An unchanged inherited skill is not re-judged.
    assert inert_new_skills(inert, inert, tasks) == ()


def test_classify_operator() -> None:
    base = HarnessSpec(name="p", system_prompt="A" * 50)
    assert classify_operator(base, base) == "none"
    assert classify_operator(base, HarnessSpec(name="c", system_prompt="B" * 50)) == "prompt"
    assert (
        classify_operator(base, HarnessSpec(name="c", system_prompt="A" * 50, skills=(skill(),)))
        == "skills"
    )
    assert (
        classify_operator(
            base, HarnessSpec(name="c", system_prompt="A" * 50, samples_per_attempt=2)
        )
        == "budget"
    )
    both = HarnessSpec(name="c", system_prompt="B" * 50, max_attempts=2)
    assert classify_operator(base, both) == "prompt+budget"


class ScriptedClient:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def generate(self, prompt: str, **_kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(
            text=json.dumps(self.payloads.pop(0)),
            model="fake",
            eval_count=1,
            prompt_eval_count=1,
            eval_duration_ns=1,
            total_duration_ns=1,
        )


def _payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "c",
        "system_prompt": SEED_HARNESS.system_prompt,
        "max_attempts": 3,
        "samples_per_attempt": 3,
        "mutation_note": "n",
        "skills": [skill().to_dict()],
    }
    base.update(overrides)
    return base


def test_proposer_rejects_task_names_then_accepts_general_rules(
    tasks: tuple[Task, ...],
) -> None:
    leaky = _payload(skills=[{**skill().to_dict(), "rule": "Handle flatten_nested lists deeply."}])
    client = ScriptedClient([leaky, _payload()])
    spec = propose_mutation(
        cast(LLMClient, client), SEED_HARNESS, "fb", forbidden=forbidden_terms(tasks)
    )
    assert spec.skills == (skill(),)
    assert "benchmark identifiers" in client.prompts[1]


def test_proposer_gives_up_when_every_attempt_leaks(tasks: tuple[Task, ...]) -> None:
    leaky = _payload(system_prompt="Fix flatten_nested carefully and completely please.")
    client = ScriptedClient([leaky, leaky, leaky])
    with pytest.raises(ProposalError, match="benchmark identifiers"):
        propose_mutation(
            cast(LLMClient, client), SEED_HARNESS, "fb", forbidden=forbidden_terms(tasks)
        )


def test_trigger_vocabulary_is_dev_only_and_excludes_identifiers(
    tmp_path: Path, tasks: tuple[Task, ...]
) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    dev = load_suite(tmp_path / "dev", "dev")
    terms = forbidden_terms((*dev, *tasks))
    vocab = trigger_vocabulary(dev, terms)
    assert vocab
    assert len(vocab) <= 40
    assert "implementation" in vocab
    assert not set(vocab) & {"flatten", "clamp", "median", "chunked"}
    # Words that occur only in anchor-only tasks never appear.
    anchor_only_words = {"transpose", "rotate", "ipv4", "dedupe"}
    assert not set(vocab) & anchor_only_words
    assert trigger_vocabulary((), ()) == ()


def test_inert_skill_is_a_retryable_proposal_error(tasks: tuple[Task, ...]) -> None:
    parent = SEED_HARNESS
    dead = _payload(skills=[skill("dead", ("qqqq zzzz",)).to_dict()])
    client = ScriptedClient([dead, _payload()])
    spec = propose_mutation(
        cast(LLMClient, client),
        parent,
        "fb",
        vocabulary=("implementation", "returns"),
        inert_check=lambda candidate: inert_new_skills(candidate, parent, tasks),
    )
    assert spec.skills == (skill(),)
    assert "match no task text" in client.prompts[1]
    assert "implementation, returns" in client.prompts[0]


def test_omitted_skills_key_means_unchanged() -> None:
    parent = HarnessSpec(name="p", system_prompt="A" * 50, skills=(skill(),))
    payload = _payload(system_prompt="B" * 50)
    del payload["skills"]
    spec = propose_mutation(cast(LLMClient, ScriptedClient([payload])), parent, "fb")
    assert spec.skills == parent.skills
    assert spec.system_prompt == "B" * 50


def test_archive_dev_samples_form_a_running_mean(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    try:
        spec = HarnessSpec(name="x", system_prompt="A" * 50)
        archive.add(spec, dev_score=1.0, anchor_score=0.5, details={"dev_samples": [1.0]})
        record = archive.record_dev_sample(spec.hash, 0.5)
        assert record.dev_score == 0.75
        assert record.details["dev_samples"] == [1.0, 0.5]
        assert archive.record_dev_sample(spec.hash, 0.0).dev_score == 0.5
        with pytest.raises(ArchiveError, match="unknown"):
            archive.record_dev_sample("nope", 1.0)
        # A legacy record without samples seeds the mean from its stored score.
        legacy = HarnessSpec(name="y", system_prompt="B" * 50)
        archive.add(legacy, dev_score=0.8, anchor_score=None)
        assert archive.record_dev_sample(legacy.hash, 0.6).dev_score == pytest.approx(0.7)
    finally:
        archive.close()
