from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from janet.bench.materialize import materialize
from janet.bench.seeds import SEEDS
from janet.config import Config, LoopConfig, PathsConfig
from janet.eval.task import Task, load_suite
from janet.governor import ResourceGovernor
from janet.harness.archive import Archive
from janet.harness.bandit import BanditState
from janet.harness.spec import SEED_HARNESS
from janet.llm import LLMClient, LLMResponse
from janet.loop.generation import LoopError, heldout_score, run_generation, run_loop
from janet.probe import Budget

_MUTATION_PROMPT = (
    "You are a careful Python engineer. Import every module you use. "
    "Return only the complete corrected solution in one ```python code block. "
    "Preserve the required function signature."
)


class FakeLoopClient:
    def __init__(self, reference: str) -> None:
        self.reference = reference
        self.mutation_calls = 0
        self.task_calls = 0

    def generate(
        self,
        _prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        **_kwargs: Any,
    ) -> LLMResponse:
        if json_mode:
            self.mutation_calls += 1
            return LLMResponse(
                text=json.dumps(
                    {
                        "name": "import-discipline",
                        "system_prompt": _MUTATION_PROMPT,
                        "max_attempts": 3,
                        "samples_per_attempt": 1,
                        "mutation_note": "add import discipline",
                    }
                ),
                model="fake",
                eval_count=1,
                prompt_eval_count=1,
                eval_duration_ns=1,
                total_duration_ns=1,
            )
        self.task_calls += 1
        return LLMResponse(
            text=self.reference,
            model="fake",
            eval_count=1,
            prompt_eval_count=1,
            eval_duration_ns=1,
            total_duration_ns=1,
        )


def _config(tmp_path: Path, *, promote: bool) -> Config:
    return Config(
        paths=PathsConfig(
            runs=tmp_path / "runs",
            archive=tmp_path / "archive",
            archive_db=tmp_path / "archive" / "archive.sqlite3",
            events=tmp_path / "events.jsonl",
            anchor=tmp_path / "anchor",
            bandit=tmp_path / "runs" / "bandit.json",
        ),
        loop=LoopConfig(promote=promote),
    )


def _tasks(tmp_path: Path) -> tuple[Task, Task]:
    materialize(tmp_path / "dev", "dev", force=True)
    materialize(tmp_path / "anchor", "anchor", force=True)
    dev_task = load_suite(tmp_path / "dev", "dev")[0]
    anchor_task = load_suite(tmp_path / "anchor", "anchor")[0]
    return dev_task, anchor_task


def reference_for(task: Task) -> str:
    return (task.source_dir / "reference.py").read_text(encoding="utf-8")


def test_run_generation_promotes_an_improvement(tmp_path: Path, budget: Budget) -> None:
    dev_task, anchor_task = _tasks(tmp_path)
    config = _config(tmp_path, promote=True)
    archive = Archive(config.paths.archive)
    try:
        archive.ensure_seed(SEED_HARNESS, dev_score=0.0, anchor_score=0.0)
        client = FakeLoopClient(reference_for(dev_task))
        bandit = BanditState.fresh()
        result = run_generation(
            archive,
            cast(LLMClient, client),
            ResourceGovernor(budget),
            config,
            tmp_path / "run",
            (dev_task,),
            (anchor_task,),
            generation=1,
            limit=1,
            bandit=bandit,
        )
        assert result.improved
        assert result.promoted
        assert result.bandit_arm == "prompt"
        assert result.reward == 1.0
        assert bandit.arms["prompt"].pulls == 1
        assert result.dev_score == 1.0
        assert result.anchor_score == 1.0
        assert result.candidate_hash is not None
        best = archive.best()
        assert best is not None
        assert best.hash == result.candidate_hash
        rolled = archive.rollback()
        assert rolled is not None
        assert rolled.system_prompt == _MUTATION_PROMPT
        assert client.mutation_calls == 1
    finally:
        archive.close()


def test_run_generation_archives_a_non_improvement(tmp_path: Path, budget: Budget) -> None:
    dev_task, anchor_task = _tasks(tmp_path)
    config = _config(tmp_path, promote=True)
    archive = Archive(config.paths.archive)
    try:
        archive.ensure_seed(SEED_HARNESS, dev_score=1.0, anchor_score=1.0)
        client = FakeLoopClient(reference_for(dev_task))
        result = run_generation(
            archive,
            cast(LLMClient, client),
            ResourceGovernor(budget),
            config,
            tmp_path / "run",
            (dev_task,),
            (anchor_task,),
            generation=1,
            limit=1,
        )
        assert not result.improved
        assert not result.promoted
        assert result.candidate_hash is not None
        assert len(archive.list()) == 2
    finally:
        archive.close()


def test_run_generation_promotes_on_an_anchor_tie(tmp_path: Path, budget: Budget) -> None:
    dev_task, anchor_task = _tasks(tmp_path)
    config = _config(tmp_path, promote=True)
    archive = Archive(config.paths.archive)
    try:
        archive.ensure_seed(SEED_HARNESS, dev_score=1.0, anchor_score=0.0)
        client = FakeLoopClient(reference_for(dev_task))
        result = run_generation(
            archive,
            cast(LLMClient, client),
            ResourceGovernor(budget),
            config,
            tmp_path / "run",
            (dev_task,),
            (anchor_task,),
            generation=1,
            limit=1,
        )
        assert result.dev_score == 1.0
        assert result.anchor_score == 1.0
        assert result.improved
        assert result.promoted
    finally:
        archive.close()


def test_run_loop_seeds_and_runs(tmp_path: Path, budget: Budget) -> None:
    dev_task, anchor_task = _tasks(tmp_path)
    config = _config(tmp_path, promote=False)
    client = FakeLoopClient(reference_for(dev_task))
    results = run_loop(
        cast(LLMClient, client),
        ResourceGovernor(budget),
        config,
        (dev_task,),
        (anchor_task,),
        generations=1,
        limit=1,
    )
    assert len(results) == 1
    assert results[0].dev_score == 1.0
    runs = list((tmp_path / "runs").glob("run-*"))
    assert len(runs) == 1
    assert (runs[0] / "generations.json").is_file()
    archive = Archive(config.paths.archive)
    try:
        assert len(archive.list()) >= 2
        assert config.paths.bandit.is_file()
        bandit = BanditState.load(config.paths.bandit)
        assert sum(stats.pulls for stats in bandit.arms.values()) == 1
        assert results[0].bandit_arm == "prompt"
    finally:
        archive.close()


def test_run_generation_records_a_proposal_error_reward(
    tmp_path: Path, budget: Budget, monkeypatch: pytest.MonkeyPatch
) -> None:
    from janet.harness.mutate import ProposalError

    config = _config(tmp_path, promote=False)
    archive = Archive(config.paths.archive)
    try:
        archive.ensure_seed(SEED_HARNESS, dev_score=0.5, anchor_score=0.5)
        bandit = BanditState.fresh()

        def explode(*_args: object, **_kwargs: object) -> None:
            raise ProposalError("no valid mutation")

        monkeypatch.setattr("janet.loop.generation.propose_mutation", explode)
        result = run_generation(
            archive,
            cast(LLMClient, object()),
            ResourceGovernor(budget),
            config,
            tmp_path / "run",
            (),
            (),
            generation=1,
            bandit=bandit,
        )
        assert not result.improved
        assert result.bandit_arm == "prompt"
        assert result.reward == -0.25
        assert bandit.arms["prompt"].pulls == 1
        assert len(archive.list()) == 1
    finally:
        archive.close()


def test_paired_parent_remeasure_blocks_a_stale_baseline(tmp_path: Path, budget: Budget) -> None:
    """A parent recorded with an unlucky 0.0 must not make any candidate look better."""

    dev_task, anchor_task = _tasks(tmp_path)
    config = _config(tmp_path, promote=True)
    archive = Archive(config.paths.archive)
    try:
        archive.ensure_seed(SEED_HARNESS, dev_score=0.0, anchor_score=1.0)
        result = run_generation(
            archive,
            cast(LLMClient, FakeLoopClient(reference_for(dev_task))),
            ResourceGovernor(budget),
            config,
            tmp_path / "run",
            (dev_task,),
            (anchor_task,),
            generation=1,
            limit=1,
        )
        assert result.parent_dev == 1.0  # fresh measurement, not the stale 0.0
        assert result.dev_score == 1.0
        assert not result.improved
        assert not result.promoted
        assert result.operator == "prompt+budget"  # fake proposer also drops samples 3 -> 1
        seed = archive.get(SEED_HARNESS.hash)
        assert seed is not None
        assert seed.dev_score == 0.5  # running mean of the stale 0.0 and the fresh 1.0
    finally:
        archive.close()


def test_accepted_variant_becomes_the_next_parent(tmp_path: Path, budget: Budget) -> None:
    """Recursion: lineage moves forward from the accepted variant, not the seed."""

    dev_task, anchor_task = _tasks(tmp_path)
    config = _config(tmp_path, promote=False)
    archive = Archive(config.paths.archive)
    try:
        archive.ensure_seed(SEED_HARNESS, dev_score=1.0, anchor_score=0.0)
        client = cast(LLMClient, FakeLoopClient(reference_for(dev_task)))
        kwargs: dict[str, Any] = {"limit": 1}
        first = run_generation(
            archive,
            client,
            ResourceGovernor(budget),
            config,
            tmp_path / "run",
            (dev_task,),
            (anchor_task,),
            generation=1,
            **kwargs,
        )
        assert first.improved
        assert first.candidate_hash is not None
        accepted = archive.get(first.candidate_hash)
        assert accepted is not None
        assert accepted.status == "accepted"
        current = archive.current()
        assert current is not None
        assert current.hash == first.candidate_hash
        # The fake proposer returns the same spec again; relative to the new parent
        # that is "no change", which proves the parent moved off the seed.
        second = run_generation(
            archive,
            client,
            ResourceGovernor(budget),
            config,
            tmp_path / "run",
            (dev_task,),
            (anchor_task,),
            generation=2,
            **kwargs,
        )
        assert second.parent_hash == first.candidate_hash
        assert second.error is not None
        assert "no valid mutation" in second.error
        lineage = archive.lineage(first.candidate_hash)
        assert [record.hash for record in lineage] == [SEED_HARNESS.hash, first.candidate_hash]
    finally:
        archive.close()


def test_heldout_score_excludes_dev_tasks() -> None:
    class Row:
        def __init__(self, task_id: str, score: float) -> None:
            self.task_id = task_id
            self.score = score

    class Report:
        rows = (Row("dev_a", 1.0), Row("new_b", 0.5), Row("new_c", 0.0))

    assert heldout_score(Report(), {"dev_a"}) == 0.25
    assert heldout_score(Report(), {"dev_a", "new_b", "new_c"}) is None


def test_generation_records_heldout_anchor_in_details(tmp_path: Path, budget: Budget) -> None:
    dev_task, anchor_task = _tasks(tmp_path)
    config = _config(tmp_path, promote=False)
    archive = Archive(config.paths.archive)
    try:
        archive.ensure_seed(SEED_HARNESS, dev_score=1.0, anchor_score=0.0)
        result = run_generation(
            archive,
            cast(LLMClient, FakeLoopClient(reference_for(dev_task))),
            ResourceGovernor(budget),
            config,
            tmp_path / "run",
            (dev_task,),
            (anchor_task,),
            generation=1,
            limit=1,
        )
        assert result.candidate_hash is not None
        record = archive.get(result.candidate_hash)
        assert record is not None
        assert "anchor_heldout" in record.details
    finally:
        archive.close()


def test_anchor_gain_with_small_dev_dip_is_accepted(tmp_path: Path, budget: Budget) -> None:
    """dev_margin lets a strict anchor gain through despite sampling noise on dev."""

    dev_task, anchor_task = _tasks(tmp_path)
    config = Config(
        paths=_config(tmp_path, promote=False).paths,
        loop=LoopConfig(promote=False, dev_margin=0.03),
    )
    archive = Archive(config.paths.archive)
    try:
        archive.ensure_seed(SEED_HARNESS, dev_score=1.0, anchor_score=0.0)
        # Parent dev (fresh) = 1.0 and candidate dev = 1.0; anchor 0.0 -> 1.0.
        result = run_generation(
            archive,
            cast(LLMClient, FakeLoopClient(reference_for(dev_task))),
            ResourceGovernor(budget),
            config,
            tmp_path / "run",
            (dev_task,),
            (anchor_task,),
            generation=1,
            limit=1,
        )
        assert result.improved
        assert result.anchor_score == 1.0
        assert result.parent_anchor == 0.0
    finally:
        archive.close()


def test_run_loop_refuses_an_archive_from_a_different_anchor(
    tmp_path: Path, budget: Budget
) -> None:
    dev_task, anchor_task = _tasks(tmp_path)
    config = _config(tmp_path, promote=False)
    client = cast(LLMClient, FakeLoopClient(reference_for(dev_task)))
    run_loop(
        client,
        ResourceGovernor(budget),
        config,
        (dev_task,),
        (anchor_task,),
        generations=1,
        limit=1,
    )
    materialize(tmp_path / "anchor", "anchor", seeds=SEEDS[:3], force=True)
    with pytest.raises(LoopError, match="different anchor"):
        run_loop(
            client,
            ResourceGovernor(budget),
            config,
            (dev_task,),
            (anchor_task,),
            generations=1,
            limit=1,
        )


def test_run_loop_requires_an_anchor(tmp_path: Path, budget: Budget) -> None:
    dev_task, _ = _tasks(tmp_path)
    config = _config(tmp_path, promote=False)
    with pytest.raises(LoopError, match="anchor"):
        run_loop(
            cast(LLMClient, FakeLoopClient(reference_for(dev_task))),
            ResourceGovernor(budget),
            config,
            (dev_task,),
            (),
            generations=1,
        )
