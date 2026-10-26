"""The generation loop: propose, evaluate, select, archive, optionally promote.

Selection is paired and Pareto:

* the parent is re-measured on dev in the *same* generation, so selection does
  not chase the parent's lucky past sample (winner's curse);
* the candidate must improve dev without regressing the anchor, or improve the
  deterministic hidden anchor while dev stays within ``loop.dev_margin`` of the
  fresh parent measurement (a deliberate allowance for sampling noise).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from janet.bench.baseline import run_baseline
from janet.bench.repair_bench import RepairBenchReport, run_repair_benchmark
from janet.config import Config
from janet.eval.task import Task
from janet.events import EventLog
from janet.governor import ResourceGovernor
from janet.harness.archive import Archive, ArchiveError, HarnessRecord
from janet.harness.bandit import OPERATOR_INSTRUCTIONS, BanditState, reward_for
from janet.harness.guard import forbidden_terms, inert_new_skills, trigger_vocabulary
from janet.harness.mutate import ProposalError, build_feedback, propose_mutation
from janet.harness.spec import SEED_HARNESS, HarnessSpec
from janet.llm import LLMClient
from janet.loop.decoupling import Decoupling, detect
from janet.preempt import StopCheck


class LoopError(RuntimeError):
    """Raised when the loop cannot take a safe step."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    generation: int
    parent_hash: str
    candidate_hash: str | None
    parent_dev: float | None
    dev_score: float | None
    anchor_score: float | None
    improved: bool
    promoted: bool
    error: str | None
    parent_anchor: float | None = None
    anchor_heldout: float | None = None
    parent_heldout: float | None = None
    operator: str | None = None
    strict_blocked: bool = False
    bandit_arm: str | None = None
    reward: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "parent_hash": self.parent_hash,
            "candidate_hash": self.candidate_hash,
            "parent_dev": self.parent_dev,
            "parent_anchor": self.parent_anchor,
            "dev_score": self.dev_score,
            "anchor_score": self.anchor_score,
            "anchor_heldout": self.anchor_heldout,
            "parent_heldout": self.parent_heldout,
            "improved": self.improved,
            "promoted": self.promoted,
            "operator": self.operator,
            "strict_blocked": self.strict_blocked,
            "bandit_arm": self.bandit_arm,
            "reward": self.reward,
            "error": self.error,
        }


def anchor_fingerprint(anchor_root: Path) -> str:
    manifest = anchor_root.expanduser() / "manifest.json"
    if not manifest.is_file():
        raise LoopError(f"anchor manifest missing: {manifest}")
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def heldout_score(report: Any, dev_ids: set[str]) -> float | None:
    """Mean anchor score over tasks the dev split never contained.

    The anchor also re-tests dev tasks on hidden inputs, which measures input
    generalization only. Gains on tasks the loop never optimized against are the
    honest transfer signal, so they are tracked separately.
    """

    scores = [row.score for row in report.rows if row.task_id not in dev_ids]
    return sum(scores) / len(scores) if scores else None


def classify_operator(parent: HarnessSpec, candidate: HarnessSpec) -> str:
    """Which part of the harness a mutation touched (input to the P6 bandit)."""

    changed: list[str] = []
    if candidate.system_prompt.strip() != parent.system_prompt.strip():
        changed.append("prompt")
    if candidate.content_dict().get("skills") != parent.content_dict().get("skills"):
        changed.append("skills")
    if (candidate.max_attempts, candidate.samples_per_attempt) != (
        parent.max_attempts,
        parent.samples_per_attempt,
    ):
        changed.append("budget")
    return "+".join(changed) or "none"


def _seed(
    archive: Archive,
    client: LLMClient,
    governor: ResourceGovernor,
    config: Config,
    run_dir: Path,
    tasks_dev: tuple[Task, ...],
    tasks_anchor: tuple[Task, ...],
    *,
    fingerprint: str,
    limit: int | None,
    dev_suite: str = "",
    stop: StopCheck | None = None,
) -> HarnessRecord:
    existing = archive.seed()
    if existing is not None:
        if existing.details.get("anchor_manifest") != fingerprint:
            raise LoopError(
                "the archive was built against a different anchor suite; scores are not "
                "comparable. Point paths.archive at a new directory."
            )
        return existing
    seed = SEED_HARNESS
    dev_report = run_repair_benchmark(
        tasks_dev,
        config,
        governor,
        run_dir / "seed" / "dev",
        client=client,
        limit=limit,
        max_attempts=seed.max_attempts,
        samples_per_attempt=seed.samples_per_attempt,
        system_prompt=seed.system_prompt,
        skills=seed.skills,
        stop=stop,
        compare=False,
    )
    anchor_report = run_baseline(
        tasks_anchor,
        config,
        governor,
        run_dir / "seed" / "anchor",
        client=client,
        system_prompt=seed.system_prompt,
        skills=seed.skills,
        stop=stop,
    )
    details: dict[str, Any] = {
        "dev": dev_report.to_dict(),
        "anchor": anchor_report.to_dict(),
        "feedback": build_feedback(dev_report, anchor_report.summary.mean_score),
        "dev_samples": [dev_report.summary.repair_mean],
        "dev_suite": dev_suite,
        "anchor_manifest": fingerprint,
        "anchor_heldout": heldout_score(anchor_report, {task.id for task in tasks_dev}),
    }
    archive.ensure_seed(
        seed,
        dev_score=dev_report.summary.repair_mean,
        anchor_score=anchor_report.summary.mean_score,
        details=details,
    )
    created = archive.get(seed.hash)
    if created is None:  # pragma: no cover - defensive
        raise LoopError("seed was not archived")
    return created


def _rejected(
    generation: int,
    parent: HarnessRecord,
    error: str,
    *,
    candidate: HarnessSpec | None = None,
    operator: str | None = None,
    bandit_arm: str | None = None,
    reward: float | None = None,
) -> GenerationResult:
    return GenerationResult(
        generation=generation,
        parent_hash=parent.hash,
        candidate_hash=candidate.hash if candidate else None,
        parent_dev=parent.dev_score,
        parent_anchor=parent.anchor_score,
        dev_score=None,
        anchor_score=None,
        improved=False,
        promoted=False,
        operator=operator,
        bandit_arm=bandit_arm,
        reward=reward,
        error=error,
    )


def run_generation(
    archive: Archive,
    client: LLMClient,
    governor: ResourceGovernor,
    config: Config,
    run_dir: Path,
    tasks_dev: tuple[Task, ...],
    tasks_anchor: tuple[Task, ...],
    *,
    generation: int,
    limit: int | None = None,
    fingerprint: str = "",
    dev_suite: str = "",
    strict_heldout: bool = False,
    stop: StopCheck | None = None,
    bandit: BanditState | None = None,
) -> GenerationResult:
    parent = archive.current()
    if parent is None:
        raise LoopError("archive is empty; seed it first")
    arm: str | None = None
    focus: str | None = None
    if bandit is not None:
        arm = bandit.choose(config.loop.bandit_exploration)
        focus = OPERATOR_INSTRUCTIONS[arm]
    feedback = str(parent.details.get("feedback") or f"Dev mean {parent.dev_score}")
    forbidden = forbidden_terms((*tasks_dev, *tasks_anchor))
    # Vocabulary comes from dev tasks only, so anchor text is never shown.
    vocabulary = trigger_vocabulary(tasks_dev, forbidden)
    try:
        candidate = propose_mutation(
            client,
            parent.spec,
            feedback,
            forbidden=forbidden,
            vocabulary=vocabulary,
            inert_check=lambda spec: inert_new_skills(
                spec, parent.spec, (*tasks_dev, *tasks_anchor)
            ),
            max_retries=config.loop.max_attempts + 1,
            focus=focus,
        )
    except ProposalError as exc:
        reward = reward_for(improved=False, error=str(exc), anchor_score=None, parent_anchor=None)
        if bandit is not None and arm is not None:
            bandit.record(arm, reward)
        return _rejected(generation, parent, str(exc), bandit_arm=arm, reward=reward)
    operator = classify_operator(parent.spec, candidate)

    gen_dir = run_dir / f"gen-{generation:03d}"
    # Paired baseline: re-measure the parent now, under the same conditions.
    parent_report = run_repair_benchmark(
        tasks_dev,
        config,
        governor,
        gen_dir / "parent-dev",
        client=client,
        limit=limit,
        max_attempts=parent.spec.max_attempts,
        samples_per_attempt=parent.spec.samples_per_attempt,
        system_prompt=parent.spec.system_prompt,
        skills=parent.spec.skills,
        stop=stop,
        compare=False,
    )
    parent_dev = parent_report.summary.repair_mean
    archive.record_dev_sample(parent.hash, parent_dev, suite=dev_suite)
    parent_anchor = parent.anchor_score or 0.0

    dev_report: RepairBenchReport = run_repair_benchmark(
        tasks_dev,
        config,
        governor,
        gen_dir / "dev",
        client=client,
        limit=limit,
        max_attempts=candidate.max_attempts,
        samples_per_attempt=candidate.samples_per_attempt,
        system_prompt=candidate.system_prompt,
        skills=candidate.skills,
        stop=stop,
        compare=False,
    )
    dev_score = dev_report.summary.repair_mean
    anchor_score: float | None = None
    anchor_report: Any = None
    dev_floor = parent_dev - config.loop.dev_margin
    if dev_score >= dev_floor:
        # The anchor is evaluated one-shot so repair feedback cannot leak hidden
        # inputs. It measures whether the persistent harness change generalizes.
        anchor_report = run_baseline(
            tasks_anchor,
            config,
            governor,
            gen_dir / "anchor",
            client=client,
            system_prompt=candidate.system_prompt,
            skills=candidate.skills,
            stop=stop,
        )
        anchor_score = anchor_report.summary.mean_score

    improved = anchor_score is not None and (
        (dev_score > parent_dev and anchor_score >= parent_anchor)
        or (dev_score >= dev_floor and anchor_score > parent_anchor)
    )
    heldout = (
        heldout_score(anchor_report, {task.id for task in tasks_dev})
        if anchor_report is not None
        else None
    )
    parent_heldout_raw = parent.details.get("anchor_heldout")
    parent_heldout = (
        float(parent_heldout_raw)
        if isinstance(parent_heldout_raw, (int, float)) and not isinstance(parent_heldout_raw, bool)
        else None
    )
    strict_blocked = False
    if (
        improved
        and strict_heldout
        and (heldout is None or (parent_heldout is not None and heldout < parent_heldout))
    ):
        # Decoupling was flagged: dev alone no longer earns acceptance.
        improved = False
        strict_blocked = True
    details: dict[str, Any] = {
        "dev": dev_report.to_dict(),
        "anchor": anchor_report.to_dict() if anchor_report is not None else None,
        "feedback": build_feedback(dev_report, anchor_score),
        "dev_samples": [dev_score],
        "dev_suite": dev_suite,
        "anchor_manifest": fingerprint,
        "operator": operator,
        "bandit_arm": arm,
        "paired_parent_dev": parent_dev,
        "anchor_heldout": heldout,
        "parent_heldout": parent_heldout,
        "strict_blocked": strict_blocked,
    }
    reward = reward_for(
        improved=improved,
        error=None,
        anchor_score=anchor_score,
        parent_anchor=parent_anchor,
        heldout_score=heldout,
        parent_heldout=parent_heldout,
    )
    details["reward"] = reward
    if bandit is not None and arm is not None:
        bandit.record(arm, reward)
    archive.add(candidate, dev_score=dev_score, anchor_score=anchor_score, details=details)
    if improved:
        archive.accept(candidate.hash)
    promoted = False
    if improved and config.loop.promote:
        archive.promote(candidate.hash)
        promoted = True
    return GenerationResult(
        generation=generation,
        parent_hash=parent.hash,
        candidate_hash=candidate.hash,
        parent_dev=parent_dev,
        parent_anchor=parent_anchor,
        anchor_heldout=heldout,
        parent_heldout=parent_heldout,
        dev_score=dev_score,
        anchor_score=anchor_score,
        improved=improved,
        promoted=promoted,
        operator=operator,
        strict_blocked=strict_blocked,
        bandit_arm=arm,
        reward=reward,
        error=None,
    )


def run_loop(
    client: LLMClient,
    governor: ResourceGovernor,
    config: Config,
    tasks_dev: tuple[Task, ...],
    tasks_anchor: tuple[Task, ...],
    *,
    generations: int,
    limit: int | None = None,
    anchor_root: Path | None = None,
    dev_suite: str = "",
    stop: StopCheck | None = None,
) -> tuple[GenerationResult, ...]:
    if generations < 1:
        raise LoopError("generations must be >= 1")
    if not tasks_anchor:
        raise LoopError("the hidden anchor is empty; materialize it first")
    fingerprint = anchor_fingerprint(anchor_root or config.paths.anchor)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.runs.expanduser() / f"run-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = EventLog(config.paths.events.expanduser(), run_id=run_dir.name)
    archive = Archive(config.paths.archive.expanduser())
    bandit = BanditState.load(config.paths.bandit.expanduser())
    try:
        seed = _seed(
            archive,
            client,
            governor,
            config,
            run_dir,
            tasks_dev,
            tasks_anchor,
            fingerprint=fingerprint,
            limit=limit,
            dev_suite=dev_suite,
            stop=stop,
        )
        log.emit(
            "seed.ready",
            hash=seed.hash,
            dev_score=seed.dev_score,
            anchor_score=seed.anchor_score,
        )
        results: list[GenerationResult] = []
        decoupling: Decoupling = detect([])
        for generation in range(1, generations + 1):
            if governor.should_yield():
                time.sleep(5.0)
            result = run_generation(
                archive,
                client,
                governor,
                config,
                run_dir,
                tasks_dev,
                tasks_anchor,
                generation=generation,
                limit=limit,
                fingerprint=fingerprint,
                dev_suite=dev_suite,
                strict_heldout=decoupling.flagged,
                stop=stop,
                bandit=bandit,
            )
            bandit.save(config.paths.bandit.expanduser())
            results.append(result)
            log.emit("generation.finished", **result.to_dict())
            decoupling = detect(archive.heldout_points())
            log.emit("decoupling.checked", **decoupling.to_dict())
            # Persist after every generation so an interrupted overnight run keeps its work.
            (run_dir / "generations.json").write_text(
                json.dumps([item.to_dict() for item in results], indent=2) + "\n",
                encoding="utf-8",
            )
        return tuple(results)
    except ArchiveError as exc:
        raise LoopError(str(exc)) from exc
    finally:
        archive.close()
