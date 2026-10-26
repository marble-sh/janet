"""Confirmation on the fresh split: an unbiased transfer measurement.

The fresh tasks are never used for selection. After a run, compare a variant
with the seed (and any other reference) one-shot on tasks neither was tuned on.
The one-shot protocol is deterministic (temperature 0), so a difference is a
property of the harness, not of sampling luck.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from janet.bench.baseline import run_baseline
from janet.config import Config
from janet.eval.task import Task
from janet.governor import ResourceGovernor
from janet.harness.archive import Archive, ArchiveError
from janet.llm import LLMClient
from janet.preempt import StopCheck


@dataclass(frozen=True, slots=True)
class Confirmation:
    variant_hash: str
    seed_hash: str
    tasks: int
    seed_score: float
    variant_score: float
    seed_passed: int
    variant_passed: int
    per_task: tuple[tuple[str, float, float], ...]

    @property
    def delta(self) -> float:
        return self.variant_score - self.seed_score

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_hash": self.variant_hash,
            "seed_hash": self.seed_hash,
            "tasks": self.tasks,
            "seed_score": round(self.seed_score, 6),
            "variant_score": round(self.variant_score, 6),
            "delta": round(self.delta, 6),
            "seed_passed": self.seed_passed,
            "variant_passed": self.variant_passed,
            "per_task": [
                {"task": task, "seed": round(seed, 4), "variant": round(variant, 4)}
                for task, seed, variant in self.per_task
            ],
        }


def confirm_variant(
    archive: Archive,
    variant_hash: str,
    tasks: tuple[Task, ...],
    client: LLMClient,
    governor: ResourceGovernor,
    config: Config,
    run_dir: Path,
    *,
    stop: StopCheck | None = None,
) -> Confirmation:
    if not tasks:
        raise ArchiveError("the fresh suite is empty")
    variant = archive.get(variant_hash)
    seed = archive.seed()
    if variant is None:
        raise ArchiveError(f"unknown variant: {variant_hash}")
    if seed is None:
        raise ArchiveError("archive has no seed")
    reports = {}
    for label, record in (("seed", seed), ("variant", variant)):
        reports[label] = run_baseline(
            tasks,
            config,
            governor,
            run_dir / label,
            client=client,
            system_prompt=record.spec.system_prompt,
            skills=record.spec.skills,
            stop=stop,
        )
    seed_rows = {row.task_id: row.score for row in reports["seed"].rows}
    variant_rows = {row.task_id: row.score for row in reports["variant"].rows}
    per_task = tuple(
        (task_id, seed_rows[task_id], variant_rows[task_id]) for task_id in sorted(seed_rows)
    )
    return Confirmation(
        variant_hash=variant.hash,
        seed_hash=seed.hash,
        tasks=len(per_task),
        seed_score=reports["seed"].summary.mean_score,
        variant_score=reports["variant"].summary.mean_score,
        seed_passed=reports["seed"].summary.passed,
        variant_passed=reports["variant"].summary.passed,
        per_task=per_task,
    )
