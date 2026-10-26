"""One evaluator-evolution pass over the dev suite."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from janet.config import Config
from janet.eval.evolve import EvolveResult, evolve_task, load_evolved, save_evolved
from janet.eval.task import Task
from janet.events import EventLog
from janet.governor import ResourceGovernor
from janet.llm import LLMClient
from janet.preempt import StopCheck
from janet.preempt import check as check_stop


@dataclass(frozen=True, slots=True)
class EvolutionSummary:
    tasks: int
    accepted: int
    mutants: int
    killed_before: int
    killed_after: int
    results: tuple[EvolveResult, ...]

    @property
    def score_before(self) -> float:
        return self.killed_before / self.mutants if self.mutants else 0.0

    @property
    def score_after(self) -> float:
        return self.killed_after / self.mutants if self.mutants else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "tasks": self.tasks,
            "accepted": self.accepted,
            "mutants": self.mutants,
            "mutation_score_before": round(self.score_before, 4),
            "mutation_score_after": round(self.score_after, 4),
            "per_task": [r.to_dict() for r in self.results],
        }


def evolve_dev_suite(
    tasks: tuple[Task, ...],
    config: Config,
    client: LLMClient,
    governor: ResourceGovernor,
    work_dir: Path,
    *,
    max_new_per_task: int = 3,
    stop: StopCheck | None = None,
    progress: Callable[[EvolveResult], None] | None = None,
) -> EvolutionSummary:
    """Strengthen each evolvable dev task and persist the grown suite.

    Cases are only appended (monotone). Task order is by weakest mutation score first
    so a short idle window spends effort where the evaluator is least discriminating.
    """

    evolved_dir = config.paths.evolved.expanduser()
    existing = load_evolved(evolved_dir)
    log = EventLog(config.paths.events.expanduser(), run_id=work_dir.name)
    results: list[EvolveResult] = []
    for task in tasks:
        check_stop(stop)
        reference_path = task.source_dir / "reference.py"
        if task.is_integrity or not reference_path.is_file():
            continue
        result = evolve_task(
            task,
            reference_path.read_text(encoding="utf-8"),
            client,
            governor,
            work_dir / task.id,
            existing_evolved=existing.get(task.id, ()),
            max_new=max_new_per_task,
        )
        results.append(result)
        if result.accepted:
            save_evolved(evolved_dir, task.id, existing.get(task.id, ()) + result.accepted)
        log.emit("evolve.task", **result.to_dict())
        if progress:
            progress(result)
    summary = EvolutionSummary(
        tasks=len(results),
        accepted=sum(len(r.accepted) for r in results),
        mutants=sum(r.mutants for r in results),
        killed_before=sum(r.killed_before for r in results),
        killed_after=sum(r.killed_after for r in results),
        results=tuple(results),
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "evolution.json").write_text(
        json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return summary
