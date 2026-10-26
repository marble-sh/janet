"""Benchmark the bounded repair loop against the one-shot baseline."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from janet.bench.baseline import BaselineReport, run_baseline
from janet.config import Config
from janet.eval.repair import RepairAttempt, RepairResult, run_repair
from janet.eval.task import Task
from janet.governor import ResourceGovernor
from janet.harness.skills import Skill
from janet.llm import LLMClient
from janet.preempt import StopCheck
from janet.preempt import check as check_stop


@dataclass(frozen=True, slots=True)
class RepairBenchRow:
    task_id: str
    family: str
    baseline_score: float
    repair_score: float
    attempts: int
    passed: bool
    improved: bool
    regressed: bool
    eval_tokens: int
    wall_s: float
    error: str | None
    artifact_dir: str

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "baseline_score": round(self.baseline_score, 6),
            "repair_score": round(self.repair_score, 6),
            "attempts": self.attempts,
            "passed": self.passed,
            "improved": self.improved,
            "regressed": self.regressed,
            "eval_tokens": self.eval_tokens,
            "wall_s": round(self.wall_s, 3),
            "error": self.error,
            "artifact_dir": self.artifact_dir,
        }


@dataclass(frozen=True, slots=True)
class RepairBenchSummary:
    tasks: int
    baseline_passed: int
    repair_passed: int
    baseline_mean: float
    repair_mean: float
    improved: int
    regressed: int
    total_eval_tokens: int
    total_wall_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "tasks": self.tasks,
            "baseline_passed": self.baseline_passed,
            "repair_passed": self.repair_passed,
            "baseline_mean": round(self.baseline_mean, 6),
            "repair_mean": round(self.repair_mean, 6),
            "improved": self.improved,
            "regressed": self.regressed,
            "total_eval_tokens": self.total_eval_tokens,
            "total_wall_s": round(self.total_wall_s, 3),
        }


@dataclass(frozen=True, slots=True)
class RepairBenchReport:
    run_id: str
    model: str
    split: str
    rows: tuple[RepairBenchRow, ...]
    summary: RepairBenchSummary
    baseline: BaselineReport | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "split": self.split,
            "created_at": self.created_at,
            "summary": self.summary.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "baseline_summary": (
                self.baseline.summary.to_dict() if self.baseline is not None else None
            ),
        }


def _write_attempts(run_dir: Path, repair: RepairResult) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in repair.task_id)
    directory = run_dir / "attempts" / safe
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".txt" if repair.family == "integrity" else ".py"
    for attempt in repair.attempts:
        path = directory / f"attempt-{attempt.attempt}{suffix}"
        path.write_text(attempt.candidate, encoding="utf-8")
    return str(directory.relative_to(run_dir))


def _summary(
    rows: list[RepairBenchRow],
    wall_s: float,
) -> RepairBenchSummary:
    if not rows:
        return RepairBenchSummary(0, 0, 0, 0.0, 0.0, 0, 0, 0, wall_s)
    return RepairBenchSummary(
        tasks=len(rows),
        baseline_passed=sum(1 for row in rows if row.baseline_score >= 1.0),
        repair_passed=sum(1 for row in rows if row.passed),
        baseline_mean=sum(row.baseline_score for row in rows) / len(rows),
        repair_mean=sum(row.repair_score for row in rows) / len(rows),
        improved=sum(1 for row in rows if row.improved),
        regressed=sum(1 for row in rows if row.regressed),
        total_eval_tokens=sum(row.eval_tokens for row in rows),
        total_wall_s=wall_s,
    )


def render_markdown(report: RepairBenchReport) -> str:
    summary = report.summary
    lines = [
        f"# Bounded repair — {report.run_id}",
        "",
        f"- Model: `{report.model}`",
        f"- Split: `{report.split}`",
        f"- Tasks: {summary.tasks}",
        f"- One-shot passed: {summary.baseline_passed}",
        f"- Repair passed: {summary.repair_passed}",
        f"- One-shot mean: {summary.baseline_mean:.3f}",
        f"- Repair mean: {summary.repair_mean:.3f}",
        f"- Improved: {summary.improved}; regressed: {summary.regressed}",
        f"- Eval tokens: {summary.total_eval_tokens}",
        f"- Wall time: {summary.total_wall_s:.1f}s",
        "",
        "| Task | Baseline | Repair | Attempts | Passed | Improved |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in report.rows:
        lines.append(
            f"| `{row.task_id}` | {row.baseline_score:.2f} | "
            f"{row.repair_score:.2f} | {row.attempts} | "
            f"{'yes' if row.passed else 'no'} | {'yes' if row.improved else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def run_repair_benchmark(
    tasks: tuple[Task, ...],
    config: Config,
    governor: ResourceGovernor,
    run_dir: Path,
    *,
    client: LLMClient | None = None,
    limit: int | None = None,
    only: set[str] | None = None,
    max_attempts: int = 3,
    samples_per_attempt: int = 2,
    system_prompt: str | None = None,
    skills: Sequence[Skill] = (),
    stop: StopCheck | None = None,
    compare: bool = True,
    progress: Callable[[str, RepairAttempt], None] | None = None,
) -> RepairBenchReport:
    run_dir = run_dir.expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "work").mkdir(exist_ok=True)
    client = client or LLMClient(config.llm)
    selected = [task for task in tasks if only is None or task.id in only]
    if limit is not None:
        selected = selected[: max(0, limit)]

    baseline: BaselineReport | None = None
    baseline_by_id: dict[str, float] = {}
    if compare:
        baseline = run_baseline(
            tuple(selected),
            config,
            governor,
            run_dir / "baseline",
            client=client,
            system_prompt=system_prompt,
            skills=skills,
            stop=stop,
        )
        baseline_by_id = {row.task_id: row.score for row in baseline.rows}

    rows: list[RepairBenchRow] = []
    started = time.monotonic()
    for task in selected:
        check_stop(stop)
        if governor.should_yield():
            time.sleep(5.0)

        def on_attempt(attempt: RepairAttempt, _task_id: str = task.id) -> None:
            if progress is not None:
                progress(_task_id, attempt)

        repair = run_repair(
            task,
            client,
            governor,
            run_dir / "work",
            max_attempts=max_attempts,
            samples_per_attempt=samples_per_attempt,
            system_prompt=system_prompt,
            skills=skills,
            progress=on_attempt if progress is not None else None,
        )
        artifact_dir = _write_attempts(run_dir, repair)
        baseline_score = baseline_by_id.get(task.id, 0.0)
        rows.append(
            RepairBenchRow(
                task_id=task.id,
                family=task.family,
                baseline_score=baseline_score,
                repair_score=repair.final_score,
                attempts=len(repair.attempts),
                passed=repair.final.passed,
                improved=compare and repair.final_score > baseline_score,
                regressed=compare and repair.final_score < baseline_score,
                eval_tokens=repair.total_eval_tokens,
                wall_s=repair.wall_s,
                error=repair.final.error,
                artifact_dir=artifact_dir,
            )
        )
    wall_s = time.monotonic() - started
    report = RepairBenchReport(
        run_id=run_dir.name,
        model=config.llm.policy_model,
        split=selected[0].split if selected else "none",
        rows=tuple(rows),
        summary=_summary(rows, wall_s),
        baseline=baseline,
    )
    (run_dir / "repair.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "repair.md").write_text(render_markdown(report), encoding="utf-8")
    return report


__all__ = [
    "RepairBenchReport",
    "RepairBenchRow",
    "RepairBenchSummary",
    "render_markdown",
    "run_repair_benchmark",
]
