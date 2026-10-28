"""Meta dashboard over the archive, bandit, and process memory (P6.5).

Read-only: the report never selects, accepts, or mutates a variant. It states
what the loop measured, what it cost, how diverse it is, and what is missing.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from janet.config import Config
from janet.eval.stats import bootstrap_slope_ci, theil_sen
from janet.harness.archive import Archive
from janet.harness.bandit import BanditState
from janet.loop.decoupling import stale_objective_probe
from janet.memory import ProcessMemory

_MAX_ROWS = 20


@dataclass(frozen=True, slots=True)
class MetaReport:
    generated_at: str
    generations: int
    capability: tuple[dict[str, Any], ...]
    improvement_rate: tuple[dict[str, Any], ...]
    meta_slope: float | None
    meta_slope_ci: tuple[float, float] | None
    transfer: float | None
    fresh: tuple[dict[str, Any], ...]
    diversity: dict[str, Any]
    integrity: dict[str, Any]
    resource: dict[str, Any]
    bandit: dict[str, Any]
    memory: dict[str, Any]
    stale_probe: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "generations": self.generations,
            "capability": list(self.capability),
            "improvement_rate": list(self.improvement_rate),
            "meta_slope": self.meta_slope,
            "meta_slope_ci": list(self.meta_slope_ci) if self.meta_slope_ci else None,
            "transfer": self.transfer,
            "fresh": list(self.fresh),
            "diversity": self.diversity,
            "integrity": self.integrity,
            "resource": self.resource,
            "bandit": self.bandit,
            "memory": self.memory,
            "stale_probe": self.stale_probe,
        }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _dev_summary(record: Any) -> dict[str, Any]:
    return _mapping(_mapping(record.details.get("dev")).get("summary"))


def _capability_mean(record: Any) -> float | None:
    """Anchor mean over capability tasks only; integrity is reported separately."""

    anchor = _mapping(record.details.get("anchor"))
    rows = anchor.get("rows")
    if isinstance(rows, list):
        scores = [
            _number(row.get("score"))
            for row in rows
            if isinstance(row, dict) and row.get("family") != "integrity"
        ]
        measured = [score for score in scores if score is not None]
        if measured:
            return sum(measured) / len(measured)
    return _number(_mapping(anchor.get("summary")).get("mean_score"))


def build_meta_report(config: Config, *, now: datetime | None = None) -> MetaReport:
    clock = now or datetime.now(UTC)
    archive = Archive(config.paths.archive.expanduser())
    try:
        records = list(reversed(archive.list(limit=100_000)))
        elites = archive.frontier()
        roots: list[str] = []
        for elite in elites:
            chain = archive.lineage(elite.hash)
            if chain:
                roots.append(chain[0].hash)
    finally:
        archive.close()
    evaluated = [record for record in records if record.dev_score is not None]

    capability: list[dict[str, Any]] = []
    for generation, record in enumerate(evaluated, start=1):
        anchor = _capability_mean(record)
        if anchor is None:
            continue
        summary = _dev_summary(record)
        capability.append(
            {
                "generation": generation,
                "variant_hash": record.hash,
                "anchor": anchor,
                "dev": _number(record.dev_score),
                "tokens": _number(summary.get("total_eval_tokens")),
                "wall_s": _number(summary.get("total_wall_s")),
            }
        )

    improvement_rate: list[dict[str, Any]] = []
    for previous, current in pairwise(capability):
        delta = current["anchor"] - previous["anchor"]
        tokens = current["tokens"]
        wall = current["wall_s"]
        improvement_rate.append(
            {
                "generation": current["generation"],
                "delta_anchor": delta,
                "per_1k_tokens": (delta / (tokens / 1000.0)) if tokens and tokens > 0 else None,
                "per_minute": (delta / (wall / 60.0)) if wall and wall > 0 else None,
            }
        )
    xs = [
        float(row["generation"])
        for row in improvement_rate
        if isinstance(row["per_1k_tokens"], float)
    ]
    ys = [
        float(row["per_1k_tokens"])
        for row in improvement_rate
        if isinstance(row["per_1k_tokens"], float)
    ]
    slope = theil_sen(xs, ys) if len(xs) >= 2 else None
    slope_ci = (
        bootstrap_slope_ci(
            xs,
            ys,
            resamples=config.loop.bootstrap_resamples,
            alpha=config.loop.alpha_total,
            seed=0,
        )
        if len(xs) >= 3
        else None
    )

    fresh: list[dict[str, Any]] = []
    for generation, record in enumerate(evaluated, start=1):
        payload = _mapping(record.details.get("fresh"))
        if not payload:
            continue
        fresh.append(
            {
                "generation": generation,
                "variant_hash": record.hash,
                "fresh_score": _number(payload.get("variant_score")),
                "delta": _number(payload.get("delta")),
                "release": payload.get("release"),
            }
        )
    transfer = fresh[-1]["delta"] if fresh else None

    operators = Counter(record.operator for record in evaluated if record.operator)
    families = Counter(
        record.coverage_cell.split("|")[0]
        for record in evaluated
        if record.coverage_cell and record.coverage_cell != "unknown|0|0"
    )
    root_counts = Counter(roots)
    diversity = {
        "evaluated": len(evaluated),
        "frontier_cells": len(elites),
        "operators": dict(sorted(operators.items())),
        "families": dict(sorted(families.items())),
        "largest_lineage_share": (
            max(root_counts.values()) / len(elites) if elites and root_counts else None
        ),
    }

    integrity_tasks = 0
    integrity_passed = 0
    violations: list[dict[str, Any]] = []
    for record in evaluated:
        rows = _mapping(record.details.get("dev")).get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("family") != "integrity":
                continue
            integrity_tasks += 1
            score = _number(row.get("repair_score")) or 0.0
            if score >= 1.0:
                integrity_passed += 1
            else:
                violations.append(
                    {"variant": record.hash[:12], "task": row.get("task_id"), "score": score}
                )
    integrity = {
        "tasks": integrity_tasks,
        "honest_abstention_rate": (integrity_passed / integrity_tasks if integrity_tasks else None),
        "violations": len(violations),
        "examples": violations[:5],
    }

    walls = [row["wall_s"] for row in capability if row["wall_s"] is not None]
    tokens = [row["tokens"] for row in capability if row["tokens"] is not None]
    p95_wall = _p95([float(value) for value in walls])
    resource = {
        "samples": len(walls),
        "p95_wall_s": p95_wall,
        "max_wall_s": max(walls) if walls else None,
        "budget_wall_s": float(config.budget.max_wall_seconds),
        "within_wall_budget": (
            p95_wall <= config.budget.max_wall_seconds if p95_wall is not None else None
        ),
        "p95_eval_tokens": _p95([float(value) for value in tokens]),
        "cpu_ram_vram": {
            "available": False,
            "reason": "the governor does not persist utilization samples",
        },
    }

    bandit = BanditState.load(config.paths.bandit.expanduser()).to_dict()

    amnesia_path = config.paths.runs.expanduser() / "memory-amnesia.json"
    amnesia: Any = None
    if amnesia_path.is_file():
        try:
            amnesia = json.loads(amnesia_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            amnesia = {"error": "unreadable"}
    memory_path = config.paths.memory.expanduser()
    if memory_path.is_file():
        with ProcessMemory(memory_path) as store:
            memory: dict[str, Any] = {
                "available": True,
                "path": str(memory_path),
                "stats": store.stats(),
                "retrieval": store.retrieval_summary(),
                "amnesia": amnesia,
            }
    else:
        memory = {"available": False, "path": str(memory_path), "amnesia": amnesia}

    return MetaReport(
        generated_at=clock.isoformat(),
        generations=len(evaluated),
        capability=tuple(capability),
        improvement_rate=tuple(improvement_rate),
        meta_slope=slope,
        meta_slope_ci=slope_ci,
        transfer=transfer,
        fresh=tuple(fresh),
        diversity=diversity,
        integrity=integrity,
        resource=resource,
        bandit=bandit,
        memory=memory,
        stale_probe=stale_objective_probe().to_dict(),
    )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: MetaReport) -> str:
    lines = [
        "# Janet meta report",
        "",
        f"Generated: {report.generated_at}",
        f"Evaluated variants: {report.generations}",
        "",
        "## Capability",
        "",
        "| gen | anchor | dev | eval tokens | wall s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report.capability[-_MAX_ROWS:]:
        lines.append(
            f"| {row['generation']} | {_fmt(row['anchor'])} | {_fmt(row['dev'])} | "
            f"{_fmt(row['tokens'], 0)} | {_fmt(row['wall_s'], 1)} |"
        )
    if not report.capability:
        lines.append("| — | — | — | — | — |")
    lines.extend(
        [
            "",
            "## Improvement rate",
            "",
            "| gen | Δ anchor | per 1k tokens | per minute |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in report.improvement_rate[-_MAX_ROWS:]:
        lines.append(
            f"| {row['generation']} | {_fmt(row['delta_anchor'])} | "
            f"{_fmt(row['per_1k_tokens'])} | {_fmt(row['per_minute'])} |"
        )
    if not report.improvement_rate:
        lines.append("| — | — | — | — |")
    ci = (
        f"[{report.meta_slope_ci[0]:.4f}, {report.meta_slope_ci[1]:.4f}]"
        if report.meta_slope_ci
        else "—"
    )
    lines.extend(
        [
            "",
            f"Meta slope (per 1k tokens per generation): {_fmt(report.meta_slope, 4)}; "
            f"bootstrap CI {ci}",
            f"Transfer (latest fresh delta vs seed): {_fmt(report.transfer)}",
            "",
            "## Diversity",
            "",
            f"- Evaluated: {report.diversity['evaluated']}",
            f"- Frontier cells: {report.diversity['frontier_cells']}",
            f"- Operators: {report.diversity['operators']}",
            f"- Families: {report.diversity['families']}",
            f"- Largest lineage share: {_fmt(report.diversity['largest_lineage_share'])}",
            "",
            "## Integrity",
            "",
            f"- Tasks: {report.integrity['tasks']}",
            f"- Honest-abstention rate: {_fmt(report.integrity['honest_abstention_rate'])}",
            f"- Violations: {report.integrity['violations']}",
            "",
            "## Resources",
            "",
            f"- Samples: {report.resource['samples']}",
            f"- p95 wall: {_fmt(report.resource['p95_wall_s'], 1)} s "
            f"(budget {_fmt(report.resource['budget_wall_s'], 0)} s, "
            f"within={report.resource['within_wall_budget']})",
            f"- p95 eval tokens: {_fmt(report.resource['p95_eval_tokens'], 0)}",
            f"- CPU/RAM/VRAM: {report.resource['cpu_ram_vram']['reason']}",
            "",
            "## Bandit",
            "",
            f"- Reward version: {report.bandit.get('reward_version')}",
            f"- Total pulls: {report.bandit.get('total_pulls')}",
            "",
            "| arm | pulls | mean | mean cost |",
            "|---|---:|---:|---:|",
        ]
    )
    arms = report.bandit.get("arms")
    if isinstance(arms, list):
        for arm in arms:
            if not isinstance(arm, dict):
                continue
            lines.append(
                f"| {arm.get('name')} | {arm.get('pulls')} | "
                f"{_fmt(arm.get('mean_reward'))} | {_fmt(arm.get('mean_cost'), 1)} |"
            )
    lines.extend(["", "## Memory", ""])
    if report.memory.get("available"):
        stats = _mapping(report.memory.get("stats"))
        retrieval = _mapping(report.memory.get("retrieval"))
        lines.extend(
            [
                f"- Lessons: {stats.get('lessons')}",
                f"- Skills: {stats.get('skills')}",
                f"- Evidence: {stats.get('evidence')}",
                f"- Retrieval events: {retrieval.get('events')}",
                f"- Retrieval hit rate: {_fmt(retrieval.get('hit_rate'))}",
                f"- Amnesia result: {report.memory.get('amnesia') or 'not run'}",
            ]
        )
    else:
        lines.append(f"- Not available at `{report.memory.get('path')}`")
        lines.append(f"- Amnesia result: {report.memory.get('amnesia') or 'not run'}")
    lines.extend(
        [
            "",
            "## Stale-objective probe",
            "",
            f"- Shift: {report.stale_probe['shift']}",
            f"- Detection latency: {report.stale_probe['detection_latency']}",
            f"- Reframed: {report.stale_probe['reframed']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: MetaReport, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    markdown_path = directory / "report.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


__all__ = ["MetaReport", "build_meta_report", "render_markdown", "write_report"]
