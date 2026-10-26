"""The idle work ladder: what Janet does when nobody is using the machine.

Every job is optional, bounded, and preemptible at a task boundary. The ladder is
deliberately shallow: a background process that can silently rewrite its own
evaluator needs a small, inspectable action set.

1. ``generation`` — one harness mutation, measured by the normal paired gate.
2. ``evolve`` — add mutation-killing inputs to the weakest dev task's evaluator.
3. ``confirm`` — re-check the current accepted variant on the never-selected
   fresh split (at most once per variant).
4. ``calibrate`` — measure the advisory judge against the verifier.

The daemon alternates the first two across sessions so a long generation cannot
starve evaluator work (and vice versa).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from janet.config import Config
from janet.eval.evolve import EvolveResult, evolve_task, load_evolved, save_evolved
from janet.eval.judge import build_samples, calibrate
from janet.eval.task import Case, Task
from janet.events import EventLog
from janet.governor import ResourceGovernor
from janet.harness.archive import Archive, ArchiveError
from janet.harness.bandit import BanditState
from janet.llm import LLMClient, LLMError
from janet.loop.confirm import confirm_variant
from janet.loop.generation import LoopError, run_generation
from janet.preempt import Preempted, StopCheck, check

logger = logging.getLogger(__name__)

EVOLVE_MAX_NEW = 2
EVOLVE_MUTANTS = 16
EVOLVE_ROUNDS = 1
CALIBRATE_TASKS = 4
MIN_GENERATION_SECONDS = 1800.0
MIN_CONFIRM_SECONDS = 300.0
MIN_CALIBRATE_SECONDS = 180.0


def _int(raw: dict[str, Any], key: str, default: int = 0) -> int:
    value = raw.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _str(raw: dict[str, Any], key: str, default: str = "") -> str:
    value = raw.get(key)
    return value if isinstance(value, str) else default


def _opt_str(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    return value if isinstance(value, str) else None


@dataclass(slots=True)
class PassiveState:
    """Durable cursor and schedule for the background ladder."""

    sessions: int = 0
    generation: int = 0
    cursor: int = 0
    current_hash: str = ""
    last_confirm_hash: str = ""
    last_confirm_at: str | None = None
    last_calibrate_at: str | None = None
    last_error: str = ""
    jobs: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> PassiveState:
        if not path.is_file():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        jobs_raw = raw.get("jobs")
        jobs: dict[str, int] = {}
        if isinstance(jobs_raw, dict):
            for key, value in jobs_raw.items():
                if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool):
                    jobs[key] = value
        return cls(
            sessions=_int(raw, "sessions"),
            generation=_int(raw, "generation"),
            cursor=_int(raw, "cursor"),
            current_hash=_str(raw, "current_hash"),
            last_confirm_hash=_str(raw, "last_confirm_hash"),
            last_confirm_at=_opt_str(raw, "last_confirm_at"),
            last_calibrate_at=_opt_str(raw, "last_calibrate_at"),
            last_error=_str(raw, "last_error"),
            jobs=jobs,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(self), indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)


@dataclass(frozen=True, slots=True)
class SessionContext:
    dev: tuple[Task, ...]  # with evolved cases applied
    raw_dev: tuple[Task, ...]  # seed tasks, for the evaluator's own evolution
    anchor: tuple[Task, ...]
    fresh: tuple[Task, ...]
    dev_suite: str


@dataclass(frozen=True, slots=True)
class JobOutcome:
    job: str
    ok: bool
    detail: str
    seconds: float
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "job": self.job,
            "ok": self.ok,
            "detail": self.detail,
            "seconds": round(self.seconds, 3),
            "data": self.data,
        }


@dataclass(frozen=True, slots=True)
class SessionReport:
    jobs: tuple[JobOutcome, ...]
    preempted: bool
    seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "jobs": [job.to_dict() for job in self.jobs],
            "preempted": self.preempted,
            "seconds": round(self.seconds, 3),
        }


def _now() -> datetime:
    return datetime.now(UTC)


def due(last: str | None, hours: float, now: datetime | None = None) -> bool:
    """Whether a periodic job should run; an unreadable timestamp is treated as due."""

    if hours <= 0:
        return True
    if last is None:
        return True
    try:
        stamp = datetime.fromisoformat(last)
    except ValueError:
        return True
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return (now or _now()) - stamp >= timedelta(hours=hours)


def select_task(
    tasks: tuple[Task, ...], evolved: dict[str, tuple[Case, ...]], cursor: int
) -> Task | None:
    """Rotate through evolvable tasks, least-covered first."""

    eligible = [
        task
        for task in tasks
        if not task.is_integrity
        and task.entrypoint
        and (task.source_dir / "reference.py").is_file()
    ]
    if not eligible:
        return None
    counts = {task.id: len(evolved.get(task.id, ())) for task in eligible}
    lowest = min(counts.values())
    pool = [task for task in eligible if counts[task.id] == lowest]
    return pool[cursor % len(pool)]


def _evolve_once(
    task: Task,
    config: Config,
    client: LLMClient,
    governor: ResourceGovernor,
    evolved: dict[str, tuple[Case, ...]],
    work_dir: Path,
    stop: StopCheck | None,
) -> JobOutcome:
    started = time.monotonic()
    check(stop)
    existing = evolved.get(task.id, ())
    reference = (task.source_dir / "reference.py").read_text(encoding="utf-8")
    result: EvolveResult = evolve_task(
        task,
        reference,
        client,
        governor,
        work_dir / task.id,
        existing_evolved=existing,
        max_new=EVOLVE_MAX_NEW,
        mutant_limit=EVOLVE_MUTANTS,
        rounds=EVOLVE_ROUNDS,
    )
    if result.accepted:
        updated = existing + result.accepted
        evolved[task.id] = updated
        save_evolved(config.paths.evolved.expanduser(), task.id, updated)
    detail = (
        f"{task.id}: +{len(result.accepted)} case(s), "
        f"mutation score {result.score_before:.2f}->{result.score_after:.2f}"
    )
    return JobOutcome(
        job="evolve",
        ok=True,
        detail=detail,
        seconds=time.monotonic() - started,
        data=result.to_dict(),
    )


def _generation_once(
    config: Config,
    client: LLMClient,
    governor: ResourceGovernor,
    context: SessionContext,
    state: PassiveState,
    work_dir: Path,
    stop: StopCheck | None,
) -> JobOutcome:
    started = time.monotonic()
    check(stop)
    archive = Archive(config.paths.archive.expanduser())
    try:
        if archive.seed() is None:
            return JobOutcome(
                job="generation",
                ok=False,
                detail="the archive is not seeded yet; skipping the generation ladder rung",
                seconds=time.monotonic() - started,
            )
        bandit = BanditState.load(config.paths.bandit.expanduser())
        result = run_generation(
            archive,
            client,
            governor,
            config,
            work_dir,
            context.dev,
            context.anchor,
            generation=state.generation + 1,
            dev_suite=context.dev_suite,
            stop=stop,
            bandit=bandit,
        )
        # Persist the learned operator policy even when the candidate is rejected.
        bandit.save(config.paths.bandit.expanduser())
        current = archive.current()
        state.current_hash = current.hash if current is not None else ""
    finally:
        archive.close()
    state.generation += 1
    outcome = "accepted" if result.improved else "rejected"
    detail = (
        f"gen {result.generation} {outcome}: dev {result.parent_dev}->{result.dev_score}, "
        f"anchor {result.parent_anchor}->{result.anchor_score}"
    )
    return JobOutcome(
        job="generation",
        ok=True,
        detail=detail,
        seconds=time.monotonic() - started,
        data=result.to_dict(),
    )


def _confirm_once(
    config: Config,
    client: LLMClient,
    governor: ResourceGovernor,
    context: SessionContext,
    state: PassiveState,
    work_dir: Path,
    stop: StopCheck | None,
) -> JobOutcome:
    started = time.monotonic()
    if not context.fresh:
        return JobOutcome("confirm", False, "the fresh suite is empty", 0.0)
    check(stop)
    archive = Archive(config.paths.archive.expanduser())
    try:
        current = archive.current()
        seed = archive.seed()
        if current is None or seed is None:
            return JobOutcome(
                "confirm", False, "the archive has no current variant", time.monotonic() - started
            )
        if current.hash == seed.hash:
            return JobOutcome(
                "confirm",
                False,
                "only the seed exists; nothing to confirm",
                time.monotonic() - started,
            )
        if current.hash == state.last_confirm_hash:
            return JobOutcome(
                "confirm",
                False,
                f"{current.hash[:8]} was already confirmed",
                time.monotonic() - started,
            )
        confirmation = confirm_variant(
            archive,
            current.hash,
            context.fresh,
            client,
            governor,
            config,
            work_dir,
            stop=stop,
        )
    finally:
        archive.close()
    state.last_confirm_hash = current.hash
    state.last_confirm_at = _now().isoformat()
    state.current_hash = current.hash
    return JobOutcome(
        "confirm",
        True,
        f"{current.hash[:8]} fresh delta {confirmation.delta:+.3f}",
        time.monotonic() - started,
        confirmation.to_dict(),
    )


def _calibrate_once(
    config: Config,
    client: LLMClient,
    governor: ResourceGovernor,
    context: SessionContext,
    state: PassiveState,
    work_dir: Path,
    stop: StopCheck | None,
) -> JobOutcome:
    started = time.monotonic()
    check(stop)
    tasks = tuple(task for task in context.dev if not task.is_integrity)[:CALIBRATE_TASKS]
    if not tasks:
        return JobOutcome("calibrate", False, "no dev tasks to calibrate on", 0.0)
    samples = build_samples(tasks, governor, work_dir / "samples", mutants_per_task=1)
    result = calibrate(samples, tasks, client, config.llm.judge_model)
    state.last_calibrate_at = _now().isoformat()
    return JobOutcome(
        "calibrate",
        True,
        f"judge kappa={result.kappa:.2f} trustworthy={result.trustworthy}",
        time.monotonic() - started,
        result.to_dict(),
    )


def _next_job(
    config: Config,
    context: SessionContext,
    state: PassiveState,
    evolve_done: int,
    generations_done: int,
    remaining: float,
    now: datetime,
) -> str | None:
    # A newly accepted variant is verified before more compute is spent building
    # on it. The time interval is only a fallback for state written before the
    # current-hash field existed (or a manual confirmation).
    new_variant = bool(state.current_hash) and state.current_hash != state.last_confirm_hash
    confirm_ready = (
        context.fresh
        and remaining >= MIN_CONFIRM_SECONDS
        and (
            new_variant
            or (
                not state.current_hash
                and due(state.last_confirm_at, config.daemon.confirm_interval_hours, now)
            )
        )
    )
    if confirm_ready:
        return "confirm"
    generation_ready = (
        generations_done < config.daemon.generations_per_session
        and remaining >= MIN_GENERATION_SECONDS
    )
    evolve_ready = evolve_done < config.daemon.evolve_per_session and remaining > 0
    order = ["generation", "evolve"] if state.sessions % 2 == 0 else ["evolve", "generation"]
    for job in order:
        if job == "generation" and generation_ready:
            return job
        if job == "evolve" and evolve_ready:
            return job
    if remaining >= MIN_CALIBRATE_SECONDS and due(
        state.last_calibrate_at, config.daemon.calibrate_interval_hours, now
    ):
        return "calibrate"
    return None


def run_session(
    config: Config,
    client: LLMClient,
    governor: ResourceGovernor,
    context: SessionContext,
    state: PassiveState,
    state_path: Path,
    work_root: Path,
    *,
    stop: StopCheck | None = None,
    session_seconds: float | None = None,
    log: EventLog | None = None,
    now: datetime | None = None,
) -> SessionReport:
    """Run one bounded ladder session, saving state after every completed job."""

    budget_seconds = (
        session_seconds if session_seconds is not None else config.daemon.session_seconds
    )
    started = time.monotonic()
    deadline = started + budget_seconds
    clock = now or _now()
    stamp = clock.strftime("%Y%m%dT%H%M%SZ")

    def session_stop() -> bool:
        if stop is not None and stop():
            return True
        return time.monotonic() >= deadline

    evolved = load_evolved(config.paths.evolved.expanduser())
    outcomes: list[JobOutcome] = []
    preempted = False
    evolve_done = 0
    generations_done = 0
    try:
        while True:
            check(session_stop)
            remaining = deadline - time.monotonic()
            job = _next_job(config, context, state, evolve_done, generations_done, remaining, clock)
            if job is None:
                break
            if job == "generation":
                outcome = _generation_once(
                    config,
                    client,
                    governor,
                    context,
                    state,
                    work_root / stamp / "generation",
                    session_stop,
                )
                generations_done += 1
            elif job == "evolve":
                task = select_task(context.raw_dev, evolved, state.cursor)
                if task is None:
                    break
                outcome = _evolve_once(
                    task,
                    config,
                    client,
                    governor,
                    evolved,
                    work_root / stamp / "evolve",
                    session_stop,
                )
                state.cursor += 1
                evolve_done += 1
            elif job == "confirm":
                outcome = _confirm_once(
                    config,
                    client,
                    governor,
                    context,
                    state,
                    work_root / stamp / "confirm",
                    session_stop,
                )
            else:
                outcome = _calibrate_once(
                    config,
                    client,
                    governor,
                    context,
                    state,
                    work_root / stamp / "calibrate",
                    session_stop,
                )
            outcomes.append(outcome)
            state.jobs[outcome.job] = state.jobs.get(outcome.job, 0) + 1
            state.save(state_path)
            if log is not None:
                log.emit("daemon.job", **outcome.to_dict())
    except Preempted:
        preempted = True
    except (ArchiveError, LLMError, LoopError, OSError) as exc:
        # Background work must not take the daemon down; the next session retries.
        logger.exception("background job failed")
        state.last_error = f"{type(exc).__name__}: {exc}"
        if log is not None:
            log.emit("daemon.error", error=state.last_error)
    finally:
        state.save(state_path)
    return SessionReport(
        jobs=tuple(outcomes),
        preempted=preempted,
        seconds=time.monotonic() - started,
    )
