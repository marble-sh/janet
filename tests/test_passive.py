from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from helpers import make_sign_task

from janet.config import Config, DaemonConfig, PathsConfig
from janet.eval.evolve import load_evolved
from janet.eval.task import Case, Task
from janet.governor import ResourceGovernor
from janet.llm import LLMResponse
from janet.loop.passive import (
    JobOutcome,
    PassiveState,
    SessionContext,
    _calibrate_once,
    _confirm_once,
    _next_job,
    due,
    run_session,
    select_task,
)
from janet.probe import Budget


class FakeProposer:
    def __init__(self, text: str = '{"inputs": [[0]]}') -> None:
        self.text = text

    def generate(self, _prompt: str, **_kwargs: Any) -> LLMResponse:
        return LLMResponse(
            text=self.text,
            model="fake",
            eval_count=1,
            prompt_eval_count=1,
            eval_duration_ns=1,
            total_duration_ns=1,
        )


def _config(tmp_path: Path, **daemon: Any) -> Config:
    defaults: dict[str, Any] = {
        "session_seconds": 600.0,
        "evolve_per_session": 1,
        "generations_per_session": 0,
        "confirm_interval_hours": 999.0,
        "calibrate_interval_hours": 999.0,
    }
    defaults.update(daemon)
    return Config(
        paths=PathsConfig(
            runs=tmp_path / "runs",
            archive=tmp_path / "archive",
            events=tmp_path / "events.jsonl",
            evolved=tmp_path / "evolved",
            daemon=tmp_path / "daemon",
            bandit=tmp_path / "runs" / "bandit.json",
        ),
        daemon=DaemonConfig(**defaults),
    )


def _context(task: Task, *, fresh: bool = False) -> SessionContext:
    return SessionContext(
        dev=(task,),
        raw_dev=(task,),
        anchor=(),
        fresh=(task,) if fresh else (),
        dev_suite="",
    )


def test_due_handles_never_open_and_bad_timestamps() -> None:
    now = datetime(2026, 10, 25, tzinfo=UTC)
    assert due(None, 24, now)
    assert due((now - timedelta(hours=25)).isoformat(), 24, now)
    assert not due((now - timedelta(hours=1)).isoformat(), 24, now)
    assert due("not a timestamp", 24, now)
    assert due(None, 0, now)


def test_select_task_prefers_the_least_evolved_and_rotates(tmp_path: Path) -> None:
    first = make_sign_task(tmp_path, task_id="a")
    second = make_sign_task(tmp_path, task_id="b")
    assert select_task((first, second), {"a": (Case(args=(1,), expect=1),)}, 0) == second
    assert select_task((first, second), {}, 0) == first
    assert select_task((first, second), {}, 1) == second
    assert select_task((), {}, 0) is None


def test_passive_state_round_trip_and_robust_load(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = PassiveState(
        sessions=3,
        generation=2,
        cursor=4,
        current_hash="current",
        last_confirm_hash="abc",
        last_confirm_at="2026-10-25T00:00:00+00:00",
        last_calibrate_at=None,
        last_error="boom",
        jobs={"evolve": 2},
    )
    state.save(path)
    assert PassiveState.load(path) == state

    path.write_text(
        json.dumps({"sessions": "x", "jobs": {"a": 2, "b": "x", "c": True}}),
        encoding="utf-8",
    )
    loaded = PassiveState.load(path)
    assert loaded.sessions == 0
    assert loaded.jobs == {"a": 2}

    path.write_text("not json", encoding="utf-8")
    assert PassiveState.load(path).sessions == 0
    assert PassiveState.load(tmp_path / "missing.json").sessions == 0


def test_job_and_session_payloads_are_json_safe() -> None:
    outcome = JobOutcome("evolve", True, "ok", 1.2345, {"a": 1})
    payload = outcome.to_dict()
    assert payload["seconds"] == 1.234
    assert json.dumps(payload)


def test_run_session_runs_the_evolve_rung(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    config = _config(tmp_path)
    state = PassiveState()
    state_path = tmp_path / "state.json"
    report = run_session(
        config,
        FakeProposer(),  # type: ignore[arg-type]
        ResourceGovernor(budget),
        _context(task),
        state,
        state_path,
        tmp_path / "work",
        session_seconds=120.0,
    )
    assert not report.preempted
    assert [job.job for job in report.jobs] == ["evolve"]
    assert state.cursor == 1
    assert state.jobs["evolve"] == 1
    assert load_evolved(config.paths.evolved)[task.id]
    assert PassiveState.load(state_path).cursor == 1


def test_run_session_yields_on_stop(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    config = _config(tmp_path)
    state = PassiveState()
    report = run_session(
        config,
        FakeProposer(),  # type: ignore[arg-type]
        ResourceGovernor(budget),
        _context(task),
        state,
        tmp_path / "state.json",
        tmp_path / "work",
        stop=lambda: True,
    )
    assert report.preempted
    assert report.jobs == ()


def test_run_session_skips_an_unseeded_generation(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    config = _config(
        tmp_path,
        evolve_per_session=0,
        generations_per_session=1,
        session_seconds=2000.0,
    )
    state = PassiveState(last_calibrate_at=datetime.now(UTC).isoformat())
    report = run_session(
        config,
        FakeProposer(),  # type: ignore[arg-type]
        ResourceGovernor(budget),
        _context(task),
        state,
        tmp_path / "state.json",
        tmp_path / "work",
        session_seconds=2000.0,
    )
    assert len(report.jobs) == 1
    assert report.jobs[0].job == "generation"
    assert report.jobs[0].ok is False
    assert "not seeded" in report.jobs[0].detail


def test_next_job_alternates_the_first_two_rungs(tmp_path: Path) -> None:
    task = make_sign_task(tmp_path / "tasks")
    config = _config(tmp_path, generations_per_session=1, evolve_per_session=1)
    now = datetime(2026, 10, 25, tzinfo=UTC)
    even = _next_job(config, _context(task), PassiveState(sessions=0), 0, 0, 2000.0, now)
    odd = _next_job(config, _context(task), PassiveState(sessions=1), 0, 0, 2000.0, now)
    assert even == "generation"
    assert odd == "evolve"


def test_next_job_confirms_a_new_variant_before_building_on_it(tmp_path: Path) -> None:
    task = make_sign_task(tmp_path / "tasks")
    config = _config(tmp_path, generations_per_session=1, evolve_per_session=1)
    now = datetime(2026, 10, 25, tzinfo=UTC)
    recent = now.isoformat()
    state = PassiveState(
        sessions=0,
        current_hash="new",
        last_confirm_hash="old",
        last_confirm_at=recent,
    )
    assert _next_job(config, _context(task, fresh=True), state, 0, 0, 2000.0, now) == "confirm"
    state.last_confirm_hash = "new"
    assert _next_job(config, _context(task, fresh=True), state, 0, 0, 2000.0, now) == "generation"


def test_next_job_falls_back_to_the_interval_without_a_current_hash(tmp_path: Path) -> None:
    task = make_sign_task(tmp_path / "tasks")
    config = _config(tmp_path, generations_per_session=1, evolve_per_session=1)
    now = datetime(2026, 10, 25, tzinfo=UTC)
    state = PassiveState(current_hash="", last_confirm_at=None)
    assert _next_job(config, _context(task, fresh=True), state, 0, 0, 2000.0, now) == "confirm"


def test_confirm_without_a_fresh_split_is_a_noop(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    outcome = _confirm_once(
        _config(tmp_path),
        FakeProposer(),  # type: ignore[arg-type]
        ResourceGovernor(budget),
        _context(task),
        PassiveState(),
        tmp_path / "work",
        None,
    )
    assert outcome.ok is False
    assert "fresh" in outcome.detail


def test_calibrate_without_dev_tasks_is_a_noop(tmp_path: Path, budget: Budget) -> None:
    context = SessionContext(dev=(), raw_dev=(), anchor=(), fresh=(), dev_suite="")
    outcome = _calibrate_once(
        _config(tmp_path, calibrate_interval_hours=0.0),
        FakeProposer(),  # type: ignore[arg-type]
        ResourceGovernor(budget),
        context,
        PassiveState(),
        tmp_path / "work",
        None,
    )
    assert outcome.ok is False


def test_calibrate_runs_when_due(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    state = PassiveState()
    outcome = _calibrate_once(
        _config(tmp_path, calibrate_interval_hours=0.0),
        FakeProposer('{"correct": true}'),  # type: ignore[arg-type]
        ResourceGovernor(budget),
        _context(task),
        state,
        tmp_path / "work",
        None,
    )
    assert outcome.ok is True
    assert state.last_calibrate_at is not None
    assert "kappa" in outcome.detail
