from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from helpers import make_sign_task

from janet.config import Config, PathsConfig
from janet.eval.evolve import load_evolved, save_evolved
from janet.eval.task import Case
from janet.governor import ResourceGovernor
from janet.llm import LLMResponse
from janet.loop.evolution import evolve_dev_suite
from janet.preempt import Preempted
from janet.probe import Budget


class FakeProposer:
    def __init__(self, text: str = '{"inputs": [[0]]}') -> None:
        self.text = text
        self.calls = 0

    def generate(self, _prompt: str, **_kwargs: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=self.text,
            model="fake",
            eval_count=1,
            prompt_eval_count=1,
            eval_duration_ns=1,
            total_duration_ns=1,
        )


def _config(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            runs=tmp_path / "runs",
            archive=tmp_path / "archive",
            events=tmp_path / "events.jsonl",
            evolved=tmp_path / "evolved",
        )
    )


def test_evolve_dev_suite_saves_cases_and_a_report(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    config = _config(tmp_path)
    work_dir = tmp_path / "work"
    seen: list[object] = []
    summary = evolve_dev_suite(
        (task,),
        config,
        FakeProposer(),  # type: ignore[arg-type]
        ResourceGovernor(budget),
        work_dir,
        max_new_per_task=1,
        progress=seen.append,
    )
    assert summary.tasks == 1
    assert summary.accepted >= 1
    assert len(seen) == 1
    assert load_evolved(config.paths.evolved)[task.id]
    payload = json.loads((work_dir / "evolution.json").read_text(encoding="utf-8"))
    assert payload["tasks"] == 1
    assert "evolve.task" in Path(config.paths.events).read_text(encoding="utf-8")


def test_evolve_dev_suite_appends_to_existing_cases(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    config = _config(tmp_path)
    save_evolved(config.paths.evolved, task.id, (Case(args=(5,), expect=1),))
    evolve_dev_suite(
        (task,),
        config,
        FakeProposer(),  # type: ignore[arg-type]
        ResourceGovernor(budget),
        tmp_path / "work",
        max_new_per_task=1,
    )
    cases = load_evolved(config.paths.evolved)[task.id]
    assert cases[0].args == (5,)
    assert len(cases) >= 1


def test_evolve_dev_suite_stops_on_request(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    with pytest.raises(Preempted):
        evolve_dev_suite(
            (task,),
            _config(tmp_path),
            FakeProposer(),  # type: ignore[arg-type]
            ResourceGovernor(budget),
            tmp_path / "work",
            stop=lambda: True,
        )


def test_evolve_dev_suite_skips_integrity_tasks(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    integrity = replace(task, kind="integrity", entrypoint=None)
    summary = evolve_dev_suite(
        (integrity,),
        _config(tmp_path),
        FakeProposer(),  # type: ignore[arg-type]
        ResourceGovernor(budget),
        tmp_path / "work",
    )
    assert summary.tasks == 0
    assert summary.accepted == 0
