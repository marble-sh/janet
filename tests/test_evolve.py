from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from helpers import make_sign_task

from janet.eval.evolve import (
    MAX_EVOLVED_PER_TASK,
    apply_evolved,
    evolve_task,
    load_evolved,
    save_evolved,
    suite_fingerprint,
)
from janet.eval.task import Case, Task
from janet.governor import ResourceGovernor
from janet.llm import LLMError, LLMResponse
from janet.probe import Budget


class FakeProposer:
    def __init__(self, text: str = '{"inputs": [[0]]}') -> None:
        self.text = text
        self.calls = 0

    def generate(
        self,
        _prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        **_kwargs: Any,
    ) -> LLMResponse:
        self.calls += 1
        if not json_mode:
            raise AssertionError("evolve_task must use JSON mode")
        return LLMResponse(
            text=self.text,
            model="fake",
            eval_count=1,
            prompt_eval_count=1,
            eval_duration_ns=1,
            total_duration_ns=1,
        )


class ExplodingProposer:
    def generate(self, *_args: Any, **_kwargs: Any) -> LLMResponse:
        raise LLMError("model down")


def _reference(task: Task) -> str:
    return (task.source_dir / "reference.py").read_text(encoding="utf-8")


def test_evolve_accepts_a_mutant_killing_input(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    client = FakeProposer()
    result = evolve_task(
        task,
        _reference(task),
        client,  # type: ignore[arg-type]
        ResourceGovernor(budget),
        tmp_path / "work",
        max_new=1,
        mutant_limit=12,
        rounds=1,
    )
    assert result.mutants > 0
    assert result.killed_after > result.killed_before
    assert [case.args for case in result.accepted] == [(0,)]
    assert result.accepted[0].expect == 0
    assert client.calls == 1


def test_evolve_rejects_a_duplicate_input(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    result = evolve_task(
        task,
        _reference(task),
        FakeProposer('{"inputs": [[2]]}'),  # type: ignore[arg-type]
        ResourceGovernor(budget),
        tmp_path / "work",
        max_new=1,
        mutant_limit=12,
        rounds=1,
    )
    assert result.accepted == ()
    assert result.rejected["duplicate"] == 1


def test_evolve_rejects_bad_arity_json_and_contract(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    governor = ResourceGovernor(budget)
    no_error_contract = replace(task, prompt="Return the sign of an integer.")
    for target, text, reason in (
        (task, '{"inputs": [[1, 2]]}', "bad_arity"),
        (task, "not json", "bad_json"),
        (no_error_contract, '{"inputs": [["x"]]}', "out_of_contract"),
    ):
        result = evolve_task(
            target,
            _reference(task),
            FakeProposer(text),  # type: ignore[arg-type]
            governor,
            tmp_path / f"work-{reason}",
            max_new=1,
            mutant_limit=12,
            rounds=1,
        )
        assert result.accepted == ()
        assert result.rejected[reason] >= 1


def test_evolve_handles_llm_errors(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    result = evolve_task(
        task,
        _reference(task),
        ExplodingProposer(),  # type: ignore[arg-type]
        ResourceGovernor(budget),
        tmp_path / "work",
        max_new=1,
        mutant_limit=12,
        rounds=1,
    )
    assert result.accepted == ()
    assert result.rejected["llm_error"] == 1


def test_evolve_stops_when_the_room_is_full(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    existing = tuple(Case(args=(i,), expect=0) for i in range(MAX_EVOLVED_PER_TASK))
    client = FakeProposer()
    result = evolve_task(
        task,
        _reference(task),
        client,  # type: ignore[arg-type]
        ResourceGovernor(budget),
        tmp_path / "work",
        existing_evolved=existing,
        max_new=4,
        mutant_limit=12,
        rounds=2,
    )
    assert result.accepted == ()
    assert client.calls == 0


def test_evolve_skips_tasks_without_an_entrypoint(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    from dataclasses import replace

    integrity = replace(task, kind="integrity", entrypoint=None)
    result = evolve_task(
        integrity,
        _reference(task),
        FakeProposer(),  # type: ignore[arg-type]
        ResourceGovernor(budget),
        tmp_path / "work",
    )
    assert result.mutants == 0
    assert result.accepted == ()


def test_evolved_cases_round_trip_and_apply(tmp_path: Path) -> None:
    directory = tmp_path / "evolved"
    assert load_evolved(directory) == {}
    assert suite_fingerprint(directory) == ""
    cases = (Case(args=(0,), expect=0),)
    save_evolved(directory, "sign", cases)
    assert load_evolved(directory)["sign"] == cases
    assert suite_fingerprint(directory)

    task = make_sign_task(tmp_path / "tasks")
    applied = apply_evolved((task,), directory)
    assert applied[0].hidden_cases == cases
    assert task.hidden_cases == ()


def test_suite_fingerprint_changes_with_the_cases(tmp_path: Path) -> None:
    directory = tmp_path / "evolved"
    save_evolved(directory, "a", (Case(args=(1,), expect=1),))
    first = suite_fingerprint(directory)
    save_evolved(directory, "a", (Case(args=(1,), expect=1), Case(args=(2,), expect=2)))
    assert suite_fingerprint(directory) != first


def test_save_evolved_is_stable_json(tmp_path: Path) -> None:
    directory = tmp_path / "evolved"
    save_evolved(directory, "a", (Case(args=(1,), expect={"x": 1}),))
    text = (directory / "a.json").read_text(encoding="utf-8")
    assert json.loads(text) == [{"args": [1], "expect": {"x": 1}}]


def test_evolve_requires_visible_cases(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    from dataclasses import replace

    empty = replace(task, visible_cases=())
    result = evolve_task(
        empty,
        _reference(task),
        FakeProposer(),  # type: ignore[arg-type]
        ResourceGovernor(budget),
        tmp_path / "work",
    )
    assert result.accepted == ()
    assert result.mutants == 0
