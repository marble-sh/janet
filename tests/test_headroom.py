from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from janet.bench.headroom import HeadroomError, HeadroomReport, assert_headroom, headroom_report
from janet.bench.headroom_seeds import headroom_seeds
from janet.bench.materialize import materialize
from janet.eval.task import Task, load_suite


def _run_reference(task: Task) -> None:
    assert task.entrypoint is not None
    namespace: dict[str, Any] = {}
    reference = (task.source_dir / "reference.py").read_text(encoding="utf-8")
    exec(compile(reference, str(task.source_dir / "reference.py"), "exec"), namespace)
    function = namespace[task.entrypoint]
    for case in task.cases:
        expected = case.expect
        if isinstance(expected, dict) and "error" in expected:
            with pytest.raises(Exception) as raised:
                function(*case.args)
            assert type(raised.value).__name__ == expected["error"]
        else:
            assert function(*case.args) == expected


def _suites(
    tmp_path: Path,
) -> tuple[dict[str, Task], dict[str, Task], dict[str, Task], HeadroomReport]:
    dev = load_suite(materialize(tmp_path / "dev", "dev", force=True).root, "dev")
    anchor = load_suite(materialize(tmp_path / "anchor", "anchor", force=True).root, "anchor")
    fresh = load_suite(materialize(tmp_path / "fresh", "fresh", force=True).root, "fresh")
    return (
        {task.id: task for task in dev},
        {task.id: task for task in anchor},
        {task.id: task for task in fresh},
        headroom_report(dev, anchor, fresh),
    )


def test_headroom_seeds_are_stratified() -> None:
    seeds = headroom_seeds()
    assert len(seeds) == 6
    assert all(seed.difficulty >= 4 for seed in seeds)
    splits = {
        "dev": sum(not seed.anchor_only and not seed.fresh_only for seed in seeds),
        "anchor": sum(seed.anchor_only for seed in seeds),
        "fresh": sum(seed.fresh_only for seed in seeds),
    }
    assert splits == {"dev": 2, "anchor": 2, "fresh": 2}


def test_references_pass_their_materialized_cases(tmp_path: Path) -> None:
    dev, anchor, fresh, _ = _suites(tmp_path)
    for seed in headroom_seeds():
        if seed.fresh_only:
            task = fresh[seed.id]
        elif seed.anchor_only:
            task = anchor[seed.id]
        else:
            task = dev[seed.id]
        _run_reference(task)


def test_known_hard_answers(tmp_path: Path) -> None:
    dev, anchor, fresh, _ = _suites(tmp_path)
    assert dev["weighted_interval_schedule"].cases[0].expect == [
        9,
        [[1, 3, 5], [3, 5, 4]],
    ]
    assert dev["token_bucket_allow"].cases[0].expect == [True, True, False, True]
    assert anchor["topological_order"].cases[0].expect == [0, 1, 2, 3]
    assert anchor["min_cost_grid"].cases[0].expect == 7
    assert fresh["word_ladder_length"].cases[0].expect == 5
    assert fresh["json_pointer_get"].cases[0].expect == 20


def test_headroom_report_and_gate(tmp_path: Path) -> None:
    _, _, _, report = _suites(tmp_path)
    assert report.dev_hard >= 2
    assert report.anchor_hard >= 4
    assert report.fresh_hard >= 2
    assert "headroom" in report.hard_families
    assert_headroom(report)
    payload = report.to_dict()
    assert payload["min_difficulty"] == 3


def test_headroom_gate_rejects_a_thin_suite() -> None:
    report = HeadroomReport(
        dev_tasks=1,
        anchor_tasks=1,
        fresh_tasks=1,
        dev_hard=1,
        anchor_hard=1,
        fresh_hard=1,
        min_difficulty=3,
        hard_families=("headroom",),
    )
    with pytest.raises(HeadroomError, match="dev has"):
        assert_headroom(report)
