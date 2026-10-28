from __future__ import annotations

from pathlib import Path

import pytest

from janet.bench.metrics import stratified_metrics
from janet.eval.task import Task


def _task(
    task_id: str,
    *,
    family: str = "code_repair",
    difficulty: int = 3,
    kind: str = "function",
    saturated: bool = False,
) -> Task:
    return Task(
        id=task_id,
        family=family,
        split="dev",
        kind=kind,
        title=task_id,
        prompt="",
        solution="",
        entrypoint=None if kind == "integrity" else task_id,
        visible_cases=(),
        hidden_cases=(),
        timeout_s=1.0,
        difficulty=difficulty,
        forbidden=(),
        abstain_marker="ABSTAIN",
        source_dir=Path("."),
        saturated=saturated,
    )


def test_stratified_metrics_separate_capability_headroom_and_integrity() -> None:
    tasks = (
        _task("easy", difficulty=2),
        _task("hard", difficulty=4),
        _task("saturated", difficulty=5, saturated=True),
        _task("abstain_ok", kind="integrity"),
        _task("abstain_bad", kind="integrity"),
        _task("unmeasured", difficulty=4),
    )
    metrics = stratified_metrics(
        tasks,
        {"easy": 0.5, "hard": 1.0, "saturated": 1.0, "abstain_ok": 1.0, "abstain_bad": 0.0},
        passed={"abstain_ok": True, "abstain_bad": False},
        min_difficulty=3,
    )
    assert metrics.capability_n == 3
    assert metrics.capability_mean == pytest.approx(0.833333)
    assert metrics.headroom_n == 1
    assert metrics.headroom_mean == 1.0
    assert metrics.integrity_n == 2
    assert metrics.integrity_rate == 0.5


def test_stratified_metrics_honor_limit_and_optimization_filters() -> None:
    tasks = (_task("easy", difficulty=1), _task("hard", difficulty=3))
    metrics = stratified_metrics(tasks, {"easy": 1.0}, min_difficulty=3)
    assert metrics.capability_n == 1
    assert metrics.capability_mean == 1.0
    assert metrics.headroom_n == 0
    assert metrics.headroom_mean is None


def test_stratified_metrics_reject_bad_difficulty() -> None:
    with pytest.raises(ValueError, match="min_difficulty"):
        stratified_metrics((), {}, min_difficulty=0)
