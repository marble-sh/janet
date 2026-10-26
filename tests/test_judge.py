from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from helpers import make_sign_task

from janet.eval.judge import (
    Calibration,
    Sample,
    build_samples,
    calibrate,
    cohen_kappa,
    judge_once,
)
from janet.governor import ResourceGovernor
from janet.llm import LLMError, LLMResponse
from janet.probe import Budget


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="fake",
        eval_count=1,
        prompt_eval_count=1,
        eval_duration_ns=1,
        total_duration_ns=1,
    )


class VerdictClient:
    def __init__(self, verdict: bool | None = None, *, text: str | None = None) -> None:
        self.verdict = verdict
        self.text = text
        self.calls = 0

    def generate(self, _prompt: str, **_kwargs: Any) -> LLMResponse:
        self.calls += 1
        if self.text is not None:
            return _response(self.text)
        return _response(json.dumps({"correct": self.verdict}))


class SourceVerdictClient:
    def generate(self, prompt: str, **_kwargs: Any) -> LLMResponse:
        return _response(json.dumps({"correct": "BUG" not in prompt}))


class ExplodingClient:
    def generate(self, *_args: Any, **_kwargs: Any) -> LLMResponse:
        raise LLMError("down")


def test_cohen_kappa_known_values() -> None:
    assert cohen_kappa([]) == 0.0
    assert cohen_kappa([(True, True), (False, False)]) == 1.0
    assert cohen_kappa([(True, False), (False, True)]) == -1.0
    assert cohen_kappa([(True, True)]) == 0.0
    assert cohen_kappa([(True, True), (True, True)]) == 0.0


def test_trustworthy_requires_agreement_and_both_classes() -> None:
    base = Calibration(
        n=10,
        unparsable=0,
        accuracy=0.8,
        balanced_accuracy=0.8,
        kappa=0.5,
        true_positive_rate=0.8,
        true_negative_rate=0.8,
        positives=5,
    )
    assert base.trustworthy
    assert not replace(base, kappa=0.3).trustworthy
    assert not replace(base, positives=0).trustworthy
    assert not replace(base, positives=10).trustworthy


def test_judge_once_parses_a_boolean_verdict(tmp_path: Path) -> None:
    task = make_sign_task(tmp_path / "tasks")
    client = VerdictClient(verdict=False)
    assert judge_once(client, "judge", task, "code") is False  # type: ignore[arg-type]
    assert client.calls == 1


@pytest.mark.parametrize(
    "client",
    [
        VerdictClient(text="not json"),
        VerdictClient(text='{"correct": "yes"}'),
        VerdictClient(text="[1, 2]"),
        ExplodingClient(),
    ],
)
def test_judge_once_returns_none_on_bad_output(tmp_path: Path, client: object) -> None:
    task = make_sign_task(tmp_path / "tasks")
    assert judge_once(client, "judge", task, "code") is None  # type: ignore[arg-type]


def test_build_samples_labels_references_and_mutants(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    samples = build_samples(
        (task,), ResourceGovernor(budget), tmp_path / "work", mutants_per_task=2
    )
    assert samples
    assert samples[0].origin == "reference"
    assert samples[0].label is True
    assert all(sample.task_id == task.id for sample in samples)
    assert len(samples) <= 3


def test_build_samples_skips_integrity_tasks(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    integrity = replace(task, kind="integrity")
    assert build_samples((integrity,), ResourceGovernor(budget), tmp_path / "work") == []


def test_build_samples_skips_tasks_without_a_reference(tmp_path: Path, budget: Budget) -> None:
    task = make_sign_task(tmp_path / "tasks")
    (task.source_dir / "reference.py").unlink()
    assert build_samples((task,), ResourceGovernor(budget), tmp_path / "work") == []


def test_calibrate_detects_a_useful_judge(tmp_path: Path) -> None:
    task = make_sign_task(tmp_path / "tasks")
    samples = [
        Sample(task.id, "GOOD", True, "reference"),
        Sample(task.id, "BUG", False, "mutant"),
        Sample(task.id, "GOOD", True, "reference"),
        Sample(task.id, "BUG", False, "mutant"),
    ]
    result = calibrate(samples, (task,), SourceVerdictClient(), "judge")  # type: ignore[arg-type]
    assert result.n == 4
    assert result.unparsable == 0
    assert result.accuracy == 1.0
    assert result.kappa == 1.0
    assert result.trustworthy


def test_calibrate_counts_unparsable_verdicts(tmp_path: Path) -> None:
    task = make_sign_task(tmp_path / "tasks")
    samples = [Sample(task.id, "GOOD", True, "reference")]
    result = calibrate(
        samples,
        (task,),
        VerdictClient(text="not json"),  # type: ignore[arg-type]
        "judge",
    )
    assert result.n == 0
    assert result.unparsable == 1
    assert not result.trustworthy


def test_calibrate_counts_llm_errors(tmp_path: Path) -> None:
    task = make_sign_task(tmp_path / "tasks")
    samples = [Sample(task.id, "GOOD", True, "reference")]
    result = calibrate(samples, (task,), ExplodingClient(), "judge")  # type: ignore[arg-type]
    assert result.n == 0
    assert result.unparsable == 1


def test_calibration_to_dict_is_json_safe() -> None:
    result = Calibration(2, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1)
    payload = result.to_dict()
    assert payload["trustworthy"] is True
    assert json.dumps(payload)
