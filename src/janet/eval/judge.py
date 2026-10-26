"""Advisory LLM judge, calibrated against the program verifier.

ADR 0001: a judge never accepts or promotes anything. This module only measures how
far a judge can be trusted, by comparing its verdicts with the verifier's on
programs whose correctness is known (the reference, and mutants of it that the
verifier classifies by actually running them). A judge that cannot beat chance
on that set is recorded as untrustworthy; nothing in the loop consumes its verdicts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from janet.eval.mutants import generate_mutants
from janet.eval.runner import run_function_task
from janet.eval.task import Task
from janet.governor import ResourceGovernor
from janet.llm import LLMClient, LLMError

JUDGE_SYSTEM = (
    "You are a strict code reviewer. Decide whether the code correctly implements the "
    'specification for ALL valid inputs. Reply only with JSON: {"correct": true|false}.'
)


@dataclass(frozen=True, slots=True)
class Sample:
    task_id: str
    source: str
    label: bool  # the verifier's verdict (ground truth)
    origin: str


@dataclass(frozen=True, slots=True)
class Calibration:
    n: int
    unparsable: int
    accuracy: float
    balanced_accuracy: float
    kappa: float
    true_positive_rate: float
    true_negative_rate: float
    positives: int

    @property
    def trustworthy(self) -> bool:
        """Moderate agreement beyond chance (kappa >= 0.4) on a non-degenerate set."""

        return self.kappa >= 0.4 and 0 < self.positives < self.n

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "unparsable": self.unparsable,
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "kappa": round(self.kappa, 4),
            "true_positive_rate": round(self.true_positive_rate, 4),
            "true_negative_rate": round(self.true_negative_rate, 4),
            "positives": self.positives,
            "trustworthy": self.trustworthy,
        }


def cohen_kappa(pairs: Sequence[tuple[bool, bool]]) -> float:
    """Agreement between (verifier, judge) beyond chance; 0.0 when undefined."""

    n = len(pairs)
    if n == 0:
        return 0.0
    observed = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def build_samples(
    tasks: Sequence[Task],
    governor: ResourceGovernor,
    workspace: Path,
    *,
    mutants_per_task: int = 2,
) -> list[Sample]:
    """Reference plus mutants, each labelled by running the real verifier."""

    samples: list[Sample] = []
    for task in tasks:
        reference_path = task.source_dir / "reference.py"
        if task.is_integrity or not reference_path.is_file():
            continue
        reference = reference_path.read_text(encoding="utf-8")
        candidates = [("reference", reference)]
        for mutant in generate_mutants(reference, limit=12, seed=task.id)[:mutants_per_task]:
            candidates.append((f"mutant:{mutant.description}", mutant.source))
        for origin, source in candidates:
            result = run_function_task(task, source, governor, workspace)
            samples.append(Sample(task.id, source, result.passed, origin))
    return samples


def judge_once(client: LLMClient, model: str, task: Task, source: str) -> bool | None:
    prompt = f"Specification:\n{task.prompt}\n\nCode:\n```python\n{source}\n```"
    try:
        response = client.generate(
            prompt, system=JUDGE_SYSTEM, model=model, json_mode=True, temperature=0.0
        )
        verdict = json.loads(response.text).get("correct")
    except (LLMError, json.JSONDecodeError, AttributeError):
        return None
    return verdict if isinstance(verdict, bool) else None


def calibrate(
    samples: Sequence[Sample], tasks: Sequence[Task], client: LLMClient, model: str
) -> Calibration:
    by_id = {t.id: t for t in tasks}
    pairs: list[tuple[bool, bool]] = []
    unparsable = 0
    for sample in samples:
        verdict = judge_once(client, model, by_id[sample.task_id], sample.source)
        if verdict is None:
            unparsable += 1
        else:
            pairs.append((sample.label, verdict))
    positives = sum(1 for label, _ in pairs if label)
    negatives = len(pairs) - positives
    tp = sum(1 for label, judged in pairs if label and judged)
    tn = sum(1 for label, judged in pairs if not label and not judged)
    tpr = tp / positives if positives else 0.0
    tnr = tn / negatives if negatives else 0.0
    accuracy = (tp + tn) / len(pairs) if pairs else 0.0
    balanced = (tpr + tnr) / 2 if positives and negatives else accuracy
    return Calibration(
        n=len(pairs),
        unparsable=unparsable,
        accuracy=accuracy,
        balanced_accuracy=balanced,
        kappa=cohen_kappa(pairs),
        true_positive_rate=tpr,
        true_negative_rate=tnr,
        positives=positives,
    )
