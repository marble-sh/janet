"""Admission pipeline for lessons and skills (P5.2).

Only the local gates live here: retrievable-text checks, trigger specificity,
and deterministic test shape. The paired dev check and the anchor check need
the policy model and are performed by the loop before it calls this module
(P5.3); their result arrives as ``reason`` text or as an admission rejection.

A rejected entry stays in the store as ``quarantined`` and is never retrieved.
``retired`` is terminal: a retired entry is not re-admitted here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from janet.eval.task import Task
from janet.harness.guard import find_text_leaks, forbidden_terms
from janet.memory.store import MemoryError, ProcessMemory, SubjectKind

ADMISSION_VERSION = "p5.2-1"

_MIN_RULE = 20
_MAX_RULE = 400
_MAX_TESTS = 16
_TEST_FIELDS = frozenset({"args", "expect", "error"})
_UNSAFE = (
    (
        re.compile(r"\b(?:subprocess|socket|urllib|requests|httpx)\b"),
        "mentions network or subprocess",
    ),
    (
        re.compile(r"\b(?:eval|exec|compile|__import__)\s*\("),
        "contains dynamic execution",
    ),
    (re.compile(r"https?://"), "embeds a URL"),
)
_TERM_RE = re.compile(r"[a-z0-9_]{2,32}")


class AdmissionError(MemoryError):
    """Raised when the admission pipeline is called incorrectly."""


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    status: str
    reasons: tuple[str, ...]
    evidence_id: int
    subject_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "admitted": self.admitted,
            "status": self.status,
            "reasons": list(self.reasons),
            "evidence_id": self.evidence_id,
            "subject_id": self.subject_id,
        }


def _dev_texts(tasks: Sequence[Task]) -> tuple[str, ...]:
    return tuple(
        f"{task.prompt}\n{task.solution}".lower() for task in tasks if not task.is_integrity
    )


def _text_reasons(label: str, text: str, tasks: Sequence[Task]) -> tuple[str, ...]:
    reasons: list[str] = []
    leaks = find_text_leaks(text, forbidden_terms(tasks))
    if leaks:
        reasons.append(f"{label} names a benchmark task or function: {', '.join(leaks)}")
    lowered = text.lower()
    for pattern, message in _UNSAFE:
        if pattern.search(lowered):
            reasons.append(f"{label} {message}")
    return tuple(reasons)


def _size_reasons(label: str, text: str) -> tuple[str, ...]:
    if not _MIN_RULE <= len(text.strip()) <= _MAX_RULE:
        return (f"{label} must be {_MIN_RULE}..{_MAX_RULE} characters",)
    return ()


def _trigger_reasons(triggers: Sequence[str], tasks: Sequence[Task]) -> tuple[str, ...]:
    texts = _dev_texts(tasks)
    if not texts:
        return ("admission requires at least one non-integrity dev task",)
    reasons: list[str] = []
    half = len(texts) / 2
    for trigger in triggers:
        hits = sum(1 for text in texts if trigger in text)
        if hits == 0:
            reasons.append(f"trigger {trigger!r} is inert: no dev task matches")
        elif hits > half:
            reasons.append(f"trigger {trigger!r} is generic: matches {hits}/{len(texts)} dev tasks")
    return tuple(reasons)


def _lesson_term_reasons(trigger_terms: str, tasks: Sequence[Task]) -> tuple[str, ...]:
    terms = tuple(dict.fromkeys(_TERM_RE.findall(trigger_terms.lower())))
    if not terms:
        return ("trigger_terms has no usable terms",)
    texts = _dev_texts(tasks)
    if not texts:
        return ("admission requires at least one non-integrity dev task",)
    half = len(texts) / 2
    hits = {term: sum(1 for text in texts if term in text) for term in terms}
    if all(count == 0 for count in hits.values()):
        return ("trigger_terms are inert: no dev task matches",)
    if all(count > half for count in hits.values()):
        return ("trigger_terms are generic: every term matches most dev tasks",)
    return ()


def _test_reasons(tests: Any) -> tuple[str, ...]:
    if not isinstance(tests, list) or not tests:
        return ("skill has no tests",)
    if len(tests) > _MAX_TESTS:
        return (f"skill has more than {_MAX_TESTS} tests",)
    reasons: list[str] = []
    for index, test in enumerate(tests):
        where = f"test {index}"
        if not isinstance(test, dict):
            reasons.append(f"{where} is not an object")
            continue
        unknown = set(test) - _TEST_FIELDS
        if unknown:
            reasons.append(f"{where} has unknown fields {sorted(unknown)}")
        if not isinstance(test.get("args"), list):
            reasons.append(f"{where} is missing a list 'args'")
        has_expect = "expect" in test
        has_error = "error" in test
        if has_expect == has_error:
            reasons.append(f"{where} must have exactly one of 'expect' or 'error'")
        elif has_error and (not isinstance(test["error"], str) or not test["error"]):
            reasons.append(f"{where} 'error' must be a non-empty string")
        try:
            json.dumps(test, sort_keys=True)
        except (TypeError, ValueError):
            reasons.append(f"{where} is not JSON-serializable")
    return tuple(reasons)


def _finalize(
    memory: ProcessMemory,
    subject_kind: SubjectKind,
    subject_id: int | str,
    reasons: tuple[str, ...],
    *,
    run_id: str,
    verifier_version: str,
    raw: dict[str, Any],
) -> AdmissionDecision:
    record = (
        memory.get_skill(subject_id)
        if subject_kind == "skill" and isinstance(subject_id, str)
        else memory.get_lesson(subject_id)
        if subject_kind == "lesson" and isinstance(subject_id, int)
        else None
    )
    if record is None:  # pragma: no cover - defensive
        raise AdmissionError(f"unknown {subject_kind}: {subject_id}")
    current = record.status
    if current == "retired":
        return AdmissionDecision(
            admitted=False,
            status="retired",
            reasons=(*reasons, "entry is retired"),
            evidence_id=0,
            subject_id=str(subject_id),
        )
    if reasons:
        if current in {"candidate", "verified"}:
            memory.set_status(subject_kind, subject_id, "quarantined")
        evidence_id = memory.record_evidence(
            subject_kind,
            subject_id,
            run_id=run_id,
            split="dev",
            verifier_version=verifier_version,
            test_ids=(),
            raw={"reasons": list(reasons), **raw},
        )
        return AdmissionDecision(
            admitted=False,
            status="quarantined",
            reasons=reasons,
            evidence_id=evidence_id,
            subject_id=str(subject_id),
        )
    if current == "quarantined":
        memory.set_status(subject_kind, subject_id, "candidate")
    memory.set_status(subject_kind, subject_id, "verified")
    evidence_id = memory.record_evidence(
        subject_kind,
        subject_id,
        run_id=run_id,
        split="dev",
        verifier_version=verifier_version,
        test_ids=(),
        raw={"reasons": [], **raw},
    )
    return AdmissionDecision(
        admitted=True,
        status="verified",
        reasons=(),
        evidence_id=evidence_id,
        subject_id=str(subject_id),
    )


def _require_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id.strip():
        raise AdmissionError("run_id must be a non-empty string")
    return run_id.strip()


def admit_skill(
    memory: ProcessMemory,
    *,
    name: str,
    triggers: list[str] | tuple[str, ...],
    rule: str,
    tests: Any,
    provenance: dict[str, Any],
    tasks: Sequence[Task],
    run_id: str,
    verifier_version: str = ADMISSION_VERSION,
    parent_id: str | None = None,
) -> AdmissionDecision:
    """Run the local admission gates and store the outcome.

    Shape violations raise ``MemoryError`` from the store. Semantic violations
    (task names, unsafe text, inert or generic triggers, missing tests)
    quarantine the entry instead.
    """

    run_id = _require_run_id(run_id)
    subject_id = memory.add_skill(
        name=name,
        triggers=triggers,
        rule=rule,
        tests=tests,
        provenance=provenance,
        parent_id=parent_id,
    )
    record = memory.get_skill(subject_id)
    if record is None:  # pragma: no cover - defensive
        raise AdmissionError(f"skill disappeared after insert: {subject_id}")
    text = f"{record.name}\n{record.rule}\n{' '.join(record.triggers)}"
    reasons = (
        *_text_reasons("skill", text, tasks),
        *_size_reasons("skill rule", record.rule),
        *_trigger_reasons(record.triggers, tasks),
        *_test_reasons(record.tests),
    )
    return _finalize(
        memory,
        "skill",
        subject_id,
        reasons,
        run_id=run_id,
        verifier_version=verifier_version,
        raw={"name": record.name, "parent_id": parent_id},
    )


def admit_lesson(
    memory: ProcessMemory,
    *,
    task_id: str,
    task_family: str,
    failure_signature: str,
    trigger_terms: str,
    rule: str,
    patch_summary: str,
    provenance: dict[str, Any],
    tasks: Sequence[Task],
    run_id: str,
    verifier_version: str = ADMISSION_VERSION,
    dev_gain: float | None = None,
    anchor_delta: float | None = None,
    fresh_delta: float | None = None,
) -> AdmissionDecision:
    """Same pipeline for lessons, scoped to a dev task that actually exists."""

    run_id = _require_run_id(run_id)
    subject_id = memory.add_lesson(
        task_id=task_id,
        task_family=task_family,
        failure_signature=failure_signature,
        trigger_terms=trigger_terms,
        rule=rule,
        patch_summary=patch_summary,
        provenance=provenance,
        dev_gain=dev_gain,
        anchor_delta=anchor_delta,
        fresh_delta=fresh_delta,
    )
    record = memory.get_lesson(subject_id)
    if record is None:  # pragma: no cover - defensive
        raise AdmissionError(f"lesson disappeared after insert: {subject_id}")
    known = {task.id: task for task in tasks if not task.is_integrity}
    reasons: tuple[str, ...] = ()
    task = known.get(record.task_id)
    if task is None:
        reasons = (f"lesson task {record.task_id!r} is not a non-integrity dev task",)
    elif task.family != record.task_family:
        reasons = (
            f"lesson family {record.task_family!r} does not match task family {task.family!r}",
        )
    text = "\n".join(
        (
            record.failure_signature,
            record.trigger_terms,
            record.rule,
            record.patch_summary,
        )
    )
    reasons = (
        *reasons,
        *_text_reasons("lesson", text, tasks),
        *_size_reasons("lesson rule", record.rule),
        *_lesson_term_reasons(record.trigger_terms, tasks),
    )
    return _finalize(
        memory,
        "lesson",
        subject_id,
        reasons,
        run_id=run_id,
        verifier_version=verifier_version,
        raw={"task_id": record.task_id, "task_family": record.task_family},
    )
