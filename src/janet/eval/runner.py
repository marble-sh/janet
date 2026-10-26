"""Run a task against a candidate solution in the sandbox."""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from janet.eval.audit import audit_source
from janet.eval.task import Case, Task, TaskError
from janet.governor import ResourceGovernor
from janet.probe import MIB
from janet.sandbox import Sandbox, SandboxError, SandboxLimits

_CANDIDATE_MEM = 512 * MIB
_CANDIDATE_CORES = 2


@dataclass(frozen=True, slots=True)
class Failure:
    """A failed case with enough detail for repair, but no expected value."""

    args: tuple[Any, ...]
    kind: str
    actual: str

    def to_dict(self) -> dict[str, object]:
        return {
            "args": list(self.args),
            "kind": self.kind,
            "actual": self.actual,
        }


@dataclass(frozen=True, slots=True)
class TaskResult:
    task_id: str
    family: str
    split: str
    score: float
    passed: bool
    output: str
    error: str | None
    duration_s: float
    cases_passed: int
    cases_total: int
    response_text: str
    failures: tuple[Failure, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "score": round(self.score, 6),
            "passed": self.passed,
            "error": self.error,
            "duration_s": round(self.duration_s, 3),
            "cases_passed": self.cases_passed,
            "cases_total": self.cases_total,
            "failure_count": len(self.failures),
        }


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(actual, float) or isinstance(expected, float):
            return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
        return int(actual) == int(expected)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _values_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _values_equal(actual[key], expected[key]) for key in actual
        )
    return bool(actual == expected)


def _compare(actual: dict[str, Any], expected: Any) -> bool:
    if isinstance(expected, dict) and "error" in expected and len(expected) == 1:
        wanted = str(expected["error"])
        return (
            actual.get("ok") is False
            and isinstance(actual.get("error"), str)
            and str(actual["error"]).split(":", 1)[0].strip() == wanted
        )
    if actual.get("ok") is not True:
        return False
    return _values_equal(actual.get("value"), expected)


def _integrity_score(task: Task, text: str) -> tuple[bool, str | None]:
    marker = task.abstain_marker.lower()
    has_marker = marker in text.lower()
    forbidden_hit = False
    for pattern in task.forbidden:
        try:
            if re.search(pattern, text, re.IGNORECASE):
                forbidden_hit = True
                break
        except re.error as exc:  # pragma: no cover - validated at load time
            raise TaskError(f"invalid forbidden regex {pattern!r}: {exc}") from exc
    if not has_marker:
        return (False, "missing abstention marker")
    if forbidden_hit:
        return (False, "forbidden answer pattern")
    return (True, None)


def _safe_dirname(task_id: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in task_id)


def run_function_task(
    task: Task,
    candidate_source: str,
    governor: ResourceGovernor,
    workspace_root: Path,
) -> TaskResult:
    cases: tuple[Case, ...] = task.cases
    if not cases:
        return TaskResult(
            task_id=task.id,
            family=task.family,
            split=task.split,
            score=0.0,
            passed=False,
            output="",
            error="task has no cases",
            duration_s=0.0,
            cases_passed=0,
            cases_total=0,
            response_text=candidate_source,
        )
    flags = audit_source(candidate_source)
    if flags:
        return TaskResult(
            task_id=task.id,
            family=task.family,
            split=task.split,
            score=0.0,
            passed=False,
            output="",
            error="audit: " + "; ".join(flags[:4]),
            duration_s=0.0,
            cases_passed=0,
            cases_total=len(cases),
            response_text=candidate_source,
            failures=(Failure(args=(), kind="audit", actual="; ".join(flags[:4])),),
        )
    workspace = workspace_root / _safe_dirname(task.id)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text(candidate_source, encoding="utf-8")
    worker_source = Path(__file__).with_name("worker.py").read_text(encoding="utf-8")
    (workspace / "worker.py").write_text(worker_source, encoding="utf-8")
    (workspace / "inputs.json").write_text(
        json.dumps([list(case.args) for case in cases], default=repr),
        encoding="utf-8",
    )

    started = time.monotonic()
    try:
        sandbox = Sandbox(workspace, governor)
        run = sandbox.run(
            [
                "/usr/bin/python3",
                "worker.py",
                "solution.py",
                "inputs.json",
                task.entrypoint or "",
            ],
            limits=SandboxLimits(
                timeout_s=task.timeout_s,
                mem_bytes=_CANDIDATE_MEM,
                cpu_cores=_CANDIDATE_CORES,
            ),
        )
    except SandboxError as exc:
        return TaskResult(
            task_id=task.id,
            family=task.family,
            split=task.split,
            score=0.0,
            passed=False,
            output="",
            error=f"sandbox: {exc}",
            duration_s=time.monotonic() - started,
            cases_passed=0,
            cases_total=len(cases),
            response_text=candidate_source,
        )
    duration = time.monotonic() - started
    if not run.ok:
        return TaskResult(
            task_id=task.id,
            family=task.family,
            split=task.split,
            score=0.0,
            passed=False,
            output=run.stdout,
            error=(run.stderr or run.killed_reason or "candidate failed").strip(),
            duration_s=duration,
            cases_passed=0,
            cases_total=len(cases),
            response_text=candidate_source,
        )
    try:
        payload = json.loads(run.stdout)
    except json.JSONDecodeError as exc:
        return TaskResult(
            task_id=task.id,
            family=task.family,
            split=task.split,
            score=0.0,
            passed=False,
            output=run.stdout,
            error=f"worker returned non-JSON: {exc}",
            duration_s=duration,
            cases_passed=0,
            cases_total=len(cases),
            response_text=candidate_source,
        )
    if not isinstance(payload, dict):
        payload = {"error": "worker payload must be an object"}
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(cases):
        return TaskResult(
            task_id=task.id,
            family=task.family,
            split=task.split,
            score=0.0,
            passed=False,
            output=run.stdout,
            error=str(payload.get("error", "worker returned the wrong number of results")),
            duration_s=duration,
            cases_passed=0,
            cases_total=len(cases),
            response_text=candidate_source,
        )
    passed = 0
    failures: list[Failure] = []
    for actual, case in zip(results, cases, strict=True):
        if isinstance(actual, dict) and _compare(actual, case.expect):
            passed += 1
            continue
        if not isinstance(actual, dict):
            failures.append(Failure(args=case.args, kind="malformed", actual=repr(actual)))
            continue
        expects_error = isinstance(case.expect, dict) and "error" in case.expect
        actual_ok = actual.get("ok") is True
        if expects_error:
            if actual_ok:
                failures.append(
                    Failure(
                        args=case.args,
                        kind="missing_error",
                        actual=repr(actual.get("value")),
                    )
                )
            else:
                failures.append(
                    Failure(
                        args=case.args,
                        kind="wrong_error",
                        actual=str(actual.get("error")),
                    )
                )
        elif actual_ok:
            failures.append(
                Failure(
                    args=case.args,
                    kind="wrong_value",
                    actual=repr(actual.get("value")),
                )
            )
        else:
            failures.append(
                Failure(
                    args=case.args,
                    kind="unexpected_error",
                    actual=str(actual.get("error")),
                )
            )
    score = passed / len(cases)
    return TaskResult(
        task_id=task.id,
        family=task.family,
        split=task.split,
        score=score,
        passed=passed == len(cases),
        output=run.stdout,
        error=None,
        duration_s=duration,
        cases_passed=passed,
        cases_total=len(cases),
        response_text=candidate_source,
        failures=tuple(failures),
    )


def run_integrity_task(task: Task, response_text: str) -> TaskResult:
    started = time.monotonic()
    passed, error = _integrity_score(task, response_text)
    return TaskResult(
        task_id=task.id,
        family=task.family,
        split=task.split,
        score=1.0 if passed else 0.0,
        passed=passed,
        output=response_text,
        error=error,
        duration_s=time.monotonic() - started,
        cases_passed=1 if passed else 0,
        cases_total=1,
        response_text=response_text,
        failures=(
            () if passed else (Failure(args=(), kind="integrity", actual=error or "rejected"),)
        ),
    )


def run_task(
    task: Task,
    candidate: str,
    governor: ResourceGovernor,
    workspace_root: Path,
) -> TaskResult:
    if task.is_integrity:
        return run_integrity_task(task, candidate)
    return run_function_task(task, candidate, governor, workspace_root)
