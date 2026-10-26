"""Propose harness mutations with the local model, then validate them."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from janet.bench.repair_bench import RepairBenchReport
from janet.harness.guard import find_leaks
from janet.harness.spec import HarnessError, HarnessSpec
from janet.llm import LLMClient, LLMError

MUTATOR_SYSTEM = (
    "You improve the harness of a code-repair agent. Return only a JSON object. "
    "Never invent benchmark results, never mention hidden tests, and never ask "
    "for expected outputs. Make one targeted change that should generalize."
)

_REQUIRED_KEYS = (
    "name",
    "system_prompt",
    "max_attempts",
    "samples_per_attempt",
    "mutation_note",
    "skills",
)


class ProposalError(RuntimeError):
    """Raised when the proposer cannot produce a valid, changed harness."""


def build_feedback(report: RepairBenchReport, anchor_score: float | None = None) -> str:
    lines = [
        f"Dev tasks: {report.summary.tasks}",
        f"Dev mean: {report.summary.repair_mean:.4f}",
        f"Passed: {report.summary.repair_passed}/{report.summary.tasks}",
    ]
    if anchor_score is None:
        lines.append("Held-out anchor: not evaluated (details withheld).")
    else:
        lines.append(
            f"Held-out anchor score: {anchor_score:.4f} (inputs and expected outputs withheld)."
        )
    lines.append("Per-task scores (task id, family, score, attempts):")
    for row in report.rows:
        lines.append(
            f"- {row.task_id} | {row.family} | {row.repair_score:.3f} | {row.attempts} attempts"
        )
    failures = [row for row in report.rows if row.repair_score < 1.0]
    if failures:
        lines.append("Failures to address:")
        for row in failures:
            lines.append(
                f"- {row.task_id}: score {row.repair_score:.3f}; "
                f"last error: {row.error or 'wrong output'}"
            )
    lines.append("The hidden anchor's inputs and expected outputs are withheld from this feedback.")
    return "\n".join(lines)


def _mutation_prompt(
    parent: HarnessSpec,
    feedback: str,
    error: str | None,
    vocabulary: Sequence[str] = (),
    focus: str | None = None,
) -> str:
    change = focus or (
        "Make ONE change: edit the system_prompt, OR add/edit/remove one skill, "
        "OR change the repair budget. Keep everything else identical."
    )
    lines = [
        "Current harness:",
        parent.to_json().strip(),
        "",
        "Dev feedback:",
        feedback,
        "",
        "Return a JSON object with exactly these keys:",
        ", ".join(_REQUIRED_KEYS),
        "",
        "Constraints:",
        f"- {change}",
        "- system_prompt: 40..4000 characters.",
        "- skills: a list (max 6) of {name, triggers, rule}. triggers are 1..5",
        "  lowercase words/phrases; the rule is injected only when a trigger appears",
        "  in the task text. rule: 20..400 characters.",
        "- max_attempts: 1..5; samples_per_attempt: 1..4.",
        "- mutation_note: <= 500 characters, explaining the hypothesis.",
        "- name: a short label for this variant.",
        "- Describe the failure MODE as a general rule (for example 'validate",
        "  inputs before computing' or 'import every module you use'). NEVER name",
        "  a task, function, or test; such rules are rejected as hard-coding.",
        "- Skill triggers are matched as substrings of the task text, so they must",
        "  be words that really occur there. Prefer these: " + ", ".join(vocabulary) + "."
        if vocabulary
        else "- Skill triggers are matched as substrings of the task text.",
        "- Generic advice such as 'be careful' or 'optimize' is not useful.",
        "- Do not mention hidden tests, expected outputs, or benchmark gaming.",
    ]
    if error:
        lines.extend(["", f"The previous proposal was rejected: {error}", "Try again."])
    return "\n".join(lines)


def propose_mutation(
    client: LLMClient,
    parent: HarnessSpec,
    feedback: str,
    *,
    forbidden: Sequence[str] = (),
    vocabulary: Sequence[str] = (),
    inert_check: Callable[[HarnessSpec], Sequence[str]] | None = None,
    max_retries: int = 2,
    focus: str | None = None,
) -> HarnessSpec:
    error: str | None = None
    for _ in range(max_retries + 1):
        prompt = _mutation_prompt(parent, feedback, error, vocabulary, focus)
        try:
            response = client.generate(
                prompt,
                system=MUTATOR_SYSTEM,
                json_mode=True,
                temperature=0.3,
            )
            data = json.loads(response.text)
            if isinstance(data, dict) and "skills" not in data:
                # An omitted key means "unchanged", not "delete the library".
                data["skills"] = [skill.to_dict() for skill in parent.skills]
            candidate = HarnessSpec.from_dict(data, parent=parent.hash)
        except (LLMError, json.JSONDecodeError, HarnessError) as exc:
            error = str(exc)
            continue
        if candidate.hash == parent.hash:
            error = "the proposal did not change behavior; make a real change"
            continue
        leaks = find_leaks(candidate, forbidden)
        if leaks:
            error = f"the proposal names benchmark identifiers {list(leaks)}; use general rules"
            continue
        inert = inert_check(candidate) if inert_check is not None else ()
        if inert:
            error = (
                f"skills {list(inert)} have triggers that match no task text, so they "
                "would never apply; choose triggers from the suggested vocabulary"
            )
            continue
        return candidate
    raise ProposalError(f"no valid mutation after {max_retries + 1} attempts: {error}")
