"""Evaluator co-evolution with an anti-collusion gate.

The evaluator being improved is a task's test suite. Rules that keep it honest:

* the model proposes **inputs only**; expected outputs come from the trusted
  reference run in the sandbox, so a proposer cannot dictate "correct";
* an input is accepted only if it kills at least one reference mutant that the
  suite so far does not (the mutation-kill gate), so trivial or redundant inputs
  are worthless to a proposer;
* the suite is monotone: cases are only ever added, never removed or edited;
* only the dev split evolves. The anchor and fresh splits stay frozen as the
  external grounding that detects a dev evaluator drifting away from the target.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from janet.eval.execute import execute_cases
from janet.eval.mutants import Mutant, generate_mutants
from janet.eval.runner import _compare
from janet.eval.task import Case, Task
from janet.governor import ResourceGovernor
from janet.llm import LLMClient, LLMError

MAX_PROPOSALS = 12
MAX_ARG_CHARS = 1500
MAX_EVOLVED_PER_TASK = 8

PROPOSER_SYSTEM = (
    "You design test inputs for a Python function. Return only JSON of the form "
    '{"inputs": [[arg1, arg2, ...], ...]}. Each inner list is one call\'s positional '
    "arguments. Choose edge cases and tricky cases that a plausible wrong implementation "
    "would get wrong. Respect the stated contract; do not include expected outputs."
)


@dataclass(frozen=True, slots=True)
class EvolveResult:
    task_id: str
    mutants: int
    killed_before: int
    killed_after: int
    proposed: int
    accepted: tuple[Case, ...]
    rejected: dict[str, int] = field(default_factory=dict)

    @property
    def score_before(self) -> float:
        return self.killed_before / self.mutants if self.mutants else 0.0

    @property
    def score_after(self) -> float:
        return self.killed_after / self.mutants if self.mutants else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "mutants": self.mutants,
            "killed_before": self.killed_before,
            "killed_after": self.killed_after,
            "score_before": round(self.score_before, 4),
            "score_after": round(self.score_after, 4),
            "proposed": self.proposed,
            "accepted": len(self.accepted),
            "rejected": dict(self.rejected),
        }


def _case_to_json(case: Case) -> dict[str, Any]:
    return {"args": list(case.args), "expect": case.expect}


def load_evolved(directory: Path) -> dict[str, tuple[Case, ...]]:
    """Evolved cases by task id (empty if the directory does not exist)."""

    out: dict[str, tuple[Case, ...]] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[path.stem] = tuple(Case(args=tuple(e["args"]), expect=e["expect"]) for e in data)
    return out


def save_evolved(directory: Path, task_id: str, cases: tuple[Case, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{task_id}.json").write_text(
        json.dumps([_case_to_json(c) for c in cases], indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def apply_evolved(tasks: tuple[Task, ...], directory: Path) -> tuple[Task, ...]:
    """Add evolved cases to dev tasks as *hidden* cases (not shown in prompts)."""

    from dataclasses import replace

    evolved = load_evolved(directory)
    return tuple(
        replace(t, hidden_cases=t.hidden_cases + evolved[t.id]) if t.id in evolved else t
        for t in tasks
    )


def suite_fingerprint(directory: Path) -> str:
    """Stable hash of the evolved evaluator state ('' when nothing has evolved)."""

    evolved = load_evolved(directory)
    if not any(evolved.values()):
        return ""
    canonical = json.dumps(
        {k: [_case_to_json(c) for c in v] for k, v in sorted(evolved.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _parse_inputs(
    text: str, arity: int, existing: set[str]
) -> tuple[list[list[Any]], Counter[str]]:
    rejected: Counter[str] = Counter()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        rejected["bad_json"] += 1
        return [], rejected
    raw = data.get("inputs") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        rejected["bad_shape"] += 1
        return [], rejected
    inputs: list[list[Any]] = []
    for item in raw[:MAX_PROPOSALS]:
        if not isinstance(item, list) or len(item) != arity:
            rejected["bad_arity"] += 1
            continue
        key = json.dumps(item, sort_keys=True)
        if len(key) > MAX_ARG_CHARS:
            rejected["too_large"] += 1
        elif key in existing:
            rejected["duplicate"] += 1
        else:
            existing.add(key)
            inputs.append(item)
    return inputs, rejected


def _killed(
    mutant_results: list[dict[str, Any]] | None, expected: list[Any], indices: range
) -> bool:
    if mutant_results is None:
        return True  # crash/timeout is an observable difference
    return any(not _compare(mutant_results[i], expected[i]) for i in indices)


def evolve_task(
    task: Task,
    reference: str,
    client: LLMClient,
    governor: ResourceGovernor,
    workspace: Path,
    *,
    existing_evolved: tuple[Case, ...] = (),
    max_new: int = 4,
    mutant_limit: int = 24,
    rounds: int = 2,
) -> EvolveResult:
    """Strengthen one task's suite; returns the new cases (never mutates anything)."""

    entry = task.entrypoint
    if entry is None or not task.visible_cases:
        return EvolveResult(task.id, 0, 0, 0, 0, ())
    mutants: tuple[Mutant, ...] = generate_mutants(reference, limit=mutant_limit, seed=task.id)
    base_cases = (*task.visible_cases, *task.hidden_cases, *existing_evolved)
    base_inputs = [list(c.args) for c in base_cases]
    base_expected = [c.expect for c in base_cases]

    # Which mutants does the current suite already kill?
    killed: set[int] = set()
    for m in mutants:
        base_run = execute_cases(m.source, entry, base_inputs, governor, workspace / f"m{m.index}")
        if _killed(base_run, base_expected, range(len(base_inputs))):
            killed.add(m.index)
    killed_before = len(killed)

    arity = len(task.visible_cases[0].args)
    seen = {json.dumps(list(c.args), sort_keys=True) for c in base_cases}
    rejected: Counter[str] = Counter()
    accepted: list[Case] = []
    proposed = 0
    allow_error = "ValueError" in task.prompt
    room = min(max_new, MAX_EVOLVED_PER_TASK - len(existing_evolved))

    for _ in range(max(0, rounds)):
        if room - len(accepted) <= 0 or len(killed) == len(mutants):
            break
        examples = "\n".join(json.dumps(list(c.args)) for c in task.visible_cases)
        prompt = (
            f"{task.prompt}\n\nExample argument lists:\n{examples}\n\n"
            f"Propose {MAX_PROPOSALS} new argument lists as JSON."
        )
        try:
            response = client.generate(
                prompt, system=PROPOSER_SYSTEM, json_mode=True, temperature=0.7
            )
        except LLMError:
            rejected["llm_error"] += 1
            break
        inputs, bad = _parse_inputs(response.text, arity, seen)
        rejected.update(bad)
        proposed += len(inputs) + sum(bad.values())
        if not inputs:
            continue
        ref_results = execute_cases(reference, entry, inputs, governor, workspace / "ref")
        if ref_results is None:
            rejected["reference_failed"] += len(inputs)
            continue
        usable: list[tuple[list[Any], Any]] = []
        for args, ref in zip(inputs, ref_results, strict=True):
            if ref.get("ok") is True:
                usable.append((args, ref.get("value")))
            elif allow_error and str(ref.get("error", "")).startswith("ValueError"):
                usable.append((args, {"error": "ValueError"}))
            else:
                rejected["out_of_contract"] += 1
        if not usable:
            continue
        use_inputs = [a for a, _ in usable]
        use_expected = [e for _, e in usable]
        survivors = [m for m in mutants if m.index not in killed]
        kill_map: dict[int, set[int]] = {}
        for m in survivors:
            mut_run = execute_cases(
                m.source, entry, use_inputs, governor, workspace / f"p{m.index}"
            )
            if mut_run is None:
                # A hard failure cannot be attributed to one input; crediting every
                # input would make the gate accept redundant cases.
                continue
            for i in range(len(use_inputs)):
                if _killed(mut_run, use_expected, range(i, i + 1)):
                    kill_map.setdefault(i, set()).add(m.index)
        for i, (args, expect) in enumerate(usable):
            if len(accepted) >= room:
                break
            new = kill_map.get(i, set()) - killed
            if not new:
                rejected["no_new_kill"] += 1
                continue
            killed |= new
            accepted.append(Case(args=tuple(args), expect=expect))

    return EvolveResult(
        task_id=task.id,
        mutants=len(mutants),
        killed_before=killed_before,
        killed_after=len(killed),
        proposed=proposed,
        accepted=tuple(accepted),
        rejected=dict(rejected),
    )
