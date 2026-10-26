"""Measure the one-shot baseline on every unassigned headroom task.

Each task is shown its visible examples (as a dev task would be) and scored on the
visible plus held-out cases. Output: JSON of {task_id: score} on stdout.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

from janet.bench.baseline import run_baseline
from janet.bench.hard_seeds import all_hard_seeds
from janet.bench.materialize import materialize
from janet.config import load_config
from janet.eval.task import load_suite
from janet.governor import ResourceGovernor
from janet.llm import LLMClient
from janet.probe import compute_budget, probe_machine


def main() -> int:
    config = load_config(Path("configs/default.toml"))
    seeds = all_hard_seeds()
    with tempfile.TemporaryDirectory(prefix="janet-probe-") as tmp:
        root = Path(tmp)
        dev = {
            t.id: t for t in load_suite(materialize(root / "dev", "dev", seeds=seeds).root, "dev")
        }
        anchor = {
            t.id: t
            for t in load_suite(materialize(root / "anchor", "anchor", seeds=seeds).root, "anchor")
        }
        tasks = tuple(
            dataclasses.replace(dev[tid], hidden_cases=anchor[tid].hidden_cases)
            for tid in sorted(dev)
        )
        governor = ResourceGovernor(compute_budget(probe_machine(), config.budget))
        report = run_baseline(tasks, config, governor, root / "run", client=LLMClient(config.llm))
        scores = {row.task_id: round(row.score, 4) for row in report.rows}
    json.dump(scores, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
