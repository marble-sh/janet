# Benchmark metrics

Janet reports three strata and never blends them.

| Metric | Definition | Use |
|---|---|---|
| **Capability** | Mean score over measured non-integrity tasks | Primary capability claim; prefer the hidden anchor suite |
| **Headroom** | Mean over measured non-saturated tasks with `difficulty >= loop.min_difficulty` | Optimization signal only |
| **Integrity** | Fraction of integrity tasks that abstain correctly | Honesty metric, reported separately |

## Why

- The headroom subset is selected using the policy's own outcomes and the
  difficulty floor, so it is circular as a capability estimate.
- Saturated and easy tasks inflate an all-task mean; integrity tasks measure
  honesty, not capability.
- The hidden anchor is never used for selection, so anchor capability is the
  only defensible generalization claim. Fresh is reserved for final
  confirmation.

## Reporting rules

1. Always report the measured `n` next to each mean.
2. Label the difficulty-gated dev number **headroom**, never **capability**.
3. Do not average integrity into capability or headroom.
4. Use a paired CI for any before/after claim; a point estimate alone is not a
   result.
5. `src/janet/bench/baseline.py` emits these fields under `metrics` in
   `baseline.json`; `janet bench baseline` prints them alongside the summary.
