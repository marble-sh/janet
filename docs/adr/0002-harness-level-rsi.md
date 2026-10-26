# ADR 0002 — Recursion at the harness layer, not the weights

**Status:** accepted 2026-10-23

## Context

Full weight-level recursive self-improvement is bounded by grounding,
collapse, and compute, and is not credible on one machine under a 50% budget.
The tractable, externally inspectable self-modification surface is the harness:
prompts, skills, tool policy, routing, stopping rules, and the search policy
that proposes changes. DGM, Gödel Agent, Voyager, VASO, and SkillAxe all
demonstrate improvement at this layer under empirical validation.

## Decision

Janet's mutable surface is the harness, skill library, and improvement policy.
Model weights are frozen in v1. Every candidate is a declarative variant with a
canonical hash, stored as a git commit, and evaluated by the same verifier stack
as its parent. The anchor suite, evaluator, governor, sandbox, and promotion
gate are outside the writable surface.

## Consequences

- Gains are inspectable, diffable, and rollback-able.
- The system cannot claim open-ended RSI; it can measure bounded recursion.
- Weight-level self-distillation is deferred to an optional P7 with a tiny
  model, and only if the harness loop is stable.
