# ADR 0001 — Verification first, judges advisory

**Status:** accepted 2026-10-23

## Context

The survey's central finding is that a self-improvement loop's ceiling is the
quality of its evaluator, and that demonstrated improvement tracks the
verification hierarchy: formal verifiers, execution feedback, learned judges,
intrinsic signals. LLM-as-judge systems carry position, verbosity, and
self-enhancement biases; self-rewarding loops preferentially reinforce
high-confidence mistakes.

## Decision

Acceptance decisions are made only by program verifiers (tests, objective
functions, validity checkers). An LLM judge may be used to generate proposals,
structured critiques, or lessons, but it may never be the signal that promotes a
variant. If a judge is used, it must come from a different model family than the
policy, and its agreement with a small human-labelled calibration set is
reported.

## Consequences

- Slower, narrower early progress; no free-form "taste" evaluation.
- The integrity and anchor suites are load-bearing infrastructure, not extras.
- The project can honestly report negative results instead of judge-approved
  regressions.
