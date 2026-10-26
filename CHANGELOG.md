# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **P6 bandit reward v2:** the reward now uses the held-out anchor delta (anchor
  tasks the dev split never saw) when both sides have it. A total-anchor gain
  with a flat held-out score earns +0.25 instead of +1.0; a held-out regression
  is penalized even if the total anchor improved. Total anchor is only a
  fallback. `REWARD_VERSION = 2` resets old state so incomparable means are not
  mixed; the fresh split is never used in the reward.

## [0.4.0] - 2026-10-26

### Added

- **P6 operator bandit (first slice):** a UCB1 bandit over four mutation
  operators (`prompt`, `skill`, `budget`, `prompt+skill`) chooses the focus given
  to the proposer and learns from the paired acceptance gate. State persists in
  `paths.bandit`; `janet bandit` shows pulls and mean reward.
- Rewards are shaped conservatively: accepted +1.0, anchor improved but blocked
  +0.5, unchanged 0.0, regressed −0.5, dev-floor rejection −0.05, proposal
  error −0.25. The gate remains the only acceptance source.

### Fixed

- A newly accepted variant is now fresh-confirmed before the daemon starts
  another generation; the 24-hour interval is only a fallback for old state.
  `PassiveState.current_hash` records the latest accepted variant, `_next_job`
  prioritizes `confirm`, and `daemon status` reports it.

## [0.3.0] - 2026-10-25

### Added

- **P3.3 headroom benchmark:** 18 graded tasks whose one-shot baseline sits near
  0.5 on the hidden split; `loop.min_difficulty` selects them while always keeping
  integrity tasks; an AST audit flags candidates that inspect evaluator state.
- **P4 evaluator co-evolution:** `janet evolve` proposes test *inputs* and keeps
  only inputs that kill a reference mutant the current suite does not; expected
  outputs come from the trusted reference. Evolved cases are monotone, hashed
  into `dev_suite`, and applied as hidden cases.
- **Judge calibration:** `janet calibrate` measures the advisory 3B judge
  against the program verifier with Cohen's kappa. The judge is never in the
  acceptance path.
- **Passive daemon:** `janet daemon start|stop|status|once`. Sessions begin only
  after an idle period, alternate harness generations and evaluator evolution,
  confirm accepted variants on the fresh split at most once, and yield at the
  next task boundary when foreground Janet work appears. A transient systemd
  unit caps CPU, memory, and nice priority.
- `janet bench baseline` / `janet bench repair` now apply the evolved evaluator
  to the dev split.

### Changed

- The model server and daemon each hold a 300% CPU quota, leaving two cores of
the 8-core soft budget for the desktop.
- The daemon session cap is one hour because a full generation does not fit in
  twenty minutes on this machine.

### Fixed

- The evaluator's mutation-kill gate no longer credits every proposed input when
  a mutant fails hard on one of them; unattributable crashes are skipped.

## [0.2.0] - 2026-10-23

### Added

- 7 anchor-only (held-out bug) tasks and 6 fresh tasks that are never used for
  selection; `janet bench materialize --split fresh`, `janet archive confirm`,
  `janet archive lineage`.
- Harness skills (name, triggers, rule) included in the harness hash, with
  per-task retrieval; integrity tasks never receive skills.
- Leakage guard rejecting proposals that name benchmark task ids or
  entrypoints; inert-skill rejection; dev-derived trigger vocabulary.
- Paired parent re-measurement each generation, running-mean dev scores,
  `loop.dev_margin`, accepted-variant parent selection, anchor-manifest
  fingerprint, and the `anchor_heldout` metric.

### Changed

- **BREAKING (behavior):** the loop's parent is the latest accepted variant, not
  the best dev score; selection is paired. P3 results are superseded; see
  `docs/state/p32-skills-heldout-2026-10-23.md`.

### Fixed

- Trigger vocabulary no longer splits long words into fragments.

## [0.1.0] - 2026-10-23

### Added

- Phase 0 substrate: hardware probe, resource governor with a rolling spike
  budget, bubblewrap sandbox, TOML configuration, append-only event log, and
  the `janet` CLI (`probe`, `doctor`, `run`).
- AMD/ROCm GPU detection alongside `nvidia-smi`, and a VRAM budget with soft
  and spike caps.
- `num_gpu` / `num_ctx` model controls so the local 7B model stays under the
  VRAM cap.
- Phase 1: seed benchmark (10 code-repair + 4 integrity tasks), hidden anchor
  outside the repo with a SHA-256 manifest, black-box worker, cost/token
  accounting, and `janet bench baseline`.
- Phase 2: bounded generate → verify → repair loop with failure feedback that
  withholds expected outputs, best-of-N verifier selection, and
  `janet bench repair`.
- Phase 3: declarative `HarnessSpec`, SQLite + git variant archive, mutation
  proposer, Pareto generation loop, and `janet run` / `janet archive`.
- 207 tests, 86.8% coverage; strict mypy, ruff, and build green.
