# Janet — Verification-First Recursive Self-Improvement

**Status:** P0–P4 and the passive daemon (P4.5) are COMPLETE (2026-10-25).
The 6-generation measurement and fresh-split confirmation are in
`docs/state/p4-evaluator-2026-10-25.md`: one accepted variant, anchor +0.017,
fresh delta −0.030 (no transfer), judge kappa 0.0 (untrustworthy). P5 (process
memory) and P6 (meta-loop / operator bandit) are next. The P3 trajectory below
is superseded by `docs/state/p32-skills-heldout-2026-10-23.md`.
**Recommended executor:** DeepSeek V4.1 Flash for implementation; Claude Opus 5
for the P0/P3/P4/P6 review gates; Claude Sonnet 5.5 for evaluation design.
**Launch line:** `uv run janet run --config configs/default.toml --generations 24`

## 1. What the survey licenses

The design follows Chen, Wang, and Qu (2026), *Recursive Self-Improvement in
AI: From Bounded Self-Refinement to Autonomous Research Loops*, DOI
[10.48550/arXiv.2607.07663](https://doi.org/10.48550/arXiv.2607.07663):

- Every self-improvement loop is a claim that some signal can substitute for
  human judgment; the loop's ceiling is the quality of that substitute.
- Demonstrated strength tracks the verification hierarchy: formal verifiers >
  execution feedback > learned judges > intrinsic signals. Janet accepts only
  program verifiers; judges are advisory.
- Pure closed loops degrade (Mirror Loop, Shumailov, Zenil). Janet therefore
  carries an explicit exogenous grounding budget and a hidden, human-signed
  anchor suite.
- The tractable self-modification surface is the harness, skills, and search
  policy (DGM, Gödel Agent, Voyager, VASO, SkillAxe), not the weights.
- Evaluator co-evolution is the field's bet; anti-collusion instrumentation is
  built before the evaluator is allowed to evolve (Red Queen, A-Evolve).
- Process-level gains (verified skills, methodology memory) are the capital;
  result-level gains are operating expenditure.
- Integrity must be measured separately: unsolvable tasks score honest
  abstention, not fabricated completion (SciIntegrity).

## 2. Thesis and scope

Janet is a bounded, verifier-anchored RSI laboratory. It recursively improves
its own harness, skill library, and improvement policy against a hidden anchor
suite, and reports whether its rate of improvement is itself improving.

**Recursive** here means the system modifies the artifacts that determine how
it solves and improves (prompts, skills, routing, stopping rules, mutation
weights), and those modifications are evaluated by the same loop with a fixed
external anchor.

**Not in scope:** training a frontier model, claiming open-ended RSI,
autonomous deployment, or sending task data off the machine.

## 3. Resolved machine budget (2026-10-23)

| Resource | Available | Soft cap | Spike cap | Enforcement |
|---|---:|---:|---:|---|
| CPU | 16 cores | 8 cores | 12 cores | affinity + cgroup + monitor |
| RAM | 61.7 GiB total, ~18.5 GiB available | 14.0 GiB | 16.0 GiB | `RLIMIT_AS` + cgroup + monitor |
| VRAM | AMD RX 6600, 7.98 GiB | 3.99 GiB | 4.79 GiB | `num_gpu` offload cap + ROCm check |
| Disk | 32 GiB free | 16 GiB | — | run rotation + GC |
| Spike allowance | — | 300 s per rolling hour | — | rolling `SpikeBudget` |

There is **no NVIDIA GPU**; the machine has an AMD RX 6600 (ROCm) plus the
Ryzen iGPU. The model server runs in a transient user systemd unit with
`CPUQuota=800%`, `MemoryMax=12G`, and `Nice=10`. A 7B model at
`num_gpu=20`/`num_ctx=8192` uses ~4.01 GB VRAM and ~6.4 GB RSS, staying under
the soft caps. A second model is loaded only after unloading the first
(`OLLAMA_MAX_LOADED_MODELS=1`).

## 4. Runtime models

The runtime is **local-only**. `allow_remote_task_data=false` is enforced in
config; task content and failure traces never leave the machine. The build
harness (this session) is hybrid.

| Role | Model | Why |
|---|---|---|
| Policy / solver | `qwen2.5-coder:7b-instruct-q4_K_M` | Best code quality that fits under 4 GiB VRAM with partial offload; ~16 tok/s |
| Proposer / mutator | same as policy (sequential) | Avoids a second resident model under RAM/VRAM caps |
| Judge (advisory) | `llama3.2:3b-instruct-q4_K_M` | Different family; never in the acceptance path; loaded only on demand |
| Embeddings / novelty | none (TF-IDF) in v1 | Avoids torch and extra VRAM |
| Meta-policy | UCB1 bandit | Auditable; no LLM rationalization |
| Static safety | AST + allowlist | No model dependence |

Build executors per phase are in §7. The default implementation model is
DeepSeek V4.1 Flash; premium models are reserved for review gates.

## 5. Architecture

```
Immutable safety kernel (budget, sandbox, anchor hashes, promotion gate)
        |
Orchestrator -> select parent -> proposer mutates harness/skills/policy
        -> static checks -> sandbox -> dev verifiers
        -> hidden anchor + fresh suites -> archive (SQLite + git)
        -> lessons + skill admission -> bandit updates -> report
```

Mutable: system prompt, skill documents + code + tests, model routing,
stopping rules, tool policy, context policy, mutation-operator weights.

Immutable: governor, sandbox, anchor suite + manifest, acceptance verifiers,
promotion gate. Candidates write only to `runs/<id>/candidate/`; anchor tests
live outside the sandbox (`~/.local/share/janet/anchor`) and are never mounted.

### The loop

1. Sample a parent from the archive (Pareto over capability, cost, novelty;
   MAP-Elites preserves diversity).
2. Retrieve verified lessons and admitted skills for the task family.
3. Proposer emits a bounded mutation (prompt diff, new skill, routing change,
   stopping rule).
4. Static checks: allowed paths, diff size, no evaluator/anchor edits, no
   network, AST scan.
5. Dev suite in the bubblewrap sandbox; program verifiers only.
6. If gates pass, run the hidden anchor and periodically the fresh suite.
7. Archive; update lessons; update the bandit.
8. Promote only through the human gate by default, with git commit + rollback.
9. Every N generations, run the stale-objective probe and compute meta-metrics.

## 6. Verification and measurement

**Task families (P1):**

- Code repair: seeded Python bugs, hidden pytest suite.
- Algorithm design: bin packing, sorting networks, regex golf, scheduling;
  objective + validity checker.
- Harness meta-tasks: produce a prompt/skill that maximizes a frozen solver's
  pass rate on held-out tasks (the recursive bridge).
- Integrity suite: unsolvable or missing-data tasks; correct behavior is an
  honest abstention. Score separately.
- Stale-objective probe (P6): dev metric decoupled from the true target after
  a shift; measure detection and reframing.

**Splits:** `dev` visible, `anchor` hidden and human-signed, `fresh` rotated
and never used for selection, `integrity` mixed in.

**Metrics:**

- `capability(g)` on the hidden anchor.
- `fresh_capability(g)` on tasks never seen by the loop.
- `improvement_rate(g)` = capability gain per unit compute.
- `meta_improvement` = slope of `improvement_rate` over generations, with a
  bootstrap CI. This is the RSI signal.
- `transfer` = fresh-task gain versus the seed harness.
- `grounding_ratio` = exogenous signal / total signal, with a floor.
- `diversity` = archive coverage + distinct operators + task-family spread.
- `integrity` = 1 − violations/generations; honest abstention rate.
- `resource_compliance` = p95 CPU/RAM/VRAM vs the contract.

**Anti-Goodhart controls:** hidden anchor, scalar-only evaluation API, no
network, randomized instances, dev/anchor correlation monitor, mutation-kill
gate for generated tests, and a human promotion gate.

## 7. Phases, models, and acceptance

| Phase | Deliverable | Runtime model | Build executor | Review gate | Status |
|---|---|---|---|---|---|
| **P0** Substrate | probe, governor, sandbox, config, events, CLI | none | **DeepSeek V4.1 Flash** | **Claude Opus 5** | **DONE** |
| **P1** Benchmark | 3 families + integrity + hidden anchor + baseline | Qwen2.5-Coder-7B | **DeepSeek V4.1 Flash** | **Claude Sonnet 5.5** | **DONE** |
| **P2** Bounded refinement | generate → execute → parse failure → repair | Qwen2.5-Coder-7B | **DeepSeek V4.1 Flash** | — | **DONE** |
| **P3** Archive + harness mutation | variant spec, mutations, SQLite+git archive, Pareto/MAP-Elites, rollback | Qwen2.5-Coder-7B | **DeepSeek V4 Pro** design + **Flash** implementation | **Claude Opus 5** | **DONE** |
| **P3.2** Harder anchor + skills | held-out bugs, verified skill admission, paired selection, fresh split | Qwen2.5-Coder-7B | **DeepSeek V4.1 Flash** | **Claude Sonnet 5.5** | **DONE (negative: no transfer)** |
| **P3.3** Headroom benchmark | graded multi-function / spec-to-code tasks; one-shot baseline near 0.5 | Qwen2.5-Coder-7B | **Claude Sonnet 5.5** design + **Flash** implementation | **Claude Opus 5** | **DONE** |
| **P4** Evaluator co-evolution | test generation, mutation-kill gate, judge calibration, decoupling detector | 7B + Llama-3.2-3B judge | **Claude Sonnet 5.5** design + **Flash** implementation | **Claude Opus 5** | **DONE (negative transfer)** |
| **P4.5** Passive daemon | idle-only background ladder, preemptible, systemd-capped | 7B + Llama-3.2-3B judge | **DeepSeek V4.1 Flash** | **Claude Sonnet 5.5** | **DONE (running)** |
| **P5** Process memory | skills with tests, lessons, consolidation, amnesia test | 7B + reflector | **DeepSeek V4.1 Flash** | **Claude Sonnet 5.5** | |
| **P6** Meta-loop | UCB1 bandit, fresh tasks, stale-objective probe, RSI dashboard | all | **DeepSeek V4 Pro** design + **Flash** implementation | **Claude Opus 5** | **IN PROGRESS (operator bandit done)** |
| **P7** Optional weights | QLoRA on a 0.5–3B model, night-only, rollback | tiny model + PEFT | **GPT-5.3-Codex** or **Sonnet 5.5** | **Claude Opus 5** | DEFERRED |

### P0 acceptance evidence

- `uv run janet probe` reports 16 cores, 14 GiB RAM budget, 3.99 GiB VRAM
  budget, 16 GiB disk budget, 300 s/hour spike allowance.
- `uv run janet doctor` passes all required checks: Python 3.14.8, uv, git,
  bwrap, disk, AMD GPU, sandbox network denial, two local models.
- The governor kills a 300 MB allocation under a 64 MB limit; the sandbox
  blocks network and hides `/home`; the timeout kills a sleeping child.
- 110 tests pass; coverage 86.45%; ruff, mypy strict, and `uv build` green.
- The model server is capped: 4.01 GB VRAM, under the 3.99 GiB soft cap, with
  a 4.79 GiB spike ceiling.

### P1 acceptance evidence

- Seed suite: 14 tasks (10 code repair, 4 integrity), materialized from
  `src/janet/bench/seeds.py` with a trusted reference per task. The candidate
  sees the buggy code and visible cases; the evaluator holds expected outputs.
- Dev baseline: 12/14 passed, mean score 0.905, 749 eval tokens, 62 s, 11.7
  tok/s.
- Hidden anchor (outside the repo, manifest-verified): 12/14 passed, mean score
  0.923, 55 s, 13.8 tok/s.
- Integrity: 4/4 on both splits (honest abstention).
- Persistent failures: `flatten_nested` (drops string leaves) and
  `parse_duration` (iterates characters). Both are genuine model errors, not
  verifier artifacts.
- 170 tests, 87.9% coverage; ruff, mypy strict, and build green.
- Limitation: the anchor holds out inputs, not bugs. The dev/anchor scores are
  nearly identical, so P3 must add held-out bugs to make the anchor a harder
  generalization test.

### P2 acceptance evidence

- Full dev repair: baseline 0.9048 → repair 0.9524, 13/14 passed, 2 improved,
  0 regressed, 3361 eval tokens, 289 s.
- Best-of-N (3 samples per attempt) fixed `flatten_nested` in one run; sampling
  variance still leaves `parse_duration` partial. The gain is real but the 7B
  model is not reliable.
- The repair prompt contains only inputs and the candidate's own output; expected
  outputs are withheld. Integrity repair preserves the abstention contract.

### P3 acceptance evidence

- Archive: SQLite index + git commit per variant; `janet archive list/show`,
  `promote`, `rollback`.
- Trajectory: seed 0.9048/0.8988 → `2b87a277` 0.9762/0.9345 → `8f763e61`
  1.0000/0.9345 → `95e014bc` 0.9762/rejected. Net +9.5 dev, +3.6 anchor.
- The best variant is archived but **not promoted** (`loop.promote=false`); the
  promotion command is in `docs/state/p3-loop-2026-10-23.md`.
- Limitation: the winning prompt names `flatten_nested`; the anchor holds out
  inputs, not bugs, so this is narrow. P3.2 adds held-out bugs and skills.

### P3.2 acceptance evidence

- 7 anchor-only (held-out bug) tasks, 6 fresh tasks, skills, leakage guard,
  paired parent re-measurement, accepted-parent recursion, `confirm`/`lineage`.
- 8 generations, 2 accepted; both won on dev tasks re-tested on anchor inputs.
  Fresh-split delta **0.000** for both accepted variants: no transfer.
- Conclusion: the suite had no headroom; P3.3 builds one.

### P3.3 acceptance evidence

- 18 graded tasks (multi-function, spec-to-code, larger-file bugs), difficulty 3.
- One-shot baseline at `min_difficulty=3`: dev **0.900** (10 tasks),
  anchor **0.768** (16), fresh **0.633** (6).
- AST audit flags evaluator-state inspection; `loop.min_difficulty = 3` keeps
  integrity tasks while dropping the saturated easy tasks.

### P4 acceptance evidence

- Evaluator evolution: 6 dev tasks, 118 mutants, mutation score **0.686 →
  0.737**, 4 accepted kill-cases (`cron_field_match` 3, `knapsack_pick` 1).
- Dev one-shot after evolution: **0.867** (was 0.900); repair seed 0.933.
- 6 generations: 1 accepted (`d2a036af`): dev 0.933, anchor **0.768 → 0.785**.
  Generations 2–3 rejected on anchor (0.679, 0.706); 4–6 rejected on the dev
  floor without running the anchor (dev 0.933 / 0.900 / 0.867 vs parents
  0.967 / 0.933 / 0.933).
- Fresh-split confirmation: seed **0.633** vs variant **0.604**, delta **−0.030**;
  no transfer. The variant is not promoted.
- Judge calibration: kappa **0.0** (TPR 1.0, TNR 0.0); untrustworthy and advisory.
- Decoupling false positive from 1-ULP dev differences fixed; tie epsilon and
  near-zero-gain guard, with regression tests.
- 369 tests, 87.13% coverage; ruff, mypy strict, and build green.
- The 6-generation measurement, fresh confirmation, calibration, and daemon
  launch are recorded in `docs/state/p4-evaluator-2026-10-25.md`.

## 8. Failure modes and controls

| Failure mode | Control | Test |
|---|---|---|
| Self-confirming loop | generator/judge separation; program verifiers accept; hidden anchor | biased judge cannot promote |
| Model collapse | no weight training in v1; grounding ratio; fresh tasks; entropy monitors | collapse simulation |
| Diversity collapse | MAP-Elites + novelty + multiple operators + fresh families | coverage over 20 generations |
| Frame lock-in | human-signed anchor; stale-objective probe; human gate | decoupled dev metric triggers detector |
| Reward hacking / exfiltration | anchor outside sandbox; scalar-only API; no network; audit hook | malicious candidate is blocked and flagged |
| Integrity failure | unsolvable tasks scored on honest abstention | SciIntegrity-style suite |
| Corrupted skill propagation | admission tests, versioning, quarantine, rollback | poisoned skill rejected |
| Resource exhaustion | governor, kill switch, checkpoint/resume | memory-hog test |

## 9. Definition of done

A ≥20-generation overnight run on this machine, p95 resource use within the
contract, and a report with anchor capability, improvement rate, meta-slope,
transfer to fresh tasks, diversity, integrity, cost, and at least one accepted
self-modification that survives a rollback test. If there is no gain, the
report says so with evidence. A negative result is still a successful outcome
for a measurement instrument.

## 10. Open risks

- 7B-class local models may be too weak for harness-level mutations to
  compound; P1's baseline will tell us, and the fallback is a stronger
  smaller-fine-tune or a hybrid meta-proposer with sanitized metadata only.
- Partial GPU offload can change with driver/Ollama updates; `doctor` must
  re-check VRAM before each run.
- Disk is tight (32 GiB free); the anchor and run artifacts must be small, and
  model downloads are frozen at the current two.
- Single-user isolation is best-effort; the sandbox hides `/home` and blocks
  network, but the safety kernel is not a separate UID. Documented as a
  limitation, not a guarantee.

## 11. Key sources

FunSearch, AlphaEvolve, Darwin Gödel Machine, Gödel Agent, Voyager, VASO,
SkillAxe, Mirror Loop, A-Evolve-Training, Meta-Agent Challenge, Red Queen
Gödel Machine, SciIntegrity, Self-Refine, Reflexion, STaR, SPIN, Absolute Zero,
R-Zero, Self-Rewarding LMs, and the survey's verification-hierarchy argument.
