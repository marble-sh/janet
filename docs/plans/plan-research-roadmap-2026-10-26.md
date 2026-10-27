# Research-to-implementation roadmap — hardware-bounded Janet

**Status:** ACCEPTED (2026-10-26) — A/B retained, C/D removed, two 10-generation confirmations
**Inputs:** 223-paper audit, 23 A-tier implement-now papers, 116 B-tier design
inputs. The active library is 139 papers (A/B only; 82 C and 2 D were deleted),
with 138 arXiv TeX sources.
**Companion:** `docs/plans/plan-p5-p6-2026-10-26.md`.
**Audit:** `docs/state/reference-download-2026-10-26/validation-report.md`.

---

## 1. What the research says, in one page

The evidence does **not** support "run more generations and the loop will
compound." It supports a narrower claim:

1. **The evaluator is the ceiling.** Program verifiers are stronger than learned
   judges; judges are advisory only. The judge at kappa 0.0 is consistent with
   the literature. Keep it out of acceptance.
2. **The harness/skill/memory surface is the tractable one for a frozen 7B.**
   Gödel-style agents, DGM, Voyager, SkillAxe, SkillRevise, Pythalab-MERA,
   Audited Skill-Graph, and ISM all improve behavior without changing weights.
3. **External memory only helps when it is verified, bounded, retrieved
   contextually, and consolidated.** Unstructured trace accumulation degrades
   performance. This is the P5 design constraint.
4. **The bottleneck is proposer diversity and benchmark headroom, not the
   acceptance gate.** Current accepted variants do not transfer to fresh tasks,
   and the bandit has only ~22 pulls. Adding gate complexity without harder tasks
   will produce more precise measurements of no transfer.
5. **Statistical control is cheap and necessary.** SGM-style e-processes,
   always-valid release, and bootstrap CIs are CPU-only and prevent overclaiming
   from small paired samples.
6. **Domain-specific and training-time papers are mostly background.** They
   justify the design boundary (no training, no frontier APIs, no robotics/
   clinical/vision scope) but should not drive code.

The 23 A-tier papers cluster into four implementation levers:

- **P5 memory/skills:** Pythalab-MERA, Audited Skill-Graph, EXG, ISM, SkillAxe,
  SkillRevise, Trajectory-Informed Memory, Reflexion.
- **P6 meta-loop:** SGM, Polaris, Harness Flaws, GEAR, BenchEvolver,
  Always-Valid Inference.
- **Evaluator:** Ask Don't Judge, CoSPlay, Falsification Not Exposure,
  LLM Judges Have Dark Current, Reliability without Validity, ACE,
  AutoPyVerifier, Hardening Agent Benchmarks.
- **Search/selection:** FunSearch.

The most important non-A evidence is the B-tier set on memory consolidation,
judge reliability, reward hacking, quality-diversity, and skill security.

---

## 2. Hardware envelope

| Resource | Soft cap | Spike cap | Implication |
|---|---:|---:|---|
| CPU | 8 cores | 12 cores | one generation at a time; no parallel model calls |
| RAM | 14 GiB | 16 GiB | one 7B q4 + one 3B judge, sequential |
| VRAM | 3.99 GiB | 4.79 GiB | 7B q4 at `num_gpu=20`; no second resident model |
| Disk | 16 GiB | — | research corpus 0.5 GiB; memory DB <100 MB; rotate runs |
| Spike | 300 s/hour | — | daemon sessions only when idle; preempt at task boundary |
| Model | Qwen2.5-Coder-7B q4 | — | ~16 tok/s; 8,192 context; 2,048 max output |
| Judge | Llama-3.2-3B q4 | — | on demand; advisory only |
| Network | none at runtime | — | all research and model downloads are build-time |

**Hard exclusions:** no weight training or fine-tuning (P7 deferred), no
embeddings/vector DB, no multi-GPU, no distributed search, no frontier API, no
remote task data, no second resident model.

---

## 3. Workstreams

### W1 — Benchmark headroom (highest priority, parallel with P5)

**Problem:** dev is ~0.9; one-shot anchor is ~0.77; fresh is ~0.63; accepted
variants do not transfer. A saturated dev split cannot distinguish harness
changes.

**Plan:**

- Adopt BenchEvolver's solution-centric escalation: evolve a trusted reference
  solution through structured transformations, derive the task statement and
  tests from the evolved solution, and accept a task only if the reference still
  passes and the parent algorithm fails.
- Add ACE/CoSPlay-style adversarial test generation: runtime errors,
  non-termination, side effects, edge cases, and property checks. Expected
  outputs always come from the trusted reference.
- Add held-out bug types, multi-function/spec-to-code tasks, and algorithm
  design tasks with objective checkers.
- Calibrate difficulty to a target one-shot baseline of 0.4–0.6 on dev; keep
  anchor and fresh harder than dev.
- Add contamination checks: no task text in memory/skills, no public-benchmark
  memorization, no anchor leakage, no fresh leakage.
- Rotate fresh tasks from a pre-authored pool; never use fresh for selection.

**Files:** `src/janet/bench/catalog.py`, `hard_seeds.py`, `materialize.py`,
`src/janet/suites.py`, `src/janet/eval/evolve.py`, tests.

**Acceptance:** one-shot dev 0.4–0.6; anchor and fresh gaps measurable; evolved
tests kill at least one reference mutant; no task/anchor/fresh leakage; two
10-generation runs can show a non-zero transfer signal.

### W2 — P5 process memory

Follow `plan-p5-p6-2026-10-26.md` §3.

**Sequencing:** store/schema → admission → retrieval → consolidation →
amnesia split/test.

**Acceptance:** memory-on beats memory-off on the hidden memory split with a
paired CI excluding zero; poisoned skill blocked; <100 ms p95 retrieval;
<100 MB DB; no anchor regression.

### W3 — P6 meta-loop

Follow `plan-p5-p6-2026-10-26.md` §4.

**Sequencing:** six operators/reward v3 → frontier/MAP-Elites → e-process gate
→ CUSUM/stale-objective → meta dashboard → two 10-generation confirmations.

**Acceptance:** each operator dispatchable; frontier coverage preserved; null
candidate rejected on synthetic data; strong paired improvement accepted;
CUSUM detects a synthetic gap; meta-slope CI reported; no anchor regression.

### W4 — Evaluator hardening

**Plan:**

- Judge calibration: use the LLM Judges Dark Current psychometric datasheet and
  Reliability without Validity protocol. Report kappa, TPR, TNR, consistency,
  and bias; never use the judge for acceptance.
- Binary-question diagnostics (Ask Don't Judge) for judge reasoning, not for
  acceptance.
- Reward-hacking detectors: dev-anchor gap, suspicious pass patterns, static
  AST audit, cheap trajectory probes, adversarial hacker-fixer tasks.
- Always-valid release for fresh confirmation; no optional-stopping p-hacking.
- Integrity suite: unsolvable/missing-data tasks scored on honest abstention;
  SciIntegrity-style scoring separate from capability.
- Oracle taxonomy: unit test, property, invariant, differential, formal; record
  which verifier type accepted each task.

**Files:** `src/janet/eval/judge.py`, `audit.py`, `evolve.py`,
`src/janet/loop/confirm.py`, `src/janet/suites.py`, tests.

**Acceptance:** judge kappa reported; judge cannot promote; adversarial
hacker-fixer tests blocked; integrity score reported separately; fresh release
uses always-valid inference; no false acceptance on a synthetic null.

### W5 — Harness mutation diversity

**Plan:**

- Trace-to-flaw attribution: normalize failed trajectories, align steps to
  harness artifacts, consolidate recurring diagnoses, map to scoped operators.
- Skill revision from failing traces (SkillRevise/Polaris), not generic
  critique.
- Skill quality dimensions: quality impact, trigger precision, instruction
  compliance, solution-path coverage.
- Anti-duplicate novelty: reject a mutation whose TF-IDF/Jaccard similarity to
  an existing variant is too high.
- Operator prompts that force distinct hypotheses: validation, exception
  handling, state, complexity, testability, stopping.
- Keep the diff bounded and the guard strict.

**Files:** `src/janet/harness/mutate.py`, `guard.py`, `skills.py`,
`src/janet/loop/generation.py`, `src/janet/loop/passive.py`, tests.

**Acceptance:** at least 3 distinct operators produce accepted or near-accepted
variants; duplicate rate below a declared threshold; no inert skill; no task
name; no evaluator/anchor edit.

### W6 — Safety and skill admission

**Plan:**

- Quarantine every candidate skill until tests + paired dev + anchor evidence
  pass.
- Static audit for network, subprocess, eval, filesystem escape, and
  evaluator-state inspection.
- MalSkillBench-style malicious skill tests; SkillMutator-style cross-modal
  attacks; SkillHarness-style safe skill wrapping.
- Data-leakage checks for tool-using tasks; no task content in logs that leave
  the machine (none leave).
- Layered mutability: bound mutation rate and require rollback evidence.
- Keep the unfireable-kernel concept as a design reference, not an
  implementation (no Rust/Z3/Kani in this phase).

**Acceptance:** poisoned skill rejected and logged; malicious skill cannot enter
retrieval; sandbox still blocks network and hides `/home`; rollback works.

### W7 — Resource governance and reporting

**Plan:**

- Add budget admission before each generation; estimate tokens/seconds from the
  parent run; skip or shrink the generation if the estimate exceeds the
  remaining session budget.
- Cache the parent dev baseline within a generation; reuse it for paired
  comparison.
- Early stop when dev falls below the floor; do not run the anchor.
- Reuse the loaded model across repair attempts; no model reload per task.
- Rotate runs and memory DB; keep disk under the 16 GiB budget.
- Report p50/p95 CPU/RAM/VRAM/disk and token/time cost per generation.
- Keep one generation + two evolve rungs per daemon session.

**Files:** `src/janet/governor.py`, `src/janet/daemon.py`,
`src/janet/loop/passive.py`, `src/janet/config.py`, `configs/default.toml`,
tests.

**Acceptance:** two 10-generation confirmations (20 total) with p95 within the
contract; no OOM; no unbounded disk; daemon preempts at the task boundary;
report includes resource compliance.

---

## 4. Five-week sequence (two confirmation runs)

Assumes one implementation stream (Flash) and one review gate (Opus/Sonnet) per
week, with at most one overnight run per week.

| Week | Primary | Parallel | Checkpoint |
|---|---|---|---|
| 1 | W1 benchmark headroom; W2 P5.1–P5.3 | index/search hygiene | one-shot dev 0.4–0.6; memory store + admission |
| 2 | W2 P5.4–P5.5; W3 P6.1–P6.2 | W4 judge/reward-hacking baseline | amnesia test; six operators; frontier |
| 3 | W3 P6.3–P6.5; W5 mutation diversity | W6 skill security | e-process gate; CUSUM; poisoned skill blocked |
| 4 | W3 P6.6 first 10-generation confirmation | W7 resource report; docs | transfer/meta-slope report; p95 budget |
| 5 | W3 P6.7 second 10-generation confirmation | checkpoint review | final transfer/meta-slope report; rollback check |

Do not run the first 10-generation confirmation until:

- the benchmark has headroom,
- the memory amnesia test passes,
- the poison test passes,
- the e-process gate passes synthetic null/alternative tests.

---

## 5. Experiment portfolio

| ID | Question | Design | Kill criterion |
|---|---|---|---|
| E1 | Did benchmark escalation restore headroom? | one-shot dev vs anchor vs fresh | dev remains >0.9 with no anchor gain |
| E2 | Does memory help? | memory on/off on hidden memory split, paired bootstrap | CI includes 0 |
| E3 | Which operator pays? | six-arm UCB1, min 3 pulls, cost-adjusted reward | no arm separation after two 10-generation runs |
| E4 | Does the gate control false acceptance? | synthetic null + strong alternative, e-process/fixed-sample | any null accepted |
| E5 | Can we detect reward hacking? | adversarial hacker-fixer tasks + dev-anchor gap | hacker accepted as improvement |
| E6 | Can a poisoned skill enter? | malicious lesson/skill injection | quarantined entry retrieved |
| E7 | Does memory/skills add cost? | p95 retrieval latency, prompt tokens, RAM/disk | >100 ms or >100 MB or prompt cap breach |
| E8 | Is meta-improvement real? | two 10-generation runs; bootstrap CI of slope | CI includes 0; report negative |

---

## 6. Decision gates

1. **Benchmark gate:** if one-shot dev is still saturated, stop P5/P6 expansion
   and fix W1. More loop iterations cannot fix a ceiling.
2. **Memory gate:** if memory-on does not beat memory-off on the hidden split,
   keep the store as a research artifact but do not wire it into generation.
3. **Operator gate:** if operators do not separate after two 10-generation runs, the
   proposer is the bottleneck; invest in W5 before adding arms.
4. **Gate gate:** if the e-process accepts a synthetic null or rejects a strong
   alternative, do not run the confirmation; fix the statistics.
5. **Safety gate:** if any quarantined skill is retrieved or any malicious skill
   passes, stop and fix W6.
6. **Resource gate:** if p95 exceeds the contract for three sessions, reduce
   batch size or stop; do not override the governor.
7. **Honesty gate:** if fresh delta is negative, report no transfer. A negative
   result is a successful measurement.

---

## 7. What not to build

- No QLoRA/fine-tuning/P7 in this cycle; the frozen-model contract is the point.
- No embeddings, vector DB, or second model; SQLite FTS5 is enough.
- No multi-agent orchestration, debate, or ensemble judges; one model at a time.
- No large-scale RL, synthetic-data training, or test-time weight updates.
- No frontier API calls; task data never leaves the machine.
- No edits to the anchor, acceptance verifiers, governor, sandbox, or promotion
  gate.
- No "judge decides" path; the judge is advisory.
- No removal of the 2 D-tier papers until the user approves; they remain
  flagged in `reference-validation.csv`.

---

## 8. Research evidence map

| Workstream | A-tier | B-tier support |
|---|---|---|
| W1 benchmark | BenchEvolver, ACE, CoSPlay | Verification Horizon, How Many Tries, When Good Verifiers Go Bad, SciIntegrity |
| W2 P5 memory | Pythalab-MERA, Audited Skill-Graph, EXG, ISM, SkillAxe, SkillRevise, Trajectory-Informed Memory, Reflexion | MemGym, Beyond Perplexity, Experience Graphs, Group-Evolving Agents, From Context to Skills |
| W3 P6 meta-loop | SGM, Polaris, Harness Flaws, GEAR, Always-Valid Inference | Statistical Limits, SAHOO, Layered Mutability, DEI, Heuresis, EvoTest, Live-SWE-agent |
| W4 evaluator | Ask Don't Judge, LLM Judges Dark Current, Reliability without Validity, Falsification, AutoPyVerifier, Hardening Agent Benchmarks | When Good Verifiers Go Bad, Cheap Reward Hacking Detection, Reward Hacking in LM Agents, Can LLM-as-a-Judge Verify Rubrics |
| W5 mutation | Harness Flaws, Polaris, SkillRevise, SkillAxe | Live-SWE-agent, STOP, Escher-Loop, Combee, SePO |
| W6 safety | Hardening Agent Benchmarks | MalSkillBench, SkillMutator, SkillHarness, Data Leakage Risks, Penetration Tests, Layered Mutability |
| W7 resources | Always-Valid Inference | OptPO, HS-STaR, Surprisal-Guided Selection, Multi-Agent Compute Efficiency |

---

## 9. Accepted decisions (2026-10-26)

1. **Memory split:** approved — new hidden, human-signed 12-task `memory` split
   outside the repo; fresh remains confirmation-only.
2. **Operator scope:** approved — six arms (`prompt`, `skill_add`,
   `skill_revise`, `budget`, `memory_policy`, `prompt+skill`).
3. **Frontier:** approved — replace greedy `current()` with a bounded
   MAP-Elites/frontier and novelty-aware parent selection.
4. **C/D papers:** deleted — 84 papers removed; the library now contains only
   the 23 A-tier and 116 B-tier papers. The full audit remains in `docs/state/`.
5. **Run cadence:** two 10-generation confirmations with a checkpoint between;
   stop early if the benchmark or memory gate fails.
6. **P7:** deferred — no weight training or fine-tuning in this cycle.

---

## 10. Deliverables

- `docs/plans/plan-p5-p6-2026-10-26.md` — detailed P5/P6 brief.
- `docs/plans/plan-research-roadmap-2026-10-26.md` — this roadmap.
- `docs/state/reference-download-2026-10-26/validation-report.md` — audit.
- `research/reference-validation.csv` — per-paper tier and reason.
- `research/reference-validation.md` — A-tier reading list.
- Searchable corpus: `research/<DOI>/paper.md`; tessera source
  `janet-research-docs`; code atlas `janet` and `janet-research`.
- After implementation: updated `docs/how-janet-learns.md`, `README.md`, and a
  state save-off with the two 10-generation confirmation results.
