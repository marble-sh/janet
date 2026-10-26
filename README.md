# Janet

A verification-first, resource-bounded recursive self-improvement laboratory.

Janet improves its own harness, skill library, and improvement policy against a
hidden, human-signed anchor suite. It is deliberately **bounded**: every
acceptance decision is made by a program verifier, learned judges are advisory,
and the loop reports whether its rate of improvement is itself improving.

The design is grounded in Chen, Wang, and Qu (2026), *Recursive
Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research
Loops*, DOI
[10.48550/arXiv.2607.07663](https://doi.org/10.48550/arXiv.2607.07663).

## Status

P0–P4 are complete: substrate, verification-first benchmark with a hidden
anchor, bounded repair loop, recursive harness loop (archive + mutation +
Pareto selection), and evaluator co-evolution (mutant-killing inputs, judge
calibration, decoupling detection). P5 (process memory) and P6 (meta-loop) are
next. See
[`docs/plans/plan-rsi-janet-2026-10-23.md`](docs/plans/plan-rsi-janet-2026-10-23.md).

## Quick start

```bash
uv sync
uv run janet probe
uv run janet doctor
uv run janet run --dry-run
uv run janet bench tasks
uv run janet bench baseline --limit 4
uv run janet bench repair --limit 4 --attempts 3 --samples 3
```

## Recursive loop

```bash
# The hidden anchor lives outside the repo and is manifest-verified.
uv run janet bench materialize --root ~/.local/share/janet/anchor --split anchor --force

# One generation: propose a harness mutation, evaluate dev, evaluate the
# anchor when dev does not regress, archive the result.
uv run janet run --generations 1

# Inspect the archive; promotion is the human gate.
uv run janet archive list
uv run janet archive show <hash>
uv run janet archive promote <hash>
uv run janet archive rollback

# Grow the dev evaluator: propose inputs, keep only those that kill a reference
# mutant the suite does not already kill. Expected outputs come from the trusted
# reference, never from the proposer.
uv run janet evolve

# Calibrate the advisory judge against the verifier (kappa; never in the
# acceptance path).
uv run janet calibrate --split dev --limit 4

# Inspect the P6 operator bandit: which kind of mutation has been earning its
# compute.
uv run janet bandit
```

Selection is Pareto: dev must not regress, and the candidate must improve dev
or tie dev while improving the hidden anchor. The best verified variant is
archived but not promoted by default (`loop.promote = false`).

## Publishing

Janet submits committed work as bot-authored pull requests through the trusted
`janet-publish` entry point. The publisher refuses protected paths, never
pushes `main`, and never merges. See
[`docs/publishing.md`](docs/publishing.md).

## Passive background daemon

When the machine is idle, Janet can improve herself in the background. A session
starts only after a quiet period, runs at most one harness generation plus a
couple of evaluator-evolution rungs, and yields at the next task boundary as
soon as any foreground `janet` work appears (or the machine crosses the spike
load). Fresh-split confirmation and judge calibration run at most once per
variant/interval.

```bash
uv run janet daemon start     # transient systemd user unit, capped
uv run janet daemon status
uv run janet daemon stop
uv run janet daemon once      # one session now, ignoring idleness (smoke test)
```

The unit is `janet-daemon` with `CPUQuota=300%`, `MemoryMax=10G`,
`MemoryHigh=8G`, and `Nice=15`; the model server holds the other 300% of the
8-core soft budget, leaving two cores for the desktop. State and logs live under `runs/daemon/`, and the ladder's
cursor survives restarts. A transient unit does **not** survive a reboot:
restart the model server, then `janet daemon start`.

## Resource contract

Janet targets at most 50% of total machine resources, with a rolling spike
allowance of 300 seconds per hour. The governor clamps against **available**
memory as a safety measure, so a busy machine gets a smaller budget rather than
being pushed into swap. See `configs/default.toml` and `src/janet/governor.py`.

On this machine the resolved soft caps are 8 CPU cores, 14 GiB RAM, 3.99 GiB
VRAM (AMD RX 6600), and 16 GiB disk.

## Local model server

The runtime is local-only. The model server is a transient user systemd unit
with CPU and memory caps:

```bash
systemd-run --user --unit janet-ollama --collect \
  -p MemoryMax=12G -p MemoryHigh=10G -p CPUQuota=300% -p Nice=10 \
  --setenv=OLLAMA_HOST=127.0.0.1:11434 \
  --setenv=OLLAMA_MAX_LOADED_MODELS=1 \
  --setenv=OLLAMA_NUM_PARALLEL=1 \
  --setenv=OLLAMA_KEEP_ALIVE=10m \
  ollama serve
```

The 300% model-server quota plus the daemon's 300% leaves two of the 8-core
soft budget for the desktop. The governor still pauses on memory pressure and
high load.

Stop it before measuring a clean budget, because a loaded model consumes the
free VRAM the probe would otherwise count:

```bash
systemctl --user stop janet-ollama
uv run janet probe
```

The 7B policy model is partially offloaded (`num_gpu = 20`, `num_ctx = 8192`)
to keep VRAM at ~4.01 GB, under the 3.99 GiB soft cap. Do not raise `num_gpu`
without re-checking `rocm-smi --showpids`.

## Development

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest --cov
uv build
```

## Layout

- `src/janet/probe.py` — hardware and tool discovery (NVIDIA and ROCm),
  budget calculation
- `src/janet/governor.py` — rlimits, CPU affinity, process-tree monitoring, spikes
- `src/janet/sandbox.py` — `bubblewrap` sandbox (read-only root, no network)
- `src/janet/config.py` — TOML configuration
- `src/janet/events.py` — append-only JSONL event log
- `src/janet/eval/mutants.py` — AST single-site mutants of a reference
- `src/janet/eval/evolve.py` — mutation-kill gate for generated test inputs
- `src/janet/eval/judge.py` — advisory judge calibration (Cohen's kappa)
- `src/janet/loop/passive.py` — the idle work ladder
- `src/janet/daemon.py` — idle detection, lock/pid, systemd unit management
- `src/janet/suites.py` — dev/anchor/fresh loading with evolved cases
- `configs/default.toml` — the default budget, model routing, loop, and daemon
- `docs/adr/` — architecture decisions
