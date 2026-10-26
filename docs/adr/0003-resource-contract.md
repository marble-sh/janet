# ADR 0003 — Resource contract: total hardware, 300 s/hour spikes

**Status:** accepted 2026-10-23

## Context

The machine is an AMD Ryzen 7 7700 (16 threads), 61.7 GiB RAM, an AMD Radeon
RX 6600 (7.98 GiB VRAM) via ROCm, and 32 GiB free disk. There is no NVIDIA GPU.
The user requires total hardware load at or below 50%, with at most 300 seconds
per rolling hour above the soft cap.

## Decision

The governor resolves the soft caps to 8 CPU cores, 14 GiB RAM, 3.99 GiB VRAM,
and 16 GiB disk. Spike ceilings are 12 cores, 16 GiB RAM, and 4.79 GiB VRAM.
Spike time is tracked in a rolling one-hour window and the run is killed when
the allowance is exhausted. RAM is additionally clamped to available memory
minus a 2 GiB reserve, so a busy machine gets a smaller budget instead of
swapping. The model server runs in a transient user systemd unit with
`CPUQuota=800%`, `MemoryMax=12G`, and `Nice=10`; the 7B model is partially
offloaded with `num_gpu=20` and `num_ctx=8192`, which keeps VRAM at ~4.01 GB.

## Consequences

- The governor is stricter than the contract on CPU/RAM (static cgroup caps);
  spikes are available to the loop's child processes, not to the model server.
- The budget must be recomputed after the model server is stopped, because a
  loaded model consumes the free VRAM the probe would otherwise count.
- Any new model, server, or background process must be placed under the same
  caps; an ungoverned subprocess is a bug.
