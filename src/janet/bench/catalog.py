"""The complete seed catalog: the easy families plus the headroom family."""

from __future__ import annotations

from janet.bench.hard_seeds import hard_seeds
from janet.bench.seeds import SEEDS, TaskSeed

ALL_SEEDS: tuple[TaskSeed, ...] = SEEDS + hard_seeds()
