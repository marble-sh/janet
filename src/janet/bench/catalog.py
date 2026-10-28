"""The complete seed catalog: the easy families plus the headroom family."""

from __future__ import annotations

from janet.bench.hard_seeds import hard_seeds
from janet.bench.headroom_seeds import headroom_seeds
from janet.bench.seeds import SEEDS, TaskSeed
from janet.bench.shape_seeds import shape_seeds

ALL_SEEDS: tuple[TaskSeed, ...] = SEEDS + hard_seeds() + headroom_seeds() + shape_seeds()
