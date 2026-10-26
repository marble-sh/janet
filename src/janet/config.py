"""Configuration loading for Janet.

All configuration is TOML. Unknown keys are rejected so a typo cannot silently
disable a budget or a privacy guard.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

DEFAULT_CONFIG_PATH: Final = Path("configs/default.toml")


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    """The resource contract.

    Fractions are of total hardware. The governor additionally clamps memory
    against what is actually available, so a busy machine gets a smaller
    budget instead of being pushed into swap.
    """

    cpu_fraction: float = 0.5
    mem_fraction: float = 0.5
    disk_fraction: float = 0.5
    gpu_fraction: float = 0.5
    spike_seconds_per_hour: float = 300.0
    spike_cpu_fraction: float = 0.75
    spike_mem_fraction: float = 0.6
    spike_gpu_fraction: float = 0.6
    min_cpu_cores: int = 1
    max_cpu_cores: int = 8
    max_mem_gib: float = 14.0
    max_disk_gib: float = 20.0
    max_vram_gib: float = 8.0
    max_wall_seconds: int = 1200
    reserve_mem_gib: float = 2.0
    reserve_vram_gib: float = 0.5


@dataclass(frozen=True, slots=True)
class PathsConfig:
    runs: Path = Path("runs")
    archive: Path = Path("runs/archive")
    archive_db: Path = Path("runs/archive.sqlite3")
    events: Path = Path("runs/events.jsonl")
    anchor: Path = Path.home() / ".local/share/janet/anchor"
    fresh: Path = Path.home() / ".local/share/janet/fresh"
    evolved: Path = Path("runs/evolved")
    daemon: Path = Path("runs/daemon")
    bandit: Path = Path("runs/bandit.json")


@dataclass(frozen=True, slots=True)
class LLMConfig:
    mode: str = "local"
    base_url: str = "http://127.0.0.1:11434"
    policy_model: str = "qwen2.5-coder:7b-instruct-q4_K_M"
    judge_model: str = "llama3.2:3b-instruct-q4_K_M"
    temperature: float = 0.0
    max_tokens: int = 2048
    context_tokens: int = 8192
    num_gpu: int = 0
    timeout_seconds: float = 300.0
    allow_remote_task_data: bool = False


@dataclass(frozen=True, slots=True)
class LoopConfig:
    max_generations: int = 24
    max_attempts: int = 3
    mutation_max_chars: int = 4000
    promote: bool = False
    # Dev may dip by this much against the paired parent (sampling noise) when
    # the candidate earns acceptance through a strict hidden-anchor gain.
    dev_margin: float = 0.03
    # Easy tasks are saturated for the 7B policy; the loop selects on difficulty >= this.
    min_difficulty: int = 3
    # UCB1 exploration constant for the P6 operator bandit.
    bandit_exploration: float = 0.7


@dataclass(frozen=True, slots=True)
class DaemonConfig:
    """The idle-only background worker.

    The daemon is a convenience layer over the same governed loop: it only starts
    when the machine is idle and yields as soon as foreground work appears. The
    systemd caps are a second fence under the governor, not a replacement for it.
    """

    enabled: bool = True
    session_seconds: float = 3600.0
    idle_seconds: float = 90.0
    check_seconds: float = 30.0
    cooldown_seconds: float = 60.0
    max_load_fraction: float = 0.75
    min_free_mem_gib: float = 4.0
    min_free_disk_gib: float = 3.0
    confirm_interval_hours: float = 24.0
    calibrate_interval_hours: float = 168.0
    evolve_per_session: int = 2
    generations_per_session: int = 1
    cpu_quota_percent: int = 300
    memory_max_gib: float = 10.0
    memory_high_gib: float = 8.0
    nice: int = 15


@dataclass(frozen=True, slots=True)
class Config:
    budget: BudgetConfig = BudgetConfig()
    paths: PathsConfig = PathsConfig()
    llm: LLMConfig = LLMConfig()
    loop: LoopConfig = LoopConfig()
    daemon: DaemonConfig = DaemonConfig()


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return cast("dict[str, Any]", value)


def _validate_budget(config: BudgetConfig) -> None:
    for name in ("cpu_fraction", "mem_fraction", "disk_fraction", "gpu_fraction"):
        value = getattr(config, name)
        if not 0.0 < value <= 1.0:
            raise ConfigError(f"budget.{name} must be in (0, 1], got {value}")
    if config.spike_seconds_per_hour < 0.0:
        raise ConfigError("budget.spike_seconds_per_hour must be >= 0")
    for name in (
        "spike_cpu_fraction",
        "spike_mem_fraction",
        "spike_gpu_fraction",
    ):
        value = getattr(config, name)
        if not 0.0 < value <= 1.0:
            raise ConfigError(f"budget.{name} must be in (0, 1], got {value}")
    if config.spike_cpu_fraction < config.cpu_fraction:
        raise ConfigError("budget.spike_cpu_fraction must be >= budget.cpu_fraction")
    if config.spike_mem_fraction < config.mem_fraction:
        raise ConfigError("budget.spike_mem_fraction must be >= budget.mem_fraction")
    if config.spike_gpu_fraction < config.gpu_fraction:
        raise ConfigError("budget.spike_gpu_fraction must be >= budget.gpu_fraction")
    if config.min_cpu_cores < 1:
        raise ConfigError("budget.min_cpu_cores must be >= 1")
    if config.max_cpu_cores < config.min_cpu_cores:
        raise ConfigError("budget.max_cpu_cores must be >= budget.min_cpu_cores")
    for name in ("max_mem_gib", "max_disk_gib", "max_vram_gib", "max_wall_seconds"):
        if getattr(config, name) <= 0:
            raise ConfigError(f"budget.{name} must be > 0")
    for name in ("reserve_mem_gib", "reserve_vram_gib"):
        if getattr(config, name) < 0.0:
            raise ConfigError(f"budget.{name} must be >= 0")


def _build_budget(raw: dict[str, Any]) -> BudgetConfig:
    try:
        config = BudgetConfig(**_table(raw, "budget"))
    except TypeError as exc:
        raise ConfigError(f"invalid [budget]: {exc}") from exc
    _validate_budget(config)
    return config


def _build_paths(raw: dict[str, Any]) -> PathsConfig:
    values = dict(_table(raw, "paths"))
    for key in (
        "runs",
        "archive",
        "archive_db",
        "events",
        "anchor",
        "fresh",
        "evolved",
        "daemon",
        "bandit",
    ):
        if key in values:
            values[key] = Path(str(values[key])).expanduser()
    try:
        return PathsConfig(**values)
    except TypeError as exc:
        raise ConfigError(f"invalid [paths]: {exc}") from exc


def _build_llm(raw: dict[str, Any]) -> LLMConfig:
    try:
        config = LLMConfig(**_table(raw, "llm"))
    except TypeError as exc:
        raise ConfigError(f"invalid [llm]: {exc}") from exc
    if config.mode not in {"local", "hybrid", "remote"}:
        raise ConfigError("llm.mode must be one of: local, hybrid, remote")
    if config.mode == "remote" and not config.allow_remote_task_data:
        raise ConfigError(
            "llm.mode = 'remote' requires allow_remote_task_data = true; "
            "task content must not leave the machine by default"
        )
    if config.temperature < 0.0:
        raise ConfigError("llm.temperature must be >= 0")
    if config.max_tokens <= 0 or config.context_tokens <= 0:
        raise ConfigError("llm.max_tokens and llm.context_tokens must be > 0")
    if config.timeout_seconds <= 0.0:
        raise ConfigError("llm.timeout_seconds must be > 0")
    if config.num_gpu < 0:
        raise ConfigError("llm.num_gpu must be >= 0")
    return config


def _build_loop(raw: dict[str, Any]) -> LoopConfig:
    try:
        config = LoopConfig(**_table(raw, "loop"))
    except TypeError as exc:
        raise ConfigError(f"invalid [loop]: {exc}") from exc
    if config.max_generations < 1:
        raise ConfigError("loop.max_generations must be >= 1")
    if config.max_attempts < 1:
        raise ConfigError("loop.max_attempts must be >= 1")
    if config.mutation_max_chars < 1:
        raise ConfigError("loop.mutation_max_chars must be >= 1")
    if config.min_difficulty < 1:
        raise ConfigError("loop.min_difficulty must be >= 1")
    if not 0.0 <= config.dev_margin <= 0.1:
        raise ConfigError("loop.dev_margin must be in [0, 0.1]")
    if config.bandit_exploration < 0.0:
        raise ConfigError("loop.bandit_exploration must be >= 0")
    return config


def _build_daemon(raw: dict[str, Any]) -> DaemonConfig:
    try:
        config = DaemonConfig(**_table(raw, "daemon"))
    except TypeError as exc:
        raise ConfigError(f"invalid [daemon]: {exc}") from exc
    for name in ("session_seconds", "check_seconds"):
        if getattr(config, name) <= 0:
            raise ConfigError(f"daemon.{name} must be > 0")
    if config.idle_seconds < 0 or config.cooldown_seconds < 0:
        raise ConfigError("daemon idle_seconds and cooldown_seconds must be >= 0")
    if not 0.0 < config.max_load_fraction <= 1.0:
        raise ConfigError("daemon.max_load_fraction must be in (0, 1]")
    for name in ("min_free_mem_gib", "min_free_disk_gib"):
        if getattr(config, name) < 0:
            raise ConfigError(f"daemon.{name} must be >= 0")
    for name in ("confirm_interval_hours", "calibrate_interval_hours"):
        if getattr(config, name) < 0:
            raise ConfigError(f"daemon.{name} must be >= 0")
    if config.evolve_per_session < 0 or config.generations_per_session < 0:
        raise ConfigError("daemon per-session job counts must be >= 0")
    if config.cpu_quota_percent <= 0:
        raise ConfigError("daemon.cpu_quota_percent must be > 0")
    if config.memory_max_gib <= 0 or config.memory_high_gib <= 0:
        raise ConfigError("daemon memory caps must be > 0")
    if config.memory_high_gib > config.memory_max_gib:
        raise ConfigError("daemon.memory_high_gib must be <= daemon.memory_max_gib")
    if not 0 <= config.nice <= 19:
        raise ConfigError("daemon.nice must be in [0, 19]")
    return config


def load_config(path: Path | None = None) -> Config:
    """Load and validate a TOML configuration.

    With no path, returns the documented defaults. A missing explicit path is
    an error rather than a silent fallback.
    """

    if path is None:
        return Config()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    return Config(
        budget=_build_budget(raw),
        paths=_build_paths(raw),
        llm=_build_llm(raw),
        loop=_build_loop(raw),
        daemon=_build_daemon(raw),
    )
