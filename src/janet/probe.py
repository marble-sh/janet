"""Hardware and tool discovery, and the concrete resource budget.

The probe is deliberately dependency-free (no psutil, no torch): it reads
``/proc`` and shells out to ``nvidia-smi`` only when present.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from janet.config import BudgetConfig

GIB: Final = 1024**3
MIB: Final = 1024**2
MIN_RUNNABLE_MEMORY: Final = 256 * MIB
_TOOL_NAMES: Final = (
    "ollama",
    "llama-server",
    "bwrap",
    "systemd-run",
    "git",
    "uv",
    "tessera",
    "nvidia-smi",
    "rocm-smi",
)


class ResourceError(RuntimeError):
    """Raised when the machine cannot safely run the requested work."""


@dataclass(frozen=True, slots=True)
class CpuInfo:
    total_cores: int
    allowed_cores: int


@dataclass(frozen=True, slots=True)
class MemInfo:
    total_bytes: int
    available_bytes: int


@dataclass(frozen=True, slots=True)
class GpuInfo:
    index: int
    name: str
    total_bytes: int
    free_bytes: int
    vendor: str = "unknown"


@dataclass(frozen=True, slots=True)
class MachineProbe:
    cpu: CpuInfo
    mem: MemInfo
    gpus: tuple[GpuInfo, ...]
    disk_free_bytes: int
    load_average: tuple[float, float, float]
    tools: dict[str, str | None]
    python_version: str
    platform: str

    def to_dict(self) -> dict[str, object]:
        return {
            "cpu": {
                "total_cores": self.cpu.total_cores,
                "allowed_cores": self.cpu.allowed_cores,
            },
            "mem": {
                "total_bytes": self.mem.total_bytes,
                "available_bytes": self.mem.available_bytes,
                "total_gib": round(self.mem.total_bytes / GIB, 2),
                "available_gib": round(self.mem.available_bytes / GIB, 2),
            },
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "vendor": gpu.vendor,
                    "total_bytes": gpu.total_bytes,
                    "free_bytes": gpu.free_bytes,
                }
                for gpu in self.gpus
            ],
            "disk_free_bytes": self.disk_free_bytes,
            "disk_free_gib": round(self.disk_free_bytes / GIB, 2),
            "load_average": list(self.load_average),
            "tools": dict(self.tools),
            "python_version": self.python_version,
            "platform": self.platform,
        }


@dataclass(frozen=True, slots=True)
class Budget:
    """The resolved, concrete budget for this run."""

    cpu_cores: int
    cpu_cores_spike: int
    mem_bytes: int
    mem_bytes_spike: int
    disk_bytes: int
    max_wall_seconds: float
    spike_seconds_per_hour: float
    vram_bytes: int = 0
    vram_bytes_spike: int = 0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "cpu_cores": self.cpu_cores,
            "cpu_cores_spike": self.cpu_cores_spike,
            "mem_bytes": self.mem_bytes,
            "mem_gib": round(self.mem_bytes / GIB, 2),
            "mem_bytes_spike": self.mem_bytes_spike,
            "mem_gib_spike": round(self.mem_bytes_spike / GIB, 2),
            "disk_bytes": self.disk_bytes,
            "disk_gib": round(self.disk_bytes / GIB, 2),
            "vram_bytes": self.vram_bytes,
            "vram_gib": round(self.vram_bytes / GIB, 2),
            "vram_bytes_spike": self.vram_bytes_spike,
            "vram_gib_spike": round(self.vram_bytes_spike / GIB, 2),
            "max_wall_seconds": self.max_wall_seconds,
            "spike_seconds_per_hour": self.spike_seconds_per_hour,
            "notes": list(self.notes),
        }


def parse_meminfo(text: str) -> MemInfo:
    """Parse the subset of ``/proc/meminfo`` we need."""

    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            values[key.strip()] = int(parts[0]) * 1024
        except ValueError:
            continue
    try:
        return MemInfo(total_bytes=values["MemTotal"], available_bytes=values["MemAvailable"])
    except KeyError as exc:
        raise ValueError(f"missing {exc.args[0]} in /proc/meminfo") from exc


def parse_nvidia_smi(text: str) -> tuple[GpuInfo, ...]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits``."""

    gpus: list[GpuInfo] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index = int(parts[0])
            total_mib = int(float(parts[2]))
            free_mib = int(float(parts[3]))
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                index=index,
                name=parts[1],
                total_bytes=total_mib * MIB,
                free_bytes=free_mib * MIB,
                vendor="nvidia",
            )
        )
    return tuple(gpus)


def parse_rocm_smi_json(text: str) -> tuple[GpuInfo, ...]:
    """Parse ``rocm-smi --json --showmeminfo vram --showproductname``."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ()
    if not isinstance(payload, dict):
        return ()
    gpus: list[GpuInfo] = []
    for key, value in payload.items():
        if not key.startswith("card") or not isinstance(value, dict):
            continue
        try:
            total = int(value["VRAM Total Memory (B)"])
            used = int(value["VRAM Total Used Memory (B)"])
        except (KeyError, TypeError, ValueError):
            continue
        suffix = key.removeprefix("card")
        index = int(suffix) if suffix.isdigit() else len(gpus)
        name = value.get("Card Series", key)
        gpus.append(
            GpuInfo(
                index=index,
                name=str(name),
                total_bytes=total,
                free_bytes=max(0, total - used),
                vendor="amd",
            )
        )
    return tuple(gpus)


def _read_meminfo() -> MemInfo:
    try:
        return parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResourceError(f"cannot read /proc/meminfo: {exc}") from exc


def _allowed_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _probe_gpus() -> tuple[GpuInfo, ...]:
    nvidia = shutil.which("nvidia-smi")
    if nvidia is not None:
        try:
            completed = subprocess.run(
                [
                    nvidia,
                    "--query-gpu=index,name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if completed.returncode == 0:
                gpus = parse_nvidia_smi(completed.stdout)
                if gpus:
                    return gpus
        except (OSError, subprocess.TimeoutExpired):
            pass
    rocm = shutil.which("rocm-smi")
    if rocm is None:
        return ()
    try:
        completed = subprocess.run(
            [rocm, "--json", "--showmeminfo", "vram", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if completed.returncode != 0:
        return ()
    return parse_rocm_smi_json(completed.stdout)


def _load_average() -> tuple[float, float, float]:
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:
        return (0.0, 0.0, 0.0)
    return (one, five, fifteen)


def _which_tools() -> dict[str, str | None]:
    return {name: shutil.which(name) for name in _TOOL_NAMES}


def probe_machine(root: Path | None = None) -> MachineProbe:
    """Read the current machine state."""

    root = Path.cwd() if root is None else root
    try:
        disk_free = shutil.disk_usage(root).free
    except OSError:
        disk_free = 0
    return MachineProbe(
        cpu=CpuInfo(total_cores=os.cpu_count() or 1, allowed_cores=_allowed_cores()),
        mem=_read_meminfo(),
        gpus=_probe_gpus(),
        disk_free_bytes=disk_free,
        load_average=_load_average(),
        tools=_which_tools(),
        python_version=platform.python_version(),
        platform=platform.platform(),
    )


def compute_budget(probe: MachineProbe, config: BudgetConfig) -> Budget:
    """Turn fractions into concrete, safe limits.

    CPU is a fraction of the allowed cores. Memory is the *smallest* of the
    soft fraction, the absolute clamp, and what is actually available minus a
    reserve. That last term is intentional: the contract is about load on the
    machine, and a busy machine must get a smaller budget rather than swap.
    """

    allowed = max(1, probe.cpu.allowed_cores)
    notes: list[str] = []

    soft_cpu = math.floor(allowed * config.cpu_fraction)
    soft_cpu = max(config.min_cpu_cores, min(config.max_cpu_cores, soft_cpu))
    if (
        soft_cpu >= config.max_cpu_cores
        and math.floor(allowed * config.cpu_fraction) > config.max_cpu_cores
    ):
        notes.append(f"cpu clamped to max_cpu_cores={config.max_cpu_cores}")
    spike_cpu = max(soft_cpu, min(allowed, math.ceil(allowed * config.spike_cpu_fraction)))

    soft_mem = int(probe.mem.total_bytes * config.mem_fraction)
    available_cap = max(0, probe.mem.available_bytes - int(config.reserve_mem_gib * GIB))
    absolute_cap = int(config.max_mem_gib * GIB)
    mem = min(soft_mem, available_cap, absolute_cap)
    if mem == available_cap and available_cap < min(soft_mem, absolute_cap):
        notes.append("mem clamped by available memory minus reserve")
    elif mem == absolute_cap and absolute_cap < min(soft_mem, available_cap):
        notes.append(f"mem clamped to max_mem_gib={config.max_mem_gib}")
    if mem < MIN_RUNNABLE_MEMORY:
        raise ResourceError(
            f"only {mem / MIB:.0f} MiB of memory budget available "
            f"(reserve {config.reserve_mem_gib:.1f} GiB); refusing to start"
        )

    spike_mem = min(
        int(probe.mem.total_bytes * config.spike_mem_fraction),
        max(mem, available_cap),
    )

    disk = min(
        int(probe.disk_free_bytes * config.disk_fraction),
        int(config.max_disk_gib * GIB),
    )
    if disk <= 0:
        raise ResourceError("no disk budget available")

    vram = 0
    spike_vram = 0
    if probe.gpus:
        gpu = max(probe.gpus, key=lambda item: item.total_bytes)
        soft_vram = int(gpu.total_bytes * config.gpu_fraction)
        vram_available_cap = max(0, gpu.free_bytes - int(config.reserve_vram_gib * GIB))
        vram_absolute_cap = int(config.max_vram_gib * GIB)
        vram = min(soft_vram, vram_available_cap, vram_absolute_cap)
        if vram == vram_available_cap and vram_available_cap < min(soft_vram, vram_absolute_cap):
            notes.append("vram clamped by free VRAM minus reserve")
        elif vram == vram_absolute_cap and vram_absolute_cap < min(soft_vram, vram_available_cap):
            notes.append(f"vram clamped to max_vram_gib={config.max_vram_gib}")
        spike_vram = min(
            int(gpu.total_bytes * config.spike_gpu_fraction),
            max(vram, vram_available_cap),
        )
        if vram == 0:
            notes.append("no VRAM budget; the model must run on CPU")

    return Budget(
        cpu_cores=soft_cpu,
        cpu_cores_spike=spike_cpu,
        mem_bytes=mem,
        mem_bytes_spike=spike_mem,
        disk_bytes=disk,
        max_wall_seconds=float(config.max_wall_seconds),
        spike_seconds_per_hour=config.spike_seconds_per_hour,
        vram_bytes=vram,
        vram_bytes_spike=spike_vram,
        notes=tuple(notes),
    )


def default_python() -> str:
    """The interpreter used by the limit wrapper and task subprocesses."""

    return sys.executable
