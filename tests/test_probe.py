from __future__ import annotations

import pytest

from janet.config import BudgetConfig
from janet.probe import (
    GIB,
    MIB,
    Budget,
    CpuInfo,
    GpuInfo,
    MachineProbe,
    MemInfo,
    ResourceError,
    compute_budget,
    parse_meminfo,
    parse_nvidia_smi,
    parse_rocm_smi_json,
)


def make_probe(
    *,
    total_gib: float = 64.0,
    available_gib: float = 32.0,
    allowed_cores: int = 16,
    disk_gib: float = 100.0,
) -> MachineProbe:
    return MachineProbe(
        cpu=CpuInfo(total_cores=allowed_cores, allowed_cores=allowed_cores),
        mem=MemInfo(
            total_bytes=int(total_gib * GIB),
            available_bytes=int(available_gib * GIB),
        ),
        gpus=(),
        disk_free_bytes=int(disk_gib * GIB),
        load_average=(0.1, 0.1, 0.1),
        tools={},
        python_version="3.14.0",
        platform="Linux",
    )


def test_parse_meminfo() -> None:
    info = parse_meminfo(
        "MemTotal:       64000000 kB\nMemFree:         1000000 kB\nMemAvailable:   20000000 kB\n"
    )
    assert info.total_bytes == 64_000_000 * 1024
    assert info.available_bytes == 20_000_000 * 1024


def test_parse_meminfo_missing_key() -> None:
    with pytest.raises(ValueError, match="MemTotal"):
        parse_meminfo("MemFree: 1 kB\n")


def test_parse_nvidia_smi() -> None:
    gpus = parse_nvidia_smi("0, NVIDIA RTX 4090, 24564, 23000\n")
    assert len(gpus) == 1
    assert gpus[0].index == 0
    assert gpus[0].name == "NVIDIA RTX 4090"
    assert gpus[0].total_bytes == 24564 * MIB
    assert gpus[0].free_bytes == 23000 * MIB


def test_parse_nvidia_smi_skips_malformed_lines() -> None:
    gpus = parse_nvidia_smi("nonsense\n0, GPU, 100, 50\n")
    assert len(gpus) == 1
    assert gpus[0].vendor == "nvidia"


def test_parse_rocm_smi_json() -> None:
    payload = (
        '{"card0": {"VRAM Total Memory (B)": "8573157376", '
        '"VRAM Total Used Memory (B)": "1000000000", '
        '"Card Series": "AMD Radeon RX 6600"}, '
        '"card1": {"VRAM Total Memory (B)": "536870912", '
        '"VRAM Total Used Memory (B)": "16000000", '
        '"Card Series": "AMD Radeon 610M"}}'
    )
    gpus = parse_rocm_smi_json(payload)
    assert len(gpus) == 2
    assert gpus[0].vendor == "amd"
    assert gpus[0].total_bytes == 8_573_157_376
    assert gpus[0].free_bytes == 8_573_157_376 - 1_000_000_000
    assert gpus[1].index == 1


def test_parse_rocm_smi_json_rejects_bad_payload() -> None:
    assert parse_rocm_smi_json("not json") == ()
    assert parse_rocm_smi_json("[]") == ()
    assert parse_rocm_smi_json('{"card0": {"bad": "shape"}}') == ()


def test_cpu_budget_is_half_and_spike_is_three_quarters() -> None:
    config = BudgetConfig(cpu_fraction=0.5, max_cpu_cores=8, spike_cpu_fraction=0.75)
    budget = compute_budget(make_probe(allowed_cores=16), config)
    assert budget.cpu_cores == 8
    assert budget.cpu_cores_spike == 12


def test_memory_is_clamped_by_absolute_max() -> None:
    config = BudgetConfig(max_mem_gib=14.0, reserve_mem_gib=2.0)
    budget = compute_budget(make_probe(total_gib=64.0, available_gib=40.0), config)
    assert budget.mem_bytes == 14 * GIB
    assert any("max_mem_gib" in note for note in budget.notes)


def test_memory_is_clamped_by_available_minus_reserve() -> None:
    config = BudgetConfig(max_mem_gib=32.0, reserve_mem_gib=2.0)
    budget = compute_budget(make_probe(total_gib=64.0, available_gib=10.0), config)
    assert budget.mem_bytes == 8 * GIB
    assert any("available" in note for note in budget.notes)


def test_memory_spike_never_exceeds_available_cap() -> None:
    config = BudgetConfig(
        max_mem_gib=14.0,
        reserve_mem_gib=2.0,
        spike_mem_fraction=0.9,
    )
    budget = compute_budget(make_probe(total_gib=64.0, available_gib=12.0), config)
    assert budget.mem_bytes == 10 * GIB
    assert budget.mem_bytes_spike == 10 * GIB


def test_tight_machine_is_refused() -> None:
    config = BudgetConfig(reserve_mem_gib=2.0)
    with pytest.raises(ResourceError, match="refusing"):
        compute_budget(make_probe(available_gib=1.0), config)


def test_disk_budget_is_clamped() -> None:
    config = BudgetConfig(max_disk_gib=20.0)
    budget = compute_budget(make_probe(disk_gib=100.0), config)
    assert budget.disk_bytes == 20 * GIB


def test_vram_budget_is_half_of_total() -> None:
    probe = make_probe()
    probe = MachineProbe(
        cpu=probe.cpu,
        mem=probe.mem,
        gpus=(
            GpuInfo(
                index=0,
                name="AMD Radeon RX 6600",
                total_bytes=8_573_157_376,
                free_bytes=7_400_000_000,
                vendor="amd",
            ),
        ),
        disk_free_bytes=probe.disk_free_bytes,
        load_average=probe.load_average,
        tools=probe.tools,
        python_version=probe.python_version,
        platform=probe.platform,
    )
    config = BudgetConfig(
        gpu_fraction=0.5,
        max_vram_gib=8.0,
        reserve_vram_gib=0.5,
        spike_gpu_fraction=0.6,
    )
    budget = compute_budget(probe, config)
    assert budget.vram_bytes == 8_573_157_376 // 2
    assert budget.vram_bytes_spike >= budget.vram_bytes


def test_vram_is_zero_without_a_gpu() -> None:
    budget = compute_budget(make_probe(), BudgetConfig())
    assert budget.vram_bytes == 0
    assert budget.vram_bytes_spike == 0


def test_budget_serialization() -> None:
    budget = Budget(
        cpu_cores=2,
        cpu_cores_spike=3,
        mem_bytes=4 * GIB,
        mem_bytes_spike=5 * GIB,
        disk_bytes=6 * GIB,
        max_wall_seconds=10.0,
        spike_seconds_per_hour=300.0,
    )
    payload = budget.to_dict()
    assert payload["cpu_cores"] == 2
    assert payload["mem_gib"] == 4.0
    assert payload["vram_bytes"] == 0
    assert payload["notes"] == []


def test_gpu_serialization() -> None:
    probe = MachineProbe(
        cpu=CpuInfo(total_cores=4, allowed_cores=4),
        mem=MemInfo(total_bytes=8 * GIB, available_bytes=4 * GIB),
        gpus=(GpuInfo(index=0, name="GPU", total_bytes=8 * GIB, free_bytes=6 * GIB),),
        disk_free_bytes=100 * GIB,
        load_average=(0.0, 0.0, 0.0),
        tools={"bwrap": "/usr/bin/bwrap"},
        python_version="3.14.0",
        platform="Linux",
    )
    payload = probe.to_dict()
    gpus = payload["gpus"]
    assert isinstance(gpus, list)
    assert gpus[0]["name"] == "GPU"
    assert gpus[0]["vendor"] == "unknown"
