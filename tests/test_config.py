from __future__ import annotations

from pathlib import Path

import pytest

from janet.config import (
    BudgetConfig,
    ConfigError,
    LLMConfig,
    load_config,
)


def test_defaults_are_the_documented_contract() -> None:
    config = load_config()
    assert config.budget.cpu_fraction == 0.5
    assert config.budget.gpu_fraction == 0.5
    assert config.budget.spike_seconds_per_hour == 300.0
    assert config.llm.mode == "local"
    assert config.llm.allow_remote_task_data is False
    assert config.loop.promote is False


def test_loads_repository_default_file() -> None:
    config = load_config(Path("configs/default.toml"))
    assert config.llm.policy_model.startswith("qwen2.5-coder")
    assert config.llm.judge_model.startswith("llama3.2")
    assert config.paths.anchor == Path.home() / ".local/share/janet/anchor"


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[budget\ncpu_fraction = 0.5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "typo.toml"
    path.write_text("[budget]\ncpu_fractions = 0.5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid \\[budget\\]"):
        load_config(path)


def test_remote_mode_requires_explicit_task_data_permission(tmp_path: Path) -> None:
    path = tmp_path / "remote.toml"
    path.write_text(
        '[llm]\nmode = "remote"\nallow_remote_task_data = false\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="allow_remote_task_data"):
        load_config(path)


def test_paths_expand_user(tmp_path: Path) -> None:
    path = tmp_path / "paths.toml"
    path.write_text('[paths]\nanchor = "~/anchor"\n', encoding="utf-8")
    config = load_config(path)
    assert config.paths.anchor == Path.home() / "anchor"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpu_fraction", 0.0),
        ("mem_fraction", 1.5),
        ("disk_fraction", -0.1),
        ("gpu_fraction", 0.0),
        ("spike_seconds_per_hour", -1.0),
        ("max_cpu_cores", 0),
        ("max_mem_gib", 0.0),
        ("max_vram_gib", 0.0),
        ("reserve_mem_gib", -1.0),
        ("reserve_vram_gib", -1.0),
    ],
)
def test_invalid_budget_values_are_rejected(tmp_path: Path, field: str, value: float) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(f"[budget]\n{field} = {value}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_spike_fraction_must_exceed_soft_fraction() -> None:
    with pytest.raises(ConfigError, match="spike_cpu_fraction"):
        from janet.config import _validate_budget

        _validate_budget(BudgetConfig(cpu_fraction=0.5, spike_cpu_fraction=0.4))


def test_dev_margin_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "margin.toml"
    path.write_text("[loop]\ndev_margin = 0.5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="dev_margin"):
        load_config(path)
    assert load_config().loop.dev_margin == 0.03


def test_spike_gpu_must_cover_soft_gpu() -> None:
    with pytest.raises(ConfigError, match="spike_gpu_fraction"):
        from janet.config import _validate_budget

        _validate_budget(BudgetConfig(gpu_fraction=0.5, spike_gpu_fraction=0.4))


def test_llm_defaults_are_local_and_private() -> None:
    llm = LLMConfig()
    assert llm.mode == "local"
    assert llm.allow_remote_task_data is False
