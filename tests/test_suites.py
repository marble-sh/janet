from __future__ import annotations

from pathlib import Path

import pytest

from janet.bench.materialize import materialize
from janet.config import Config, ConfigError, LoopConfig, PathsConfig
from janet.eval.evolve import load_evolved, save_evolved
from janet.eval.task import Case
from janet.suites import load_dev_suite, load_fresh, load_suites


def _config(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            runs=tmp_path / "runs",
            archive=tmp_path / "archive",
            archive_db=tmp_path / "archive.sqlite3",
            events=tmp_path / "events.jsonl",
            anchor=tmp_path / "anchor",
            fresh=tmp_path / "fresh",
            evolved=tmp_path / "evolved",
            daemon=tmp_path / "daemon",
            bandit=tmp_path / "runs" / "bandit.json",
        ),
        loop=LoopConfig(min_difficulty=3),
    )


def test_load_dev_suite_applies_evolved_cases(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dev = load_dev_suite(config)
    assert dev.raw
    assert dev.tasks == dev.raw
    assert dev.fingerprint == ""
    assert dev.root == config.paths.runs / "dev-suite"

    target = dev.raw[0]
    cases = (Case(args=(0,), expect=0),)
    save_evolved(config.paths.evolved, target.id, cases)
    loaded = load_dev_suite(config)
    updated = next(task for task in loaded.tasks if task.id == target.id)
    assert updated.hidden_cases == cases
    assert loaded.fingerprint

    raw = load_dev_suite(config, include_evolved=False)
    assert load_evolved(config.paths.evolved)[target.id] == cases
    assert next(task for task in raw.raw if task.id == target.id).hidden_cases == ()
    assert raw.fingerprint == ""


def test_load_suites_verifies_the_hidden_anchor(tmp_path: Path) -> None:
    config = _config(tmp_path)
    materialize(config.paths.anchor, "anchor", force=True)
    suites = load_suites(config)
    assert suites.dev
    assert suites.anchor
    assert suites.dev_suite == ""
    assert suites.anchor_root == config.paths.anchor


def test_load_suites_rejects_a_missing_anchor_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.paths.anchor.mkdir(parents=True)
    with pytest.raises(ConfigError, match="anchor manifest"):
        load_suites(config)


def test_load_fresh_verifies_the_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    materialize(config.paths.fresh, "fresh", force=True)
    tasks = load_fresh(config)
    assert tasks
    assert all(task.split == "fresh" for task in tasks)


def test_load_fresh_rejects_a_missing_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.paths.fresh.mkdir(parents=True)
    with pytest.raises(ConfigError, match="fresh manifest"):
        load_fresh(config)
