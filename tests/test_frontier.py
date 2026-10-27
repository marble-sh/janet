from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from janet.harness.archive import Archive, ArchiveError
from janet.harness.spec import HarnessSpec


def _spec(name: str, prompt: str, *, parent: str | None) -> HarnessSpec:
    return HarnessSpec(name=name, system_prompt=prompt, parent=parent)


def _seed(archive: Archive) -> str:
    seed = _spec(
        "seed",
        "Validate inputs before computing and report every failure clearly.",
        parent=None,
    )
    archive.ensure_seed(seed, dev_score=0.5, anchor_score=0.5, coverage_cell="base|2|0")
    return seed.hash


def _variant(
    archive: Archive,
    seed_hash: str,
    *,
    name: str,
    prompt: str,
    cell: str,
    dev: float,
    anchor: float,
    accept: bool = True,
) -> str:
    spec = _spec(name, prompt, parent=seed_hash)
    variant = archive.add(
        spec,
        dev_score=dev,
        anchor_score=anchor,
        coverage_cell=cell,
        operator="prompt",
        cost=100.0,
    )
    if accept:
        archive.accept(variant)
    return variant


def test_one_elite_per_occupied_cell(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    try:
        seed = _seed(archive)
        same_cell = "code_repair|3|0"
        weak = _variant(
            archive, seed, name="weak", prompt="B" * 50, cell=same_cell, dev=0.6, anchor=0.5
        )
        strong = _variant(
            archive, seed, name="strong", prompt="C" * 50, cell=same_cell, dev=0.7, anchor=0.8
        )
        other = _variant(
            archive, seed, name="other", prompt="D" * 50, cell="parsing|2|1", dev=0.6, anchor=0.6
        )
        elites = {record.hash for record in archive.frontier()}
        assert strong in elites
        assert other in elites
        assert weak not in elites
        assert archive.get(weak).frontier_status == "member"  # type: ignore[union-attr]
        assert archive.get(strong).frontier_status == "elite"  # type: ignore[union-attr]
    finally:
        archive.close()


def test_rejected_variants_never_become_elites(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    try:
        seed = _seed(archive)
        rejected = _variant(
            archive,
            seed,
            name="rejected",
            prompt="E" * 50,
            cell="fresh|4|2",
            dev=0.1,
            anchor=0.1,
            accept=False,
        )
        assert archive.get(rejected).frontier_status == "none"  # type: ignore[union-attr]
        assert rejected not in {record.hash for record in archive.frontier()}
    finally:
        archive.close()


def test_selection_covers_exploit_and_explore(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    try:
        seed = _seed(archive)
        for index, cell in enumerate(("a|3|0", "b|3|0", "c|3|0")):
            _variant(
                archive,
                seed,
                name=f"v{index}",
                prompt=chr(ord("F") + index) * 50,
                cell=cell,
                dev=0.5 + index * 0.1,
                anchor=0.5 + index * 0.1,
            )
        elite_hashes = {record.hash for record in archive.frontier()}
        assert len(elite_hashes) == 4  # the seed occupies its own cell
        assert len(elite_hashes) == len(archive.frontier())
        chosen = [archive.select_parent(generation).hash for generation in range(1, 13)]  # type: ignore[union-attr]
        assert set(chosen) <= elite_hashes
        assert len(set(chosen)) >= 2
        top = max(archive.frontier(), key=lambda record: record.anchor_score or 0.0)
        assert top.hash in chosen
        assert archive.select_parent(1) == archive.select_parent(1)
    finally:
        archive.close()


def test_novelty_is_distance_to_the_closest_variant(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    try:
        seed = _seed(archive)
        duplicate_tokens = _variant(
            archive,
            seed,
            name="same-tokens",
            prompt="Validate, inputs before computing and report every failure clearly.",
            cell="x|3|0",
            dev=0.5,
            anchor=0.5,
        )
        fresh = _variant(
            archive,
            seed,
            name="different",
            prompt="Zebra quantum xylophone plays loudly under violet moons.",
            cell="y|3|0",
            dev=0.5,
            anchor=0.5,
        )
        assert archive.get(duplicate_tokens).novelty == pytest.approx(0.0)  # type: ignore[union-attr]
        assert archive.get(fresh).novelty == pytest.approx(1.0)  # type: ignore[union-attr]
    finally:
        archive.close()


def test_record_carries_operator_cost_and_cell(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    try:
        seed = _seed(archive)
        variant = _variant(
            archive, seed, name="v", prompt="Q" * 50, cell="code_repair|3|1", dev=0.5, anchor=0.5
        )
        record = archive.get(variant)
        assert record is not None
        assert record.operator == "prompt"
        assert record.cost == pytest.approx(100.0)
        assert record.coverage_cell == "code_repair|3|1"
        payload = record.to_dict()
        assert payload["operator"] == "prompt"
        assert payload["frontier_status"] == "elite"
    finally:
        archive.close()


def test_migration_adds_frontier_columns(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    connection = sqlite3.connect(root / "archive.sqlite3")
    connection.execute(
        """
        CREATE TABLE variants (
            hash TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            parent_hash TEXT,
            spec_json TEXT NOT NULL,
            dev_score REAL,
            anchor_score REAL,
            status TEXT NOT NULL DEFAULT 'candidate',
            created_at TEXT NOT NULL,
            details_json TEXT
        )
        """
    )
    connection.commit()
    connection.close()
    archive = Archive(root)
    try:
        seed = _seed(archive)
        variant = _variant(
            archive, seed, name="v", prompt="R" * 50, cell="code_repair|3|0", dev=0.5, anchor=0.5
        )
        assert archive.get(variant).coverage_cell == "code_repair|3|0"  # type: ignore[union-attr]
    finally:
        archive.close()


def test_invalid_selection_options_are_rejected(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    try:
        _seed(archive)
        with pytest.raises(ArchiveError, match="explore_ratio"):
            archive.select_parent(1, explore_ratio=1.5)
        with pytest.raises(ArchiveError, match="cost"):
            archive.add(
                _spec("bad", "S" * 50, parent="x"), dev_score=0.0, anchor_score=0.0, cost=-1.0
            )
    finally:
        archive.close()
