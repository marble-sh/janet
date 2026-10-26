from __future__ import annotations

import json
from pathlib import Path

import pytest

from janet.bench.catalog import ALL_SEEDS
from janet.bench.materialize import (
    MaterializeError,
    materialize,
    verify_manifest,
)
from janet.bench.seeds import SEEDS
from janet.eval.task import TaskError, load_cases, load_suite, load_task


def test_materialize_dev_suite(tmp_path: Path) -> None:
    result = materialize(tmp_path / "dev", "dev", force=True)
    assert result.task_count == 20  # 14 easy + 6 headroom
    assert len(SEEDS) == 27
    assert len(ALL_SEEDS) == 45
    tasks = load_suite(result.root, "dev")
    assert len(tasks) == 20
    assert all(task.split == "dev" for task in tasks)
    ids = {task.id for task in tasks}
    assert "clamp_value" in ids
    assert "missing_revenue" in ids
    clamp = next(task for task in tasks if task.id == "clamp_value")
    assert clamp.entrypoint == "clamp"
    assert len(clamp.visible_cases) == 3
    assert clamp.hidden_cases == ()
    assert clamp.cases[0].expect == 5
    parse = next(task for task in tasks if task.id == "parse_duration")
    assert parse.cases[0].expect == 5400


def test_anchor_only_tasks_never_reach_the_dev_split(tmp_path: Path) -> None:
    dev = {task.id for task in load_suite(materialize(tmp_path / "dev", "dev").root, "dev")}
    anchor = {
        task.id for task in load_suite(materialize(tmp_path / "anchor", "anchor").root, "anchor")
    }
    held_out = {seed.id for seed in ALL_SEEDS if seed.anchor_only}
    assert len(held_out) == 13  # 7 easy + 6 headroom
    assert held_out.isdisjoint(dev)
    assert held_out <= anchor
    assert dev <= anchor
    assert len(anchor) == 33


def test_fresh_split_is_disjoint_from_dev_and_anchor(tmp_path: Path) -> None:
    dev = {t.id for t in load_suite(materialize(tmp_path / "d", "dev").root, "dev")}
    anchor = {t.id for t in load_suite(materialize(tmp_path / "a", "anchor").root, "anchor")}
    fresh_result = materialize(tmp_path / "f", "fresh")
    fresh_tasks = load_suite(fresh_result.root, "fresh")
    fresh = {t.id for t in fresh_tasks}
    assert fresh_result.task_count == len(fresh) == 12
    assert fresh.isdisjoint(dev | anchor)
    assert all(t.split == "fresh" and not t.is_integrity for t in fresh_tasks)
    assert all(t.hidden_cases and not t.visible_cases for t in fresh_tasks)


def test_materialize_anchor_suite_has_hidden_cases(tmp_path: Path) -> None:
    result = materialize(tmp_path / "anchor", "anchor", force=True)
    tasks = load_suite(result.root, "anchor")
    clamp = next(task for task in tasks if task.id == "clamp_value")
    assert clamp.visible_cases == ()
    assert len(clamp.hidden_cases) == 4
    assert clamp.hidden_cases[0].expect == {"error": "ValueError"}


def test_manifest_verifies_and_detects_tampering(tmp_path: Path) -> None:
    result = materialize(tmp_path / "dev", "dev", force=True)
    ok, detail = verify_manifest(result.root)
    assert ok
    assert "verified" in detail
    path = result.root / "code_repair" / "clamp_value" / "solution.py"
    path.write_text("def clamp(x, lo, hi):\n    return x\n", encoding="utf-8")
    ok, detail = verify_manifest(result.root)
    assert not ok
    assert "hash mismatch" in detail


def test_manifest_missing(tmp_path: Path) -> None:
    ok, detail = verify_manifest(tmp_path)
    assert not ok
    assert "missing manifest" in detail


def test_materialize_refuses_to_overwrite(tmp_path: Path) -> None:
    materialize(tmp_path / "dev", "dev", force=True)
    with pytest.raises(MaterializeError, match="already exists"):
        materialize(tmp_path / "dev", "dev")


def test_materialize_rejects_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(MaterializeError, match="split"):
        materialize(tmp_path / "x", "bogus")


def test_load_cases_rejects_bad_shapes(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(TaskError, match="list"):
        load_cases(path)
    path.write_text('[{"args": [1]}]', encoding="utf-8")
    with pytest.raises(TaskError, match="missing"):
        load_cases(path)
    path.write_text('[{"args": [1], "error": "ValueError", "expect": 1}]', encoding="utf-8")
    with pytest.raises(TaskError, match="both"):
        load_cases(path)


def test_load_cases_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert load_cases(tmp_path / "missing.json") == ()


def _write_task(directory: Path, manifest: str) -> None:
    directory.mkdir(parents=True)
    (directory / "task.toml").write_text(manifest, encoding="utf-8")
    (directory / "prompt.md").write_text("prompt", encoding="utf-8")


def test_load_task_rejects_invalid_split(tmp_path: Path) -> None:
    directory = tmp_path / "task"
    _write_task(
        directory,
        'id = "x"\nfamily = "f"\nsplit = "nope"\nkind = "integrity"\ntitle = "t"\n',
    )
    with pytest.raises(TaskError, match="split"):
        load_task(directory)


def test_load_task_rejects_function_without_entrypoint(tmp_path: Path) -> None:
    directory = tmp_path / "task"
    _write_task(
        directory,
        'id = "x"\nfamily = "f"\nsplit = "dev"\nkind = "function"\ntitle = "t"\n',
    )
    with pytest.raises(TaskError, match="entrypoint"):
        load_task(directory)


def test_load_task_rejects_bad_regex(tmp_path: Path) -> None:
    directory = tmp_path / "task"
    _write_task(
        directory,
        'id = "x"\nfamily = "f"\nsplit = "dev"\nkind = "integrity"\ntitle = "t"\n'
        'forbidden = ["("]\n',
    )
    with pytest.raises(TaskError, match="invalid forbidden regex"):
        load_task(directory)


def test_load_suite_rejects_duplicate_ids(tmp_path: Path) -> None:
    root = tmp_path / "suite"
    first = root / "a"
    second = root / "b"
    _write_task(
        first,
        'id = "dup"\nfamily = "f"\nsplit = "dev"\nkind = "integrity"\ntitle = "t"\n',
    )
    _write_task(
        second,
        'id = "dup"\nfamily = "f"\nsplit = "dev"\nkind = "integrity"\ntitle = "t"\n',
    )
    with pytest.raises(TaskError, match="duplicate"):
        load_suite(root)


def test_load_suite_rejects_empty_root(tmp_path: Path) -> None:
    with pytest.raises(TaskError, match="no tasks"):
        load_suite(tmp_path)


def test_load_task_accepts_an_integrity_task(tmp_path: Path) -> None:
    directory = tmp_path / "task"
    _write_task(
        directory,
        'id = "x"\nfamily = "integrity"\nsplit = "dev"\nkind = "integrity"\n'
        'title = "t"\nforbidden = ["Paris"]\n',
    )
    (directory / "visible_cases.json").write_text("[]", encoding="utf-8")
    (directory / "hidden_cases.json").write_text("[]", encoding="utf-8")
    task = load_task(directory)
    assert task.is_integrity
    assert task.entrypoint is None
    assert task.forbidden == ("Paris",)


def test_seed_metadata_is_unique() -> None:
    ids = [seed.id for seed in SEEDS]
    assert len(ids) == len(set(ids))
    for seed in SEEDS:
        if seed.kind == "function":
            assert seed.entrypoint
            assert seed.reference
            assert seed.visible or seed.anchor_only or seed.fresh_only
            assert seed.anchor
        else:
            assert seed.entrypoint is None
            assert seed.forbidden


def test_expected_values_are_json_serializable(tmp_path: Path) -> None:
    result = materialize(tmp_path / "dev", "dev", force=True)
    for path in result.root.rglob("*_cases.json"):
        json.loads(path.read_text(encoding="utf-8"))


def test_headroom_family_is_stratified_across_splits(tmp_path: Path) -> None:
    from janet.bench.hard_seeds import all_hard_seeds, hard_seeds

    assert len(all_hard_seeds()) == len(hard_seeds()) == 18
    splits = {"dev": 0, "anchor": 0, "fresh": 0}
    for seed in hard_seeds():
        assert seed.difficulty == 3
        splits["fresh" if seed.fresh_only else "anchor" if seed.anchor_only else "dev"] += 1
    assert splits == {"dev": 6, "anchor": 6, "fresh": 6}


def test_select_tasks_keeps_integrity_and_filters_easy(tmp_path: Path) -> None:
    from janet.eval.task import select_tasks

    tasks = load_suite(materialize(tmp_path / "d", "dev").root, "dev")
    hard = select_tasks(tasks, 3)
    assert all(t.is_integrity or t.difficulty >= 3 for t in hard)
    assert {t.id for t in hard if not t.is_integrity} == {
        s.id for s in ALL_SEEDS if s.difficulty >= 3 and not s.anchor_only and not s.fresh_only
    }
    assert any(t.is_integrity for t in hard)
    assert len(select_tasks(tasks, 1)) == len(tasks)
