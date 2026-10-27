from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from janet.cli import main
from janet.config import Config, PathsConfig
from janet.harness.archive import Archive
from janet.harness.bandit import BanditState
from janet.harness.spec import HarnessSpec
from janet.memory import ProcessMemory
from janet.report import build_meta_report, render_markdown, write_report

_TASK = "parse_duration"


def _config(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            runs=tmp_path / "runs",
            archive=tmp_path / "archive",
            archive_db=tmp_path / "archive" / "archive.sqlite3",
            bandit=tmp_path / "bandit.json",
            memory=tmp_path / "memory.sqlite3",
        )
    )


def _details(
    anchor: float,
    *,
    tokens: int = 1000,
    wall: float = 10.0,
    operator: str = "prompt",
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "anchor": {"summary": {"mean_score": anchor}},
        "dev": {
            "summary": {"total_eval_tokens": tokens, "total_wall_s": wall},
            "rows": rows or [],
        },
        "operator": operator,
    }


def _spec(name: str, letter: str, parent: str | None) -> HarnessSpec:
    return HarnessSpec(name=name, system_prompt=letter * 50, parent=parent)


def _fixture(tmp_path: Path) -> Config:
    config = _config(tmp_path)
    archive = Archive(config.paths.archive)
    try:
        seed = _spec("seed", "A", None)
        archive.ensure_seed(
            seed,
            dev_score=0.5,
            anchor_score=0.5,
            details=_details(0.5),
            coverage_cell="code_repair|3|0",
            cost=1000,
        )
        first = _spec("first", "B", seed.hash)
        archive.add(
            first,
            dev_score=0.6,
            anchor_score=0.6,
            details=_details(0.6),
            coverage_cell="code_repair|3|0",
            operator="prompt",
            cost=1000,
        )
        archive.accept(first.hash)
        second = _spec("second", "C", seed.hash)
        archive.add(
            second,
            dev_score=0.8,
            anchor_score=0.8,
            details=_details(
                0.8,
                rows=[
                    {"family": "integrity", "task_id": "missing_revenue", "repair_score": 1.0},
                    {"family": "integrity", "task_id": "bad_abstain", "repair_score": 0.0},
                ],
            ),
            coverage_cell="parsing|2|1",
            operator="skill_add",
            cost=1000,
        )
        archive.accept(second.hash)
        third = _spec("third", "D", seed.hash)
        archive.add(
            third,
            dev_score=0.9,
            anchor_score=1.0,
            details=_details(1.0, operator="budget"),
            coverage_cell="fresh|4|0",
            operator="budget",
            cost=1000,
        )
        archive.accept(third.hash)
        archive.record_confirmation(
            second.hash,
            {"variant_score": 0.9, "delta": 0.4, "release": {"decision": "accept"}},
        )
    finally:
        archive.close()
    bandit = BanditState.fresh()
    bandit.record("prompt", 1.0, cost=100.0)
    bandit.record("skill_add", 0.5, cost=50.0)
    bandit.save(config.paths.bandit)
    with ProcessMemory(config.paths.memory) as memory:
        lesson = memory.add_lesson(
            task_id=_TASK,
            task_family="code_repair",
            failure_signature="invalid suffix",
            trigger_terms="duration suffix",
            rule="Validate the duration suffix before multiplying the numeric value.",
            patch_summary="Reject the suffix.",
            provenance={"run": "run-1"},
        )
        memory.set_status("lesson", lesson, "verified")
        memory.record_retrieval(
            "lesson",
            lesson,
            task_id=_TASK,
            task_family="code_repair",
            query="duration suffix",
            score=0.8,
            rank=0,
        )
    return config


def test_build_meta_report_from_archive_bandit_and_memory(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    report = build_meta_report(config)
    assert report.generations == 4
    anchors = [row["anchor"] for row in report.capability]
    assert anchors == [0.5, 0.6, 0.8, 1.0]
    rates = [row["per_1k_tokens"] for row in report.improvement_rate]
    assert rates == pytest.approx([0.1, 0.2, 0.2])
    assert report.meta_slope is not None and report.meta_slope > 0
    assert report.meta_slope_ci is not None
    assert report.transfer == 0.4
    assert report.diversity["frontier_cells"] == 3
    assert report.diversity["operators"] == {"prompt": 1, "skill_add": 1, "budget": 1}
    assert report.integrity["tasks"] == 2
    assert report.integrity["honest_abstention_rate"] == 0.5
    assert report.integrity["violations"] == 1
    assert report.resource["samples"] == 4
    assert report.resource["p95_wall_s"] == 10.0
    assert report.bandit["total_pulls"] == 2
    assert report.memory["available"] is True
    assert report.memory["stats"]["lessons"] == {"verified": 1}
    assert report.memory["retrieval"]["hit_rate"] == 1.0
    assert report.stale_probe["reframed"] is True
    payload = report.to_dict()
    assert json.loads(json.dumps(payload))["generations"] == 4


def test_markdown_and_files_are_written(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    report = build_meta_report(config)
    markdown = render_markdown(report)
    assert "# Janet meta report" in markdown
    assert "## Bandit" in markdown
    assert "skill_add" in markdown
    assert "## Stale-objective probe" in markdown
    json_path, markdown_path = write_report(report, tmp_path / "out")
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert json.loads(json_path.read_text(encoding="utf-8"))["generations"] == 4


def test_report_handles_an_empty_archive(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = build_meta_report(config)
    assert report.generations == 0
    assert report.capability == ()
    assert report.meta_slope is None
    assert report.transfer is None
    assert report.memory["available"] is False
    assert "## Capability" in render_markdown(report)


def test_report_command_writes_json_and_markdown(tmp_path: Path, capsys: Any) -> None:
    config = _fixture(tmp_path)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[paths]\n"
        f"runs = {json.dumps(str(config.paths.runs))}\n"
        f"archive = {json.dumps(str(config.paths.archive))}\n"
        f"archive_db = {json.dumps(str(config.paths.archive_db))}\n"
        f"bandit = {json.dumps(str(config.paths.bandit))}\n"
        f"memory = {json.dumps(str(config.paths.memory))}\n",
        encoding="utf-8",
    )
    out = tmp_path / "report-out"
    assert (
        main(["report", "--meta", "--config", str(config_path), "--json", "--out", str(out)]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["generations"] == 4
    assert (out / "report.json").is_file()
    assert (out / "report.md").is_file()
