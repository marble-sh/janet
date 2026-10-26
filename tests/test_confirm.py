from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from janet.bench.materialize import materialize
from janet.cli import main
from janet.config import Config
from janet.eval.task import Task, load_suite
from janet.governor import ResourceGovernor
from janet.harness.archive import Archive, ArchiveError
from janet.harness.spec import SEED_HARNESS, HarnessSpec
from janet.llm import LLMClient, LLMResponse
from janet.loop.confirm import confirm_variant
from janet.probe import Budget

MARKER = "VARIANT-MARKER"


class MarkerClient:
    """Returns the reference only when the system prompt carries the marker."""

    def __init__(self, references: dict[str, str]) -> None:
        self.references = references

    def generate(self, prompt: str, *, system: str | None = None, **_kw: Any) -> LLMResponse:
        text = "def broken(:\n"
        if system and MARKER in system:
            for entry, reference in self.references.items():
                if f"{entry}(" in prompt:
                    text = reference
                    break
        return LLMResponse(
            text=text,
            model="fake",
            eval_count=1,
            prompt_eval_count=1,
            eval_duration_ns=1,
            total_duration_ns=1,
        )


def _fresh_tasks(tmp_path: Path) -> tuple[Task, ...]:
    materialize(tmp_path / "fresh", "fresh", force=True)
    return load_suite(tmp_path / "fresh", "fresh")


def _references(tasks: tuple[Task, ...]) -> dict[str, str]:
    return {
        str(task.entrypoint): (task.source_dir / "reference.py").read_text(encoding="utf-8")
        for task in tasks
    }


def test_confirm_measures_variant_against_seed(tmp_path: Path, budget: Budget) -> None:
    tasks = _fresh_tasks(tmp_path)[:2]
    archive = Archive(tmp_path / "archive")
    try:
        archive.ensure_seed(SEED_HARNESS, dev_score=0.5, anchor_score=0.5)
        variant = HarnessSpec(
            name="v",
            system_prompt=f"{MARKER} " + SEED_HARNESS.system_prompt,
            parent=SEED_HARNESS.hash,
        )
        archive.add(variant, dev_score=0.9, anchor_score=0.9)
        result = confirm_variant(
            archive,
            variant.hash,
            tasks,
            cast(LLMClient, MarkerClient(_references(tasks))),
            ResourceGovernor(budget),
            Config(),
            tmp_path / "run",
        )
        assert result.tasks == 2
        assert result.seed_score == 0.0
        assert result.variant_score == 1.0
        assert result.delta == 1.0
        assert (result.seed_passed, result.variant_passed) == (0, 2)
        payload = result.to_dict()
        assert payload["delta"] == 1.0
        assert len(cast(list[Any], payload["per_task"])) == 2
    finally:
        archive.close()


def test_confirm_rejects_bad_inputs(tmp_path: Path, budget: Budget) -> None:
    tasks = _fresh_tasks(tmp_path)[:1]
    archive = Archive(tmp_path / "archive")
    client = cast(LLMClient, MarkerClient({}))
    governor = ResourceGovernor(budget)
    try:
        with pytest.raises(ArchiveError, match="fresh suite is empty"):
            confirm_variant(archive, "x", (), client, governor, Config(), tmp_path / "r")
        with pytest.raises(ArchiveError, match="unknown variant"):
            confirm_variant(archive, "x", tasks, client, governor, Config(), tmp_path / "r")
        archive.add(SEED_HARNESS, dev_score=0.5, anchor_score=0.5)
        archive.seed()
        with pytest.raises(ArchiveError, match="unknown variant"):
            confirm_variant(archive, "nope", tasks, client, governor, Config(), tmp_path / "r")
    finally:
        archive.close()


def test_cli_confirm_requires_a_materialized_fresh_suite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[paths]\n"
        f"archive = {json.dumps(str(tmp_path / 'archive'))}\n"
        f"fresh = {json.dumps(str(tmp_path / 'missing-fresh'))}\n",
        encoding="utf-8",
    )
    assert main(["archive", "confirm", "abc", "--config", str(config)]) == 2
    assert "fresh manifest check failed" in capsys.readouterr().err


def test_cli_lineage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f"[paths]\narchive = {json.dumps(str(tmp_path / 'archive'))}\n", encoding="utf-8"
    )
    archive = Archive(tmp_path / "archive")
    child = HarnessSpec(name="child", system_prompt="B" * 50, parent=SEED_HARNESS.hash)
    archive.add(SEED_HARNESS, dev_score=0.5, anchor_score=0.5)
    archive.add(child, dev_score=0.6, anchor_score=0.6)
    archive.close()
    assert main(["archive", "lineage", child.hash, "--config", str(config)]) == 0
    chain = json.loads(capsys.readouterr().out)
    assert [item["hash"] for item in chain] == [SEED_HARNESS.hash, child.hash]
    assert main(["archive", "lineage", "nope", "--config", str(config)]) == 2
