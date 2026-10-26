"""Load the working evaluation suites from a configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from janet.bench.materialize import materialize, verify_manifest
from janet.config import Config, ConfigError
from janet.eval.evolve import apply_evolved, suite_fingerprint
from janet.eval.task import Task, load_suite, select_tasks


@dataclass(frozen=True, slots=True)
class DevSuite:
    tasks: tuple[Task, ...]  # with evolved cases applied as hidden cases
    raw: tuple[Task, ...]  # the seed tasks, before evolved cases
    fingerprint: str  # fingerprint of the evolved evaluator ('' = unevolved)
    root: Path


@dataclass(frozen=True, slots=True)
class Suites:
    dev: tuple[Task, ...]
    anchor: tuple[Task, ...]
    dev_suite: str
    anchor_root: Path


def load_dev_suite(config: Config, *, include_evolved: bool = True) -> DevSuite:
    """Dev tasks, optionally with the evolved evaluator applied.

    The dev suite is rematerialized from the seed catalog on every call, so a stale
    on-disk copy can never be scored; evolved cases are layered on top as hidden
    cases and identified by ``fingerprint`` so measurements stay comparable.

    ``include_evolved=False`` is for the evaluator's own evolution pass: that code
    takes the existing cases explicitly, so applying them here too would duplicate
    every input.
    """

    floor = config.loop.min_difficulty
    dev_root = config.paths.runs.expanduser() / "dev-suite"
    materialize(dev_root, "dev", force=True)
    evolved_dir = config.paths.evolved.expanduser()
    raw = select_tasks(load_suite(dev_root, "dev"), floor)
    tasks = apply_evolved(raw, evolved_dir) if include_evolved else raw
    return DevSuite(
        tasks=tasks,
        raw=raw,
        fingerprint=suite_fingerprint(evolved_dir) if include_evolved else "",
        root=dev_root,
    )


def load_fresh(config: Config) -> tuple[Task, ...]:
    """The never-selected fresh split, after verifying its manifest."""

    ok, detail = verify_manifest(config.paths.fresh)
    if not ok:
        raise ConfigError(
            f"fresh manifest check failed: {detail}. Materialize it with "
            "`janet bench materialize --root ~/.local/share/janet/fresh "
            "--split fresh --force`."
        )
    return select_tasks(load_suite(config.paths.fresh, "fresh"), config.loop.min_difficulty)


def load_suites(config: Config) -> Suites:
    """Dev (with evolved cases applied) and the verified hidden anchor."""

    dev = load_dev_suite(config)
    ok, detail = verify_manifest(config.paths.anchor)
    if not ok:
        raise ConfigError(
            f"anchor manifest check failed: {detail}. Materialize it with "
            "`janet bench materialize --root ~/.local/share/janet/anchor "
            "--split anchor --force`."
        )
    anchor = select_tasks(load_suite(config.paths.anchor, "anchor"), config.loop.min_difficulty)
    return Suites(
        dev=dev.tasks,
        anchor=anchor,
        dev_suite=dev.fingerprint,
        anchor_root=config.paths.anchor,
    )
