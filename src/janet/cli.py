"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from janet import __version__
from janet.bench.baseline import run_baseline
from janet.bench.catalog import ALL_SEEDS
from janet.bench.materialize import materialize, verify_manifest
from janet.bench.repair_bench import run_repair_benchmark
from janet.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from janet.daemon import (
    DaemonError,
    daemon_start,
    daemon_status,
    daemon_stop,
    format_status,
    run_foreground,
    run_once,
)
from janet.eval.evolve import apply_evolved
from janet.eval.judge import build_samples, calibrate
from janet.eval.repair import RepairAttempt
from janet.eval.task import TaskError, load_suite, select_tasks
from janet.github_app import GitHubAppError, load_app_config
from janet.governor import GovernorError, ResourceGovernor
from janet.harness.archive import Archive, ArchiveError
from janet.harness.bandit import BanditState, format_bandit
from janet.llm import LLMClient, LLMError
from janet.loop.confirm import confirm_variant
from janet.loop.evolution import evolve_dev_suite
from janet.loop.generation import LoopError, run_loop
from janet.preempt import Preempted, file_stop
from janet.probe import (
    GIB,
    Budget,
    MachineProbe,
    ResourceError,
    compute_budget,
    probe_machine,
)
from janet.publish import PublishError, publish
from janet.sandbox import Sandbox, SandboxError
from janet.suites import load_dev_suite, load_fresh, load_suites


def _load(path: str | None) -> Config:
    if path is not None:
        return load_config(Path(path))
    default = DEFAULT_CONFIG_PATH
    return load_config(default if default.is_file() else None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="janet", description=__doc__)
    parser.add_argument("--version", action="version", version=f"janet {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="show machine state and the resolved budget")
    probe.add_argument("--config", default=None)
    probe.add_argument("--json", action="store_true", dest="as_json")

    doctor = subparsers.add_parser("doctor", help="check tools, sandbox, models, and budget")
    doctor.add_argument("--config", default=None)
    doctor.add_argument("--json", action="store_true", dest="as_json")

    run = subparsers.add_parser("run", help="run the improvement loop")
    run.add_argument("--config", default=None)
    run.add_argument("--generations", type=int, default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--dry-run", action="store_true")

    evolve = subparsers.add_parser(
        "evolve", help="grow the dev evaluator with mutation-killing test inputs"
    )
    evolve.add_argument("--config", default=None)
    evolve.add_argument("--limit", type=int, default=None, help="max tasks this pass")
    evolve.add_argument("--only", default=None, help="comma-separated task ids")
    evolve.add_argument("--max-new", type=int, default=3, help="new cases per task")
    evolve.add_argument("--run-id", default=None)
    evolve.add_argument("--stop-file", default=None, help="yield when this file appears")

    calibrate = subparsers.add_parser(
        "calibrate", help="measure the advisory judge against the program verifier"
    )
    calibrate.add_argument("--config", default=None)
    calibrate.add_argument("--split", choices=["dev", "anchor", "fresh"], default="dev")
    calibrate.add_argument("--limit", type=int, default=6, help="max tasks")
    calibrate.add_argument("--mutants", type=int, default=2, help="mutants per task")
    calibrate.add_argument("--run-id", default=None)

    bandit = subparsers.add_parser("bandit", help="show the operator-bandit state")
    bandit.add_argument("--config", default=None)
    bandit.add_argument("--json", action="store_true", dest="as_json")

    daemon = subparsers.add_parser("daemon", help="idle-only background self-improvement")
    daemon_sub = daemon.add_subparsers(dest="daemon_command", required=True)
    daemon_start_cmd = daemon_sub.add_parser("start", help="start the background daemon")
    daemon_start_cmd.add_argument("--config", default=None)
    daemon_stop_cmd = daemon_sub.add_parser("stop", help="stop the background daemon")
    daemon_stop_cmd.add_argument("--config", default=None)
    daemon_status_cmd = daemon_sub.add_parser("status", help="show daemon state")
    daemon_status_cmd.add_argument("--config", default=None)
    daemon_status_cmd.add_argument("--json", action="store_true", dest="as_json")
    daemon_run_cmd = daemon_sub.add_parser("run", help="run the daemon in the foreground")
    daemon_run_cmd.add_argument("--config", default=None)
    daemon_run_cmd.add_argument("--session-seconds", type=float, default=None)
    daemon_run_cmd.add_argument("--max-sessions", type=int, default=None)
    daemon_once_cmd = daemon_sub.add_parser(
        "once", help="run one ladder session now, regardless of idle state"
    )
    daemon_once_cmd.add_argument("--config", default=None)
    daemon_once_cmd.add_argument("--session-seconds", type=float, default=None)

    archive = subparsers.add_parser("archive", help="inspect and promote harness variants")
    archive_sub = archive.add_subparsers(dest="archive_command", required=True)
    archive_list = archive_sub.add_parser("list", help="list archived variants")
    archive_list.add_argument("--config", default=None)
    archive_list.add_argument("--limit", type=int, default=50)
    archive_show = archive_sub.add_parser("show", help="show a variant")
    archive_show.add_argument("hash")
    archive_show.add_argument("--config", default=None)
    archive_promote = archive_sub.add_parser("promote", help="promote a variant")
    archive_promote.add_argument("hash")
    archive_promote.add_argument("--config", default=None)
    archive_rollback = archive_sub.add_parser("rollback", help="print the last promoted harness")
    archive_rollback.add_argument("--config", default=None)
    archive_confirm = archive_sub.add_parser(
        "confirm", help="compare a variant with the seed on the never-selected fresh split"
    )
    archive_confirm.add_argument("hash")
    archive_confirm.add_argument("--config", default=None)
    archive_lineage = archive_sub.add_parser("lineage", help="root-to-variant chain")
    archive_lineage.add_argument("hash")
    archive_lineage.add_argument("--config", default=None)

    bench = subparsers.add_parser("bench", help="materialize and run the seed benchmark")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)
    bench_materialize = bench_sub.add_parser(
        "materialize", help="write the dev or hidden anchor suite"
    )
    bench_materialize.add_argument("--root", required=True)
    bench_materialize.add_argument("--split", choices=["dev", "anchor", "fresh"], required=True)
    bench_materialize.add_argument("--force", action="store_true")
    bench_baseline = bench_sub.add_parser("baseline", help="one-shot baseline on a suite")
    bench_baseline.add_argument("--config", default=None)
    bench_baseline.add_argument("--split", choices=["dev", "anchor", "fresh"], default="dev")
    bench_baseline.add_argument(
        "--min-difficulty", type=int, default=None, help="default: loop.min_difficulty"
    )
    bench_baseline.add_argument("--limit", type=int, default=None)
    bench_baseline.add_argument("--only", default=None, help="comma-separated task ids")
    bench_baseline.add_argument("--root", default=None)
    bench_baseline.add_argument("--run-id", default=None)
    bench_repair = bench_sub.add_parser(
        "repair", help="bounded repair loop compared with the one-shot baseline"
    )
    bench_repair.add_argument("--config", default=None)
    bench_repair.add_argument("--limit", type=int, default=None)
    bench_repair.add_argument("--only", default=None, help="comma-separated task ids")
    bench_repair.add_argument("--run-id", default=None)
    bench_repair.add_argument("--attempts", type=int, default=3)
    bench_repair.add_argument("--samples", type=int, default=2)
    bench_repair.add_argument("--no-compare", action="store_true")
    bench_sub.add_parser("tasks", help="list the seed tasks")

    publish_cmd = subparsers.add_parser(
        "publish",
        help="publish committed work as a bot-authored pull request (trusted install)",
    )
    publish_cmd.add_argument(
        "--config", default=None, help="publish config (default: configs/publish.toml)"
    )
    publish_cmd.add_argument("--root", default=None, help="repository root (default: cwd)")
    publish_cmd.add_argument("--title", default=None)
    publish_cmd.add_argument("--body", default=None)
    publish_cmd.add_argument("--dry-run", action="store_true")

    return parser


def _budget(config: Config) -> tuple[MachineProbe, Budget]:
    machine = probe_machine()
    return machine, compute_budget(machine, config.budget)


def _print_probe(machine: MachineProbe, budget: Budget) -> None:
    print(f"CPU: {machine.cpu.allowed_cores} allowed / {machine.cpu.total_cores} total cores")
    print(
        "RAM: "
        f"{machine.mem.available_bytes / GIB:.2f} GiB available / "
        f"{machine.mem.total_bytes / GIB:.2f} GiB total"
    )
    if machine.gpus:
        for gpu in machine.gpus:
            print(
                f"GPU {gpu.index} ({gpu.vendor}): {gpu.name}, "
                f"{gpu.free_bytes / GIB:.2f}/{gpu.total_bytes / GIB:.2f} GiB free"
            )
    else:
        print("GPU: none detected")
    print(f"Disk free: {machine.disk_free_bytes / GIB:.2f} GiB")
    print(f"Load average: {machine.load_average[0]:.2f}")
    print()
    print(f"Budget CPU: {budget.cpu_cores} cores (spike to {budget.cpu_cores_spike})")
    print(
        f"Budget RAM: {budget.mem_bytes / GIB:.2f} GiB "
        f"(spike to {budget.mem_bytes_spike / GIB:.2f} GiB)"
    )
    if budget.vram_bytes:
        print(
            f"Budget VRAM: {budget.vram_bytes / GIB:.2f} GiB "
            f"(spike to {budget.vram_bytes_spike / GIB:.2f} GiB)"
        )
    else:
        print("Budget VRAM: none (CPU-only model)")
    print(f"Budget disk: {budget.disk_bytes / GIB:.2f} GiB")
    print(f"Spike allowance: {budget.spike_seconds_per_hour:.0f} s per rolling hour")
    for note in budget.notes:
        print(f"note: {note}")


def _probe_payload(machine: MachineProbe, budget: Budget) -> dict[str, object]:
    return {"machine": machine.to_dict(), "budget": budget.to_dict()}


def _model_names(base_url: str) -> list[str]:
    url = base_url.rstrip("/") + "/api/tags"
    with urllib.request.urlopen(url, timeout=3) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("models", [])
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for model in models:
        if isinstance(model, dict):
            name = model.get("name") or model.get("model")
            if isinstance(name, str):
                names.append(name)
    return names


def _check_models(config: Config) -> tuple[bool, str]:
    try:
        names = _model_names(config.llm.base_url)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, f"model server unreachable at {config.llm.base_url}: {exc}"
    missing = [
        model
        for model in (config.llm.policy_model, config.llm.judge_model)
        if not any(name == model or name.startswith(model + ":") for name in names)
    ]
    if missing:
        return False, "missing models: " + ", ".join(missing)
    return True, f"{len(names)} model(s) available"


def _check_sandbox(config: Config, budget: Budget) -> tuple[bool, str]:
    governor = ResourceGovernor(budget)
    with tempfile.TemporaryDirectory(prefix="janet-doctor-") as directory:
        try:
            sandbox = Sandbox(Path(directory), governor)
            result = sandbox.preflight()
        except SandboxError as exc:
            return False, str(exc)
    if result.ok and "network-denied" in result.stdout:
        return True, "bubblewrap jail active; network denied"
    return False, f"sandbox preflight failed: {result.stderr.strip() or result.returncode}"


def _cmd_probe(args: argparse.Namespace) -> int:
    config = _load(args.config)
    machine, budget = _budget(config)
    if args.as_json:
        print(json.dumps(_probe_payload(machine, budget), indent=2))
    else:
        _print_probe(machine, budget)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    config = _load(args.config)
    machine, budget = _budget(config)
    checks: list[dict[str, object]] = []

    def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "required": required})

    add("python", sys.version_info >= (3, 12), sys.version.split()[0])
    for tool in ("uv", "git", "bwrap"):
        path = shutil.which(tool)
        add(tool, path is not None, path or "not found")
    for tool in ("ollama", "llama-server", "systemd-run", "tessera", "rocm-smi"):
        path = shutil.which(tool)
        add(tool, path is not None, path or "not found", required=False)
    add("disk", budget.disk_bytes >= 5 * GIB, f"{budget.disk_bytes / GIB:.2f} GiB available")
    if machine.gpus:
        largest = max(machine.gpus, key=lambda gpu: gpu.total_bytes)
        add(
            "gpu",
            budget.vram_bytes > 0,
            f"{largest.vendor} {largest.name}; VRAM budget {budget.vram_bytes / GIB:.2f} GiB",
        )
    else:
        add("gpu", True, "none detected (CPU-only model)", required=False)
    ok, detail = _check_sandbox(config, budget)
    add("sandbox", ok, detail)
    ok, detail = _check_models(config)
    add("models", ok, detail, required=False)

    failed_required = [check for check in checks if check["required"] and not check["ok"]]
    if args.as_json:
        print(json.dumps({"checks": checks, "ok": not failed_required}, indent=2))
    else:
        for check in checks:
            status = "PASS" if check["ok"] else ("FAIL" if check["required"] else "WARN")
            print(f"{status:4} {check['name']:12} {check['detail']}")
        if not failed_required:
            print("\nAll required checks passed.")
    return 1 if failed_required else 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = _load(args.config)
    machine, budget = _budget(config)
    generations = args.generations if args.generations is not None else config.loop.max_generations
    if args.dry_run:
        payload: dict[str, Any] = {
            "dry_run": True,
            "generations": generations,
            "config": asdict(config),
            "budget": budget.to_dict(),
            "machine": machine.to_dict(),
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if generations < 1:
        raise ConfigError("--generations must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ConfigError("--limit must be >= 1")
    suites = load_suites(config)
    client = _local_client(config)
    governor = ResourceGovernor(budget)
    results = run_loop(
        client,
        governor,
        config,
        suites.dev,
        suites.anchor,
        generations=generations,
        limit=args.limit,
        anchor_root=suites.anchor_root,
        dev_suite=suites.dev_suite,
    )
    print(json.dumps([result.to_dict() for result in results], indent=2))
    return 0


def _cmd_archive(args: argparse.Namespace) -> int:
    config = _load(args.config)
    archive = Archive(config.paths.archive.expanduser())
    try:
        if args.archive_command == "list":
            records = archive.list(limit=args.limit)
            print(json.dumps([record.to_dict() for record in records], indent=2))
            return 0
        if args.archive_command == "show":
            record = archive.get(args.hash)
            if record is None:
                raise ArchiveError(f"unknown variant: {args.hash}")
            print(json.dumps(record.to_dict(), indent=2))
            print(record.spec.to_json())
            return 0
        if args.archive_command == "promote":
            record = archive.promote(args.hash)
            print(json.dumps(record.to_dict(), indent=2))
            return 0
        if args.archive_command == "lineage":
            chain = archive.lineage(args.hash)
            if not chain:
                raise ArchiveError(f"unknown variant: {args.hash}")
            print(json.dumps([record.to_dict() for record in chain], indent=2))
            return 0
        if args.archive_command == "confirm":
            ok, detail = verify_manifest(config.paths.fresh)
            if not ok:
                raise ConfigError(
                    f"fresh manifest check failed: {detail}. Materialize it with "
                    "`janet bench materialize --root ~/.local/share/janet/fresh "
                    "--split fresh --force`."
                )
            _machine, budget = _budget(config)
            run_dir = config.paths.runs.expanduser() / _run_id("confirm", None)
            confirmation = confirm_variant(
                archive,
                args.hash,
                select_tasks(load_suite(config.paths.fresh, "fresh"), config.loop.min_difficulty),
                _local_client(config),
                ResourceGovernor(budget),
                config,
                run_dir,
            )
            print(json.dumps(confirmation.to_dict(), indent=2))
            return 0
        if args.archive_command == "rollback":
            spec = archive.rollback()
            if spec is None:
                raise ArchiveError("no promoted harness to roll back to")
            print(spec.to_json())
            return 0
        return 2
    finally:
        archive.close()


def _local_client(config: Config, *, require: Sequence[str] = ()) -> LLMClient:
    client = LLMClient(config.llm)
    try:
        models = client.list_models()
    except LLMError as exc:
        raise ConfigError(f"local model server is unavailable: {exc}") from exc
    required = (config.llm.policy_model, *require)
    missing = [
        model
        for model in required
        if not any(name == model or name.startswith(model + ":") for name in models)
    ]
    if missing:
        raise ConfigError(
            f"model(s) {', '.join(missing)!r} are not available on {config.llm.base_url}"
        )
    return client


def _cmd_evolve(args: argparse.Namespace) -> int:
    config = _load(args.config)
    if args.limit is not None and args.limit < 1:
        raise ConfigError("--limit must be >= 1")
    if args.max_new < 1:
        raise ConfigError("--max-new must be >= 1")
    dev = load_dev_suite(config, include_evolved=False)
    tasks = dev.raw
    if args.only:
        wanted = {part.strip() for part in args.only.split(",") if part.strip()}
        tasks = tuple(task for task in tasks if task.id in wanted)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise ConfigError("no dev tasks selected for evolution")
    run_dir = config.paths.runs.expanduser() / _run_id("evolve", args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    client = _local_client(config)
    _machine, budget = _budget(config)
    governor = ResourceGovernor(budget)
    stop = file_stop(Path(args.stop_file)) if args.stop_file else None

    def progress(result: object) -> None:
        task_id = getattr(result, "task_id", "?")
        accepted = len(getattr(result, "accepted", ()))
        print(
            f"[evolve] {task_id}: +{accepted} case(s)",
            file=sys.stderr,
            flush=True,
        )

    summary = evolve_dev_suite(
        tasks,
        config,
        client,
        governor,
        run_dir,
        max_new_per_task=args.max_new,
        stop=stop,
        progress=progress,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    print(f"evolved cases: {config.paths.evolved.expanduser()}", file=sys.stderr)
    print(f"report: {run_dir / 'evolution.json'}", file=sys.stderr)
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    config = _load(args.config)
    if args.limit < 1:
        raise ConfigError("--limit must be >= 1")
    if args.mutants < 1:
        raise ConfigError("--mutants must be >= 1")
    if args.split == "dev":
        tasks = load_dev_suite(config).tasks
    elif args.split == "anchor":
        ok, detail = verify_manifest(config.paths.anchor)
        if not ok:
            raise ConfigError(f"anchor manifest check failed: {detail}")
        tasks = select_tasks(load_suite(config.paths.anchor, "anchor"), config.loop.min_difficulty)
    else:
        tasks = load_fresh(config)
    tasks = tuple(task for task in tasks if not task.is_integrity)[: args.limit]
    if not tasks:
        raise ConfigError("no function tasks selected for calibration")
    run_dir = config.paths.runs.expanduser() / _run_id("calibrate", args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    client = _local_client(config, require=[config.llm.judge_model])
    _machine, budget = _budget(config)
    governor = ResourceGovernor(budget)
    samples = build_samples(tasks, governor, run_dir / "samples", mutants_per_task=args.mutants)
    calibration = calibrate(samples, tasks, client, config.llm.judge_model)
    payload = {
        "split": args.split,
        "judge_model": config.llm.judge_model,
        "calibration": calibration.to_dict(),
        "samples": [
            {"task": sample.task_id, "origin": sample.origin, "verifier": sample.label}
            for sample in samples
        ],
    }
    (run_dir / "calibration.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if not calibration.trustworthy:
        print(
            "warning: the judge is not trustworthy on this set; it remains advisory only",
            file=sys.stderr,
        )
    print(f"report: {run_dir / 'calibration.json'}", file=sys.stderr)
    return 0


def _cmd_bandit(args: argparse.Namespace) -> int:
    config = _load(args.config)
    state = BanditState.load(config.paths.bandit.expanduser())
    if args.as_json:
        print(json.dumps(state.to_dict(), indent=2))
    else:
        print(format_bandit(state))
    return 0


def _cmd_daemon(args: argparse.Namespace) -> int:
    config = _load(args.config)
    if args.daemon_command == "run":
        sessions = run_foreground(
            config,
            session_seconds=args.session_seconds,
            max_sessions=args.max_sessions,
        )
        print(json.dumps({"sessions": sessions}, indent=2))
        return 0
    if args.daemon_command == "once":
        report = run_once(config, session_seconds=args.session_seconds)
        print(json.dumps(report.to_dict(), indent=2))
        return 0
    if args.daemon_command == "start":
        print(json.dumps(daemon_start(config, args.config), indent=2))
        return 0
    if args.daemon_command == "stop":
        print(json.dumps(daemon_stop(config), indent=2))
        return 0
    payload = daemon_status(config)
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_status(payload))
    return 0 if payload["running"] else 3


def _run_id(prefix: str, requested: str | None) -> str:
    if requested:
        raw = requested
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        raw = f"{prefix}-{stamp}"
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in raw)


def _cmd_bench_repair(args: argparse.Namespace) -> int:
    config = _load(args.config)
    if args.limit is not None and args.limit < 1:
        raise ConfigError("--limit must be >= 1")
    if args.attempts < 1:
        raise ConfigError("--attempts must be >= 1")
    if args.samples < 1:
        raise ConfigError("--samples must be >= 1")
    run_dir = config.paths.runs.expanduser() / _run_id("repair", args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    root = run_dir / "suite"
    materialize(root, "dev", force=True)
    tasks = apply_evolved(load_suite(root, "dev"), config.paths.evolved.expanduser())
    client = _local_client(config)
    _machine, budget = _budget(config)
    governor = ResourceGovernor(budget)

    def progress(task_id: str, attempt: RepairAttempt) -> None:
        print(
            f"[repair] {task_id} attempt {attempt.attempt}: "
            f"score={attempt.score:.2f} {'PASS' if attempt.passed else 'FAIL'}",
            file=sys.stderr,
            flush=True,
        )

    report = run_repair_benchmark(
        tasks,
        config,
        governor,
        run_dir,
        client=client,
        limit=args.limit,
        only={part.strip() for part in args.only.split(",")} if args.only else None,
        max_attempts=args.attempts,
        samples_per_attempt=args.samples,
        compare=not args.no_compare,
        progress=progress,
    )
    print(json.dumps(report.summary.to_dict(), indent=2))
    print(f"report: {run_dir / 'repair.json'}", file=sys.stderr)
    print(f"markdown: {run_dir / 'repair.md'}", file=sys.stderr)
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    if args.bench_command == "tasks":
        for seed in ALL_SEEDS:
            print(f"{seed.family:12} {seed.id:28} {seed.title}")
        return 0
    if args.bench_command == "materialize":
        result = materialize(Path(args.root), args.split, force=args.force)
        ok, detail = verify_manifest(result.root)
        payload = {**result.to_dict(), "manifest_ok": ok, "manifest_detail": detail}
        print(json.dumps(payload, indent=2))
        return 0 if ok else 1
    if args.bench_command == "repair":
        return _cmd_bench_repair(args)
    if args.bench_command != "baseline":
        return 2

    config = _load(args.config)
    if args.limit is not None and args.limit < 1:
        raise ConfigError("--limit must be >= 1")
    run_dir = config.paths.runs.expanduser() / _run_id("baseline", args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.split in ("anchor", "fresh"):
        default_root = config.paths.anchor if args.split == "anchor" else config.paths.fresh
        root = Path(args.root).expanduser() if args.root else default_root
        ok, detail = verify_manifest(root)
        if not ok:
            raise ConfigError(f"{args.split} manifest check failed: {detail}")
        tasks = load_suite(root, args.split)
    else:
        root = Path(args.root).expanduser() if args.root else run_dir / "suite"
        materialize(root, "dev", force=True)
        tasks = apply_evolved(load_suite(root, "dev"), config.paths.evolved.expanduser())
    floor = args.min_difficulty if args.min_difficulty is not None else config.loop.min_difficulty
    tasks = select_tasks(tasks, floor)

    client = _local_client(config)

    _machine, budget = _budget(config)
    governor = ResourceGovernor(budget)

    def progress(row: object) -> None:
        task_id = getattr(row, "task_id", "?")
        score = getattr(row, "score", 0.0)
        passed = getattr(row, "passed", False)
        print(
            f"[bench] {task_id}: score={score:.2f} {'PASS' if passed else 'FAIL'}",
            file=sys.stderr,
            flush=True,
        )

    report = run_baseline(
        tasks,
        config,
        governor,
        run_dir,
        client=client,
        limit=args.limit,
        only={part.strip() for part in args.only.split(",")} if args.only else None,
        progress=progress,
    )
    print(json.dumps(report.summary.to_dict(), indent=2))
    print(f"report: {run_dir / 'baseline.json'}", file=sys.stderr)
    print(f"markdown: {run_dir / 'baseline.md'}", file=sys.stderr)
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    app_config = load_app_config(Path(args.config) if args.config else None)
    root = Path(args.root).expanduser() if args.root else Path.cwd()

    def progress(message: str) -> None:
        print(f"[publish] {message}", file=sys.stderr, flush=True)

    result = publish(
        root,
        app_config,
        title=args.title,
        body=args.body,
        dry_run=args.dry_run,
        progress=progress,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "probe":
            return _cmd_probe(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "evolve":
            return _cmd_evolve(args)
        if args.command == "calibrate":
            return _cmd_calibrate(args)
        if args.command == "bandit":
            return _cmd_bandit(args)
        if args.command == "daemon":
            return _cmd_daemon(args)
        if args.command == "archive":
            return _cmd_archive(args)
        if args.command == "bench":
            return _cmd_bench(args)
        if args.command == "publish":
            return _cmd_publish(args)
        parser.error(f"unknown command: {args.command}")
    except Preempted:
        print("preempted", file=sys.stderr)
        return 0
    except (
        ConfigError,
        ResourceError,
        GovernorError,
        SandboxError,
        TaskError,
        LLMError,
        ArchiveError,
        LoopError,
        DaemonError,
        GitHubAppError,
        PublishError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
