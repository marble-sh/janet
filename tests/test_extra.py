from __future__ import annotations

import sys
from pathlib import Path

import pytest

from janet._limit import _parser
from janet._limit import main as limit_main
from janet.cli import (
    _check_models,
    _check_sandbox,
    main,
)
from janet.config import Config, ConfigError, load_config
from janet.events import EventLog
from janet.governor import (
    ResourceGovernor,
    RunResult,
    SpikeBudget,
    _clean_env,
    _read_proc_stat,
    _truncate,
    format_bytes,
    tree_stats,
)
from janet.probe import GIB, Budget, MemInfo, probe_machine
from janet.sandbox import Sandbox, SandboxError

# ---------- CLI ----------


def test_probe_human_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["probe"]) == 0
    out = capsys.readouterr().out
    assert "CPU:" in out
    assert "Budget RAM:" in out
    assert "Spike allowance:" in out


def test_check_models_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("janet.cli._model_names", lambda _base: ["other:latest"])
    ok, detail = _check_models(Config())
    assert not ok
    assert "missing models" in detail


def test_check_models_reports_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(_base: str) -> list[str]:
        raise OSError("down")

    monkeypatch.setattr("janet.cli._model_names", explode)
    ok, detail = _check_models(Config())
    assert not ok
    assert "unreachable" in detail


def test_check_sandbox_reports_failure(monkeypatch: pytest.MonkeyPatch, budget: Budget) -> None:
    def explode(*_args: object, **_kwargs: object) -> Sandbox:
        raise SandboxError("no jail")

    monkeypatch.setattr("janet.cli.Sandbox", explode)
    ok, detail = _check_sandbox(Config(), budget)
    assert not ok
    assert "no jail" in detail


def test_main_handles_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupt(_args: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr("janet.cli._cmd_probe", interrupt)
    assert main(["probe"]) == 130
    assert "interrupted" in capsys.readouterr().err


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["bogus"])
    assert excinfo.value.code == 2


# ---------- Config ----------


@pytest.mark.parametrize("section", ["budget", "paths", "llm", "loop"])
def test_sections_must_be_tables(tmp_path: Path, section: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(f'{section} = "oops"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=f"\\[{section}\\] must be a table"):
        load_config(path)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ('[llm]\nmode = "telepathy"\n', "llm.mode"),
        ("[llm]\ntemperature = -1.0\n", "temperature"),
        ("[llm]\nmax_tokens = 0\n", "max_tokens"),
        ("[llm]\ntimeout_seconds = 0\n", "timeout_seconds"),
        ("[loop]\nmax_generations = 0\n", "max_generations"),
        ("[loop]\nmax_attempts = 0\n", "max_attempts"),
        ("[loop]\nmutation_max_chars = 0\n", "mutation_max_chars"),
        ("[loop]\nbandit_exploration = -1.0\n", "bandit_exploration"),
    ],
)
def test_invalid_llm_and_loop_values(tmp_path: Path, body: str, match: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_paths_bandit_is_expanded(tmp_path: Path) -> None:
    path = tmp_path / "partial.toml"
    path.write_text('[paths]\nbandit = "~/janet-bandit.json"\n', encoding="utf-8")
    config = load_config(path)
    assert config.paths.bandit == Path.home() / "janet-bandit.json"


def test_max_cpu_cores_must_cover_minimum() -> None:
    from janet.config import BudgetConfig, _validate_budget

    with pytest.raises(ConfigError, match="max_cpu_cores"):
        _validate_budget(BudgetConfig(min_cpu_cores=4, max_cpu_cores=2))


def test_spike_memory_must_cover_soft_memory() -> None:
    from janet.config import BudgetConfig, _validate_budget

    with pytest.raises(ConfigError, match="spike_mem_fraction"):
        _validate_budget(BudgetConfig(mem_fraction=0.5, spike_mem_fraction=0.4))


# ---------- Governor ----------


def test_spike_budget_reset() -> None:
    spikes = SpikeBudget(allowance_seconds=10.0)
    spikes.record(5.0, 0.0)
    spikes.reset()
    assert spikes.used(0.0) == 0.0


def test_should_yield_on_low_memory(monkeypatch: pytest.MonkeyPatch, budget: Budget) -> None:
    monkeypatch.setattr(
        "janet.governor.parse_meminfo",
        lambda _text: MemInfo(total_bytes=10 * GIB, available_bytes=0),
    )
    assert ResourceGovernor(budget).should_yield() is True


def test_should_yield_on_high_load(monkeypatch: pytest.MonkeyPatch, budget: Budget) -> None:
    monkeypatch.setattr(
        "janet.governor.parse_meminfo",
        lambda _text: MemInfo(total_bytes=10 * GIB, available_bytes=10 * GIB),
    )
    monkeypatch.setattr("janet.governor.os.getloadavg", lambda: (100.0, 0.0, 0.0))
    assert ResourceGovernor(budget).should_yield() is True


def test_should_yield_when_idle(monkeypatch: pytest.MonkeyPatch, budget: Budget) -> None:
    monkeypatch.setattr(
        "janet.governor.parse_meminfo",
        lambda _text: MemInfo(total_bytes=10 * GIB, available_bytes=10 * GIB),
    )
    monkeypatch.setattr("janet.governor.os.getloadavg", lambda: (0.0, 0.0, 0.0))
    assert ResourceGovernor(budget).should_yield() is False


def test_truncate_marks_long_output() -> None:
    text = "x" * (1024 * 1024 + 10)
    assert _truncate(text).endswith("[truncated]")


def test_clean_env_is_minimal() -> None:
    env = _clean_env({"JANET_TEST": "1"})
    assert env["JANET_TEST"] == "1"
    assert "PATH" in env
    assert "PYTHONPATH" not in env


def test_format_bytes() -> None:
    assert format_bytes(2 * GIB) == "2.00 GiB"


def test_kill_tree_handles_missing_process(budget: Budget) -> None:
    governor = ResourceGovernor(budget)
    assert governor._kill_tree(999_999, "missing") == "missing"


def test_read_proc_stat_missing_pid() -> None:
    assert _read_proc_stat(999_999) == (-1, 0.0)


def test_tree_stats_missing_pid() -> None:
    stats = tree_stats(999_999)
    assert stats.rss_bytes == 0
    assert stats.cpu_seconds == 0.0


def test_governor_context_manager(monkeypatch: pytest.MonkeyPatch, budget: Budget) -> None:
    governor = ResourceGovernor(budget)
    monkeypatch.setattr(governor, "apply_to_self", lambda: None)
    with governor as entered:
        assert entered is governor


def test_governor_passes_stdin(budget: Budget) -> None:
    governor = ResourceGovernor(budget)
    result = governor.run(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"],
        timeout_s=10.0,
        stdin_text="hello from stdin\n",
    )
    assert result.ok
    assert result.stdout.strip() == "hello from stdin"


def test_governor_uses_default_timeout(budget: Budget) -> None:
    governor = ResourceGovernor(budget)
    result = governor.run([sys.executable, "-c", "print('ok')"])
    assert result.ok


def test_governor_reports_missing_executable(budget: Budget) -> None:
    governor = ResourceGovernor(budget)
    result = governor.run(["/nonexistent/janet-command"], timeout_s=1.0)
    assert not result.ok
    assert result.returncode == 127
    assert "cannot exec" in result.stderr


# ---------- Probe ----------


def test_probe_machine_on_this_host() -> None:
    probe = probe_machine(Path.cwd())
    assert probe.cpu.total_cores >= 1
    assert probe.cpu.allowed_cores >= 1
    assert probe.mem.total_bytes > 0
    assert "bwrap" in probe.tools
    assert probe.python_version


def test_probe_machine_missing_root(tmp_path: Path) -> None:
    probe = probe_machine(tmp_path / "missing")
    assert probe.disk_free_bytes == 0


# ---------- Events ----------


def test_read_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"ts": "x"}\n\n', encoding="utf-8")
    log = EventLog(path, run_id="run-0004")
    assert log.read() == [{"ts": "x"}]


# ---------- Sandbox ----------


def test_sandbox_passes_extra_env(sandbox: Sandbox) -> None:
    result = sandbox.run(
        ["/usr/bin/python3", "-c", "import os; print(os.environ['JANET_EXTRA'])"],
        env={"JANET_EXTRA": "visible"},
    )
    assert result.ok
    assert result.stdout.strip() == "visible"


def test_sandbox_reports_bwrap_setup_failure(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = RunResult(
        argv=(),
        returncode=1,
        stdout="",
        stderr="bwrap: cannot set up",
        duration_s=0.0,
        timed_out=False,
        killed_reason=None,
        peak_rss_bytes=0,
        cpu_seconds=0.0,
        spike_seconds=0.0,
    )
    monkeypatch.setattr(sandbox.governor, "run", lambda *_args, **_kwargs: fake)
    with pytest.raises(SandboxError, match="bwrap"):
        sandbox.run(["/usr/bin/python3", "-c", "pass"])


# ---------- Limit wrapper ----------


def test_limit_parser_accepts_a_command() -> None:
    args = _parser().parse_args(["--mem-bytes", "1024", "--", "/bin/true"])
    assert args.mem_bytes == 1024
    assert args.command == ["--", "/bin/true"]


def test_limit_requires_a_command() -> None:
    assert limit_main([]) == 2


def test_limit_rejects_bad_cpu_list() -> None:
    assert (
        limit_main(
            [
                "--cpus",
                "not-an-int",
                "--fsize-bytes",
                "0",
                "--nofile",
                "0",
                "--",
                "/bin/true",
            ]
        )
        == 125
    )


def test_limit_reports_exec_failure() -> None:
    assert (
        limit_main(
            [
                "--nice",
                "0",
                "--fsize-bytes",
                "0",
                "--nofile",
                "0",
                "--",
                "/nonexistent/janet-command",
            ]
        )
        == 127
    )
