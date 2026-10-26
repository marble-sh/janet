"""Tests for the host-side publisher. No network, no real credentials."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import janet.publish as publish_module
from janet.github_app import AppConfig, Installation, Transport
from janet.publish import (
    PublishError,
    PublishPlan,
    _is_trusted_entry_point,
    create_squash_commit,
    default_body,
    is_protected,
    plan_publish,
    publish,
    push_branch,
    scan_secrets,
)


def _run(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-b", "main")
    _run(repo, "config", "user.name", "Test")
    _run(repo, "config", "user.email", "test@example.com")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _run(repo, "add", "README.md")
    _run(repo, "commit", "-m", "base")
    _run(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _run(repo, "checkout", "-b", "feature")
    (repo / "feature.py").write_text("x = 1\n", encoding="utf-8")
    _run(repo, "add", "feature.py")
    _run(repo, "commit", "-m", "feature")
    return repo


def _app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        repo="marble-sh/janet",
        app_id=5480729,
        installation_id=175567250,
        private_key_path=tmp_path / "key.pem",
    )


def _fake_installation() -> Installation:
    return Installation(
        token="ghs_test",
        expires_at="2026-10-26T19:47:27Z",
        app_id=5480729,
        app_slug="janet-could-you-please",
        app_name="janet-could-you-please",
        bot_user_id=348334669,
        installation_id=175567250,
        permissions={
            "contents": "write",
            "pull_requests": "write",
            "metadata": "read",
        },
    )


def test_plan_publishes_only_the_feature_diff(tmp_path: Path) -> None:
    plan = plan_publish(_repo(tmp_path), repo="marble-sh/janet")
    assert plan.ok
    assert plan.source_branch == "feature"
    assert plan.files == ("feature.py",)
    assert plan.branch.startswith("janet/auto-")
    assert plan.protected == ()
    assert plan.secret_hits == ()


def test_plan_rejects_a_dirty_tree(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "dirty.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(PublishError, match="dirty"):
        plan_publish(repo, repo="marble-sh/janet")


def test_plan_rejects_publishing_from_main(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _run(repo, "checkout", "main")
    with pytest.raises(PublishError, match="feature branch"):
        plan_publish(repo, repo="marble-sh/janet")


def test_plan_rejects_no_changes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _run(repo, "checkout", "-b", "empty", "origin/main")
    with pytest.raises(PublishError, match="no changes"):
        plan_publish(repo, repo="marble-sh/janet")


def test_plan_rejects_a_branch_not_based_on_origin_main(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _run(repo, "checkout", "main")
    (repo / "main2.txt").write_text("2\n", encoding="utf-8")
    _run(repo, "add", "main2.txt")
    _run(repo, "commit", "-m", "main2")
    _run(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _run(repo, "checkout", "feature")
    with pytest.raises(PublishError, match="not an ancestor"):
        plan_publish(repo, repo="marble-sh/janet")


def test_plan_detects_a_protected_path(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    path = repo / "src" / "janet" / "governor.py"
    path.parent.mkdir(parents=True)
    path.write_text("# change\n", encoding="utf-8")
    _run(repo, "add", "src/janet/governor.py")
    _run(repo, "commit", "-m", "touch governor")
    plan = plan_publish(repo, repo="marble-sh/janet")
    assert plan.protected == ("src/janet/governor.py",)
    assert not plan.ok


def test_plan_detects_a_secret(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "leak.txt").write_text("ghp_" + "a" * 30 + "\n", encoding="utf-8")
    _run(repo, "add", "leak.txt")
    _run(repo, "commit", "-m", "leak")
    plan = plan_publish(repo, repo="marble-sh/janet")
    assert plan.secret_hits == (("leak.txt", "github-token"),)
    assert not plan.ok


def test_is_protected() -> None:
    assert is_protected(".github/workflows/ci.yml")
    assert is_protected("tests/test_publish.py")
    assert is_protected("src/janet/governor.py")
    assert is_protected("configs/publish.toml")
    assert not is_protected("src/janet/harness/spec.py")
    assert not is_protected("README.md")


def test_scan_secrets(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "benign.txt").write_text("hello\n", encoding="utf-8")
    assert scan_secrets(repo, ("benign.txt",)) == ()
    marker = "-----BEGIN RSA " + "PRIVATE KEY-----\n"
    (repo / "leak.txt").write_text(marker, encoding="utf-8")
    assert scan_secrets(repo, ("leak.txt",)) == (("leak.txt", "private-key"),)


def test_plan_handles_unusual_filenames(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    relative = "tests/evil\tname.py"
    path = repo / relative
    path.parent.mkdir()
    path.write_text("x = 1\n", encoding="utf-8")
    _run(repo, "add", relative)
    _run(repo, "commit", "-m", "quoted name")
    plan = plan_publish(repo, repo="marble-sh/janet")
    assert relative in plan.files
    assert relative in plan.protected


def test_trusted_entry_point(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _is_trusted_entry_point(tmp_path / "venv" / "janet" / "publish.py", repo)
    assert not _is_trusted_entry_point(repo / "src" / "janet" / "publish.py", repo)


def test_create_squash_commit_has_one_bot_authored_parent(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    plan = plan_publish(repo, repo="marble-sh/janet")
    email = "348334669+janet-could-you-please[bot]@users.noreply.github.com"
    commit = create_squash_commit(
        plan,
        author_name="Janet",
        author_email=email,
        title="Test publish",
        body="Body",
    )
    assert _run(repo, "rev-parse", f"refs/heads/{plan.branch}") == commit
    assert _run(repo, "show", "-s", "--format=%an <%ae>", commit) == f"Janet <{email}>"
    assert _run(repo, "rev-parse", f"{commit}^") == plan.base_sha
    assert "feature.py" in _run(repo, "diff", "--name-only", f"{plan.base_sha}..{commit}")


def test_publish_dry_run_does_not_authenticate(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = publish(repo, _app_config(tmp_path), dry_run=True)
    assert result.ok
    assert result.dry_run
    assert result.commit is None
    assert result.pr_url is None
    assert result.installation is None
    assert result.plan.files == ("feature.py",)


def test_publish_rejects_protected_paths(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    path = repo / "src" / "janet" / "governor.py"
    path.parent.mkdir(parents=True)
    path.write_text("# change\n", encoding="utf-8")
    _run(repo, "add", "src/janet/governor.py")
    _run(repo, "commit", "-m", "touch governor")
    with pytest.raises(PublishError, match="protected paths"):
        publish(repo, _app_config(tmp_path))


def _mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str, str]]]:
    installation = _fake_installation()
    pushes: list[tuple[str, str, str]] = []
    prs: list[tuple[str, str, str, str]] = []

    def fake_authenticate(config: AppConfig, *, transport: Transport | None = None) -> Installation:
        return installation

    def fake_push(plan: PublishPlan, commit: str, token: str) -> None:
        pushes.append((plan.branch, commit, token))

    def fake_pr(
        plan: PublishPlan,
        token: str,
        *,
        title: str,
        body: str,
        transport: Transport | None = None,
    ) -> str:
        prs.append((plan.branch, token, title, body))
        return "https://github.com/marble-sh/janet/pull/1"

    monkeypatch.setattr(publish_module, "authenticate", fake_authenticate)
    monkeypatch.setattr(publish_module, "push_branch", fake_push)
    monkeypatch.setattr(publish_module, "open_pull_request", fake_pr)
    return pushes, prs


def test_publish_opens_a_pr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path)
    pushes, prs = _mocks(monkeypatch)
    result = publish(repo, _app_config(tmp_path))
    assert result.ok
    assert result.pr_url == "https://github.com/marble-sh/janet/pull/1"
    assert result.installation == _fake_installation()
    assert pushes == [(result.plan.branch, result.commit, "ghs_test")]
    assert prs[0][0] == result.plan.branch
    assert prs[0][1] == "ghs_test"
    assert prs[0][3].startswith("Automated publication by Janet.")


def test_push_branch_isolates_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = plan_publish(_repo(tmp_path), repo="marble-sh/janet")
    calls: list[tuple[tuple[str, ...], dict[str, str] | None]] = []

    def fake_git(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
        calls.append((args, env))
        return ""

    monkeypatch.setattr(publish_module, "_git", fake_git)
    push_branch(plan, "abc123", "ghs_test")
    args, env = calls[0]
    assert "credential.helper=" in args
    assert "core.hooksPath=/dev/null" in args
    assert env is not None
    assert env["GH_TOKEN"] == "ghs_test"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_ASKPASS"] == "/bin/false"
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_CONFIG_SYSTEM"] == "/dev/null"


def test_default_body_marks_the_human_gate(tmp_path: Path) -> None:
    plan = plan_publish(_repo(tmp_path), repo="marble-sh/janet")
    body = default_body(plan)
    assert "Automated publication by Janet." in body
    assert "feature" in body
    assert "human must review and merge" in body
