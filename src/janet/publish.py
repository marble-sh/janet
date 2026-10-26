"""Host-side publication of committed work as a bot-authored pull request.

This is the only code allowed to hold the GitHub App installation token. It
runs outside the sandbox, never writes the token to disk, and never pushes to
``main``. The human review and branch protection are the merge gate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from janet.github_app import (
    API_ROOT,
    AppConfig,
    GitHubAppError,
    Installation,
    Transport,
    api_request,
    authenticate,
    load_app_config,
)

PROTECTED_PREFIXES: tuple[str, ...] = (".github/", "tests/")
PROTECTED_FILES: frozenset[str] = frozenset(
    {
        "configs/default.toml",
        "configs/publish.toml",
        "pyproject.toml",
        "uv.lock",
        "src/janet/_limit.py",
        "src/janet/bench/baseline.py",
        "src/janet/bench/catalog.py",
        "src/janet/bench/hard_seeds.py",
        "src/janet/bench/materialize.py",
        "src/janet/bench/repair_bench.py",
        "src/janet/bench/seeds.py",
        "src/janet/cli.py",
        "src/janet/config.py",
        "src/janet/daemon.py",
        "src/janet/eval/audit.py",
        "src/janet/eval/evolve.py",
        "src/janet/eval/execute.py",
        "src/janet/eval/mutants.py",
        "src/janet/eval/runner.py",
        "src/janet/eval/task.py",
        "src/janet/eval/worker.py",
        "src/janet/github_app.py",
        "src/janet/governor.py",
        "src/janet/harness/archive.py",
        "src/janet/harness/guard.py",
        "src/janet/llm.py",
        "src/janet/loop/confirm.py",
        "src/janet/loop/decoupling.py",
        "src/janet/loop/evolution.py",
        "src/janet/loop/generation.py",
        "src/janet/loop/passive.py",
        "src/janet/preempt.py",
        "src/janet/probe.py",
        "src/janet/publish.py",
        "src/janet/sandbox.py",
        "src/janet/suites.py",
    }
)

_MAX_SCAN_BYTES = 4 * 1024 * 1024
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("github-pat", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("slack-token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("aws-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("openai-key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("google-key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    (
        "generic-secret",
        re.compile(
            r"(?i)(api[_-]?key|secret|password|access[_-]?token)"
            r"\s*[:=]\s*['\"][^'\"]{8,}['\"]"
        ),
    ),
)


class PublishError(RuntimeError):
    """Raised when a publication cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class PublishPlan:
    repo: str
    repo_root: Path
    source_branch: str
    source_sha: str
    base_branch: str
    base_sha: str
    branch: str
    files: tuple[str, ...]
    commit_subjects: tuple[str, ...]
    protected: tuple[str, ...]
    secret_hits: tuple[tuple[str, str], ...]

    @property
    def ok(self) -> bool:
        return not self.protected and not self.secret_hits

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "repo": self.repo,
            "repo_root": str(self.repo_root),
            "source_branch": self.source_branch,
            "source_sha": self.source_sha,
            "base_branch": self.base_branch,
            "base_sha": self.base_sha,
            "branch": self.branch,
            "files": list(self.files),
            "commit_subjects": list(self.commit_subjects),
            "protected": list(self.protected),
            "secret_hits": [{"path": path, "kind": kind} for path, kind in self.secret_hits],
        }


@dataclass(frozen=True, slots=True)
class PublishResult:
    plan: PublishPlan
    commit: str | None
    pr_url: str | None
    dry_run: bool
    installation: Installation | None = None

    @property
    def ok(self) -> bool:
        return self.plan.ok

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "dry_run": self.dry_run,
            "plan": self.plan.to_dict(),
            "commit": self.commit,
            "pr_url": self.pr_url,
            "app": None if self.installation is None else self.installation.public_dict(),
        }


def _git(
    repo_root: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise PublishError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _git_ok(repo_root: Path, *args: str) -> bool:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def _git_paths(repo_root: Path, *args: str) -> tuple[str, ...]:
    """Read NUL-delimited paths without Git's C-style quoting."""
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PublishError(f"git {' '.join(args)} failed: {detail}")
    return tuple(
        part.decode("utf-8", "surrogateescape") for part in completed.stdout.split(b"\0") if part
    )


def _is_trusted_entry_point(module_path: Path, repo_root: Path) -> bool:
    return not module_path.resolve().is_relative_to(repo_root.resolve())


def _assert_trusted_entry_point(repo_root: Path) -> None:
    if not _is_trusted_entry_point(Path(__file__), repo_root):
        raise PublishError(
            "run the publisher from a trusted installation (`janet-publish`), "
            "not from the candidate checkout"
        )


def is_protected(path: str) -> bool:
    normalized = PurePosixPath(path).as_posix()
    if normalized in PROTECTED_FILES:
        return True
    return any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in PROTECTED_PREFIXES
    )


def scan_secrets(
    repo_root: Path,
    files: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    hits: list[tuple[str, str]] = []
    for relative in files:
        path = repo_root / relative
        if path.is_symlink() or not path.is_file():
            continue
        if path.stat().st_size > _MAX_SCAN_BYTES:
            hits.append((relative, "file-too-large-to-scan"))
            continue
        text = path.read_bytes().decode("latin-1", errors="replace")
        for label, pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                hits.append((relative, label))
    return tuple(hits)


def plan_publish(
    repo_root: Path,
    *,
    repo: str,
    base_branch: str = "main",
    branch_prefix: str = "janet/auto-",
) -> PublishPlan:
    root = repo_root.expanduser().resolve()
    if not root.exists():
        raise PublishError(f"repository root does not exist: {root}")
    status = _git(root, "status", "--porcelain")
    if status:
        raise PublishError("working tree is dirty; commit or stash changes first")
    source_branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if source_branch == base_branch:
        raise PublishError(f"refusing to publish from {base_branch}; use a feature branch")
    source_sha = _git(root, "rev-parse", "HEAD")
    base_ref = f"origin/{base_branch}"
    if not _git_ok(root, "rev-parse", "--verify", base_ref):
        raise PublishError(f"base ref not found: {base_ref}; fetch origin first")
    base_sha = _git(root, "rev-parse", base_ref)
    if not _git_ok(root, "merge-base", "--is-ancestor", base_sha, "HEAD"):
        raise PublishError(f"{base_ref} is not an ancestor of HEAD; rebase on it first")
    files = _git_paths(root, "diff", "--no-renames", "--name-only", "-z", f"{base_ref}...HEAD")
    if not files:
        raise PublishError(f"no changes between {base_ref} and HEAD")
    subjects = tuple(
        line
        for line in _git(root, "log", "--format=%h %s", f"{base_ref}..HEAD").splitlines()
        if line
    )
    protected = tuple(path for path in files if is_protected(path))
    secret_hits = scan_secrets(root, files)
    return PublishPlan(
        repo=repo,
        repo_root=root,
        source_branch=source_branch,
        source_sha=source_sha,
        base_branch=base_branch,
        base_sha=base_sha,
        branch=f"{branch_prefix}{source_sha[:12]}",
        files=files,
        commit_subjects=subjects,
        protected=protected,
        secret_hits=secret_hits,
    )


def default_body(plan: PublishPlan) -> str:
    lines = [
        "Automated publication by Janet.",
        "",
        f"- Source branch: `{plan.source_branch}`",
        f"- Source commit: `{plan.source_sha}`",
        f"- Base: `{plan.base_branch}` at `{plan.base_sha[:12]}`",
        f"- Changed files: {len(plan.files)}",
        "",
        "### Source commits",
        "",
    ]
    lines.extend(f"- {subject}" for subject in plan.commit_subjects[:20])
    if len(plan.commit_subjects) > 20:
        lines.append(f"- ... and {len(plan.commit_subjects) - 20} more")
    lines.extend(
        [
            "",
            "This pull request was opened by the Janet GitHub App. "
            "A human must review and merge it.",
        ]
    )
    return "\n".join(lines)


def create_squash_commit(
    plan: PublishPlan,
    *,
    author_name: str,
    author_email: str,
    title: str,
    body: str,
) -> str:
    tree = _git(plan.repo_root, "rev-parse", f"{plan.source_sha}^{{tree}}")
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": author_name,
            "GIT_AUTHOR_EMAIL": author_email,
            "GIT_COMMITTER_NAME": author_name,
            "GIT_COMMITTER_EMAIL": author_email,
        }
    )
    commit = _git(
        plan.repo_root,
        "commit-tree",
        tree,
        "-p",
        plan.base_sha,
        "-m",
        title,
        "-m",
        body,
        env=env,
    )
    _git(plan.repo_root, "update-ref", f"refs/heads/{plan.branch}", commit)
    return commit


def push_branch(plan: PublishPlan, commit: str, token: str) -> None:
    helper = (
        '!f() { test "$1" = get && '
        "printf '%s\\n' 'username=x-access-token' \"password=$GH_TOKEN\"; }; f"
    )
    env = os.environ.copy()
    env.update(
        {
            "GH_TOKEN": token,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "/bin/false",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        }
    )
    _git(
        plan.repo_root,
        "-c",
        "credential.helper=",
        "-c",
        f"credential.helper={helper}",
        "-c",
        "core.hooksPath=/dev/null",
        "push",
        f"https://github.com/{plan.repo}.git",
        f"{commit}:refs/heads/{plan.branch}",
        env=env,
    )


def open_pull_request(
    plan: PublishPlan,
    token: str,
    *,
    title: str,
    body: str,
    transport: Transport | None = None,
) -> str:
    send = transport or api_request
    response = send(
        "POST",
        f"{API_ROOT}/repos/{plan.repo}/pulls",
        token,
        {
            "title": title,
            "head": plan.branch,
            "base": plan.base_branch,
            "body": body,
        },
        30.0,
    )
    url = response.get("html_url")
    if not isinstance(url, str) or not url:
        raise PublishError("GitHub did not return a pull-request URL")
    return url


def publish(
    repo_root: Path,
    app_config: AppConfig,
    *,
    title: str | None = None,
    body: str | None = None,
    dry_run: bool = False,
    transport: Transport | None = None,
    progress: Callable[[str], None] | None = None,
) -> PublishResult:
    _assert_trusted_entry_point(repo_root)
    plan = plan_publish(
        repo_root,
        repo=app_config.repo,
        base_branch=app_config.base_branch,
    )
    pr_title = title or f"Publish {plan.source_branch}"
    pr_body = body or default_body(plan)
    if dry_run:
        return PublishResult(plan=plan, commit=None, pr_url=None, dry_run=True)
    if not plan.ok:
        details: list[str] = []
        if plan.protected:
            details.append("protected paths: " + ", ".join(plan.protected))
        if plan.secret_hits:
            details.append(
                "secret scan: " + ", ".join(f"{path} ({kind})" for path, kind in plan.secret_hits)
            )
        raise PublishError("; ".join(details))
    if progress is not None:
        progress(f"authenticating as GitHub App {app_config.app_id}")
    installation = authenticate(app_config, transport=transport)
    commit = create_squash_commit(
        plan,
        author_name=app_config.author_name,
        author_email=installation.bot_email,
        title=pr_title,
        body=pr_body,
    )
    if progress is not None:
        progress(f"pushing {plan.branch} as {installation.app_slug}[bot]")
    push_branch(plan, commit, installation.token)
    if progress is not None:
        progress("opening pull request")
    pr_url = open_pull_request(
        plan,
        installation.token,
        title=pr_title,
        body=pr_body,
        transport=transport,
    )
    return PublishResult(
        plan=plan,
        commit=commit,
        pr_url=pr_url,
        dry_run=False,
        installation=installation,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="janet-publish",
        description="Publish committed work from a trusted Janet installation.",
    )
    parser.add_argument("--config", required=True, help="trusted publish config")
    parser.add_argument("--root", default=".", help="candidate repository root")
    parser.add_argument("--title", default=None)
    parser.add_argument("--body", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_app_config(Path(args.config))
        result = publish(
            Path(args.root),
            config,
            title=args.title,
            body=args.body,
            dry_run=args.dry_run,
            progress=lambda message: print(f"[publish] {message}", file=sys.stderr, flush=True),
        )
    except (GitHubAppError, PublishError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
