# Plan — publishing Janet safely — 2026-10-26

**Status:** COMPLETE 2026-10-26. P2 merged in PR #1; the first real publication (PR #2, docs) merged; release `v0.5.0`.

## Goal

Make `marble-sh/janet` a public, reproducible repository that Janet can extend,
while preserving the project's safety kernel: **no unverified variant reaches
`main`**.

## Current state

- Repository: <https://github.com/marble-sh/janet> (public).
- `main` is protected: PR required, `test` check required, `@marble-sh` code-owner review, no force-push or deletion.
- CI: `.github/workflows/ci.yml`; the first run passed on GitHub-hosted Ubuntu.
- GitHub App: `janet-could-you-please` (ID 5480729), installation 175567250, installed on this repository only.
- App permissions: `contents: write`, `pull_requests: write`, `metadata: read`, `checks: read`. No administration, workflows, secrets, or actions.
- Private key: `.secrets/github-app.pem` (mode 600, gitignored).
- Publisher: `src/janet/github_app.py` and `src/janet/publish.py`, exposed as `janet publish` and the trusted `janet-publish`.
- Local gate: 436 tests, 86.40% coverage, ruff/mypy/build green.
- The third-party `research/` corpus is local-only and gitignored.
- README image removed at user request; no image asset is tracked.

## Hard constraint: why Janet does not get a blank-check token

`AGENTS.md` puts the anchor suite, acceptance verifiers, governor, sandbox, and
promotion gate outside the agent's writable surface. A token that lets the loop
push to `main` would let a candidate edit its own verifier and publish the
result. That removes the experiment's control and turns a verification-first
system into an unverified self-publisher.

Therefore the safe maximum is:

- Janet may write code and **open a pull request** on a branch.
- A human reviews and merges to `main`.
- The token is **host-side only**: never in the sandbox, never in the repo,
  never in a prompt, never in task data.

This still satisfies "Janet modifies her own source and publishes it": the
public repo receives her verified commits. The only thing withheld is the power
to approve her own changes.

## Phases

### P1 — baseline release — DONE 2026-10-26

- 0BSD `LICENSE` added and declared in `pyproject.toml`.
- Reward v2, `docs/how-janet-learns.md`, and this plan committed together.
- Local CI green: 404 tests, 87.48% coverage, ruff/mypy/build pass.
- `main` pushed as a fresh history; old tags discarded.

### P2 — `janet publish` — DONE 2026-10-26

The publisher is host-side and refuses unless:

- the working tree is clean;
- HEAD is a feature branch (never `main`);
- `origin/main` is an ancestor of HEAD (rebase first otherwise);
- the diff is non-empty;
- no changed path is in the protected surface below;
- the secret scan is clean;
- the running publisher is outside the candidate checkout (trusted install).

It squashes the diff into one commit authored as
`Janet <348334669+janet-could-you-please[bot]@users.noreply.github.com>`, pushes
`janet/auto-<sha>` over HTTPS with a one-hour installation token, and opens a PR
through the GitHub API. It never force-pushes, pushes `main`, or merges.

The publisher does **not** execute candidate code: local `uv`/pytest runs were
removed after review. GitHub Actions runs the required `test` check on the PR,
and the human merge gate remains.

The archive `promoted` gate is for harness variants; source publication is
gated by CI, the protected surface, and human review. A harness-variant
publication command is a separate future feature.

### Trusted installation

The publisher must not import the candidate checkout. Install a trusted copy in
a separate environment and run the dedicated entry point:

```bash
uv venv ~/.local/share/janet-publisher
uv pip install --python ~/.local/share/janet-publisher/bin/python \
  "git+https://github.com/marble-sh/janet@v0.5.0"
~/.local/share/janet-publisher/bin/janet-publish \
  --config ~/.config/janet/publish.toml --root /path/to/janet
```

The trusted config lives outside the candidate and should use an absolute
`private_key_path`. `janet publish` remains available from an installed package
but refuses when its own module is inside the candidate root.

### Protected surface

`src/janet/publish.py` refuses a PR that changes:

- `.github/` (CI and CODEOWNERS) and `tests/`;
- `pyproject.toml`, `uv.lock`, `configs/default.toml`, `configs/publish.toml`;
- the governor, sandbox, probe, preemption, LLM, and suite loaders;
- the verifier (`eval/runner.py`, `eval/audit.py`, `eval/task.py`,
  `eval/worker.py`), mutants/evolve/execute, and the bench seeds/materializer;
- the selection and confirmation loop (`loop/generation.py`,
  `loop/confirm.py`, `loop/decoupling.py`, `loop/evolution.py`,
  `loop/passive.py`), the archive, and the anti-hard-coding guard;
- the publisher and app-auth modules themselves.

This is deliberately conservative. A human can change any of these through a
reviewed PR; Janet can still publish the harness, skills, prompts, repair loop,
docs, and non-safety code.

### P3 — CI and branch protection — DONE 2026-10-26

- GitHub Actions runs the full gate on pushes to `main` and every pull request.
- `main` requires the `test` check, a PR, one approval, the `@marble-sh`
  code-owner review, and approval of the latest push.
- Force-pushes and deletions are blocked. Admins are not enforced, so the owner
  can bootstrap changes that the bot cannot publish.

### P4 — releases

- Human merges the PR, tags the release, and runs `gh release create`.
- If the package is published to PyPI, use **Trusted Publishing (OIDC)** rather
  than a long-lived PyPI token.

### P5 — runbook

Document token creation, rotation, revocation, and what to do if the token
leaks. Record every publish attempt in the event log.

## Authentication

The publisher uses the GitHub App, not a PAT.

| Setting | Value |
|---|---|
| App | `janet-could-you-please` |
| App ID | 5480729 |
| Installation ID | 175567250 |
| Bot user ID | 348334669 |
| Repository access | only `marble-sh/janet` |
| Contents | Read and write |
| Pull requests | Read and write |
| Metadata | Read |
| Checks | Read |
| Administration / Workflows / Secrets / Actions | No access |

`src/janet/github_app.py` mints a 540-second RS256 JWT with the app private
key, exchanges it for a one-hour installation token, and verifies the granted
permissions before use. The bot email uses the bot user ID, not the App ID, so
GitHub links the squashed commit to `janet-could-you-please[bot]`. The token is
passed to `git push` through an ephemeral credential helper; inherited helpers
are cleared, `core.hooksPath` is disabled, and global/system config is bypassed
so an existing helper or repository hook cannot capture the token. It is never
printed, written to disk, or given to the model or sandbox.

The private key is local-only: `.secrets/github-app.pem`, mode 600, gitignored.
Rotate it in the app settings and delete the old key if it leaks.

## Verification

- 436 tests, 86.40% coverage; ruff, mypy strict, and build green.
- Unit tests cover JWT payloads, token exchange, bot-identity lookup, permission
  checks, planner guards, NUL-safe paths, secret scanning, protected paths,
  squash commits, dry-run, credential isolation, and pull-request creation with
  fakes. No test touches the network.
- Real App auth was verified once: app slug `janet-could-you-please`, bot user
  ID 348334669, installation 175567250, and a token scoped to `marble-sh/janet`
  only.

## Review fixes (PR #1)

Copilot raised five issues; all were verified and fixed:

1. **Candidate imports** — the publisher refuses to run from inside the
   candidate checkout; `janet-publish` is the dedicated trusted entry point.
2. **Quoted paths** — changed paths are read NUL-delimited with `git diff -z`,
   so tabs and newlines cannot bypass the protected-prefix check.
3. **Local CI credentials** — local CI was removed; GitHub Actions is the gate,
   so candidate code never runs in the token-holding process.
4. **Credential helpers** — inherited helpers are cleared, hooks are disabled,
   and global/system config is bypassed for the push.
5. **Bot identity** — the noreply email uses bot user ID 348334669, not the App
   ID, so commits link to `janet-could-you-please[bot]`.

## Decisions (2026-10-26)

1. **License:** 0BSD — accepted.
2. **First push:** one clean baseline — done; history recreated without the research corpus.
3. **Autonomy:** PR-only human merge — accepted. Janet never pushes to `main`.
4. **Publisher identity:** GitHub App `janet-could-you-please` — accepted and
   installed. The publisher mints one-hour installation tokens; no PAT is used.

## Execution note

The first publisher change is human-bootstrapped because the publisher, App
auth, CI, and config are inside the protected surface. After merge, Janet can
publish non-protected changes through `janet publish`. Nothing in P2 runs inside
the bubblewrap sandbox.
