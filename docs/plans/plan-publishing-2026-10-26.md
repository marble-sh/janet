# Plan — publishing Janet safely — 2026-10-26

**Status:** ACCEPTED 2026-10-26. Fresh public history created after removing the third-party research corpus; P2 is blocked only on the publisher identity choice (fine-grained PAT vs GitHub App).

## Goal

Make `marble-sh/janet` a public, reproducible repository that Janet can extend,
while preserving the project's safety kernel: **no unverified variant reaches
`main`**.

## Current state

- Repository: <https://github.com/marble-sh/janet> (public, fresh single-commit history).
- Remote: `origin git@github.com:marble-sh/janet.git`.
- License: 0BSD (`LICENSE`), recognized by GitHub.
- Baseline includes reward v2, `docs/how-janet-learns.md`, and this plan.
- The third-party `research/` corpus is local-only and gitignored.
- Old release tags were intentionally not carried into the fresh history.
- README image removed at user request; no image asset is tracked.
- No CI workflow and no branch protection yet.

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

### P2 — `janet publish` (host-side command)

Add `src/janet/publish.py` and a CLI subcommand. It refuses unless all of these
hold:

- the archive variant is `promoted` (not merely `accepted`);
- the working tree is clean;
- local CI passes;
- the secret scan is clean;
- protected paths are unchanged (anchor, verifiers, governor, sandbox,
  promotion gate);
- the token has only the scopes listed below.

Then it pushes `janet/auto-<hash>` and opens a PR with `gh pr create`. It never
force-pushes and never auto-merges.

### P3 — CI and branch protection

- GitHub Actions runs: `uv sync --locked`, `ruff format --check`, `ruff check`,
  `mypy`, `pytest --cov`, `uv build`.
- Protect `main`: require the CI check, require a pull request, disallow
  force-pushes and deletions.
- Optional: require signed commits for `janet/auto-*` branches.

### P4 — releases

- Human merges the PR, tags the release, and runs `gh release create`.
- If the package is published to PyPI, use **Trusted Publishing (OIDC)** rather
  than a long-lived PyPI token.

### P5 — runbook

Document token creation, rotation, revocation, and what to do if the token
leaks. Record every publish attempt in the event log.

## Token specification (when you create it)

Use a **fine-grained personal access token**, not a classic token:

| Setting | Value |
|---|---|
| Resource owner | `marble-sh` |
| Repository access | only `marble-sh/janet` |
| Contents | Read and write |
| Pull requests | Read and write |
| Metadata | Read (mandatory) |
| Actions / Administration / Secrets / Workflows | **No access** |
| Expiration | 30–90 days |

Store it in the OS keyring or export `GH_TOKEN` for the publish command only.
`.env` is already gitignored, but do not put the token there either. Never pass
it to the model, the sandbox, or a candidate process.

The publisher reads `GH_TOKEN`, so either a fine-grained PAT or a GitHub App
installation token works without changing `janet publish`.

Recommended: a **GitHub App** named `Janet`, installed only on
`marble-sh/janet`, with Contents R/W, Pull requests R/W, and Metadata R. It
gives a real `janet[bot]` identity and one-hour installation tokens. The
trade-off is a private key to store and a JWT-to-installation-token exchange in
the publisher. A dedicated `janet-bot` account with a fine-grained PAT is the
simpler intermediate step.

## Verification

- Unit tests mock `git` and `gh`; no test touches the network.
- A dry-run mode prints the branch, PR title, and files without pushing.
- A test asserts the command refuses an unpromoted variant, a dirty tree, a
  failing CI result, a secret hit, and a protected-path edit.

## Decisions (2026-10-26)

1. **License:** 0BSD — accepted.
2. **First push:** one clean baseline — done; history recreated without the research corpus.
3. **Autonomy:** PR-only human merge — accepted. Janet never pushes to `main`.
4. **Publisher identity:** PAT vs GitHub App — **pending**. Recommend the App;
   either way `janet publish` reads `GH_TOKEN`.

## Execution note

Implement P2 with DeepSeek V4.1 Flash at high effort; have a human review the
token/App private key, CI, and branch-protection settings. Nothing in P2 should
run inside the bubblewrap sandbox.
