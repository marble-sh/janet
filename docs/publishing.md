# Publishing Janet

Janet publishes committed work as bot-authored pull requests. The publisher is
host-side, runs from a **trusted installation**, and never pushes `main` or
merges a pull request.

## Trusted entry point

Do not run the publisher from the candidate checkout. Install a trusted copy in
a separate environment and use `janet-publish`:

```bash
uv venv ~/.local/share/janet-publisher
uv pip install --python ~/.local/share/janet-publisher/bin/python \
  "git+https://github.com/marble-sh/janet@v0.5.0"
```

For a local checkout during development:

```bash
uvx --from . --with 'PyJWT[crypto]' janet-publish --help
```

The `janet publish` subcommand refuses when its own module is inside the
candidate root; `janet-publish` is the supported entry point.

## Configuration

Create a trusted config outside the candidate, for example
`~/.config/janet/publish.toml`:

```toml
[github_app]
repo = "marble-sh/janet"
app_id = 5480729
installation_id = 175567250
private_key_path = "/absolute/path/to/.secrets/github-app.pem"
author_name = "Janet"
base_branch = "main"
```

The app ID and installation ID are public identifiers. The private key is
local-only, mode `600`, and gitignored.

## Publish

From a clean feature branch based on `origin/main`:

```bash
janet-publish --config ~/.config/janet/publish.toml --root /path/to/janet --dry-run
janet-publish --config ~/.config/janet/publish.toml --root /path/to/janet
```

The dry run prints the branch, changed files, protected paths, and secret-scan
hits without authenticating or pushing.

## Guards

The publisher refuses unless:

- the working tree is clean;
- HEAD is a feature branch, never `main`;
- `origin/main` is an ancestor of HEAD;
- the diff is non-empty;
- no changed path is in the protected surface (CI, tests, dependencies,
  verifier, governor, sandbox, archive, selection loop, and the publisher
  itself);
- the secret scan is clean;
- the running publisher is outside the candidate checkout.

It does not execute candidate code. GitHub Actions runs the required `test`
check on the PR, and a human reviews and merges.

## Identity and credentials

Janet's commits are authored as:

```text
Janet <348334669+janet-could-you-please[bot]@users.noreply.github.com>
```

The publisher mints a one-hour installation token from the app private key. The
token is passed to `git push` through an ephemeral credential helper; inherited
helpers are cleared, repository hooks are disabled, and global/system git config
is bypassed. The token is never printed, written to disk, or exposed to the
model or sandbox.

## Review gate

`main` requires a pull request, the `test` check, one approval, a code-owner
review from `@marble-sh`, and approval of the latest push. Force-pushes and
deletions are blocked. The publisher cannot satisfy the code-owner review, so a
human must approve every publication.
