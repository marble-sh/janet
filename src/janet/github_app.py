"""GitHub App authentication for the host-side publisher.

The publisher runs outside the bubblewrap sandbox. It mints a short-lived
installation token from the app's private key, uses it for one publish
operation, and never writes the token to disk or passes it to the model.
"""

from __future__ import annotations

import importlib
import json
import os
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import quote

API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
DEFAULT_APP_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "publish.toml"

_JWT_BACKDATE_S = 60
_JWT_LIFETIME_S = 540
_REQUIRED_PERMISSIONS = {"contents": "write", "pull_requests": "write", "metadata": "read"}
_FORBIDDEN_PERMISSIONS = {
    "actions",
    "administration",
    "environments",
    "pages",
    "secrets",
    "variables",
    "webhooks",
    "workflows",
}


class GitHubAppError(RuntimeError):
    """Raised when the app credentials, API, or permissions are unusable."""


class Transport(Protocol):
    """One JSON request to the GitHub API."""

    def __call__(
        self,
        method: str,
        url: str,
        token: str,
        payload: dict[str, object] | None,
        timeout: float,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class AppConfig:
    repo: str
    app_id: int
    installation_id: int | None
    private_key_path: Path
    author_name: str = "Janet"
    base_branch: str = "main"

    def public_dict(self) -> dict[str, object]:
        return {
            "repo": self.repo,
            "app_id": self.app_id,
            "installation_id": self.installation_id,
            "private_key_path": str(self.private_key_path),
            "author_name": self.author_name,
            "base_branch": self.base_branch,
        }


@dataclass(frozen=True, slots=True)
class Installation:
    token: str
    expires_at: str
    app_id: int
    app_slug: str
    app_name: str
    bot_user_id: int
    installation_id: int
    permissions: dict[str, str]

    @property
    def bot_email(self) -> str:
        return f"{self.bot_user_id}+{self.app_slug}[bot]@users.noreply.github.com"

    def public_dict(self) -> dict[str, object]:
        """Everything except the token."""
        return {
            "app_id": self.app_id,
            "app_slug": self.app_slug,
            "app_name": self.app_name,
            "bot_user_id": self.bot_user_id,
            "installation_id": self.installation_id,
            "permissions": dict(self.permissions),
            "bot_email": self.bot_email,
            "expires_at": self.expires_at,
        }


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _non_empty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GitHubAppError(f"{label} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, label: str) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError) as exc:
        raise GitHubAppError(f"{label} must be an integer") from exc
    if number < 1:
        raise GitHubAppError(f"{label} must be >= 1")
    return number


def load_app_config(path: Path | None = None) -> AppConfig:
    """Load the publish config, with environment overrides for every field."""

    config_path = (path or DEFAULT_APP_CONFIG_PATH).expanduser()
    if not config_path.is_file():
        raise GitHubAppError(f"publish config not found: {config_path}")
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise GitHubAppError(f"invalid publish config {config_path}: {exc}") from exc
    section = raw.get("github_app")
    if not isinstance(section, dict):
        raise GitHubAppError("[github_app] table is required in the publish config")
    allowed = {
        "repo",
        "app_id",
        "installation_id",
        "private_key_path",
        "author_name",
        "base_branch",
    }
    unknown = set(section) - allowed
    if unknown:
        raise GitHubAppError(f"unknown [github_app] keys: {sorted(unknown)}")

    repo = _non_empty(_env("JANET_GITHUB_REPO") or section.get("repo"), "github_app.repo")
    if repo.count("/") != 1 or not all(part for part in repo.split("/")):
        raise GitHubAppError("github_app.repo must be owner/name")

    app_id = _positive_int(
        _env("JANET_GITHUB_APP_ID") or section.get("app_id"), "github_app.app_id"
    )
    raw_installation = _env("JANET_GITHUB_APP_INSTALLATION_ID") or section.get("installation_id")
    installation_id = (
        None
        if raw_installation in (None, "")
        else _positive_int(raw_installation, "installation_id")
    )
    key_value = _non_empty(
        _env("JANET_GITHUB_APP_PRIVATE_KEY_PATH") or section.get("private_key_path"),
        "github_app.private_key_path",
    )
    key_path = Path(key_value).expanduser()
    if not key_path.is_absolute():
        key_path = (Path.cwd() / key_path).resolve()

    return AppConfig(
        repo=repo,
        app_id=app_id,
        installation_id=installation_id,
        private_key_path=key_path,
        author_name=_non_empty(
            _env("JANET_GITHUB_APP_AUTHOR_NAME") or section.get("author_name", "Janet"),
            "github_app.author_name",
        ),
        base_branch=_non_empty(
            _env("JANET_GITHUB_BASE_BRANCH") or section.get("base_branch", "main"),
            "github_app.base_branch",
        ),
    )


def api_request(
    method: str,
    url: str,
    token: str,
    payload: dict[str, object] | None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Send one JSON request; never include the token in an error message."""

    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": API_VERSION,
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GitHubAppError(
            f"GitHub API {method} {url} failed: HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise GitHubAppError(f"GitHub API {method} {url} failed: {exc.reason}") from exc
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GitHubAppError(f"GitHub API {method} {url} returned non-JSON") from exc
    if not isinstance(result, dict):
        raise GitHubAppError(f"GitHub API {method} {url} returned a non-object")
    return result


def mint_app_jwt(
    config: AppConfig,
    *,
    now: float | None = None,
    encoder: Callable[..., str] | None = None,
) -> str:
    """Create a short RS256 app JWT.

    ``encoder`` is injectable so tests never need the optional cryptographic
    dependency.
    """

    if not config.private_key_path.is_file():
        raise GitHubAppError(f"GitHub App private key not found: {config.private_key_path}")
    private_key = config.private_key_path.read_bytes()
    moment = int(time.time() if now is None else now)
    payload = {
        "iat": moment - _JWT_BACKDATE_S,
        "exp": moment + _JWT_LIFETIME_S,
        "iss": str(config.app_id),
    }
    if encoder is None:
        try:
            module = importlib.import_module("jwt")
        except ImportError as exc:
            raise GitHubAppError(
                "PyJWT is required for publishing; run `uv sync --extra publish`"
            ) from exc
        encoder = cast(Callable[..., str], module.encode)
    return encoder(payload, private_key, algorithm="RS256")


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise GitHubAppError(f"GitHub response is missing {key!r}")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise GitHubAppError(f"GitHub response is missing integer {key!r}")
    return value


def _permissions(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _check_permissions(permissions: dict[str, str]) -> None:
    missing = [
        f"{name}={level}"
        for name, level in _REQUIRED_PERMISSIONS.items()
        if permissions.get(name) != level
    ]
    if missing:
        raise GitHubAppError("the GitHub App is missing permissions: " + ", ".join(missing))
    forbidden = {
        name: level
        for name, level in permissions.items()
        if name in _FORBIDDEN_PERMISSIONS and level != "none"
    }
    if forbidden:
        raise GitHubAppError(
            "the GitHub App has permissions that must be removed: "
            + ", ".join(f"{name}={level}" for name, level in sorted(forbidden.items()))
        )


def authenticate(config: AppConfig, *, transport: Transport | None = None) -> Installation:
    """Mint a one-hour installation token and verify its scope."""

    send = transport or api_request
    app_jwt = mint_app_jwt(config)
    app = send("GET", f"{API_ROOT}/app", app_jwt, None, 30.0)
    slug = _required_str(app, "slug")
    name = _required_str(app, "name")

    installation = send(
        "GET",
        f"{API_ROOT}/repos/{config.repo}/installation",
        app_jwt,
        None,
        30.0,
    )
    installation_id = _required_int(installation, "id")
    if config.installation_id is not None and config.installation_id != installation_id:
        raise GitHubAppError(
            f"configured installation {config.installation_id} does not match "
            f"the repository installation {installation_id}"
        )
    permissions = _permissions(installation.get("permissions"))
    _check_permissions(permissions)

    token_response = send(
        "POST",
        f"{API_ROOT}/app/installations/{installation_id}/access_tokens",
        app_jwt,
        {},
        30.0,
    )
    token = _required_str(token_response, "token")
    expires_at = _required_str(token_response, "expires_at")
    bot_login = quote(f"{slug}[bot]", safe="")
    bot = send("GET", f"{API_ROOT}/users/{bot_login}", token, None, 30.0)
    bot_user_id = _required_int(bot, "id")
    return Installation(
        token=token,
        expires_at=expires_at,
        app_id=config.app_id,
        app_slug=slug,
        app_name=name,
        bot_user_id=bot_user_id,
        installation_id=installation_id,
        permissions=permissions,
    )
