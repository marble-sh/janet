"""Tests for GitHub App authentication. No network or real credentials."""

from __future__ import annotations

import urllib.error
from pathlib import Path
from typing import Any

import pytest

from janet.github_app import (
    API_ROOT,
    AppConfig,
    GitHubAppError,
    Installation,
    api_request,
    authenticate,
    load_app_config,
    mint_app_jwt,
)


def _config_file(tmp_path: Path, *, key_path: Path | None = None, extra: str = "") -> Path:
    key = key_path or (tmp_path / "key.pem")
    path = tmp_path / "publish.toml"
    path.write_text(
        "[github_app]\n"
        'repo = "marble-sh/janet"\n'
        "app_id = 5480729\n"
        "installation_id = 175567250\n"
        f'private_key_path = "{key}"\n'
        'author_name = "Janet"\n'
        'base_branch = "main"\n' + extra,
        encoding="utf-8",
    )
    return path


def _app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        repo="marble-sh/janet",
        app_id=5480729,
        installation_id=175567250,
        private_key_path=tmp_path / "key.pem",
    )


def _permissions() -> dict[str, str]:
    return {
        "contents": "write",
        "pull_requests": "write",
        "metadata": "read",
        "checks": "read",
    }


def test_load_app_config(tmp_path: Path) -> None:
    key = tmp_path / "key.pem"
    key.write_bytes(b"key")
    config = load_app_config(_config_file(tmp_path, key_path=key))
    assert config.repo == "marble-sh/janet"
    assert config.app_id == 5480729
    assert config.installation_id == 175567250
    assert config.private_key_path == key
    assert config.author_name == "Janet"
    assert config.base_branch == "main"


def test_load_app_config_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = tmp_path / "other.pem"
    monkeypatch.setenv("JANET_GITHUB_REPO", "other/repo")
    monkeypatch.setenv("JANET_GITHUB_APP_ID", "42")
    monkeypatch.setenv("JANET_GITHUB_APP_INSTALLATION_ID", "7")
    monkeypatch.setenv("JANET_GITHUB_APP_PRIVATE_KEY_PATH", str(key))
    monkeypatch.setenv("JANET_GITHUB_APP_AUTHOR_NAME", "Bot")
    monkeypatch.setenv("JANET_GITHUB_BASE_BRANCH", "trunk")
    config = load_app_config(_config_file(tmp_path))
    assert config.repo == "other/repo"
    assert config.app_id == 42
    assert config.installation_id == 7
    assert config.private_key_path == key
    assert config.author_name == "Bot"
    assert config.base_branch == "trunk"


def test_load_app_config_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(GitHubAppError, match="unknown"):
        load_app_config(_config_file(tmp_path, extra="wat = 1\n"))


def test_load_app_config_rejects_missing_table(tmp_path: Path) -> None:
    path = tmp_path / "publish.toml"
    path.write_text("[other]\n", encoding="utf-8")
    with pytest.raises(GitHubAppError, match="github_app"):
        load_app_config(path)


def test_load_app_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GitHubAppError, match="not found"):
        load_app_config(tmp_path / "missing.toml")


def test_mint_app_jwt_uses_a_short_rs256_token(tmp_path: Path) -> None:
    key = tmp_path / "key.pem"
    key.write_bytes(b"private-key")
    config = _app_config(tmp_path)
    seen: dict[str, object] = {}

    def encoder(
        payload: dict[str, object],
        private_key: bytes,
        *,
        algorithm: str,
    ) -> str:
        seen.update(payload=payload, private_key=private_key, algorithm=algorithm)
        return "app-jwt"

    assert mint_app_jwt(config, now=1000.0, encoder=encoder) == "app-jwt"
    assert seen["payload"] == {"iat": 940, "exp": 1540, "iss": "5480729"}
    assert seen["private_key"] == b"private-key"
    assert seen["algorithm"] == "RS256"


def test_mint_app_jwt_rejects_a_missing_key(tmp_path: Path) -> None:
    with pytest.raises(GitHubAppError, match="private key not found"):
        mint_app_jwt(_app_config(tmp_path))


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_api_request_returns_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float) -> _Response:
        return _Response(b'{"ok": true}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert api_request("GET", f"{API_ROOT}/app", "token", None) == {"ok": True}


def test_api_request_reports_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHTTPError(Exception):
        code = 403

        def read(self) -> bytes:
            return b'{"message":"denied"}'

    def fake_urlopen(request: object, timeout: float) -> _Response:
        raise FakeHTTPError()

    monkeypatch.setattr("urllib.error.HTTPError", FakeHTTPError)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(GitHubAppError, match="HTTP 403"):
        api_request("GET", f"{API_ROOT}/app", "token", None)


def test_api_request_reports_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float) -> _Response:
        raise urllib.error.URLError("boom")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(GitHubAppError, match="boom"):
        api_request("GET", f"{API_ROOT}/app", "token", None)


def _transport(
    *,
    installation_id: int = 175567250,
    permissions: dict[str, str] | None = None,
) -> tuple[Any, list[tuple[str, str]]]:
    calls: list[tuple[str, str]] = []

    def send(
        method: str,
        url: str,
        token: str,
        payload: dict[str, object] | None,
        timeout: float,
    ) -> dict[str, Any]:
        calls.append((method, url))
        if url == f"{API_ROOT}/app":
            return {"slug": "janet-could-you-please", "name": "janet-could-you-please"}
        if url == f"{API_ROOT}/repos/marble-sh/janet/installation":
            return {
                "id": installation_id,
                "permissions": _permissions() if permissions is None else permissions,
            }
        if url == f"{API_ROOT}/app/installations/{installation_id}/access_tokens":
            return {"token": "ghs_test", "expires_at": "2026-10-26T19:47:27Z"}
        if url == f"{API_ROOT}/users/janet-could-you-please%5Bbot%5D":
            return {"id": 348334669}
        raise AssertionError(url)

    return send, calls


def test_authenticate_mints_a_scoped_installation_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("janet.github_app.mint_app_jwt", lambda config: "app-jwt")
    send, calls = _transport()
    installation = authenticate(_app_config(tmp_path), transport=send)
    assert installation.token == "ghs_test"
    assert installation.expires_at == "2026-10-26T19:47:27Z"
    assert installation.app_slug == "janet-could-you-please"
    assert installation.installation_id == 175567250
    assert installation.bot_email == (
        "348334669+janet-could-you-please[bot]@users.noreply.github.com"
    )
    assert calls == [
        ("GET", f"{API_ROOT}/app"),
        ("GET", f"{API_ROOT}/repos/marble-sh/janet/installation"),
        ("POST", f"{API_ROOT}/app/installations/175567250/access_tokens"),
        ("GET", f"{API_ROOT}/users/janet-could-you-please%5Bbot%5D"),
    ]


def test_authenticate_rejects_missing_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("janet.github_app.mint_app_jwt", lambda config: "app-jwt")
    send, _ = _transport(permissions={"contents": "write", "metadata": "read"})
    with pytest.raises(GitHubAppError, match="missing permissions"):
        authenticate(_app_config(tmp_path), transport=send)


def test_authenticate_rejects_forbidden_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("janet.github_app.mint_app_jwt", lambda config: "app-jwt")
    send, _ = _transport(permissions={**_permissions(), "workflows": "write"})
    with pytest.raises(GitHubAppError, match="must be removed"):
        authenticate(_app_config(tmp_path), transport=send)


def test_authenticate_rejects_an_installation_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("janet.github_app.mint_app_jwt", lambda config: "app-jwt")
    send, _ = _transport(installation_id=99)
    with pytest.raises(GitHubAppError, match="does not match"):
        authenticate(_app_config(tmp_path), transport=send)


def test_installation_public_dict_hides_the_token() -> None:
    installation = Installation(
        token="ghs_secret",
        expires_at="2026-10-26T19:47:27Z",
        app_id=5480729,
        app_slug="janet-could-you-please",
        app_name="janet-could-you-please",
        bot_user_id=348334669,
        installation_id=175567250,
        permissions=_permissions(),
    )
    public = installation.public_dict()
    assert "token" not in public
    assert "ghs_secret" not in str(public)
