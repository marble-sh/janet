from __future__ import annotations

import json
from typing import Any

import pytest

from janet.config import LLMConfig
from janet.llm import LLMClient, LLMError, LLMResponse, _http_post_json, is_loopback


def test_is_loopback() -> None:
    assert is_loopback("http://127.0.0.1:11434")
    assert is_loopback("http://localhost:11434")
    assert is_loopback("http://[::1]:11434")
    assert not is_loopback("https://api.example.com")


def test_remote_endpoint_is_refused_by_default() -> None:
    with pytest.raises(LLMError, match="allow_remote_task_data"):
        LLMClient(LLMConfig(base_url="https://api.example.com"))


def test_remote_endpoint_is_allowed_when_permitted() -> None:
    client = LLMClient(LLMConfig(base_url="https://api.example.com", allow_remote_task_data=True))
    assert client.endpoint == "https://api.example.com/api/generate"


def test_generate_builds_the_local_payload() -> None:
    captured: dict[str, Any] = {}

    def transport(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        captured.update({"url": url, "payload": payload, "timeout": timeout})
        return {
            "response": "print('hi')",
            "model": "test-model",
            "eval_count": 10,
            "prompt_eval_count": 4,
            "eval_duration": 2_000_000_000,
            "total_duration": 3_000_000_000,
        }

    config = LLMConfig(num_gpu=20, context_tokens=4096, max_tokens=128, temperature=0.1)
    client = LLMClient(config, transport=transport)
    response = client.generate("prompt", system="system prompt", json_mode=True)
    assert isinstance(response, LLMResponse)
    assert response.text == "print('hi')"
    assert response.tokens_per_second == 5.0
    payload = captured["payload"]
    assert payload["model"] == config.policy_model
    assert payload["system"] == "system prompt"
    assert payload["format"] == "json"
    assert payload["options"]["num_ctx"] == 4096
    assert payload["options"]["num_gpu"] == 20
    assert payload["options"]["num_predict"] == 128
    assert payload["options"]["temperature"] == 0.1
    assert captured["timeout"] == config.timeout_seconds


def test_generate_rejects_a_missing_response() -> None:
    client = LLMClient(LLMConfig(), transport=lambda _u, _p, _t: {})
    with pytest.raises(LLMError, match="missing"):
        client.generate("prompt")


def test_generate_rejects_a_non_string_response() -> None:
    client = LLMClient(LLMConfig(), transport=lambda _u, _p, _t: {"response": 3})
    with pytest.raises(LLMError, match="missing"):
        client.generate("prompt")


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_list_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(_url: str, timeout: float) -> _FakeHTTPResponse:
        assert timeout == 5
        return _FakeHTTPResponse({"models": [{"name": "a:latest"}, {"model": "b"}]})

    monkeypatch.setattr("janet.llm.urllib.request.urlopen", fake_urlopen)
    client = LLMClient(LLMConfig())
    assert client.list_models() == ["a:latest", "b"]


def test_list_models_handles_a_non_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "janet.llm.urllib.request.urlopen",
        lambda _url, timeout: _FakeHTTPResponse({"models": "nope"}),
    )
    client = LLMClient(LLMConfig())
    assert client.list_models() == []


def test_http_transport_rejects_non_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class BadResponse:
        def read(self) -> bytes:
            return b"not json"

        def __enter__(self) -> BadResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr("janet.llm.urllib.request.urlopen", lambda *_args, **_kwargs: BadResponse())
    with pytest.raises(LLMError, match="non-JSON"):
        _http_post_json("http://127.0.0.1/api", {}, 1.0)


def test_http_transport_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("down")

    monkeypatch.setattr("janet.llm.urllib.request.urlopen", boom)
    with pytest.raises(LLMError, match="request failed"):
        _http_post_json("http://127.0.0.1/api", {}, 1.0)


def test_list_models_reports_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("down")

    monkeypatch.setattr("janet.llm.urllib.request.urlopen", boom)
    with pytest.raises(LLMError, match="cannot list"):
        LLMClient(LLMConfig()).list_models()
