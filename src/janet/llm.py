"""Local-only LLM client.

The client talks to an Ollama-compatible endpoint. It refuses a non-loopback
endpoint unless the configuration explicitly permits remote task data, and it
never logs prompts or completions.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from janet.config import LLMConfig

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class LLMError(RuntimeError):
    """Raised when a local model call fails."""


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    eval_count: int
    prompt_eval_count: int
    eval_duration_ns: int
    total_duration_ns: int

    @property
    def tokens_per_second(self) -> float:
        if self.eval_duration_ns <= 0:
            return 0.0
        return self.eval_count / (self.eval_duration_ns / 1e9)


class Transport(Protocol):
    def __call__(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]: ...


def _http_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"model request failed: {exc}") from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMError(f"model returned non-JSON: {body[:200]!r}") from exc
    if not isinstance(decoded, dict):
        raise LLMError("model returned a non-object JSON payload")
    return decoded


def is_loopback(base_url: str) -> bool:
    host = urllib.parse.urlparse(base_url).hostname
    return host in _LOOPBACK_HOSTS


class LLMClient:
    """A minimal chat/generate client for a local Ollama server."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: Transport | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        if not is_loopback(config.base_url) and not config.allow_remote_task_data:
            raise LLMError(
                "remote model endpoint refused: allow_remote_task_data is false "
                f"({config.base_url})"
            )
        self.config = config
        self._transport = transport or _http_post_json
        self._sleep = sleep

    @property
    def endpoint(self) -> str:
        return self.config.base_url.rstrip("/") + "/api/generate"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model or self.config.policy_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": (self.config.temperature if temperature is None else temperature),
                "num_predict": (self.config.max_tokens if max_tokens is None else max_tokens),
                "num_ctx": self.config.context_tokens,
                "num_gpu": self.config.num_gpu,
            },
        }
        if system is not None:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        data = self._transport(self.endpoint, payload, self.config.timeout_seconds)
        text = data.get("response")
        if not isinstance(text, str):
            raise LLMError("model response is missing a string 'response' field")
        return LLMResponse(
            text=text,
            model=str(data.get("model", payload["model"])),
            eval_count=int(data.get("eval_count", 0) or 0),
            prompt_eval_count=int(data.get("prompt_eval_count", 0) or 0),
            eval_duration_ns=int(data.get("eval_duration", 0) or 0),
            total_duration_ns=int(data.get("total_duration", 0) or 0),
        )

    def list_models(self) -> list[str]:
        url = self.config.base_url.rstrip("/") + "/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMError(f"cannot list models: {exc}") from exc
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
