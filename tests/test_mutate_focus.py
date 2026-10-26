from __future__ import annotations

import json
from typing import Any

from janet.harness.mutate import _mutation_prompt, propose_mutation
from janet.harness.spec import HarnessSpec
from janet.llm import LLMResponse


class CapturingClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate(self, prompt: str, **_kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(
            text=self.text,
            model="fake",
            eval_count=1,
            prompt_eval_count=1,
            eval_duration_ns=1,
            total_duration_ns=1,
        )


def _parent() -> HarnessSpec:
    return HarnessSpec(name="parent", system_prompt="A" * 50)


def _candidate_json() -> str:
    return json.dumps(
        {
            "name": "focused",
            "system_prompt": "B" * 50,
            "max_attempts": 3,
            "samples_per_attempt": 3,
            "mutation_note": "focus test",
            "skills": [],
        }
    )


def test_a_focus_replaces_the_generic_one_change_constraint() -> None:
    focus = "Change only the system_prompt. Keep the skill library identical."
    prompt = _mutation_prompt(_parent(), "feedback", None, focus=focus)
    assert focus in prompt
    assert "Make ONE change" not in prompt
    assert "Return a JSON object" in prompt


def test_no_focus_keeps_the_generic_constraint() -> None:
    prompt = _mutation_prompt(_parent(), "feedback", None)
    assert "Make ONE change" in prompt


def test_propose_mutation_passes_the_focus_to_the_model() -> None:
    focus = "Change only the system_prompt. Keep the skill library identical."
    client = CapturingClient(_candidate_json())
    candidate = propose_mutation(
        client,  # type: ignore[arg-type]
        _parent(),
        "feedback",
        focus=focus,
    )
    assert candidate.system_prompt == "B" * 50
    assert client.prompts
    assert focus in client.prompts[0]


def test_propose_mutation_without_focus_still_works() -> None:
    client = CapturingClient(_candidate_json())
    candidate = propose_mutation(client, _parent(), "feedback")  # type: ignore[arg-type]
    assert candidate.name == "focused"
