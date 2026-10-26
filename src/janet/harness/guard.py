"""Anti-overfitting guard: a harness may not name benchmark tasks."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from janet.eval.task import Task
from janet.harness.spec import HarnessSpec


def forbidden_terms(tasks: Iterable[Task]) -> tuple[str, ...]:
    """Task ids and entrypoints; naming them is hard-coding, not generalizing."""

    terms: set[str] = set()
    for task in tasks:
        terms.add(task.id.lower())
        if task.entrypoint:
            terms.add(task.entrypoint.lower())
    return tuple(sorted(terms))


def spec_text(spec: HarnessSpec) -> str:
    parts = [spec.system_prompt]
    for skill in spec.skills:
        parts.append(skill.name)
        parts.extend(skill.triggers)
        parts.append(skill.rule)
    return "\n".join(parts).lower()


def inert_new_skills(
    candidate: HarnessSpec, parent: HarnessSpec, tasks: Iterable[Task]
) -> tuple[str, ...]:
    """Names of added/changed skills whose triggers match no task.

    An inert skill changes the hash but not behavior, so it can never be
    evaluated; accepting it would only pollute the archive.
    """

    haystacks = [
        f"{task.prompt}\n{task.solution}".lower() for task in tasks if not task.is_integrity
    ]
    known = {json.dumps(skill.to_dict(), sort_keys=True) for skill in parent.skills}
    inert = [
        skill.name
        for skill in candidate.skills
        if json.dumps(skill.to_dict(), sort_keys=True) not in known
        and not any(skill.matches(haystack) for haystack in haystacks)
    ]
    return tuple(inert)


_WORD_RE = re.compile(r"(?<![a-z])[a-z]{4,16}(?![a-z])")


def trigger_vocabulary(
    tasks: Iterable[Task], forbidden: Iterable[str], limit: int = 40
) -> tuple[str, ...]:
    """Words that occur in several task texts: valid skill triggers.

    Built from the tasks passed in (dev only, so the anchor's text is never
    revealed to the proposer). Benchmark identifiers are excluded.
    """

    banned = {part for term in forbidden for part in re.split(r"[_\W]+", term.lower())}
    counts: dict[str, int] = {}
    for task in tasks:
        if task.is_integrity:
            continue
        words = set(_WORD_RE.findall(f"{task.prompt}\n{task.solution}".lower()))
        for word in words - banned:
            counts[word] = counts.get(word, 0) + 1
    common = [(word, n) for word, n in counts.items() if n >= 2]
    common.sort(key=lambda item: (-item[1], item[0]))
    return tuple(word for word, _ in common[:limit])


def find_leaks(spec: HarnessSpec, terms: Iterable[str]) -> tuple[str, ...]:
    text = spec_text(spec)
    leaks = [
        term for term in terms if re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", text)
    ]
    return tuple(sorted(set(leaks)))
