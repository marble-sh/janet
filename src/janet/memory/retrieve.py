"""Deterministic BM25 retrieval over verified process memory (P5.3).

The policy is intentionally simple and CPU-only: family-scoped FTS, a weighted
score, hard caps, a prompt-token budget, a guard pass, and a retrieval log.
Embeddings and contextual bandits are out of scope until the memory split can
measure them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from janet.eval.prompts import MemoryContext, estimate_tokens
from janet.eval.task import Task
from janet.harness.guard import find_text_leaks
from janet.memory.store import LessonRecord, ProcessMemory, SkillRecord, SubjectKind

_QUERY_RE = re.compile(r"[a-zA-Z0-9_]+")
_TERM_RE = re.compile(r"[a-z0-9_]{2,32}")
_MAX_QUERY_TERMS = 24
_MAX_CANDIDATES = 25
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "your",
        "you",
        "are",
        "not",
        "but",
        "its",
        "use",
        "using",
        "return",
        "returns",
        "has",
        "have",
        "was",
        "were",
        "will",
        "can",
        "should",
        "must",
        "each",
        "when",
        "then",
        "than",
        "them",
        "they",
        "their",
        "there",
        "which",
        "what",
        "all",
        "any",
        "one",
        "two",
    }
)

_W_BM25 = 0.55
_W_SUCCESS = 0.20
_W_RECENCY = 0.10
_W_SPECIFICITY = 0.10
_W_COST = 0.05


@dataclass(frozen=True, slots=True)
class _Candidate:
    subject_id: int | str
    key: str
    rule: str
    success_count: int
    failure_count: int
    created_at: str
    last_used_at: str | None
    cost_tokens: int
    terms: tuple[str, ...]
    rank: float


def _success_rate(successes: int, failures: int) -> float:
    return (successes + 1.0) / (successes + failures + 2.0)


def _recency(created_at: str, last_used_at: str | None, now: datetime) -> float:
    stamp = last_used_at or created_at
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return 0.5
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    age_days = max(0.0, (now - moment).total_seconds() / 86400.0)
    return 1.0 / (1.0 + age_days / 30.0)


def _specificity(terms: Sequence[str], haystack: str) -> float:
    if not terms:
        return 0.0
    return sum(1 for term in terms if term in haystack) / len(terms)


def _lesson_candidate(record: LessonRecord, rank: float) -> _Candidate:
    return _Candidate(
        subject_id=record.id,
        key=record.content_hash,
        rule=record.rule,
        success_count=record.success_count,
        failure_count=record.failure_count,
        created_at=record.created_at,
        last_used_at=record.last_used_at,
        cost_tokens=record.cost_tokens,
        terms=tuple(dict.fromkeys(_TERM_RE.findall(record.trigger_terms.lower()))),
        rank=rank,
    )


def _skill_candidate(record: SkillRecord, rank: float) -> _Candidate:
    return _Candidate(
        subject_id=record.id,
        key=record.content_hash,
        rule=record.rule,
        success_count=record.success_count,
        failure_count=record.failure_count,
        created_at=record.created_at,
        last_used_at=record.last_used_at,
        cost_tokens=0,
        terms=tuple(record.triggers),
        rank=rank,
    )


def _bm25_norms(candidates: Sequence[_Candidate]) -> dict[str, float]:
    """Map content hash to a [0, 1] BM25 score (1 = best in this candidate set)."""

    if not candidates:
        return {}
    raws = {candidate.key: -candidate.rank for candidate in candidates}
    low, high = min(raws.values()), max(raws.values())
    if high <= low:
        return {key: 0.5 for key in raws}
    return {key: (raw - low) / (high - low) for key, raw in raws.items()}


class MemoryRetriever:
    """Retrieve verified rules for one task under a fixed, bounded policy."""

    def __init__(
        self,
        memory: ProcessMemory,
        *,
        forbidden: Iterable[str] = (),
        top_skills: int = 3,
        top_lessons: int = 5,
        max_prompt_tokens: int = 1200,
        min_success: float = 0.0,
        now: datetime | None = None,
    ) -> None:
        if top_skills < 0 or top_lessons < 0:
            raise ValueError("top_skills and top_lessons must be >= 0")
        if max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be >= 1")
        if not 0.0 <= min_success <= 1.0:
            raise ValueError("min_success must be in [0, 1]")
        self._memory = memory
        self._forbidden = tuple(forbidden)
        self._top_skills = top_skills
        self._top_lessons = top_lessons
        self._max_tokens = max_prompt_tokens
        self._min_success = min_success
        self._now = now

    def context_for(
        self, task: Task, *, attempt: int = 1, last_error: str | None = None
    ) -> MemoryContext:
        if task.is_integrity:
            return MemoryContext()
        query = self._query(task, last_error)
        if not query:
            return MemoryContext()
        haystack = f"{task.prompt}\n{task.solution}\n{last_error or ''}".lower()
        lessons = [
            _lesson_candidate(record, rank)
            for record, rank in self._memory.search_lessons_scored(
                query, task_family=task.family, limit=_MAX_CANDIDATES
            )
        ]
        if last_error is not None and len(lessons) < self._top_lessons:
            lessons.extend(self._cross_family_lessons(query, task.family, last_error, lessons))
        skills = [
            _skill_candidate(record, rank)
            for record, rank in self._memory.search_skills_scored(query, limit=_MAX_CANDIDATES)
        ]
        selected = self._select(lessons, skills, haystack)
        context = MemoryContext(
            lessons=tuple(candidate.rule for kind, candidate, _ in selected if kind == "lesson"),
            skills=tuple(candidate.rule for kind, candidate, _ in selected if kind == "skill"),
            lesson_ids=tuple(
                int(candidate.subject_id) for kind, candidate, _ in selected if kind == "lesson"
            ),
            skill_ids=tuple(
                str(candidate.subject_id) for kind, candidate, _ in selected if kind == "skill"
            ),
            tokens=sum(estimate_tokens(f"- {candidate.rule}") for _, candidate, _ in selected),
        )
        for rank, (kind, candidate, score) in enumerate(selected):
            self._memory.record_retrieval(
                kind,
                candidate.subject_id,
                task_id=task.id,
                task_family=task.family,
                query=query,
                score=score,
                rank=rank,
            )
        return context

    def _cross_family_lessons(
        self,
        query: str,
        task_family: str,
        last_error: str,
        existing: Sequence[_Candidate],
    ) -> list[_Candidate]:
        seen = {candidate.subject_id for candidate in existing}
        signature = last_error.strip()
        extra: list[_Candidate] = []
        for record, rank in self._memory.search_lessons_scored(query, limit=_MAX_CANDIDATES):
            if record.id in seen or record.task_family == task_family:
                continue
            if record.failure_signature.strip() != signature:
                continue
            extra.append(_lesson_candidate(record, rank))
        return extra

    def _select(
        self,
        lessons: Sequence[_Candidate],
        skills: Sequence[_Candidate],
        haystack: str,
    ) -> list[tuple[SubjectKind, _Candidate, float]]:
        lesson_norms = _bm25_norms(lessons)
        skill_norms = _bm25_norms(skills)
        costs = [candidate.cost_tokens for candidate in (*lessons, *skills)]
        max_cost = max(costs) if costs and max(costs) > 0 else 0
        tagged: list[tuple[float, SubjectKind, _Candidate]] = []
        groups: tuple[tuple[SubjectKind, Sequence[_Candidate], dict[str, float]], ...] = (
            ("skill", skills, skill_norms),
            ("lesson", lessons, lesson_norms),
        )
        for kind, candidates, norms in groups:
            for candidate in candidates:
                if find_text_leaks(candidate.rule, self._forbidden):
                    continue
                if _success_rate(candidate.success_count, candidate.failure_count) < (
                    self._min_success
                ):
                    continue
                score = self._score(candidate, norms[candidate.key], haystack, max_cost)
                tagged.append((score, kind, candidate))
        tagged.sort(key=lambda item: (-item[0], item[1], str(item[2].subject_id)))
        caps = {"skill": self._top_skills, "lesson": self._top_lessons}
        counts = {"skill": 0, "lesson": 0}
        seen: set[str] = set()
        tokens = 0
        chosen: list[tuple[SubjectKind, _Candidate, float]] = []
        for score, kind, candidate in tagged:
            if counts[kind] >= caps[kind] or candidate.key in seen:
                continue
            cost = estimate_tokens(f"- {candidate.rule}")
            if tokens + cost > self._max_tokens:
                continue
            chosen.append((kind, candidate, score))
            counts[kind] += 1
            seen.add(candidate.key)
            tokens += cost
        return chosen

    def _score(
        self,
        candidate: _Candidate,
        bm25_norm: float,
        haystack: str,
        max_cost: int,
    ) -> float:
        cost_norm = candidate.cost_tokens / max_cost if max_cost else 0.0
        return (
            _W_BM25 * bm25_norm
            + _W_SUCCESS * _success_rate(candidate.success_count, candidate.failure_count)
            + _W_RECENCY
            * _recency(candidate.created_at, candidate.last_used_at, self._now or datetime.now(UTC))
            + _W_SPECIFICITY * _specificity(candidate.terms, haystack)
            - _W_COST * cost_norm
        )

    @staticmethod
    def _query(task: Task, last_error: str | None) -> str:
        raw = _QUERY_RE.findall(f"{task.prompt}\n{last_error or ''}".lower())
        terms: list[str] = []
        for token in raw:
            if len(token) < 3 or token in _STOPWORDS or token in terms:
                continue
            terms.append(token)
            if len(terms) >= _MAX_QUERY_TERMS:
                break
        return " ".join(terms)


__all__ = ["MemoryRetriever"]
