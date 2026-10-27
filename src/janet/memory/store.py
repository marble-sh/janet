"""SQLite-backed process memory for verified lessons and skills.

P5.1 is deliberately small: a durable store, FTS5 retrieval, provenance, and
status transitions. Admission, retrieval policy, consolidation, and the
amnesia test build on this module in later slices.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from janet.harness.skills import Skill, SkillError

SCHEMA_VERSION = 1
SubjectKind = Literal["lesson", "skill"]
StatusLike = str

_STATUSES = frozenset({"candidate", "verified", "quarantined", "retired"})
_TRANSITIONS: dict[str, frozenset[str]] = {
    "candidate": frozenset({"verified", "quarantined", "retired"}),
    "verified": frozenset({"quarantined", "retired"}),
    "quarantined": frozenset({"candidate", "retired"}),
    "retired": frozenset(),
}
_TRIGGER_RE = re.compile(r"^[a-z0-9_ ]{2,32}$")


class MemoryError(RuntimeError):
    """Raised when a memory record is malformed or a transition is illegal."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _decode(value: str | None) -> Any:
    if value is None:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise MemoryError(f"invalid stored JSON: {exc}") from exc


def _text(name: str, value: Any, *, min_len: int = 1, max_len: int = 4000) -> str:
    if not isinstance(value, str):
        raise MemoryError(f"{name} must be a string")
    text = value.strip()
    if not min_len <= len(text) <= max_len:
        raise MemoryError(f"{name} must be {min_len}..{max_len} characters")
    return text


def _status(value: str) -> str:
    if value not in _STATUSES:
        raise MemoryError(f"status must be one of {sorted(_STATUSES)}")
    return value


def _mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MemoryError(f"{name} must be an object")
    return value


def _triggers(value: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise MemoryError("triggers must be a list of strings")
    if not 1 <= len(value) <= 5:
        raise MemoryError("triggers must have 1..5 entries")
    normalized = tuple(str(item).strip().lower() for item in value)
    for trigger in normalized:
        if not _TRIGGER_RE.match(trigger):
            raise MemoryError(f"invalid trigger {trigger!r}: use 2..32 chars of a-z0-9_ ")
    return normalized


def _hash(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[a-zA-Z0-9_]+", query.lower())
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens)


@dataclass(frozen=True, slots=True)
class LessonRecord:
    id: int
    content_hash: str
    task_id: str
    task_family: str
    failure_signature: str
    trigger_terms: str
    rule: str
    patch_summary: str
    provenance: dict[str, Any]
    dev_gain: float | None
    anchor_delta: float | None
    fresh_delta: float | None
    success_count: int
    failure_count: int
    status: str
    created_at: str
    last_used_at: str | None
    cost_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "content_hash": self.content_hash,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "failure_signature": self.failure_signature,
            "trigger_terms": self.trigger_terms,
            "rule": self.rule,
            "patch_summary": self.patch_summary,
            "provenance": self.provenance,
            "dev_gain": self.dev_gain,
            "anchor_delta": self.anchor_delta,
            "fresh_delta": self.fresh_delta,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "status": self.status,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "cost_tokens": self.cost_tokens,
        }


@dataclass(frozen=True, slots=True)
class SkillRecord:
    id: str
    name: str
    triggers: tuple[str, ...]
    rule: str
    tests: Any
    provenance: dict[str, Any]
    evidence: dict[str, Any]
    status: str
    version: int
    parent_id: str | None
    content_hash: str
    created_at: str
    last_used_at: str | None
    success_count: int
    failure_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "triggers": list(self.triggers),
            "rule": self.rule,
            "tests": self.tests,
            "provenance": self.provenance,
            "evidence": self.evidence,
            "status": self.status,
            "version": self.version,
            "parent_id": self.parent_id,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    id: int
    subject_kind: str
    subject_id: str
    run_id: str
    variant_hash: str | None
    split: str
    verifier_version: str
    test_ids: tuple[str, ...]
    score_before: float | None
    score_after: float | None
    raw: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "variant_hash": self.variant_hash,
            "split": self.split,
            "verifier_version": self.verifier_version,
            "test_ids": list(self.test_ids),
            "score_before": self.score_before,
            "score_after": self.score_after,
            "raw": self.raw,
            "created_at": self.created_at,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL UNIQUE,
    task_id TEXT NOT NULL,
    task_family TEXT NOT NULL,
    failure_signature TEXT NOT NULL,
    trigger_terms TEXT NOT NULL,
    rule TEXT NOT NULL,
    patch_summary TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    dev_gain REAL,
    anchor_delta REAL,
    fresh_delta REAL,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'candidate',
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    cost_tokens INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    triggers_json TEXT NOT NULL,
    rule TEXT NOT NULL,
    tests_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate',
    version INTEGER NOT NULL DEFAULT 1,
    parent_id TEXT,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    variant_hash TEXT,
    split TEXT NOT NULL,
    verifier_version TEXT NOT NULL,
    test_ids_json TEXT NOT NULL,
    score_before REAL,
    score_after REAL,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrieval_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_family TEXT NOT NULL,
    query TEXT NOT NULL,
    score REAL NOT NULL,
    rank INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS lessons_status_idx ON lessons(status, task_family);
CREATE INDEX IF NOT EXISTS skills_status_idx ON skills(status);
CREATE INDEX IF NOT EXISTS evidence_subject_idx ON evidence(subject_kind, subject_id);
CREATE INDEX IF NOT EXISTS retrieval_subject_idx ON retrieval_events(subject_kind, subject_id);

CREATE VIRTUAL TABLE IF NOT EXISTS lesson_fts USING fts5(
    trigger_terms, rule, failure_signature,
    content='lessons', content_rowid='id'
);
CREATE VIRTUAL TABLE IF NOT EXISTS skill_fts USING fts5(
    name, triggers_json, rule,
    content='skills', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS lessons_ai AFTER INSERT ON lessons BEGIN
    INSERT INTO lesson_fts(rowid, trigger_terms, rule, failure_signature)
    VALUES (new.id, new.trigger_terms, new.rule, new.failure_signature);
END;
CREATE TRIGGER IF NOT EXISTS lessons_ad AFTER DELETE ON lessons BEGIN
    INSERT INTO lesson_fts(lesson_fts, rowid, trigger_terms, rule, failure_signature)
    VALUES ('delete', old.id, old.trigger_terms, old.rule, old.failure_signature);
END;
CREATE TRIGGER IF NOT EXISTS lessons_au
AFTER UPDATE OF trigger_terms, rule, failure_signature ON lessons BEGIN
    INSERT INTO lesson_fts(lesson_fts, rowid, trigger_terms, rule, failure_signature)
    VALUES ('delete', old.id, old.trigger_terms, old.rule, old.failure_signature);
    INSERT INTO lesson_fts(rowid, trigger_terms, rule, failure_signature)
    VALUES (new.id, new.trigger_terms, new.rule, new.failure_signature);
END;

CREATE TRIGGER IF NOT EXISTS skills_ai AFTER INSERT ON skills BEGIN
    INSERT INTO skill_fts(rowid, name, triggers_json, rule)
    VALUES (new.rowid, new.name, new.triggers_json, new.rule);
END;
CREATE TRIGGER IF NOT EXISTS skills_ad AFTER DELETE ON skills BEGIN
    INSERT INTO skill_fts(skill_fts, rowid, name, triggers_json, rule)
    VALUES ('delete', old.rowid, old.name, old.triggers_json, old.rule);
END;
CREATE TRIGGER IF NOT EXISTS skills_au
AFTER UPDATE OF name, triggers_json, rule ON skills BEGIN
    INSERT INTO skill_fts(skill_fts, rowid, name, triggers_json, rule)
    VALUES ('delete', old.rowid, old.name, old.triggers_json, old.rule);
    INSERT INTO skill_fts(rowid, name, triggers_json, rule)
    VALUES (new.rowid, new.name, new.triggers_json, new.rule);
END;
"""


class ProcessMemory:
    """A bounded, local SQLite memory store.

    The store is intentionally policy-free: it records candidates, verified
    entries, evidence, and retrieval events. Admission and consolidation decide
    what becomes verified.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        row = self._conn.execute(
            "SELECT value FROM memory_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO memory_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()
            self.rebuild_fts()
            return
        try:
            current = int(str(row["value"]))
        except ValueError as exc:
            raise MemoryError(f"invalid memory schema version {row['value']!r}") from exc
        if current != SCHEMA_VERSION:
            raise MemoryError(
                f"unsupported memory schema version {current}; expected {SCHEMA_VERSION}"
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ProcessMemory:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def rebuild_fts(self) -> None:
        for table in ("lesson_fts", "skill_fts"):
            self._conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
        self._conn.commit()

    # ---- lessons -----------------------------------------------------

    def add_lesson(
        self,
        *,
        task_id: str,
        task_family: str,
        failure_signature: str,
        trigger_terms: str,
        rule: str,
        patch_summary: str,
        provenance: dict[str, Any],
        dev_gain: float | None = None,
        anchor_delta: float | None = None,
        fresh_delta: float | None = None,
    ) -> int:
        task_id = _text("task_id", task_id)
        task_family = _text("task_family", task_family)
        failure_signature = _text("failure_signature", failure_signature)
        trigger_terms = _text("trigger_terms", trigger_terms)
        rule = _text("rule", rule, min_len=10, max_len=2000)
        patch_summary = _text("patch_summary", patch_summary, min_len=1, max_len=2000)
        provenance = _mapping("provenance", provenance)
        content_hash = _hash(
            task_id, task_family, failure_signature, trigger_terms, rule, patch_summary
        )
        self._conn.execute(
            """
            INSERT OR IGNORE INTO lessons
                (content_hash, task_id, task_family, failure_signature, trigger_terms,
                 rule, patch_summary, provenance_json, dev_gain, anchor_delta,
                 fresh_delta, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?)
            """,
            (
                content_hash,
                task_id,
                task_family,
                failure_signature,
                trigger_terms,
                rule,
                patch_summary,
                _json(provenance),
                dev_gain,
                anchor_delta,
                fresh_delta,
                _utc_now(),
            ),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id FROM lessons WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if row is None:  # pragma: no cover - defensive
            raise MemoryError("lesson insert failed")
        return int(row["id"])

    def get_lesson(self, lesson_id: int) -> LessonRecord | None:
        row = self._conn.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        return self._lesson(row) if row is not None else None

    def list_lessons(
        self, *, status: StatusLike = "verified", task_family: str | None = None, limit: int = 50
    ) -> tuple[LessonRecord, ...]:
        _status(status)
        if limit < 1:
            raise MemoryError("limit must be >= 1")
        sql = "SELECT * FROM lessons WHERE status = ?"
        params: list[Any] = [status]
        if task_family is not None:
            sql += " AND task_family = ?"
            params.append(task_family)
        sql += " ORDER BY success_count DESC, created_at ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return tuple(self._lesson(row) for row in rows)

    def search_lessons(
        self,
        query: str,
        *,
        task_family: str | None = None,
        status: StatusLike = "verified",
        limit: int = 5,
    ) -> tuple[LessonRecord, ...]:
        scored = self.search_lessons_scored(
            query, task_family=task_family, status=status, limit=limit
        )
        return tuple(record for record, _score in scored)

    def search_lessons_scored(
        self,
        query: str,
        *,
        task_family: str | None = None,
        status: StatusLike = "verified",
        limit: int = 25,
    ) -> tuple[tuple[LessonRecord, float], ...]:
        """FTS hits ordered by ``bm25`` (lower is better), with the raw score."""

        _status(status)
        if limit < 1:
            raise MemoryError("limit must be >= 1")
        fts = _fts_query(query)
        if not fts:
            return tuple(
                (record, 0.0)
                for record in self.list_lessons(status=status, task_family=task_family, limit=limit)
            )
        sql = """
            SELECT l.*, bm25(lesson_fts) AS rank
            FROM lesson_fts JOIN lessons l ON l.id = lesson_fts.rowid
            WHERE lesson_fts MATCH ? AND l.status = ?
        """
        params: list[Any] = [fts, status]
        if task_family is not None:
            sql += " AND l.task_family = ?"
            params.append(task_family)
        sql += " ORDER BY rank ASC, l.success_count DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return tuple((self._lesson(row), float(row["rank"])) for row in rows)

    # ---- skills ------------------------------------------------------

    def add_skill(
        self,
        *,
        name: str,
        triggers: list[str] | tuple[str, ...],
        rule: str,
        tests: Any,
        provenance: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        parent_id: str | None = None,
    ) -> str:
        try:
            skill = Skill(
                name=_text("skill name", name, max_len=48),
                triggers=_triggers(triggers),
                rule=_text("skill rule", rule, min_len=20, max_len=400),
            )
            skill.validate()
        except SkillError as exc:
            raise MemoryError(str(exc)) from exc
        if not isinstance(tests, (list, dict)):
            raise MemoryError("tests must be a list or object")
        provenance = _mapping("provenance", provenance)
        evidence = _mapping("evidence", evidence or {})
        content_hash = _hash(skill.name, skill.triggers, skill.rule, tests)
        self._conn.execute(
            """
            INSERT OR IGNORE INTO skills
                (id, name, triggers_json, rule, tests_json, provenance_json,
                 evidence_json, status, version, parent_id, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'candidate', 1, ?, ?, ?)
            """,
            (
                content_hash,
                skill.name,
                _json(list(skill.triggers)),
                skill.rule,
                _json(tests),
                _json(provenance),
                _json(evidence),
                parent_id,
                content_hash,
                _utc_now(),
            ),
        )
        self._conn.commit()
        return content_hash

    def get_skill(self, skill_id: str) -> SkillRecord | None:
        row = self._conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
        return self._skill(row) if row is not None else None

    def list_skills(
        self, *, status: StatusLike = "verified", limit: int = 50
    ) -> tuple[SkillRecord, ...]:
        _status(status)
        if limit < 1:
            raise MemoryError("limit must be >= 1")
        rows = self._conn.execute(
            "SELECT * FROM skills WHERE status = ? "
            "ORDER BY success_count DESC, created_at ASC LIMIT ?",
            (status, limit),
        ).fetchall()
        return tuple(self._skill(row) for row in rows)

    def search_skills(
        self, query: str, *, status: StatusLike = "verified", limit: int = 3
    ) -> tuple[SkillRecord, ...]:
        scored = self.search_skills_scored(query, status=status, limit=limit)
        return tuple(record for record, _score in scored)

    def search_skills_scored(
        self, query: str, *, status: StatusLike = "verified", limit: int = 25
    ) -> tuple[tuple[SkillRecord, float], ...]:
        """FTS hits ordered by ``bm25`` (lower is better), with the raw score."""

        _status(status)
        if limit < 1:
            raise MemoryError("limit must be >= 1")
        fts = _fts_query(query)
        if not fts:
            return tuple((record, 0.0) for record in self.list_skills(status=status, limit=limit))
        rows = self._conn.execute(
            """
            SELECT s.*, bm25(skill_fts) AS rank
            FROM skill_fts JOIN skills s ON s.rowid = skill_fts.rowid
            WHERE skill_fts MATCH ? AND s.status = ?
            ORDER BY rank ASC, s.success_count DESC LIMIT ?
            """,
            (fts, status, limit),
        ).fetchall()
        return tuple((self._skill(row), float(row["rank"])) for row in rows)

    # ---- lifecycle ---------------------------------------------------

    def set_status(
        self, subject_kind: SubjectKind, subject_id: int | str, status: StatusLike
    ) -> None:
        _status(status)
        table = self._table(subject_kind)
        row = self._conn.execute(
            f"SELECT status FROM {table} WHERE id = ?", (subject_id,)
        ).fetchone()
        if row is None:
            raise MemoryError(f"unknown {subject_kind}: {subject_id}")
        current = str(row["status"])
        if status == current:
            return
        if status not in _TRANSITIONS[current]:
            raise MemoryError(f"illegal {subject_kind} transition {current} -> {status}")
        self._conn.execute(f"UPDATE {table} SET status = ? WHERE id = ?", (status, subject_id))
        self._conn.commit()

    def record_outcome(
        self,
        subject_kind: SubjectKind,
        subject_id: int | str,
        *,
        success: bool,
        cost_tokens: int = 0,
    ) -> None:
        table = self._table(subject_kind)
        if cost_tokens < 0:
            raise MemoryError("cost_tokens must be >= 0")
        now = _utc_now()
        if subject_kind == "lesson":
            cursor = self._conn.execute(
                f"UPDATE {table} SET "
                + ("success_count" if success else "failure_count")
                + " = "
                + ("success_count" if success else "failure_count")
                + " + 1, cost_tokens = cost_tokens + ?, last_used_at = ? WHERE id = ?",
                (cost_tokens, now, subject_id),
            )
        else:
            cursor = self._conn.execute(
                f"UPDATE {table} SET "
                + ("success_count" if success else "failure_count")
                + " = "
                + ("success_count" if success else "failure_count")
                + " + 1, last_used_at = ? WHERE id = ?",
                (now, subject_id),
            )
        if cursor.rowcount == 0:
            raise MemoryError(f"unknown {subject_kind}: {subject_id}")
        self._conn.commit()

    def record_evidence(
        self,
        subject_kind: SubjectKind,
        subject_id: int | str,
        *,
        run_id: str,
        split: str,
        verifier_version: str,
        test_ids: list[str] | tuple[str, ...],
        raw: dict[str, Any],
        variant_hash: str | None = None,
        score_before: float | None = None,
        score_after: float | None = None,
    ) -> int:
        table = self._table(subject_kind)
        run_id = _text("run_id", run_id)
        split = _text("split", split)
        verifier_version = _text("verifier_version", verifier_version)
        raw = _mapping("raw", raw)
        if not isinstance(test_ids, (list, tuple)) or not all(
            isinstance(item, str) for item in test_ids
        ):
            raise MemoryError("test_ids must be a list of strings")
        exists = self._conn.execute(f"SELECT 1 FROM {table} WHERE id = ?", (subject_id,)).fetchone()
        if exists is None:
            raise MemoryError(f"unknown {subject_kind}: {subject_id}")
        cursor = self._conn.execute(
            """
            INSERT INTO evidence
                (subject_kind, subject_id, run_id, variant_hash, split,
                 verifier_version, test_ids_json, score_before, score_after,
                 raw_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject_kind,
                str(subject_id),
                run_id,
                variant_hash,
                split,
                verifier_version,
                _json(list(test_ids)),
                score_before,
                score_after,
                _json(raw),
                _utc_now(),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def list_evidence(
        self, subject_kind: SubjectKind, subject_id: int | str, *, limit: int = 20
    ) -> tuple[EvidenceRecord, ...]:
        table = self._table(subject_kind)
        if limit < 1:
            raise MemoryError("limit must be >= 1")
        exists = self._conn.execute(f"SELECT 1 FROM {table} WHERE id = ?", (subject_id,)).fetchone()
        if exists is None:
            raise MemoryError(f"unknown {subject_kind}: {subject_id}")
        rows = self._conn.execute(
            "SELECT * FROM evidence WHERE subject_kind = ? AND subject_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (subject_kind, str(subject_id), limit),
        ).fetchall()
        return tuple(self._evidence(row) for row in rows)

    def record_retrieval(
        self,
        subject_kind: SubjectKind,
        subject_id: int | str,
        *,
        task_id: str,
        task_family: str,
        query: str,
        score: float,
        rank: int,
    ) -> None:
        self._table(subject_kind)
        if rank < 0:
            raise MemoryError("rank must be >= 0")
        self._conn.execute(
            """
            INSERT INTO retrieval_events
                (subject_kind, subject_id, task_id, task_family, query, score, rank, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject_kind,
                str(subject_id),
                _text("task_id", task_id),
                _text("task_family", task_family),
                _text("query", query, min_len=0),
                float(score),
                rank,
                _utc_now(),
            ),
        )
        self._conn.commit()

    def stats(self) -> dict[str, Any]:
        lessons = {
            str(row["status"]): int(row["n"])
            for row in self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM lessons GROUP BY status"
            )
        }
        skills = {
            str(row["status"]): int(row["n"])
            for row in self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM skills GROUP BY status"
            )
        }
        evidence_row = self._conn.execute("SELECT COUNT(*) AS n FROM evidence").fetchone()
        retrieval_row = self._conn.execute("SELECT COUNT(*) AS n FROM retrieval_events").fetchone()
        return {
            "path": str(self.path),
            "schema_version": SCHEMA_VERSION,
            "lessons": lessons,
            "skills": skills,
            "evidence": int(evidence_row["n"]) if evidence_row else 0,
            "retrieval_events": int(retrieval_row["n"]) if retrieval_row else 0,
        }

    def health(self) -> dict[str, Any]:
        integrity_row = self._conn.execute("PRAGMA integrity_check").fetchone()
        integrity = str(integrity_row[0]) if integrity_row is not None else "unknown"
        indexes: dict[str, dict[str, Any]] = {}
        for content, table in (("lessons", "lesson_fts"), ("skills", "skill_fts")):
            content_row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {content}").fetchone()
            index_row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            content_n = int(content_row["n"]) if content_row is not None else 0
            index_n = int(index_row["n"]) if index_row is not None else 0
            indexes[content] = {"content": content_n, "index": index_n, "ok": content_n == index_n}
        return {
            "path": str(self.path),
            "schema_version": SCHEMA_VERSION,
            "integrity": integrity,
            "indexes": indexes,
            "ok": integrity == "ok" and all(entry["ok"] for entry in indexes.values()),
        }

    # ---- internals ---------------------------------------------------

    @staticmethod
    def _table(subject_kind: SubjectKind) -> str:
        if subject_kind == "lesson":
            return "lessons"
        if subject_kind == "skill":
            return "skills"
        raise MemoryError("subject_kind must be 'lesson' or 'skill'")

    @staticmethod
    def _lesson(row: sqlite3.Row) -> LessonRecord:
        return LessonRecord(
            id=int(row["id"]),
            content_hash=str(row["content_hash"]),
            task_id=str(row["task_id"]),
            task_family=str(row["task_family"]),
            failure_signature=str(row["failure_signature"]),
            trigger_terms=str(row["trigger_terms"]),
            rule=str(row["rule"]),
            patch_summary=str(row["patch_summary"]),
            provenance=_mapping("provenance", _decode(str(row["provenance_json"]))),
            dev_gain=float(row["dev_gain"]) if row["dev_gain"] is not None else None,
            anchor_delta=float(row["anchor_delta"]) if row["anchor_delta"] is not None else None,
            fresh_delta=float(row["fresh_delta"]) if row["fresh_delta"] is not None else None,
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            last_used_at=str(row["last_used_at"]) if row["last_used_at"] is not None else None,
            cost_tokens=int(row["cost_tokens"]),
        )

    @staticmethod
    def _skill(row: sqlite3.Row) -> SkillRecord:
        triggers = _decode(str(row["triggers_json"]))
        if not isinstance(triggers, list):
            raise MemoryError("stored skill triggers are not a list")
        return SkillRecord(
            id=str(row["id"]),
            name=str(row["name"]),
            triggers=tuple(str(item) for item in triggers),
            rule=str(row["rule"]),
            tests=_decode(str(row["tests_json"])),
            provenance=_mapping("provenance", _decode(str(row["provenance_json"]))),
            evidence=_mapping("evidence", _decode(str(row["evidence_json"]))),
            status=str(row["status"]),
            version=int(row["version"]),
            parent_id=str(row["parent_id"]) if row["parent_id"] is not None else None,
            content_hash=str(row["content_hash"]),
            created_at=str(row["created_at"]),
            last_used_at=str(row["last_used_at"]) if row["last_used_at"] is not None else None,
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
        )

    @staticmethod
    def _evidence(row: sqlite3.Row) -> EvidenceRecord:
        test_ids = _decode(str(row["test_ids_json"]))
        if not isinstance(test_ids, list):
            raise MemoryError("stored evidence test ids are not a list")
        return EvidenceRecord(
            id=int(row["id"]),
            subject_kind=str(row["subject_kind"]),
            subject_id=str(row["subject_id"]),
            run_id=str(row["run_id"]),
            variant_hash=str(row["variant_hash"]) if row["variant_hash"] is not None else None,
            split=str(row["split"]),
            verifier_version=str(row["verifier_version"]),
            test_ids=tuple(str(item) for item in test_ids),
            score_before=float(row["score_before"]) if row["score_before"] is not None else None,
            score_after=float(row["score_after"]) if row["score_after"] is not None else None,
            raw=_mapping("raw", _decode(str(row["raw_json"]))),
            created_at=str(row["created_at"]),
        )
