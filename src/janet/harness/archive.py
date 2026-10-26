"""SQLite index + git artifact store for harness variants."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from janet.harness.spec import HarnessSpec


class ArchiveError(RuntimeError):
    """Raised when the archive cannot record or retrieve a variant."""


@dataclass(frozen=True, slots=True)
class HarnessRecord:
    hash: str
    name: str
    parent_hash: str | None
    spec: HarnessSpec
    dev_score: float | None
    anchor_score: float | None
    status: str
    created_at: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "hash": self.hash,
            "name": self.name,
            "parent_hash": self.parent_hash,
            "dev_score": self.dev_score,
            "anchor_score": self.anchor_score,
            "status": self.status,
            "created_at": self.created_at,
            "mutation_note": self.spec.mutation_note,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS variants (
    hash TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_hash TEXT,
    spec_json TEXT NOT NULL,
    dev_score REAL,
    anchor_score REAL,
    status TEXT NOT NULL DEFAULT 'candidate',
    created_at TEXT NOT NULL,
    details_json TEXT
);
"""


class Archive:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "archive.sqlite3"
        self.repo_dir = self.root / "variants"
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._init_git()

    def _init_git(self) -> None:
        if not (self.repo_dir / ".git").exists():
            self._git("init")
        self._git("config", "user.name", "Janet")
        self._git("config", "user.email", "janet@localhost")

    def _git(self, *args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(self.repo_dir), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ArchiveError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
        return completed.stdout

    def _commit(self, message: str) -> None:
        self._git("add", "-A")
        staged = subprocess.run(
            ["git", "-C", str(self.repo_dir), "diff", "--cached", "--quiet"],
            check=False,
        )
        if staged.returncode == 0:
            return
        self._git("commit", "-m", message)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Archive:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def add(
        self,
        spec: HarnessSpec,
        *,
        dev_score: float | None,
        anchor_score: float | None,
        details: dict[str, Any] | None = None,
    ) -> str:
        spec.validate()
        variant_dir = self.repo_dir / spec.hash
        variant_dir.mkdir(exist_ok=True)
        (variant_dir / "harness.json").write_text(spec.to_json(), encoding="utf-8")
        score_label = f"{dev_score:.3f}" if dev_score is not None else "na"
        self._commit(f"variant {spec.hash[:12]} {spec.name} dev={score_label}")
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO variants
                (hash, name, parent_hash, spec_json, dev_score, anchor_score, status,
                 created_at, details_json)
            VALUES (?, ?, ?, ?, ?, ?, 'candidate', ?, ?)
            ON CONFLICT(hash) DO UPDATE SET
                dev_score = excluded.dev_score,
                anchor_score = excluded.anchor_score,
                details_json = excluded.details_json
            """,
            (
                spec.hash,
                spec.name,
                spec.parent,
                spec.to_json(),
                dev_score,
                anchor_score,
                now,
                json.dumps(details or {}, sort_keys=True, default=str),
            ),
        )
        self._conn.commit()
        return spec.hash

    def get(self, variant_hash: str) -> HarnessRecord | None:
        row = self._conn.execute(
            "SELECT * FROM variants WHERE hash = ?", (variant_hash,)
        ).fetchone()
        return self._record(row) if row is not None else None

    def best(self) -> HarnessRecord | None:
        row = self._conn.execute(
            """
            SELECT * FROM variants
            WHERE dev_score IS NOT NULL
            ORDER BY dev_score DESC, anchor_score DESC, created_at ASC
            LIMIT 1
            """
        ).fetchone()
        return self._record(row) if row is not None else None

    def seed(self) -> HarnessRecord | None:
        """The seed: the earliest variant with no parent."""

        row = self._conn.execute(
            "SELECT * FROM variants WHERE parent_hash IS NULL ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        return self._record(row) if row is not None else None

    def current(self) -> HarnessRecord | None:
        """The parent for the next generation: latest accepted variant, else the seed.

        Accepted means it passed the paired Pareto gate against its own parent, so
        lineage moves forward only on verified improvement (greedy hill-climb).
        """

        row = self._conn.execute(
            "SELECT * FROM variants WHERE status IN ('accepted', 'promoted') "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return self._record(row) if row is not None else self.seed()

    def accept(self, variant_hash: str) -> HarnessRecord:
        record = self.get(variant_hash)
        if record is None:
            raise ArchiveError(f"unknown variant: {variant_hash}")
        if record.status == "candidate":
            self._conn.execute(
                "UPDATE variants SET status = 'accepted' WHERE hash = ?", (variant_hash,)
            )
            self._conn.commit()
        return self.get(variant_hash) or record

    def heldout_points(self) -> list[tuple[float, float]]:
        """(dev, held-out anchor) per evaluated variant, oldest first."""

        rows = self._conn.execute(
            "SELECT dev_score, details_json FROM variants "
            "WHERE dev_score IS NOT NULL ORDER BY created_at ASC"
        ).fetchall()
        points: list[tuple[float, float]] = []
        for row in rows:
            held = json.loads(str(row["details_json"] or "{}")).get("anchor_heldout")
            if isinstance(held, (int, float)):
                points.append((float(row["dev_score"]), float(held)))
        return points

    def lineage(self, variant_hash: str) -> tuple[HarnessRecord, ...]:
        """Root-to-variant chain (cycle-safe)."""

        chain: list[HarnessRecord] = []
        seen: set[str] = set()
        cursor: str | None = variant_hash
        while cursor is not None and cursor not in seen:
            seen.add(cursor)
            record = self.get(cursor)
            if record is None:
                break
            chain.append(record)
            cursor = record.parent_hash
        return tuple(reversed(chain))

    def list(self, limit: int = 50) -> tuple[HarnessRecord, ...]:
        rows = self._conn.execute(
            "SELECT * FROM variants ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return tuple(self._record(row) for row in rows)

    def promote(self, variant_hash: str) -> HarnessRecord:
        record = self.get(variant_hash)
        if record is None:
            raise ArchiveError(f"unknown variant: {variant_hash}")
        self._conn.execute(
            "UPDATE variants SET status = 'promoted' WHERE hash = ?", (variant_hash,)
        )
        self._conn.commit()
        current = self.repo_dir / "current.json"
        current.write_text(record.spec.to_json(), encoding="utf-8")
        self._commit(f"promote {variant_hash[:12]}")
        return self.get(variant_hash) or record

    def record_dev_sample(
        self, variant_hash: str, score: float, suite: str | None = None
    ) -> HarnessRecord:
        """Fold a fresh dev measurement into the variant's running mean.

        Dev scores are noisy (sampled repair); a single measurement makes the
        archive select on luck. The stored dev_score is the mean of all samples.
        """

        record = self.get(variant_hash)
        if record is None:
            raise ArchiveError(f"unknown variant: {variant_hash}")
        samples = list(record.details.get("dev_samples") or [])
        if suite is not None and record.details.get("dev_suite", "") != suite:
            samples = []  # the evaluator changed: older samples measured a different test
        elif not samples and record.dev_score is not None:
            samples = [record.dev_score]
        samples.append(score)
        details = {**record.details, "dev_samples": samples}
        if suite is not None:
            details["dev_suite"] = suite
        self._conn.execute(
            "UPDATE variants SET dev_score = ?, details_json = ? WHERE hash = ?",
            (
                sum(samples) / len(samples),
                json.dumps(details, sort_keys=True, default=str),
                variant_hash,
            ),
        )
        self._conn.commit()
        updated = self.get(variant_hash)
        if updated is None:  # pragma: no cover - defensive
            raise ArchiveError("variant vanished during update")
        return updated

    def rollback(self) -> HarnessSpec | None:
        current = self.repo_dir / "current.json"
        if not current.is_file():
            return None
        return HarnessSpec.from_dict(json.loads(current.read_text(encoding="utf-8")))

    def ensure_seed(
        self,
        spec: HarnessSpec,
        *,
        dev_score: float,
        anchor_score: float | None,
        details: dict[str, Any] | None = None,
    ) -> HarnessRecord:
        record = self.get(spec.hash)
        if record is not None:
            return record
        self.add(spec, dev_score=dev_score, anchor_score=anchor_score, details=details)
        created = self.get(spec.hash)
        if created is None:  # pragma: no cover - defensive
            raise ArchiveError("failed to create the seed record")
        return created

    def _record(self, row: sqlite3.Row) -> HarnessRecord:
        spec_data = json.loads(str(row["spec_json"]))
        return HarnessRecord(
            hash=str(row["hash"]),
            name=str(row["name"]),
            parent_hash=str(row["parent_hash"]) if row["parent_hash"] else None,
            spec=HarnessSpec.from_dict(spec_data),
            dev_score=float(row["dev_score"]) if row["dev_score"] is not None else None,
            anchor_score=(float(row["anchor_score"]) if row["anchor_score"] is not None else None),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            details=json.loads(str(row["details_json"] or "{}")),
        )
