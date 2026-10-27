"""SQLite index + git artifact store for harness variants."""

from __future__ import annotations

import json
import random
import re
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
    operator: str | None = None
    parent2_hash: str | None = None
    cost: float = 0.0
    novelty: float = 0.0
    coverage_cell: str | None = None
    frontier_status: str = "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "hash": self.hash,
            "name": self.name,
            "parent_hash": self.parent_hash,
            "parent2_hash": self.parent2_hash,
            "dev_score": self.dev_score,
            "anchor_score": self.anchor_score,
            "status": self.status,
            "created_at": self.created_at,
            "mutation_note": self.spec.mutation_note,
            "operator": self.operator,
            "cost": round(self.cost, 3),
            "novelty": round(self.novelty, 6),
            "coverage_cell": self.coverage_cell,
            "frontier_status": self.frontier_status,
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

_EXTRA_COLUMNS = (
    ("operator", "TEXT"),
    ("parent2_hash", "TEXT"),
    ("cost", "REAL"),
    ("novelty", "REAL"),
    ("coverage_cell", "TEXT"),
    ("frontier_status", "TEXT"),
)
_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}")
_ELIGIBLE = "(status IN ('accepted', 'promoted') OR parent_hash IS NULL)"


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
        self._migrate()
        self._conn.commit()
        self._init_git()
        self._refresh_frontier()
        self._conn.commit()

    def _migrate(self) -> None:
        columns = {str(row["name"]) for row in self._conn.execute("PRAGMA table_info(variants)")}
        for name, kind in _EXTRA_COLUMNS:
            if name not in columns:
                self._conn.execute(f"ALTER TABLE variants ADD COLUMN {name} {kind}")

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
        operator: str | None = None,
        parent2_hash: str | None = None,
        cost: float = 0.0,
        coverage_cell: str | None = None,
    ) -> str:
        spec.validate()
        if cost < 0.0:
            raise ArchiveError("cost must be >= 0")
        novelty = self._novelty(spec)
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
                 created_at, details_json, operator, parent2_hash, cost, novelty,
                 coverage_cell, frontier_status)
            VALUES (?, ?, ?, ?, ?, ?, 'candidate', ?, ?, ?, ?, ?, ?, ?, 'none')
            ON CONFLICT(hash) DO UPDATE SET
                dev_score = excluded.dev_score,
                anchor_score = excluded.anchor_score,
                details_json = excluded.details_json,
                operator = excluded.operator,
                parent2_hash = excluded.parent2_hash,
                cost = excluded.cost,
                novelty = excluded.novelty,
                coverage_cell = excluded.coverage_cell
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
                operator,
                parent2_hash,
                cost,
                novelty,
                coverage_cell,
            ),
        )
        self._refresh_frontier()
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
            self._refresh_frontier()
            self._conn.commit()
        return self.get(variant_hash) or record

    def frontier(self) -> tuple[HarnessRecord, ...]:
        """One elite per occupied MAP-Elites cell, strongest capability first."""

        rows = self._conn.execute(
            "SELECT * FROM variants WHERE frontier_status = 'elite' "
            "ORDER BY anchor_score DESC, dev_score DESC, hash ASC"
        ).fetchall()
        return tuple(self._record(row) for row in rows)

    def select_parent(self, generation: int, *, explore_ratio: float = 0.3) -> HarnessRecord | None:
        """Pick a parent from the frontier: exploit the best cells, or explore.

        Deterministic for a given generation: the same archive and generation
        always select the same parent, which keeps replay and tests honest.
        """

        if not 0.0 <= explore_ratio <= 1.0:
            raise ArchiveError("explore_ratio must be in [0, 1]")
        elites = self.frontier()
        if not elites:
            return self.current() or self.seed()
        rng = random.Random(generation)
        if rng.random() >= explore_ratio:
            top = sorted(elites, key=self._capability, reverse=True)[:3]
            return rng.choice(top)
        weights = [max(0.01, record.novelty) for record in elites]
        return rng.choices(elites, weights=weights, k=1)[0]

    def _refresh_frontier(self) -> None:
        rows = self._conn.execute(f"SELECT * FROM variants WHERE {_ELIGIBLE}").fetchall()
        groups: dict[str, list[HarnessRecord]] = {}
        for row in rows:
            record = self._record(row)
            groups.setdefault(record.coverage_cell or "unknown", []).append(record)
        elite = {max(members, key=self._capability).hash for members in groups.values()}
        self._conn.execute("UPDATE variants SET frontier_status = 'none'")
        for variant_hash in sorted(elite):
            self._conn.execute(
                "UPDATE variants SET frontier_status = 'elite' WHERE hash = ?",
                (variant_hash,),
            )
        for members in groups.values():
            for record in members:
                if record.hash not in elite:
                    self._conn.execute(
                        "UPDATE variants SET frontier_status = 'member' WHERE hash = ?",
                        (record.hash,),
                    )

    @staticmethod
    def _capability(record: HarnessRecord) -> tuple[float, float, float, str]:
        anchor = record.anchor_score if record.anchor_score is not None else -1.0
        dev = record.dev_score if record.dev_score is not None else -1.0
        return (anchor, dev, record.novelty, record.created_at)

    def _novelty(self, spec: HarnessSpec) -> float:
        """1 - the closest token Jaccard to any existing variant."""

        tokens = _spec_tokens(spec)
        similarities: list[float] = []
        for row in self._conn.execute("SELECT spec_json FROM variants"):
            try:
                other = HarnessSpec.from_dict(json.loads(str(row["spec_json"])))
            except (json.JSONDecodeError, ValueError):
                continue
            if other.hash == spec.hash:
                continue
            similarities.append(_jaccard(tokens, _spec_tokens(other)))
        return 1.0 - max(similarities) if similarities else 1.0

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
        self._refresh_frontier()
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
        self._refresh_frontier()
        self._conn.commit()
        updated = self.get(variant_hash)
        if updated is None:  # pragma: no cover - defensive
            raise ArchiveError("variant vanished during update")
        return updated

    def record_confirmation(self, variant_hash: str, payload: dict[str, Any]) -> HarnessRecord:
        """Attach a fresh-split confirmation to the variant for reporting."""

        record = self.get(variant_hash)
        if record is None:
            raise ArchiveError(f"unknown variant: {variant_hash}")
        details = {**record.details, "fresh": payload}
        self._conn.execute(
            "UPDATE variants SET details_json = ? WHERE hash = ?",
            (json.dumps(details, sort_keys=True, default=str), variant_hash),
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
        coverage_cell: str | None = None,
        cost: float = 0.0,
    ) -> HarnessRecord:
        record = self.get(spec.hash)
        if record is not None:
            return record
        self.add(
            spec,
            dev_score=dev_score,
            anchor_score=anchor_score,
            details=details,
            coverage_cell=coverage_cell,
            cost=cost,
        )
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
            operator=str(row["operator"]) if row["operator"] is not None else None,
            parent2_hash=str(row["parent2_hash"]) if row["parent2_hash"] is not None else None,
            cost=float(row["cost"]) if row["cost"] is not None else 0.0,
            novelty=float(row["novelty"]) if row["novelty"] is not None else 0.0,
            coverage_cell=(str(row["coverage_cell"]) if row["coverage_cell"] is not None else None),
            frontier_status=str(row["frontier_status"] or "none"),
        )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _spec_tokens(spec: HarnessSpec) -> frozenset[str]:
    parts = [spec.system_prompt]
    for skill in spec.skills:
        parts.extend((skill.name, *skill.triggers, skill.rule))
    return frozenset(_TOKEN_RE.findall("\n".join(parts).lower()))
