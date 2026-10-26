"""Append-only JSONL event log.

Every generation, evaluation, promotion, and integrity event is recorded here
before it can influence the archive. The log is the audit trail; SQLite is a
queryable index over the same facts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

_TS_FORMAT: Final = "%Y-%m-%dT%H:%M:%S.%fZ"


def utc_now() -> datetime:
    return datetime.now(UTC)


def format_ts(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(_TS_FORMAT)


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, _TS_FORMAT).replace(tzinfo=UTC)


@dataclass(slots=True)
class EventLog:
    path: Path
    run_id: str

    def __post_init__(self) -> None:
        self.path = self.path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, **payload: object) -> dict[str, object]:
        record: dict[str, object] = {
            "ts": format_ts(utc_now()),
            "run_id": self.run_id,
            "kind": kind,
            "payload": payload,
        }
        line = json.dumps(record, sort_keys=True, default=str)
        # O_APPEND keeps concurrent writers from interleaving partial lines.
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(descriptor, (line + "\n").encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return record

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
        return records
