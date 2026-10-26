from __future__ import annotations

from pathlib import Path

from janet.events import EventLog, format_ts, parse_ts, utc_now


def test_emit_and_read_round_trip(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl", run_id="run-0001")
    first = log.emit("generation.started", index=1)
    second = log.emit("generation.finished", index=1, score=0.5)
    assert first["run_id"] == "run-0001"
    assert second["payload"] == {"index": 1, "score": 0.5}
    records = log.read()
    assert [record["kind"] for record in records] == [
        "generation.started",
        "generation.finished",
    ]
    assert all(str(record["ts"]).endswith("Z") for record in records)


def test_read_missing_file_is_empty(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "absent.jsonl", run_id="run-0002")
    assert log.read() == []


def test_timestamp_round_trip() -> None:
    moment = utc_now()
    text = format_ts(moment)
    parsed = parse_ts(text)
    assert parsed.tzinfo is not None
    assert abs((parsed - moment).total_seconds()) < 0.001


def test_event_log_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "events.jsonl"
    log = EventLog(path, run_id="run-0003")
    log.emit("run.created")
    assert path.is_file()
