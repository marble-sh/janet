from __future__ import annotations

from pathlib import Path

import pytest

from janet.preempt import Preempted, StopFlag, check, file_stop


def test_check_accepts_no_check_and_a_false_check() -> None:
    check(None)
    check(lambda: False)


def test_check_raises_when_requested() -> None:
    with pytest.raises(Preempted):
        check(lambda: True)


def test_stop_flag_request_and_clear() -> None:
    flag = StopFlag()
    assert flag() is False
    flag.request()
    assert flag() is True
    flag.clear()
    assert flag() is False


def test_file_stop_is_absent_or_none(tmp_path: Path) -> None:
    assert file_stop(None) is None
    stop = file_stop(tmp_path / "stop")
    assert stop is not None
    assert stop() is False
    (tmp_path / "stop").write_text("stop", encoding="utf-8")
    assert stop() is True
