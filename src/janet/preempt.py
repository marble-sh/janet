"""Cooperative preemption for background work."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

StopCheck = Callable[[], bool]


class Preempted(Exception):
    """Raised when background work must yield to foreground work or a busy machine."""


def check(stop: StopCheck | None) -> None:
    if stop is not None and stop():
        raise Preempted


class StopFlag:
    """A thread-safe stop request, suitable for signal handlers."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    def __call__(self) -> bool:
        return self._event.is_set()


def file_stop(path: Path | None) -> StopCheck | None:
    """A stop check that trips when ``path`` exists (the daemon's yield signal)."""

    if path is None:
        return None
    expanded = path.expanduser()

    def stop() -> bool:
        return expanded.exists()

    return stop
