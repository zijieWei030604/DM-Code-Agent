"""Cooperative stop boundaries for synchronous, in-process Explore sessions."""

from __future__ import annotations

import threading
import time


class ExplorationStopped(BaseException):
    """Control flow, deliberately not swallowed by tool/hook Exception handlers."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


class RunControl:
    def __init__(self, cancelled: threading.Event, timeout: float) -> None:
        self.cancelled = cancelled
        self.deadline = time.monotonic() + timeout

    def check(self) -> None:
        if self.cancelled.is_set():
            raise ExplorationStopped("cancelled")
        if time.monotonic() >= self.deadline:
            raise ExplorationStopped("timed_out")

    def request_timeout(self) -> float:
        self.check()
        return min(30.0, max(0.001, self.deadline - time.monotonic()))
