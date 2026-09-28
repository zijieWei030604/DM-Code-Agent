"""Own the single background writer for the rebuildable trace projection."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path

from dm_agent.tracing.projection import TraceProjection

SYNC_INTERVAL_SECONDS = 2.0


class TraceProjectionWorker:
    """Keep the SQLite projection shortly behind append-only session logs."""

    def __init__(self, sessions_dir: Path, *, interval_seconds: float = SYNC_INTERVAL_SECONDS) -> None:
        self.sessions_dir = sessions_dir
        self.interval_seconds = interval_seconds
        self.projection = TraceProjection(sessions_dir / ".trace-index.sqlite3")
        self._stopped = asyncio.Event()

    async def run(self) -> None:
        while not self._stopped.is_set():
            await asyncio.to_thread(self.projection.sync_directory, self.sessions_dir)
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), timeout=self.interval_seconds)

    def stop(self) -> None:
        self._stopped.set()
