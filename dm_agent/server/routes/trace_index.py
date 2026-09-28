"""Read-only API for the SQLite projection of session JSONL files."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request

from dm_agent.server.projection import TraceProjectionWorker
from dm_agent.server.security import require_token

router = APIRouter(
    prefix="/api/trace-index", tags=["trace-index"], dependencies=[Depends(require_token)]
)


def _worker(request: Request) -> TraceProjectionWorker:
    return request.app.state.trace_projection


@router.get("/runs", summary="查询运行投影")
def list_runs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    """Return cross-session run summaries from the rebuildable SQLite projection."""
    return {"runs": _worker(request).projection.list_runs(limit=limit)}


@router.get("/tools", summary="查询工具调用统计")
def tool_statistics(request: Request) -> dict[str, Any]:
    return {"tools": _worker(request).projection.tool_failure_counts()}
