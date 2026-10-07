"""Ranking adapted from Hermes-LCM; see docs/lcm-search-ranking.md."""

from __future__ import annotations

from typing import Any

from .search_query import (
    AGE_DECAY_RATE,
    compute_directness_score,
    contains_risky_fts_ascii,
    count_term_matches,
    extract_quoted_phrases,
    extract_search_terms,
    should_apply_directness_rank_adjustment,
)


def prepare_rank(row: dict[str, Any], query: str, safe_query: str, *, fts: bool) -> None:
    """Upstream message/summary directness and rank adjustment constants."""
    terms = extract_search_terms(safe_query)
    phrases = extract_quoted_phrases(safe_query)
    summary = row["kind"] == "summary"
    role = row["metadata"].get("role")
    directness = compute_directness_score(row["body"], terms, phrases)
    if not summary and role == "tool" and row["body"].lstrip().startswith(("{", "[")):
        directness -= 4.0
    rank = (
        float(row["search_rank"])
        if fts
        else -float(
            sum(
                (
                    min(count_term_matches(row["body"], term), 1)
                    if contains_risky_fts_ascii(query)
                    else count_term_matches(row["body"], term)
                )
                for term in terms
            )
        )
    )
    if fts and should_apply_directness_rank_adjustment(terms, phrases):
        rank -= max(directness, 0.0) * (2e-7 if summary else 3e-7)
    row.update(
        _sort_rank=rank,
        _sort_directness=directness,
        _sort_ts=float(row["latest_at"] or row["created_at"] or 0.0),
        type="summary" if summary else "message",
        role=role,
        ranking_backend="fts5" if fts else "like",
    )


def combined_sort_key(result: dict[str, Any], sort: str, now: float) -> tuple[Any, ...]:
    """Upstream combined key, with one fixed clock value per query."""
    timestamp = float(result.get("_sort_ts") or 0.0)
    rank = result.get("_sort_rank")
    rank_value = float(rank) if rank is not None else float("inf")
    directness = float(result.get("_sort_directness") or 0.0)
    type_bias = 0 if result.get("type") == "message" else 1
    rank_tier = 1 if result.get("type") == "externalized" else 0
    role_bias = {"user": 0, "assistant": 1, "tool": 2}.get(str(result.get("role")), 1)
    effective = directness if result.get("type") == "message" else directness * 0.8
    if sort == "relevance":
        return (rank_tier, rank_value, -effective, role_bias, -timestamp, type_bias)
    if sort == "hybrid":
        age_hours = max(0.0, (now - timestamp) / 3600.0)
        blended = (
            rank_value / (1 + age_hours * AGE_DECAY_RATE) if rank is not None else float("inf")
        )
        return (
            -int(result.get("_hybrid_summary_override") or 0),
            rank_tier,
            blended,
            -effective,
            role_bias,
            -timestamp,
            type_bias,
        )
    if result.get("type") == "message":
        return (rank_tier, -timestamp, type_bias, role_bias, rank_value, 0.0, float("inf"))
    return (rank_tier, -timestamp, type_bias, 0, rank_value, 0.0, role_bias)


def rank_results(rows: list[dict[str, Any]], sort: str, *, now: float) -> None:
    if sort == "hybrid":
        maximum = max(
            (row["_sort_directness"] for row in rows if row["type"] == "message"), default=0.0
        )
        for row in rows:
            if row["type"] == "summary":
                row["_hybrid_summary_override"] = int(row["_sort_directness"] >= maximum + 8.0)
    rows.sort(key=lambda row: combined_sort_key(row, sort, now))
