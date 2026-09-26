"""Offline contracts for the staged LCM core, independent of Runtime wiring."""

from __future__ import annotations

import json
import sqlite3

import pytest

from dm_agent.memory.lcm.compaction import (
    CompactionPolicy,
    Compactor,
    ContextOverflow,
    SummaryResponse,
)
from dm_agent.memory.lcm.dag import SummaryDAG
from dm_agent.memory.lcm.fresh_tail import fresh_tail_start, safe_boundaries
from dm_agent.memory.lcm.store import LCMStore
from dm_agent.memory.lcm.tools import build_lcm_tools


@pytest.fixture
def store(tmp_path):
    value = LCMStore(tmp_path / "memory.sqlite3")
    yield value
    value.close()


def test_store_idempotent_and_immutable(store):
    branch = store.create_branch()
    record = store.append(branch, "event-1", "original")
    assert store.append(branch, "event-1", "original") == record
    with pytest.raises(ValueError, match="overwritten"):
        store.append(branch, "event-1", "replacement")
    assert store.get(branch, record)["body"] == "original"


def test_branch_excludes_future_parent_and_siblings(store):
    parent = store.create_branch()
    old = store.append(parent, "old", "visible needle")
    fork = store.create_branch(parent=parent)
    future = store.append(parent, "future", "future needle")
    sibling = store.create_branch(parent=parent)
    sibling_id = store.append(sibling, "other", "other needle")
    for forbidden in (future, sibling_id):
        with pytest.raises(ValueError, match="not visible"):
            store.get(fork, forbidden)
    assert [row["id"] for row in store.search(fork, "needle")] == [old]
    child = store.create_branch(parent=fork)
    with pytest.raises(ValueError, match="not visible"):
        store.get(child, future)


def test_summary_edges_are_atomic_and_do_not_delete_sources(store):
    branch = store.create_branch()
    source = store.append(branch, "source", "original content")
    head = store.head()
    with pytest.raises(ValueError):
        store.add_summary(branch, "invalid", [source, "missing"], metadata={})
    assert store.head() == head
    node = store.add_summary(branch, "short", [source], metadata={"depth": 0})
    parent = store.add_summary(branch, "shorter", [node], metadata={"depth": 1})
    dag = SummaryDAG(store, branch)
    assert dag.describe(parent)["sources"] == [node]
    assert dag.expand(node)["sources"] == [source]
    assert dag.expand(source)["content"] == "original content"


def test_legacy_database_is_rejected_without_modification(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE lcm_records (id TEXT)")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="Legacy LCM database"):
        LCMStore(path)


def test_compaction_persists_active_frontier_with_summary_sources(store):
    store = LCMStore(":memory:")
    try:
        compactor, ids = make_compactor(
            store,
            lambda *_: SummaryResponse("Historical observation.", {}),
            max_batches=1,
        )
        compactor.compact(ids, token_budget=compactor.tokens(ids) - 20)
        assert store.frontier(compactor.branch) == ids
        assert store.get(compactor.branch, ids[0])["kind"] == "summary"
    finally:
        store.close()


@pytest.mark.parametrize("fts_enabled", [True, False])
def test_search_literal_query_and_fallback(store, fts_enabled):
    store.fts_enabled = fts_enabled and store.fts_enabled
    branch = store.create_branch()
    record = store.append(branch, "one", "hello literal_百分比%世界")
    store.append(branch, "two", "unrelated")
    assert store.search(branch, "literal_百分比%世界")[0]["id"] == record
    assert store.search(branch, "!!!") == []
    assert store.search(branch, '" OR )') == []


def test_paginated_expand(store):
    branch = store.create_branch()
    source = store.append(branch, "one", "0123456789")
    dag = SummaryDAG(store, branch)
    first = dag.expand(source, limit=4)
    assert first["content"] == "0123"
    assert first["next_offset"] == 4
    assert first["historical"] is True
    assert first["has_more"] is True
    assert dag.expand(source, offset=8, limit=4)["next_offset"] is None
    with pytest.raises(ValueError):
        dag.expand(source, offset=-1)


def test_tool_group_boundaries_and_unmatched_calls():
    messages = [
        {"role": "user", "content": "goal"},
        {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a"},
        {"role": "tool", "tool_call_id": "b"},
        {"role": "assistant", "call_ids": ["pending"]},
    ]
    assert safe_boundaries(messages) == [0, 1, 4]
    assert fresh_tail_start(messages, keep_recent=2) == 1
    assert fresh_tail_start(messages, keep_recent=1) == 4


def test_legacy_user_tool_result_uses_explicit_ids():
    messages = [
        {"role": "assistant", "content": "action", "call_ids": ["x"]},
        {"role": "user", "content": "result", "result_ids": ["x"]},
    ]
    assert fresh_tail_start(messages, keep_recent=1) == 0
    assert safe_boundaries(messages) == [0, 2]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "tool", "content": "missing ID"}],
        [{"role": "tool", "tool_call_id": "unknown"}],
        [{"call_ids": ["x", "x"]}],
    ],
)
def test_malformed_pairing_fails_closed(messages):
    with pytest.raises(ValueError):
        safe_boundaries(messages)


def make_compactor(store, summarize, **policy):
    branch = store.create_branch()
    compactor = Compactor(
        store,
        branch,
        summarize,
        policy=CompactionPolicy(
            keep_recent=1,
            leaf_tokens=300,
            leaf_ceiling=300,
            summary_tokens=100,
            **policy,
        ),
    )
    ids = compactor.ingest(
        [{"event_id": str(index), "role": "user", "content": "x" * 1000} for index in range(5)]
    )
    return compactor, ids


def summary_rows(store, branch):
    return [
        store.get(branch, str(row[0]))
        for row in store.db.execute(
            "SELECT id FROM lcm_records WHERE branch=? AND kind='summary' ORDER BY seq", (branch,)
        )
    ]


def test_positive_batches_keep_originals_and_fresh_tail(store):
    requests = []

    def summarize(messages, max_tokens):
        requests.append((messages, max_tokens))
        return SummaryResponse("Confirmed historical observation.", {"input_tokens": 100})

    compactor, ids = make_compactor(store, summarize)
    originals = list(ids)
    compactor.compact(ids, token_budget=500)
    assert ids[-1] == originals[-1]
    assert compactor.tokens(ids) <= 500
    assert len(requests) <= 8
    assert all(store.get(compactor.branch, source)["body"] == "x" * 1000 for source in originals)
    assert compactor.calls[0]["usage"]["input_tokens"] == 100


def test_failed_llm_summary_uses_deterministic_fallback_without_losing_sources(store):
    compactor, ids = make_compactor(store, lambda *_: SummaryResponse("x" * 2000, {}))
    original = list(ids)
    with pytest.raises(ContextOverflow):
        compactor.compact(ids, token_budget=500)
    assert ids != original
    fallback = next(row for row in summary_rows(store, compactor.branch) if row["metadata"]["summary_level"] == 3)
    assert fallback["metadata"]["summary_level"] == 3
    assert store.get(compactor.branch, original[0])["body"] == "x" * 1000


def test_later_failure_keeps_committed_batch(store):
    count = 0

    def summarize(*_):
        nonlocal count
        count += 1
        if count > 1:
            raise RuntimeError("offline provider failure")
        return SummaryResponse("Historical observation.", {})

    compactor, ids = make_compactor(store, summarize)
    original = list(ids)
    with pytest.raises(ContextOverflow):
        compactor.compact(ids, token_budget=300)
    assert ids != original
    leaf = next(
        row for row in summary_rows(store, compactor.branch) if row["metadata"]["source_type"] == "messages"
    )
    assert leaf["kind"] == "summary"
    assert store.sources(compactor.branch, leaf["id"]) == original[:1]


def test_missing_event_id_is_rejected(store):
    compactor = Compactor(store, store.create_branch(), lambda *_: SummaryResponse("", {}))
    with pytest.raises(ValueError, match="event_id"):
        compactor.ingest([{"role": "user", "content": "hello"}])


def test_summary_input_budget_prevents_model_call(store):
    def forbidden(*_):
        raise AssertionError("Summary call should have been budget-blocked")

    compactor, ids = make_compactor(store, forbidden, max_request_tokens=1)
    with pytest.raises(ContextOverflow):
        compactor.compact(ids, token_budget=300)
    assert compactor.calls == []


def test_summary_spend_limit_is_per_context_build_not_lifetime(store):
    compactor, ids = make_compactor(
        store,
        lambda *_: SummaryResponse("Historical observations.", {}),
        max_total_request_tokens=1000,
        max_batches=1,
    )
    for _ in range(3):
        compactor.compact(ids, token_budget=compactor.tokens(ids) - 20)
    assert len(compactor.calls) == 3


def test_leaf_summaries_condense_only_with_same_depth_nodes(store):
    compactor, ids = make_compactor(
        store,
        lambda *_: SummaryResponse("Historical observation.", {}),
        condensation_fanin=2,
        max_batches=4,
    )
    with pytest.raises(ContextOverflow):
        compactor.compact(ids, token_budget=250)
    summaries = [
        record_id for record_id in ids if store.get(compactor.branch, record_id)["kind"] == "summary"
    ]
    assert summaries
    for record_id in summaries:
        row = store.get(compactor.branch, record_id)
        if row["metadata"]["depth"] > 0:
            sources = store.sources(compactor.branch, record_id)
            assert len(sources) == 2
            assert {
                store.get(compactor.branch, source)["metadata"]["depth"] for source in sources
            } == {row["metadata"]["depth"] - 1}
    assert sum(call["estimated_input_tokens"] for call in compactor.calls) > 1000


def test_tool_interfaces_are_bounded_and_branch_scoped(store):
    one, two = store.create_branch(), store.create_branch()
    visible = store.append(one, "v", "visible")
    hidden = store.append(two, "h", "hidden")
    tools = {tool.name: tool for tool in build_lcm_tools(store, lambda: one)}
    assert set(tools) == {"lcm_grep", "lcm_describe", "lcm_expand"}
    assert all(tool.read_only for tool in tools.values())
    assert tools["lcm_expand"].execute({"record_id": hidden}).status == "failed"
    assert tools["lcm_expand"].execute({"record_id": visible}).status == "success"


def test_grep_returns_hit_preview_offset_sequence_and_type_filter(store):
    branch = store.create_branch()
    store.append(
        branch,
        "task",
        "pytest failure investigation",
        metadata={"kind": "task", "role": "user"},
    )
    body = "setup log\n" + "x" * 600 + "\nFAILED test_login: AssertionError: token expired\n"
    result_id = store.append(
        branch,
        "test-result",
        body,
        metadata={"kind": "tool_result", "role": "user"},
    )
    tools = {tool.name: tool for tool in build_lcm_tools(store, lambda: branch)}
    response = tools["lcm_grep"].execute(
        {"query": "AssertionError", "record_types": ["tool_result"]}
    )
    assert response.status == "success"
    hit = json.loads(response.message)[0]
    assert hit["id"] == result_id
    assert hit["kind"] == "tool_result"
    assert hit["sequence"] == 2
    assert hit["offset"] > 0
    assert "AssertionError" in hit["preview"]
    expanded = tools["lcm_expand"].execute(
        {"record_id": hit["id"], "offset": hit["offset"], "limit": 500}
    )
    assert "FAILED test_login" in json.loads(expanded.message)["content"]


def test_reopen_preserves_graph(tmp_path):
    path = tmp_path / "persist.sqlite3"
    store = LCMStore(path)
    branch = store.create_branch()
    source = store.append(branch, "x", "original")
    node = store.add_summary(branch, "summary", [source], metadata={"depth": 0})
    store.close()
    reopened = LCMStore(path)
    try:
        assert reopened.sources(branch, node) == [source]
        assert reopened.get(branch, source)["body"] == "original"
    finally:
        reopened.close()
