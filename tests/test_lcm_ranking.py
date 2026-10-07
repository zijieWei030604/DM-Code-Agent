"""Offline ranking contracts derived from the pinned Hermes-LCM revision."""

import json

import pytest

from dm_agent.memory.lcm.search_ranking import combined_sort_key, rank_results
from dm_agent.memory.lcm.store import LCMStore
from dm_agent.memory.lcm.tools import build_lcm_tools


def test_upstream_negative_rank_decay_and_exact_ties():
    now = 10_000_000.0
    old = dict(type="message", role="user", _sort_rank=-12, _sort_ts=now - 3600000)
    new = dict(type="message", role="user", _sort_rank=-10, _sort_ts=now - 3600)
    assert combined_sort_key(old, "relevance", now) < combined_sort_key(new, "relevance", now)
    assert combined_sort_key(new, "hybrid", now) < combined_sort_key(old, "hybrid", now)
    assert combined_sort_key(old, "hybrid", now)[2] == -6
    # Directness is a lexicographic tie-break, not an approximate comparison.
    direct = {**new, "_sort_directness": 20}
    assert combined_sort_key(direct, "hybrid", now) < combined_sort_key(new, "hybrid", now)


def test_summary_override_exact_threshold():
    message = dict(type="message", _sort_directness=5, _sort_rank=-100)
    summary = dict(type="summary", _sort_directness=13, _sort_rank=-1)
    rows = [message, summary]
    rank_results(rows, "hybrid", now=100)
    assert rows[0] is summary
    summary["_sort_directness"] = 12.99
    rank_results(rows, "hybrid", now=100)
    assert rows[0] is message


@pytest.mark.parametrize("fts", [True, False])
def test_real_search_modes_filters_and_tool(tmp_path, monkeypatch, fts):
    store = LCMStore(tmp_path / "ranking.sqlite3")
    try:
        store.fts_enabled = fts
        branch = store.create_branch()
        monkeypatch.setattr("dm_agent.memory.lcm.store.time.time", lambda: 1000.0)
        old = store.append(branch, "old", "needle", metadata={"kind": "tool_result"})
        child = store.create_branch(parent=branch)
        monkeypatch.setattr("dm_agent.memory.lcm.store.time.time", lambda: 10_000_000.0)
        new = store.append(child, "new", "needle", metadata={"kind": "tool_result"})
        store.append(branch, "future", "needle needle needle")
        for sort in ("recency", "relevance", "hybrid"):
            rows = store.search(child, "needle", sort=sort, record_types=["tool_result"])
            assert {row["id"] for row in rows} == {old, new}
        assert store.search(child, "needle", sort="hybrid")[0]["id"] == new
        tool = next(t for t in build_lcm_tools(store, lambda: child) if t.name == "lcm_grep")
        response = tool.execute({"query": "needle", "sort": "hybrid"})
        assert response.status == "success"
        assert json.loads(response.message)[0]["id"] == new
    finally:
        store.close()


def test_relevance_finds_older_precise_record_before_limit(tmp_path):
    store = LCMStore(tmp_path / "ranking.sqlite3")
    try:
        branch = store.create_branch()
        precise = store.append(branch, "precise", "needle")
        for index in range(40):
            store.append(branch, str(index), "needle " + "unrelated " * 80)
        assert store.search(branch, "needle", sort="relevance", limit=1)[0]["id"] == precise
        assert store.search(branch, "needle", limit=1)[0]["id"] != precise
    finally:
        store.close()


def test_summary_age_uses_source_time_and_survives_reopen(tmp_path, monkeypatch):
    path = tmp_path / "time.sqlite3"
    store = LCMStore(path)
    branch = store.create_branch()
    monkeypatch.setattr("dm_agent.memory.lcm.store.time.time", lambda: 100.0)
    source = store.append(branch, "source", "needle")
    monkeypatch.setattr("dm_agent.memory.lcm.store.time.time", lambda: 200.0)
    summary = store.add_summary(branch, "needle", [source], metadata={"depth": 0})
    store.close()
    reopened = LCMStore(path)
    try:
        row = reopened.get(branch, summary)
        assert row["created_at"] == 200.0
        assert row["latest_at"] == 100.0
    finally:
        reopened.close()
