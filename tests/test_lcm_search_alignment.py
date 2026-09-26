"""Upstream query semantics with local branch and record-type isolation."""

import pytest

from dm_agent.memory.lcm.search_query import sanitize_fts5_query
from dm_agent.memory.lcm.store import LCMStore


@pytest.fixture
def store(tmp_path):
    value = LCMStore(tmp_path / "search.sqlite3")
    yield value
    value.close()


def test_multiword_and_quoted_phrase(store):
    branch = store.create_branch()
    both = store.append(branch, "both", "alpha beta")
    store.append(branch, "one", "alpha")
    separated = store.append(branch, "separated", "alpha other beta")
    assert {r["id"] for r in store.search(branch, "alpha beta")} == {both, separated}
    assert [r["id"] for r in store.search(branch, '"alpha beta"')] == [both]
    assert store.search(branch, "alpha missing") == []


def test_operators_are_literal_and_unicode_is_normalized():
    assert sanitize_fts5_query("alpha OR beta") == "alpha or beta"
    assert sanitize_fts5_query("caf\u0065\u0301") == "caf\u00e9"
    assert sanitize_fts5_query("TypeSerializer.serialize()") == "TypeSerializer serialize"


@pytest.mark.parametrize("fts", [True, False])
def test_fallback_and_branch_isolation(store, fts):
    parent = store.create_branch()
    old = store.append(parent, "old", "\u767b\u5f55 100% foo_bar")
    child = store.create_branch(parent=parent)
    store.append(parent, "future", "\u767b\u5f55 100% foo_bar")
    store.fts_enabled = fts
    assert [r["id"] for r in store.search(child, "\u767b\u5f55")] == [old]
    if not fts:
        assert [r["id"] for r in store.search(child, "foo_bar")] == [old]
        assert store.search(child, "fooXbar") == []


def test_type_filter_before_limit(store):
    branch = store.create_branch()
    wanted = store.append(branch, "wanted", "needle", metadata={"kind": "task"})
    for index in range(110):
        store.append(branch, str(index), "needle", metadata={"kind": "tool_result"})
    assert [r["id"] for r in store.search(branch, "needle", limit=1, record_types=["task"])] == [
        wanted
    ]


def test_missing_fts_table_falls_back(store):
    branch = store.create_branch()
    wanted = store.append(branch, "wanted", "needle")
    store.db.execute("DROP TABLE lcm_fts")
    assert [r["id"] for r in store.search(branch, "needle")] == [wanted]
