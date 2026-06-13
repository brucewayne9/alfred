"""Tests for org-scoped vector search (Task 7).

Stubs out chromadb (not installed in test env) and core.memory.store via
sys.modules patching so that `core.forge.search` can be imported.  Then
monkeypatches _get_collection to record what arguments are passed.
"""
from __future__ import annotations

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Stub chromadb + core.memory.store BEFORE importing search
# ---------------------------------------------------------------------------

_chromadb_stub = types.ModuleType("chromadb")
_chromadb_stub.ClientAPI = object  # type: ignore[attr-defined]
_chromadb_config_stub = types.ModuleType("chromadb.config")
_chromadb_config_stub.Settings = object  # type: ignore[attr-defined]
sys.modules.setdefault("chromadb", _chromadb_stub)
sys.modules.setdefault("chromadb.config", _chromadb_config_stub)

_store_stub = types.ModuleType("core.memory.store")
_store_stub.get_client = lambda: None  # type: ignore[attr-defined]
sys.modules.setdefault("core.memory.store", _store_stub)

from core.forge import search  # noqa: E402  (must follow stub setup)


# ---------------------------------------------------------------------------
# Fake collection
# ---------------------------------------------------------------------------


class FakeCollection:
    def __init__(self):
        self.upserts = []   # list of dicts of kwargs
        self.queries = []   # list of dicts of kwargs

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def count(self):
        # Return non-zero so search_segments doesn't short-circuit.
        return 10

    def query(self, **kwargs):
        self.queries.append(kwargs)
        # Mimic Chroma's return shape so result-shaping code doesn't crash.
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


@pytest.fixture
def fake(monkeypatch):
    coll = FakeCollection()
    monkeypatch.setattr(search, "_get_collection", lambda *a, **k: coll)
    return coll


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_upsert_tags_windows_with_org(fake):
    windows = [
        {
            "win_id": "src1_w0000",
            "start_s": 0.0,
            "end_s": 5.0,
            "text": "secret sauce",
            "speaker": "Alice",
            "seq_start": 0,
            "seq_end": 2,
        }
    ]
    search.upsert_windows("src1", windows, org="rucktalk")
    metas = []
    for call in fake.upserts:
        metas.extend(call.get("metadatas") or [])
    assert metas, "no metadatas were passed to upsert"
    assert all(m.get("org_id") == "rucktalk" for m in metas)


def test_search_filters_by_org(fake):
    search.search_segments(query="secret sauce", org="rucktalk", top_k=10)
    assert fake.queries, "query was not called"
    where = fake.queries[0].get("where")
    assert where == {"org_id": "rucktalk"}


def test_search_filters_by_org_and_source(fake):
    search.search_segments(query="x", source_id="src1", org="rucktalk")
    where = fake.queries[0].get("where")
    assert where == {"$and": [{"source_id": "src1"}, {"org_id": "rucktalk"}]}


def test_search_no_filters_is_none(fake):
    search.search_segments(query="x")
    assert fake.queries[0].get("where") is None
