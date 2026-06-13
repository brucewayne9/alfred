"""Tests for the Phase-11 search API endpoints.

Monkeypatches core.forge.search and core.forge.ingest so no live
Ollama / ChromaDB is required.

Implementation note: core.forge.search imports chromadb at module level via
core.memory.store.  In the test environment chromadb may not be installed.
We inject a stub module into sys.modules *before* the FastAPI app is built so
the lazy `from core.forge import search as forge_search` inside the endpoint
resolves to our stub.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.security.auth import require_auth


# ---------------------------------------------------------------------------
# Stub for core.forge.search — injects into sys.modules so the endpoint's
# lazy `from core.forge import search as forge_search` resolves here.
# ---------------------------------------------------------------------------

def _make_search_stub():
    stub = types.ModuleType("core.forge.search")
    stub.has_windows = MagicMock(return_value=True)
    stub.embed_source_windows = MagicMock(return_value=5)
    stub.search_segments = MagicMock(return_value=[])
    return stub


@pytest.fixture(autouse=True)
def stub_search_module():
    """Ensure core.forge.search resolves to our stub for every test.

    The endpoint resolves the module via ``from core.forge import search``,
    which reads the ``search`` *attribute* on the ``core.forge`` package — not
    ``sys.modules``. If another test file imported the real submodule first, the
    package attribute is already bound to it, so patching ``sys.modules`` alone
    leaks the real module into the endpoint. Patch both.
    """
    import core.forge as _forge_pkg

    stub = _make_search_stub()
    # Replace (or insert) the module in sys.modules for the duration of the test.
    original_mod = sys.modules.get("core.forge.search")
    original_attr = getattr(_forge_pkg, "search", None)
    sys.modules["core.forge.search"] = stub
    _forge_pkg.search = stub
    yield stub
    # Restore the original (or remove if it wasn't there before).
    if original_mod is None:
        sys.modules.pop("core.forge.search", None)
    else:
        sys.modules["core.forge.search"] = original_mod
    if original_attr is None:
        if hasattr(_forge_pkg, "search"):
            delattr(_forge_pkg, "search")
    else:
        _forge_pkg.search = original_attr


@pytest.fixture
def client(tmp_path, monkeypatch, stub_search_module):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    # Force fresh import of forge API so it picks up the stubbed search module.
    sys.modules.pop("core.api.forge", None)
    from core.api import forge
    app = FastAPI()
    forge.register(app)
    app.dependency_overrides[require_auth] = lambda: {"username": "tester", "role": "admin"}
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /forge/sources — list endpoint
# ---------------------------------------------------------------------------


def test_list_sources_returns_sources(client, monkeypatch):
    """GET /forge/sources proxies ingest.list_sources and wraps result."""
    import core.forge.ingest as _ingest
    monkeypatch.setattr(_ingest, "list_sources", lambda status=None, org=None: [
        {"id": "abc123", "status": "done", "spec": "test.mp4"},
    ])
    r = client.get("/forge/sources")
    assert r.status_code == 200
    body = r.json()
    assert "sources" in body
    assert body["sources"][0]["id"] == "abc123"


def test_list_sources_passes_status_filter(client, monkeypatch):
    """?status=done is forwarded to list_sources()."""
    import core.forge.ingest as _ingest
    captured = {}

    def _fake(status=None, org=None):
        captured["status"] = status
        return []

    monkeypatch.setattr(_ingest, "list_sources", _fake)
    client.get("/forge/sources?status=done")
    assert captured["status"] == "done"


# ---------------------------------------------------------------------------
# GET /forge/sources/{id}/search — 404
# ---------------------------------------------------------------------------


def test_search_404_unknown_source(client, monkeypatch):
    """Returns 404 when the source_id is not in the DB."""
    import core.forge.ingest as _ingest
    monkeypatch.setattr(_ingest, "get_source", lambda sid: None)
    r = client.get("/forge/sources/nonexistent/search?q=topic")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


# ---------------------------------------------------------------------------
# GET /forge/sources/{id}/search — 409 not ready
# ---------------------------------------------------------------------------


def test_search_409_source_not_done(client, monkeypatch):
    """Returns 409 when the source is still transcribing."""
    import core.forge.ingest as _ingest
    monkeypatch.setattr(_ingest, "get_source", lambda sid: {"id": sid, "status": "transcribing"})
    r = client.get("/forge/sources/src1/search?q=topic")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "not ready" in detail
    assert "transcribing" in detail


# ---------------------------------------------------------------------------
# GET /forge/sources/{id}/search — 200 happy path
# ---------------------------------------------------------------------------

_FAKE_RESULT = {
    "start_s": 120.0,
    "end_s": 155.0,
    "text": "We really pushed hard on the manufacturing process",
    "speaker": "A",
    "score": 0.82,
    "seq_start": 10,
    "seq_end": 14,
}


def test_search_200_returns_results(client, monkeypatch, stub_search_module):
    """Happy path: 200 with source_id, query, results including inline text."""
    import core.forge.ingest as _ingest
    monkeypatch.setattr(_ingest, "get_source", lambda sid: {"id": sid, "status": "done"})
    stub_search_module.has_windows = MagicMock(return_value=True)
    stub_search_module.search_segments = MagicMock(return_value=[_FAKE_RESULT])

    r = client.get("/forge/sources/src1/search?q=manufacturing")
    assert r.status_code == 200
    body = r.json()
    assert body["source_id"] == "src1"
    assert body["query"] == "manufacturing"
    results = body["results"]
    assert len(results) == 1

    hit = results[0]
    # TOPIC-02: text must be inline (no second request needed)
    assert "text" in hit and hit["text"] == _FAKE_RESULT["text"]
    assert "start_s" in hit
    assert "end_s" in hit
    assert "speaker" in hit
    assert "score" in hit
    assert "seq_start" in hit
    assert "seq_end" in hit


# ---------------------------------------------------------------------------
# GET /forge/sources/{id}/search — lazy backfill
# ---------------------------------------------------------------------------


def test_search_lazy_backfill(client, monkeypatch, stub_search_module):
    """embed_source_windows is called once when has_windows returns False."""
    import core.forge.ingest as _ingest
    monkeypatch.setattr(_ingest, "get_source", lambda sid: {"id": sid, "status": "done"})
    stub_search_module.has_windows = MagicMock(return_value=False)
    stub_search_module.search_segments = MagicMock(return_value=[])

    calls = []
    stub_search_module.embed_source_windows = MagicMock(side_effect=lambda sid: calls.append(sid))

    r = client.get("/forge/sources/oldphase10src/search?q=test")
    assert r.status_code == 200
    assert calls == ["oldphase10src"]


def test_search_no_backfill_when_windows_exist(client, monkeypatch, stub_search_module):
    """embed_source_windows is NOT called when has_windows returns True."""
    import core.forge.ingest as _ingest
    monkeypatch.setattr(_ingest, "get_source", lambda sid: {"id": sid, "status": "done"})
    stub_search_module.has_windows = MagicMock(return_value=True)
    stub_search_module.search_segments = MagicMock(return_value=[])

    calls = []
    stub_search_module.embed_source_windows = MagicMock(side_effect=lambda sid: calls.append(sid))

    r = client.get("/forge/sources/src1/search?q=test")
    assert r.status_code == 200
    assert calls == []
