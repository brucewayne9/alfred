from fastapi import FastAPI
from fastapi.testclient import TestClient
from core.api.forge import register
from core.security.auth import require_auth


def _client():
    app = FastAPI(); register(app)
    app.dependency_overrides[require_auth] = lambda: {"username": "t", "role": "team"}
    return TestClient(app)


def test_library_index_ok(monkeypatch):
    import core.forge.library as lib
    monkeypatch.setattr(lib, "list_done_jobs", lambda limit=100: [{"id": "x", "format": "leak_graphic"}])
    monkeypatch.setattr(lib, "trash_state", lambda: {"count": 0, "label": None})
    r = _client().get("/forge/library")
    body = r.json()
    assert r.status_code == 200 and body["jobs"][0]["id"] == "x"
    assert body["undo"] == {"count": 0, "label": None}


def test_undo_empty_is_404(monkeypatch):
    import core.forge.library as lib
    monkeypatch.setattr(lib, "undo_last", lambda: None)
    r = _client().post("/forge/library/undo")
    assert r.status_code == 404


def test_library_file_rejects_escape():
    r = _client().get("/forge/library/file", params={"path": "Content/Other/a.mp4"})
    assert r.status_code == 400
