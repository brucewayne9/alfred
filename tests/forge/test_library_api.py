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
    r = _client().get("/forge/library")
    assert r.status_code == 200 and r.json()["jobs"][0]["id"] == "x"


def test_library_file_rejects_escape():
    r = _client().get("/forge/library/file", params={"path": "Content/Other/a.mp4"})
    assert r.status_code == 400
