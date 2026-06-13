from fastapi import FastAPI
from fastapi.testclient import TestClient
from core.api.forge import register
from core.security.auth import require_auth
from core.forge import uploads


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()
    app = FastAPI()
    register(app)
    app.dependency_overrides[require_auth] = lambda: {"username": "t", "role": "team"}
    return TestClient(app)


def test_upload_returns_id_and_persists(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/forge/uploads", files={"file": ("hook.mp3", b"ID3bytes", "audio/mpeg")})
    assert r.status_code == 200
    uid = r.json()["upload_id"]
    assert uploads.get_upload_path(uid).read_bytes() == b"ID3bytes"


def test_upload_rejects_empty(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/forge/uploads", files={"file": ("x.mp3", b"", "audio/mpeg")})
    assert r.status_code == 400
