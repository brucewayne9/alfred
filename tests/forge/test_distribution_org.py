import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from core.api.forge import register
from core.security.auth import require_auth


def _client(tmp_path, monkeypatch, user):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()
    app = FastAPI(); register(app)
    app.dependency_overrides[require_auth] = lambda: user
    return TestClient(app)


def test_member_cannot_pack_other_orgs_job(tmp_path, monkeypatch):
    ruck = _client(tmp_path, monkeypatch, {"username": "r", "role": "member", "org": "rucktalk"})
    from core.forge import jobs as fj
    jid = fj.enqueue("leak_graphic", {"caption": "x"}, org="mainstay")
    # pack endpoint takes ?job_id=; cross-org must 404
    r = ruck.get(f"/forge/distribution/pack?job_id={jid}")
    assert r.status_code == 404


def test_member_cannot_postiz_other_orgs_job(tmp_path, monkeypatch):
    ruck = _client(tmp_path, monkeypatch, {"username": "r", "role": "member", "org": "rucktalk"})
    from core.forge import jobs as fj
    jid = fj.enqueue("leak_graphic", {"caption": "x"}, org="mainstay")
    r = ruck.post("/forge/distribution/postiz", json={"job_id": jid})
    assert r.status_code == 404
