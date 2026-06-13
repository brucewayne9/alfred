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


def test_member_cannot_list_orgs(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, {"username": "a", "role": "member", "org": "rucktalk"})
    assert c.get("/forge/orgs").status_code == 403


def test_org_admin_can_list_orgs(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, {"username": "b", "role": "org_admin", "org": "rucktalk"})
    assert c.get("/forge/orgs").status_code == 200


def test_super_admin_can_list_orgs(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, {"username": "m", "role": "super_admin", "org": "*"})
    r = c.get("/forge/orgs")
    assert r.status_code == 200
    assert {"mainstay", "rucktalk", "groundrush"} <= {o["id"] for o in r.json()["orgs"]}
