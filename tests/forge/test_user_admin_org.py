import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from core.api.forge import register
from core.security.auth import require_auth


def _client(tmp_path, monkeypatch, user):
    monkeypatch.setenv("FORGE_USERS_FILE", str(tmp_path / "u.json"))
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()
    app = FastAPI(); register(app)
    app.dependency_overrides[require_auth] = lambda: user
    return TestClient(app)


def test_org_admin_creates_user_only_in_own_org(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, {"username": "boss", "role": "org_admin", "org": "rucktalk"})
    c.post("/forge/users", json={"username": "newbie", "password": "pw12345",
                                 "role": "member", "org": "mainstay"})
    from core.forge import users
    assert users.load_users()["newbie"]["org"] == "rucktalk"


def test_org_admin_cannot_create_super_admin(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, {"username": "boss", "role": "org_admin", "org": "rucktalk"})
    c.post("/forge/users", json={"username": "x", "password": "pw12345", "role": "super_admin"})
    from core.forge import users
    assert users.load_users()["x"]["role"] == "member"


def test_super_admin_creates_user_in_any_org(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, {"username": "mike", "role": "super_admin", "org": "*"})
    c.post("/forge/users", json={"username": "x", "password": "pw12345",
                                 "role": "member", "org": "mainstay"})
    from core.forge import users
    assert users.load_users()["x"]["org"] == "mainstay"


def test_member_cannot_manage_users(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, {"username": "alice", "role": "member", "org": "rucktalk"})
    assert c.get("/forge/users").status_code == 403


def test_orgs_endpoint_lists_seeded_orgs(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, {"username": "mike", "role": "super_admin", "org": "*"})
    orgs = {o["id"] for o in c.get("/forge/orgs").json()["orgs"]}
    assert {"mainstay", "rucktalk", "groundrush"} <= orgs
