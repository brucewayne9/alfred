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


def test_member_cannot_read_other_orgs_job(tmp_path, monkeypatch):
    ruck = _client(tmp_path, monkeypatch, {"username": "r", "role": "member", "org": "rucktalk"})
    from core.forge import jobs as fj
    jid = fj.enqueue("echo", {"caption": "secret"}, org="mainstay")
    assert ruck.get(f"/forge/jobs/{jid}").status_code == 404


def test_member_can_read_own_orgs_job(tmp_path, monkeypatch):
    ruck = _client(tmp_path, monkeypatch, {"username": "r", "role": "member", "org": "rucktalk"})
    from core.forge import jobs as fj
    jid = fj.enqueue("echo", {"caption": "mine"}, org="rucktalk")
    assert ruck.get(f"/forge/jobs/{jid}").status_code == 200


def test_member_cannot_delete_other_orgs_job(tmp_path, monkeypatch):
    ruck = _client(tmp_path, monkeypatch, {"username": "r", "role": "member", "org": "rucktalk"})
    from core.forge import jobs as fj
    jid = fj.enqueue("echo", {}, org="mainstay")
    assert ruck.delete(f"/forge/jobs/{jid}").status_code == 404
    # job still exists for the owner
    mike = _client(tmp_path, monkeypatch, {"username": "mike", "role": "super_admin", "org": "*"})
    assert mike.get(f"/forge/jobs/{jid}").status_code == 200


def test_super_admin_can_read_any_job(tmp_path, monkeypatch):
    mike = _client(tmp_path, monkeypatch, {"username": "mike", "role": "super_admin", "org": "*"})
    from core.forge import jobs as fj
    jid = fj.enqueue("echo", {}, org="mainstay")
    assert mike.get(f"/forge/jobs/{jid}").status_code == 200
