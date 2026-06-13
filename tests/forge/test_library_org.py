import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.api.forge import register
from core.security.auth import require_auth


def _seed_done_job(org, caption):
    from core.forge import jobs as fj
    import json
    jid = fj.enqueue("leak_graphic", {"caption": caption}, org=org)
    fj._update(jid, status="done",
               result=json.dumps({"format": "leak_graphic", "delivered": 1,
                                  "delivered_dir": f"Content/{org}/Processed"}))
    return jid


def _client(tmp_path, monkeypatch, user):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()
    app = FastAPI(); register(app)
    app.dependency_overrides[require_auth] = lambda: user
    return TestClient(app)


def test_member_library_shows_only_own_org(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, {"username": "r", "role": "member", "org": "rucktalk"})
    _seed_done_job("mainstay", "mainstay card")
    _seed_done_job("rucktalk", "ruck card")
    cards = c.get("/forge/library").json()["jobs"]
    caps = {x["caption"] for x in cards}
    assert caps == {"ruck card"}


def test_super_admin_library_shows_all(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, {"username": "mike", "role": "super_admin", "org": "*"})
    _seed_done_job("mainstay", "mainstay card")
    _seed_done_job("rucktalk", "ruck card")
    cards = c.get("/forge/library").json()["jobs"]
    caps = {x["caption"] for x in cards}
    assert caps == {"mainstay card", "ruck card"}
