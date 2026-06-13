import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.api.forge import register
from core.security.auth import require_auth


def _setup_db(tmp_path, monkeypatch):
    """Point Forge at a fresh tmp DB and create the schema (incl. org_id).

    Must run before any ingest.create_source call so seeded rows land in the
    same DB the TestClient later reads.
    """
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()


def _client(tmp_path, monkeypatch, user):
    _setup_db(tmp_path, monkeypatch)
    app = FastAPI()
    register(app)
    app.dependency_overrides[require_auth] = lambda: user
    return TestClient(app)


def test_member_sees_only_own_org_sources(tmp_path, monkeypatch):
    _setup_db(tmp_path, monkeypatch)
    from core.forge import ingest
    ingest.create_source("url", "mainstay-clip", None, org="mainstay")
    ingest.create_source("url", "ruck-clip", None, org="rucktalk")
    ruck = _client(tmp_path, monkeypatch, {"username": "r", "role": "member", "org": "rucktalk"})
    specs = [s["spec"] for s in ruck.get("/forge/sources").json()["sources"]]
    assert specs == ["ruck-clip"]


def test_member_gets_404_on_other_orgs_source(tmp_path, monkeypatch):
    _setup_db(tmp_path, monkeypatch)
    from core.forge import ingest
    sid = ingest.create_source("url", "mainstay-clip", None, org="mainstay")
    ruck = _client(tmp_path, monkeypatch, {"username": "r", "role": "member", "org": "rucktalk"})
    assert ruck.get(f"/forge/sources/{sid}").status_code == 404


def test_super_admin_sees_all_orgs(tmp_path, monkeypatch):
    _setup_db(tmp_path, monkeypatch)
    from core.forge import ingest
    ingest.create_source("url", "mainstay-clip", None, org="mainstay")
    ingest.create_source("url", "ruck-clip", None, org="rucktalk")
    mike = _client(tmp_path, monkeypatch, {"username": "mike", "role": "super_admin", "org": "*"})
    specs = {s["spec"] for s in mike.get("/forge/sources").json()["sources"]}
    assert specs == {"mainstay-clip", "ruck-clip"}


def test_super_admin_can_focus_one_org(tmp_path, monkeypatch):
    _setup_db(tmp_path, monkeypatch)
    from core.forge import ingest
    ingest.create_source("url", "mainstay-clip", None, org="mainstay")
    ingest.create_source("url", "ruck-clip", None, org="rucktalk")
    mike = _client(tmp_path, monkeypatch, {"username": "mike", "role": "super_admin", "org": "*"})
    specs = [s["spec"] for s in mike.get("/forge/sources?org=rucktalk").json()["sources"]]
    assert specs == ["ruck-clip"]
