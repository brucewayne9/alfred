import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.api.forge import register
from core.security.auth import require_auth


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    monkeypatch.setenv("FORGE_CHROMA_DIR", str(tmp_path / "chroma"))
    from core.forge import db as _db
    _db.init_db()
    return tmp_path


def _client(user):
    app = FastAPI(); register(app)
    app.dependency_overrides[require_auth] = lambda: user
    return TestClient(app)


def test_full_isolation_member_cannot_reach_other_org(env):
    from core.forge import ingest, scorer
    # Mainstay data
    m = ingest.create_source("url", "mainstay", None, org="mainstay")
    scorer.save_candidates(m, [{"start_s": 0, "end_s": 5, "score": 90, "hook": "h",
                                "emotion": "e", "reason": "r", "caption": "c"}],
                           org="mainstay")
    # RuckTalk member
    ruck = _client({"username": "r", "role": "member", "org": "rucktalk"})
    assert ruck.get("/forge/sources").json()["sources"] == []
    assert ruck.get(f"/forge/sources/{m}").status_code == 404

    # super-admin sees it
    mike = _client({"username": "mike", "role": "super_admin", "org": "*"})
    assert len(mike.get("/forge/sources").json()["sources"]) == 1
