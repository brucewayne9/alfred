from fastapi import FastAPI
from fastapi.testclient import TestClient
from core.api.forge import register
from core.security.auth import require_auth


def _c():
    app = FastAPI(); register(app)
    app.dependency_overrides[require_auth] = lambda: {"u": 1}
    return TestClient(app)


def test_accounts_get(monkeypatch):
    import core.forge.distribution as d
    monkeypatch.setattr(d, "get_accounts", lambda: [{"handle": "@x", "platform": "TikTok"}])
    r = _c().get("/forge/distribution/accounts")
    assert r.status_code == 200 and r.json()["accounts"][0]["handle"] == "@x"


def test_mark_posted_route(monkeypatch):
    import core.forge.distribution as d
    calls = {}
    monkeypatch.setattr(d, "mark_posted", lambda pid, posted=True: calls.update(pid=pid, posted=posted))
    r = _c().post("/forge/distribution/posted", json={"post_id": "j:0", "posted": True})
    assert r.status_code == 200 and calls == {"pid": "j:0", "posted": True}
