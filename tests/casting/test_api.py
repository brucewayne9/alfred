import os, tempfile, importlib
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.mkdtemp()
    from config.settings import settings
    monkeypatch.setattr(settings, "casting_db_path", os.path.join(tmp, "casting.db"), raising=False)
    monkeypatch.setattr(settings, "casting_voices_dir", os.path.join(tmp, "voices"), raising=False)
    monkeypatch.setattr(settings, "casting_previews_dir", os.path.join(tmp, "prev"), raising=False)
    import core.casting.db as dbmod; importlib.reload(dbmod); dbmod.init_db()
    import core.casting.api_router as r; importlib.reload(r)
    # bypass auth in tests
    app = FastAPI()
    app.dependency_overrides = {}
    r.register(app, auth_dep=lambda: {"username": "test"})
    return TestClient(app)

def test_moodpack_endpoint(client):
    res = client.get("/api/casting/moodpack")
    assert res.status_code == 200
    assert len(res.json()["moods"]) == 8

def test_create_and_list_dj(client):
    res = client.post("/api/casting/djs", json={"name": "Sloan", "role": "host",
        "persona_prompt": "warm realist", "archetype_tags": ["strategist"],
        "expertise": "", "voice_source": "recorded"})
    assert res.status_code == 200, res.text
    dj_id = res.json()["id"]
    lst = client.get("/api/casting/djs").json()
    assert any(d["id"] == dj_id for d in lst)

def test_archetypes_endpoint(client):
    res = client.get("/api/casting/archetypes")
    assert res.status_code == 200 and len(res.json()) >= 6
