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

def test_get_preview_404_without_neutral(client):
    res = client.post("/api/casting/djs", json={"name": "NoClip", "role": "host",
        "persona_prompt": "x", "archetype_tags": [], "expertise": "", "voice_source": "recorded"})
    dj_id = res.json()["id"]
    res = client.get(f"/api/casting/djs/{dj_id}/preview")
    assert res.status_code == 404

def test_get_preview_serves_cached(client, monkeypatch):
    import core.casting.api_router as r
    res = client.post("/api/casting/djs", json={"name": "Cached", "role": "host",
        "persona_prompt": "x", "archetype_tags": [], "expertise": "", "voice_source": "recorded"})
    dj_id = res.json()["id"]
    # mark a neutral clip present in the db (no real wav needed; render is stubbed)
    r.db.set_mood_present(dj_id, "neutral")
    def fake_render(*, voice_wav, out_path, **kw):
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as fh:
            fh.write(b"RIFFcached")
        return out_path
    monkeypatch.setattr(r.preview_mod, "render_preview", fake_render)
    # first GET renders + caches
    res = client.get(f"/api/casting/djs/{dj_id}/preview")
    assert res.status_code == 200
    assert res.content.startswith(b"RIFF")
    # second GET serves from cache (render would raise if called again)
    monkeypatch.setattr(r.preview_mod, "render_preview",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("should be cached")))
    res = client.get(f"/api/casting/djs/{dj_id}/preview")
    assert res.status_code == 200 and res.content.startswith(b"RIFF")

def test_create_assignment_normalizes_effective_at(client, monkeypatch):
    import core.casting.api_router as r
    res = client.post("/api/casting/djs", json={"name": "Sched", "role": "host",
        "persona_prompt": "x", "archetype_tags": [], "expertise": "", "voice_source": "recorded"})
    dj_id = res.json()["id"]
    r.db.set_status(dj_id, "ready")
    # datetime-local style: no seconds
    res = client.post("/api/casting/assignments", json={"dj_id": dj_id, "station_id": 22,
        "slot": "10a-2p", "effective_at": "2026-06-07T10:00"})
    assert res.status_code == 200, res.text
    # stored form must compare correctly against a seconds-precision "now"
    due = r.db.due_assignments(now_iso="2026-06-07T10:00:05")
    assert any(a["dj_id"] == dj_id for a in due)
    # and it should be the normalized seconds form
    lst = client.get("/api/casting/assignments?station_id=22").json()
    assert any(a["effective_at"] == "2026-06-07T10:00:00" for a in lst)

def test_create_assignment_bad_effective_at(client):
    res = client.post("/api/casting/djs", json={"name": "Bad", "role": "host",
        "persona_prompt": "x", "archetype_tags": [], "expertise": "", "voice_source": "recorded"})
    dj_id = res.json()["id"]
    import core.casting.api_router as r
    r.db.set_status(dj_id, "ready")
    res = client.post("/api/casting/assignments", json={"dj_id": dj_id, "station_id": 22,
        "slot": "10a-2p", "effective_at": "not-a-date"})
    assert res.status_code == 422
