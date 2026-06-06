# tests/casting/test_db.py
import os, tempfile, importlib
import pytest

@pytest.fixture()
def db(monkeypatch):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "casting.db")
    from config.settings import settings
    monkeypatch.setattr(settings, "casting_db_path", path, raising=False)
    import core.casting.db as dbmod
    importlib.reload(dbmod)
    dbmod.init_db()
    return dbmod

def test_create_and_get_dj(db):
    dj_id = db.create_dj(name="Sloan", role="host", persona_prompt="warm realist",
                         archetype_tags=["strategist"], expertise="", voice_source="recorded")
    assert isinstance(dj_id, int)
    row = db.get_dj(dj_id)
    assert row["name"] == "Sloan"
    assert row["status"] == "draft"
    assert row["archetype_tags"] == ["strategist"]

def test_list_djs(db):
    db.create_dj(name="A", role="host", persona_prompt="", archetype_tags=[], expertise="", voice_source="recorded")
    db.create_dj(name="B", role="host", persona_prompt="", archetype_tags=[], expertise="", voice_source="recorded")
    assert len(db.list_djs()) == 2

def test_set_status_and_moods(db):
    dj_id = db.create_dj(name="A", role="host", persona_prompt="", archetype_tags=[], expertise="", voice_source="recorded")
    db.set_mood_present(dj_id, "neutral")
    db.set_mood_present(dj_id, "fired")
    db.set_status(dj_id, "ready")
    row = db.get_dj(dj_id)
    assert row["status"] == "ready"
    assert set(row["moods_present"]) == {"neutral", "fired"}

def test_assignment_roundtrip(db):
    dj_id = db.create_dj(name="A", role="host", persona_prompt="", archetype_tags=[], expertise="", voice_source="recorded")
    aid = db.create_assignment(dj_id=dj_id, station_id=22, slot="10a-2p", effective_at="2026-06-07T10:00:00")
    due = db.due_assignments(now_iso="2026-06-07T10:05:00")
    assert any(a["id"] == aid for a in due)
    db.mark_applied(aid)
    assert db.due_assignments(now_iso="2026-06-07T10:05:00") == []
