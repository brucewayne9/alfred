# tests/casting/test_scheduler.py
import os, tempfile, importlib
import pytest

@pytest.fixture()
def env(monkeypatch):
    tmp = tempfile.mkdtemp()
    from config.settings import settings
    monkeypatch.setattr(settings, "casting_db_path", os.path.join(tmp, "casting.db"), raising=False)
    import core.casting.db as dbmod; importlib.reload(dbmod); dbmod.init_db()
    import core.casting.scheduler as sch; importlib.reload(sch)
    return dbmod, sch

def test_apply_due_calls_deploy_and_marks(env, monkeypatch):
    dbmod, sch = env
    dj_id = dbmod.create_dj(name="Sloan", role="host", persona_prompt="warm",
                            archetype_tags=[], expertise="", voice_source="recorded")
    dbmod.set_mood_present(dj_id, "neutral")
    dbmod.set_status(dj_id, "ready")
    aid = dbmod.create_assignment(dj_id=dj_id, station_id=22, slot="10a-2p",
                                  effective_at="2026-06-07T10:00:00")
    deployed = {}
    monkeypatch.setattr(sch.deploy, "deploy_dj", lambda **kw: deployed.update(kw))
    applied = sch.apply_due(now_iso="2026-06-07T10:01:00")
    assert applied == 1
    assert deployed["dj_name"] == "Sloan"
    assert deployed["schedule_start"] == "10:00"
    assert deployed["schedule_end"] == "14:00"
    assert deployed["enabled"] is False
    assert dbmod.due_assignments(now_iso="2026-06-07T10:01:00") == []
    assert dbmod.get_dj(dj_id)["status"] == "live"

def test_apply_due_demotes_prior_live_dj(env, monkeypatch):
    dbmod, sch = env
    # an incumbent live DJ already applied on station 22
    old = dbmod.create_dj(name="Harry", role="host", persona_prompt="x",
                          archetype_tags=[], expertise="", voice_source="recorded")
    dbmod.set_mood_present(old, "neutral")
    dbmod.set_status(old, "ready")
    old_aid = dbmod.create_assignment(dj_id=old, station_id=22, slot="6a-10a",
                                      effective_at="2026-06-07T06:00:00")
    dbmod.mark_applied(old_aid)
    dbmod.set_status(old, "live")

    # new DJ taking the same station
    new = dbmod.create_dj(name="Sloan", role="host", persona_prompt="x",
                          archetype_tags=[], expertise="", voice_source="recorded")
    dbmod.set_mood_present(new, "neutral")
    dbmod.set_status(new, "ready")
    dbmod.create_assignment(dj_id=new, station_id=22, slot="10a-2p",
                            effective_at="2026-06-07T10:00:00")

    monkeypatch.setattr(sch.deploy, "deploy_dj", lambda **kw: None)
    sch.apply_due(now_iso="2026-06-07T10:01:00")

    assert dbmod.get_dj(old)["status"] == "ready"   # demoted
    assert dbmod.get_dj(new)["status"] == "live"     # promoted
