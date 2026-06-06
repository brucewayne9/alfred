# tests/casting/test_deploy.py
import importlib
import pytest
import core.casting.deploy as dep
import core.casting.voice as voice


@pytest.fixture()
def patched(monkeypatch):
    """Reload deploy, stub register_to_engine (no real fs writes) and capture
    every subprocess.run command. SELECT returns empty stdout by default
    (=> INSERT path)."""
    importlib.reload(dep)
    monkeypatch.setattr(dep.voice, "register_to_engine",
                        lambda dj_id, moods: {"neutral": f"cc{dj_id}_neutral"})

    ran = []

    def fake_run(cmd, **kw):
        ran.append(cmd)
        class R:
            returncode = 0
            stderr = ""
            # empty stdout for SELECT => no existing row => INSERT path
            stdout = ""
        return R()

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    return ran


# --------------------------------------------------------------------------
# slot_to_times
# --------------------------------------------------------------------------
def test_slot_to_times_am_pm():
    assert dep.slot_to_times("10a-2p") == ("10:00", "14:00")


def test_slot_to_times_pm_pm():
    assert dep.slot_to_times("6p-10p") == ("18:00", "22:00")


def test_slot_to_times_hhmm():
    assert dep.slot_to_times("10:00-14:00") == ("10:00", "14:00")


def test_slot_to_times_midnight_noon():
    # 12a = 00:00, 12p = 12:00
    assert dep.slot_to_times("12a-12p") == ("00:00", "12:00")


def test_slot_to_times_unparseable():
    with pytest.raises(ValueError):
        dep.slot_to_times("nonsense")


# --------------------------------------------------------------------------
# deploy_dj
# --------------------------------------------------------------------------
def test_deploy_disabled_builds_qwen_sql(patched):
    ran = patched
    dep.deploy_dj(dj_id=7, dj_name="Sloan", moods=["neutral"],
                  persona_prompt="warm realist", station_id=22,
                  schedule_start="10:00", schedule_end="14:00", enabled=False)

    joined = " ".join(" ".join(c) for c in ran)
    # references the real table + station
    assert "station_ai_dj_breaks" in joined
    assert "22" in joined
    # qwen provider + id-namespaced voice NAME
    assert "qwen" in joined
    assert "cc7_neutral" in joined
    # disabled => is_enabled 0
    assert "is_enabled" in joined
    insert = [c for c in ran if any("INSERT" in str(x) for x in c)]
    assert insert, "expected an INSERT (empty SELECT => insert path)"
    itext = " ".join(" ".join(c) for c in insert)
    # the is_enabled value column in the INSERT VALUES is 0
    assert ", 0," in itext or ",0," in itext
    # all shell calls wrapped in timeout
    assert all(c[0] == "timeout" for c in ran)
    assert joined.count("timeout") >= 1


def test_deploy_enabled_sets_enabled_one(patched):
    ran = patched
    dep.deploy_dj(dj_id=7, dj_name="Sloan", moods=["neutral"],
                  persona_prompt="warm realist", station_id=22,
                  schedule_start="10:00", schedule_end="14:00", enabled=True)
    insert = [c for c in ran if any("INSERT" in str(x) for x in c)]
    assert insert
    itext = " ".join(" ".join(c) for c in insert)
    # is_enabled value is 1
    assert ", 1," in itext or ",1," in itext


def test_deploy_escapes_apostrophe_in_persona(patched):
    ran = patched
    dep.deploy_dj(dj_id=7, dj_name="Sloan", moods=["neutral"],
                  persona_prompt="don't quit", station_id=22,
                  schedule_start="10:00", schedule_end="14:00")
    joined = " ".join(" ".join(c) for c in ran)
    # escaped doubled-apostrophe present, raw unescaped not present
    assert "don''t quit" in joined
    assert "don't quit" not in joined


def test_select_then_insert_when_no_existing(monkeypatch):
    importlib.reload(dep)
    monkeypatch.setattr(dep.voice, "register_to_engine",
                        lambda dj_id, moods: {"neutral": f"cc{dj_id}_neutral"})

    ran = []

    def fake_run(cmd, **kw):
        ran.append(cmd)
        is_select = any("SELECT" in str(x) for x in cmd)
        class R:
            returncode = 0
            stderr = ""
            stdout = "" if is_select else ""
        return R()

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    dep.deploy_dj(dj_id=7, dj_name="Sloan", moods=["neutral"],
                  persona_prompt="x", station_id=22,
                  schedule_start="10:00", schedule_end="14:00")

    selects = [c for c in ran if any("SELECT" in str(x) for x in c)]
    inserts = [c for c in ran if any("INSERT" in str(x) for x in c)]
    updates = [c for c in ran if any("UPDATE" in str(x) for x in c)]
    assert selects, "a SELECT must run first"
    assert inserts, "empty SELECT must lead to an INSERT"
    assert not updates


def test_select_then_update_when_existing(monkeypatch):
    importlib.reload(dep)
    monkeypatch.setattr(dep.voice, "register_to_engine",
                        lambda dj_id, moods: {"neutral": f"cc{dj_id}_neutral"})

    ran = []

    def fake_run(cmd, **kw):
        ran.append(cmd)
        is_select = any("SELECT" in str(x) for x in cmd)
        class R:
            returncode = 0
            stderr = ""
            stdout = "5\n" if is_select else ""
        return R()

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    dep.deploy_dj(dj_id=7, dj_name="Sloan", moods=["neutral"],
                  persona_prompt="x", station_id=22,
                  schedule_start="10:00", schedule_end="14:00")

    updates = [c for c in ran if any("UPDATE" in str(x) for x in c)]
    inserts = [c for c in ran if any("INSERT" in str(x) for x in c)]
    assert updates, "non-empty SELECT must lead to an UPDATE"
    assert not inserts
    utext = " ".join(" ".join(c) for c in updates)
    assert "id=5" in utext
