# tests/casting/test_deploy.py
import importlib
import pytest
import core.casting.deploy as dep


@pytest.fixture()
def patched(monkeypatch):
    """Reload deploy, stub register_to_engine (no real fs writes) and capture
    every subprocess.run command + the SQL it received on stdin. SELECT returns
    empty stdout by default (=> INSERT path).

    The SQL travels via stdin (subprocess.run(..., input=sql)), NOT as an argv
    element — ssh would otherwise re-parse it through the remote shell.
    """
    importlib.reload(dep)
    monkeypatch.setattr(dep.voice, "register_to_engine",
                        lambda dj_id, moods: {"neutral": f"cc{dj_id}_neutral"})

    calls = []  # list of {"cmd": [...], "sql": "..."}

    def fake_run(cmd, **kw):
        calls.append({"cmd": cmd, "sql": kw.get("input") or ""})
        class R:
            returncode = 0
            stderr = ""
            stdout = ""  # empty SELECT => no existing row => INSERT path
        return R()

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    return calls


def _sql_of(calls, kind):
    return [c for c in calls if kind in c["sql"]]


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
    calls = patched
    dep.deploy_dj(dj_id=7, dj_name="Sloan", moods=["neutral"],
                  persona_prompt="warm realist", station_id=22,
                  schedule_start="10:00", schedule_end="14:00", enabled=False)

    all_sql = " ".join(c["sql"] for c in calls)
    # references the real table + station
    assert "station_ai_dj_breaks" in all_sql
    assert "22" in all_sql
    # qwen provider + id-namespaced voice NAME
    assert "qwen" in all_sql
    assert "cc7_neutral" in all_sql
    # disabled => is_enabled 0 in the INSERT VALUES
    insert = _sql_of(calls, "INSERT")
    assert insert, "expected an INSERT (empty SELECT => insert path)"
    itext = insert[0]["sql"]
    assert ", 0," in itext or ",0," in itext
    # the SQL is delivered on stdin, never as a -e argv element
    assert all("-e" not in c["cmd"] for c in calls)
    # all shell calls wrapped in timeout, and go through ssh
    assert all(c["cmd"][0] == "timeout" for c in calls)
    assert all("ssh" in c["cmd"] for c in calls)


def test_deploy_enabled_sets_enabled_one(patched):
    calls = patched
    dep.deploy_dj(dj_id=7, dj_name="Sloan", moods=["neutral"],
                  persona_prompt="warm realist", station_id=22,
                  schedule_start="10:00", schedule_end="14:00", enabled=True)
    insert = _sql_of(calls, "INSERT")
    assert insert
    itext = insert[0]["sql"]
    assert ", 1," in itext or ",1," in itext


def test_deploy_escapes_apostrophe_in_persona(patched):
    calls = patched
    dep.deploy_dj(dj_id=7, dj_name="Sloan", moods=["neutral"],
                  persona_prompt="don't quit", station_id=22,
                  schedule_start="10:00", schedule_end="14:00")
    all_sql = " ".join(c["sql"] for c in calls)
    # escaped doubled-apostrophe present, raw unescaped not present
    assert "don''t quit" in all_sql
    assert "don't quit" not in all_sql


def test_select_then_insert_when_no_existing(monkeypatch):
    importlib.reload(dep)
    monkeypatch.setattr(dep.voice, "register_to_engine",
                        lambda dj_id, moods: {"neutral": f"cc{dj_id}_neutral"})

    calls = []

    def fake_run(cmd, **kw):
        sql = kw.get("input") or ""
        calls.append({"cmd": cmd, "sql": sql})
        class R:
            returncode = 0
            stderr = ""
            stdout = ""  # empty SELECT => insert
        return R()

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    dep.deploy_dj(dj_id=7, dj_name="Sloan", moods=["neutral"],
                  persona_prompt="x", station_id=22,
                  schedule_start="10:00", schedule_end="14:00")

    assert _sql_of(calls, "SELECT"), "a SELECT must run first"
    assert _sql_of(calls, "INSERT"), "empty SELECT must lead to an INSERT"
    assert not _sql_of(calls, "UPDATE")


def test_select_then_update_when_existing(monkeypatch):
    importlib.reload(dep)
    monkeypatch.setattr(dep.voice, "register_to_engine",
                        lambda dj_id, moods: {"neutral": f"cc{dj_id}_neutral"})

    calls = []

    def fake_run(cmd, **kw):
        sql = kw.get("input") or ""
        calls.append({"cmd": cmd, "sql": sql})
        is_select = "SELECT" in sql
        class R:
            returncode = 0
            stderr = ""
            stdout = "5\n" if is_select else ""
        return R()

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    dep.deploy_dj(dj_id=7, dj_name="Sloan", moods=["neutral"],
                  persona_prompt="x", station_id=22,
                  schedule_start="10:00", schedule_end="14:00")

    updates = _sql_of(calls, "UPDATE")
    assert updates, "non-empty SELECT must lead to an UPDATE"
    assert not _sql_of(calls, "INSERT")
    assert "id=5" in updates[0]["sql"]
