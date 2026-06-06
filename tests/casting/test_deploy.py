# tests/casting/test_deploy.py
import importlib
import core.casting.deploy as dep

def test_deploy_builds_scp_and_sql(monkeypatch, tmp_path):
    importlib.reload(dep)
    # create fake mood wavs
    vdir = tmp_path / "7"; vdir.mkdir()
    for m in ["neutral", "fired"]:
        (vdir / f"{m}.wav").write_bytes(b"RIFF")
    from config.settings import settings
    monkeypatch.setattr(settings, "casting_voices_dir", str(tmp_path), raising=False)

    ran = []
    def fake_run(cmd, **kw):
        ran.append(cmd)
        class R:
            returncode = 0; stdout = ""; stderr = ""
        return R()
    monkeypatch.setattr(dep.subprocess, "run", fake_run)

    dep.deploy_dj(dj_id=7, base_name="Sloan", moods=["neutral", "fired"],
                  persona_prompt="warm realist", station_id=22)

    # an scp/rsync push happened with timeout wrapping
    pushed = " ".join(" ".join(c) for c in ran)
    assert "Sloan_neutral.wav" in pushed and "Sloan_fired.wav" in pushed
    # a mariadb upsert happened referencing the breaks table + station 22
    assert "station_ai_dj_breaks" in pushed and "22" in pushed
    assert pushed.count("timeout") >= 1

def test_deploy_slugs_apostrophe_name(monkeypatch, tmp_path):
    importlib.reload(dep)
    vdir = tmp_path / "9"; vdir.mkdir()
    (vdir / "neutral.wav").write_bytes(b"RIFF")
    from config.settings import settings
    monkeypatch.setattr(settings, "casting_voices_dir", str(tmp_path), raising=False)

    ran = []
    def fake_run(cmd, **kw):
        ran.append(cmd)
        class R:
            returncode = 0; stdout = ""; stderr = ""
        return R()
    monkeypatch.setattr(dep.subprocess, "run", fake_run)

    dep.deploy_dj(dj_id=9, base_name="O'Brien", moods=["neutral"],
                  persona_prompt="warm realist", station_id=22)

    pushed = " ".join(" ".join(c) for c in ran)
    # apostrophe stripped from the wav filename
    assert "OBrien_neutral.wav" in pushed
    # the SQL must not carry a raw apostrophe from the (now sanitized) base name
    sql_cmds = [c for c in ran if any("INSERT INTO station_ai_dj_breaks" in str(x) for x in c)]
    assert sql_cmds, "no SQL upsert ran"
    sql_text = " ".join(" ".join(c) for c in sql_cmds)
    assert "'OBrien Show'" in sql_text
    assert "O'Brien" not in sql_text
