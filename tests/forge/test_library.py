import pytest
from core.forge import library, jobs as forge_jobs, db as forge_db


def test_safe_path_accepts_under_root():
    assert library._safe_library_path("Content/Mainstay-RodWave/x/a.mp4") == "Content/Mainstay-RodWave/x/a.mp4"
    assert library._safe_library_path("/Content/Mainstay-RodWave/x/a.mp4").startswith("Content/")


def test_safe_path_rejects_outside_root():
    with pytest.raises(ValueError):
        library._safe_library_path("Content/SomethingElse/a.mp4")
    with pytest.raises(ValueError):
        library._safe_library_path("Content/Mainstay-RodWave/../../etc/passwd")


def test_list_done_jobs_shapes_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "lib.db"))
    forge_db.init_db()
    jid = forge_jobs.enqueue("kinetic_lyric", {"caption": "lonely"})
    forge_jobs._update(jid, status="done",
        result='{"format":"kinetic_lyric","remix_looks":2,"variations_each":2,"delivered":6,'
               '"delivered_dirs":["Content/Mainstay-RodWave/Viral Music Verticals/Kinetic Lyric/k_1_look00",'
               '"Content/Mainstay-RodWave/Viral Music Verticals/Kinetic Lyric/k_1_look01"]}')
    rows = library.list_done_jobs()
    j = [r for r in rows if r["id"] == jid][0]
    assert j["format"] == "kinetic_lyric"
    assert j["caption"] == "lonely"
    assert len(j["dirs"]) == 2
    assert j["delivered"] == 6


def test_list_done_jobs_handles_singular_delivered_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "lib2.db"))
    forge_db.init_db()
    jid = forge_jobs.enqueue("leak_graphic", {"caption": "drop"})
    forge_jobs._update(jid, status="done",
        result='{"format":"leak_graphic","variant_count":18,"delivered":19,'
               '"delivered_dir":"Content/Mainstay-RodWave/Viral Album Videos/Processed/leak_1"}')
    j = [r for r in library.list_done_jobs() if r["id"] == jid][0]
    assert j["dirs"] == ["Content/Mainstay-RodWave/Viral Album Videos/Processed/leak_1"]
