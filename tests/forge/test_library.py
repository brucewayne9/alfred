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


def test_list_dir_files_missing_folder_returns_empty(monkeypatch):
    # A job recorded a delivered dir that never got created on Nextcloud
    # (delivery failed). Listing it must degrade to [], not raise a 500.
    import requests
    from integrations.nextcloud import client as nc

    def _raise_404(path, depth=1):
        resp = requests.Response()
        resp.status_code = 404
        raise requests.exceptions.HTTPError("404 Not Found", response=resp)

    monkeypatch.setattr(nc, "list_files", _raise_404)
    assert library.list_dir_files("Content/Mainstay-RodWave/Viral Album Videos/Processed/gone") == []


def test_list_dir_files_real_error_propagates(monkeypatch):
    # A genuine server error (500) should NOT be silently swallowed.
    import requests
    from integrations.nextcloud import client as nc

    def _raise_500(path, depth=1):
        resp = requests.Response()
        resp.status_code = 500
        raise requests.exceptions.HTTPError("500 Server Error", response=resp)

    monkeypatch.setattr(nc, "list_files", _raise_500)
    with pytest.raises(requests.exceptions.HTTPError):
        library.list_dir_files("Content/Mainstay-RodWave/x/y")
