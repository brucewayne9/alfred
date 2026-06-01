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


class _FakeNC:
    """In-memory stand-in for the Nextcloud WebDAV client: tracks the file tree
    as a set of paths so we can assert moves restore the original locations."""
    def __init__(self, paths):
        self.paths = set(paths)
        self.folders = set()

    def create_folder(self, path):
        self.folders.add(path.rstrip("/")); return {"ok": True}

    def move_file(self, src, dst):
        src = src.rstrip("/"); dst = dst.rstrip("/")
        moved = {p for p in self.paths if p == src or p.startswith(src + "/")}
        if not moved:
            moved = {src}  # a bare folder with no tracked children
        for p in moved:
            self.paths.discard(p); self.paths.add(dst + p[len(src):])
        return {"ok": True}

    def delete_file(self, path):
        path = path.rstrip("/")
        for p in {x for x in self.paths if x == path or x.startswith(path + "/")}:
            self.paths.discard(p)
        return {"ok": True}


def _patch_nc(monkeypatch, fake):
    from integrations.nextcloud import client as nc
    monkeypatch.setattr(nc, "create_folder", fake.create_folder)
    monkeypatch.setattr(nc, "move_file", fake.move_file)
    monkeypatch.setattr(nc, "delete_file", fake.delete_file)


def test_soft_delete_then_undo_restores(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "trash.db"))
    forge_db.init_db()
    orig = "Content/Mainstay-RodWave/Viral Music Verticals/Film Montage/m_1/src_1.mp4"
    fake = _FakeNC([orig]); _patch_nc(monkeypatch, fake)

    res = library.soft_delete([orig], kind="file", label="src_1.mp4")
    assert res["count"] == 1
    assert orig not in fake.paths  # moved out of the library
    assert any(p.startswith(library.TRASH_ROOT) for p in fake.paths)
    assert library.trash_state()["count"] == 1

    undone = library.undo_last()
    assert undone["restored"] == 1
    assert orig in fake.paths  # back where it was
    assert library.trash_state()["count"] == 0
    assert library.undo_last() is None


def test_batch_delete_hides_job_until_undo(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "batch.db"))
    forge_db.init_db()
    d = "Content/Mainstay-RodWave/Viral Music Verticals/Film Montage/m_9"
    jid = forge_jobs.enqueue("film_montage", {"caption": "hook"})
    forge_jobs._update(jid, status="done",
        result='{"format":"film_montage","delivered":1,"delivered_dirs":["%s"]}' % d)
    fake = _FakeNC([d + "/src_1.mp4"]); _patch_nc(monkeypatch, fake)

    assert any(r["id"] == jid for r in library.list_done_jobs())
    library.soft_delete([d], kind="batch", job_id=jid, label="hook")
    assert not any(r["id"] == jid for r in library.list_done_jobs())  # card hidden

    library.undo_last()
    assert any(r["id"] == jid for r in library.list_done_jobs())  # card back


def test_trash_prune_caps_recoverable(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "prune.db"))
    forge_db.init_db()
    monkeypatch.setattr(library, "TRASH_RETAIN", 3)
    fake = _FakeNC([]); _patch_nc(monkeypatch, fake)
    for i in range(6):
        p = f"Content/Mainstay-RodWave/x/f{i}.mp4"
        fake.paths.add(p)
        library.soft_delete([p], kind="file", label=f"f{i}", job_id=None)
    assert library.trash_state()["count"] == 3  # only the newest 3 are recoverable


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
