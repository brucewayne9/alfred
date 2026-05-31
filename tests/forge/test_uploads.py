import os
from pathlib import Path
from core.forge import uploads


def test_save_and_resolve_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_UPLOAD_DIR", str(tmp_path))
    uid = uploads.save_upload(b"ID3audiobytes", "hook.mp3")
    assert uid and "/" not in uid
    p = uploads.get_upload_path(uid)
    assert p is not None and p.exists()
    assert p.read_bytes() == b"ID3audiobytes"
    assert p.suffix == ".mp3"


def test_get_missing_upload_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_UPLOAD_DIR", str(tmp_path))
    assert uploads.get_upload_path("deadbeef") is None


def test_rejects_path_traversal_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_UPLOAD_DIR", str(tmp_path))
    uid = uploads.save_upload(b"x", "../../etc/passwd")
    p = uploads.get_upload_path(uid)
    assert p is not None and tmp_path in p.parents
