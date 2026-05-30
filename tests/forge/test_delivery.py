from pathlib import Path

import pytest


def test_deliver_creates_folder_and_uploads(monkeypatch, tmp_path):
    calls = {"folders": [], "uploads": []}
    monkeypatch.setattr(
        "core.forge.delivery.create_folder",
        lambda path: calls["folders"].append(path) or {"success": True},
    )
    monkeypatch.setattr(
        "core.forge.delivery.upload_file",
        lambda path, content, content_type=None: calls["uploads"].append((path, content)) or {"success": True},
    )
    from core.forge import delivery
    local = tmp_path / "clip.mp4"
    local.write_bytes(b"data")
    remote = delivery.deliver(local, "Viral Album Videos/Processed")
    assert remote == "Content/Mainstay-RodWave/Viral Album Videos/Processed/clip.mp4"
    assert calls["folders"] == ["Content/Mainstay-RodWave/Viral Album Videos/Processed"]
    assert calls["uploads"] == [
        ("Content/Mainstay-RodWave/Viral Album Videos/Processed/clip.mp4", b"data")
    ]


def test_deliver_honors_explicit_filename(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr("core.forge.delivery.create_folder", lambda path: {"success": True})
    monkeypatch.setattr(
        "core.forge.delivery.upload_file",
        lambda path, content, content_type=None: captured.setdefault("path", path) or {"success": True},
    )
    from core.forge import delivery
    local = tmp_path / "raw.mp4"
    local.write_bytes(b"x")
    remote = delivery.deliver(local, "Ideas", filename="renamed.mp4")
    assert remote.endswith("/Ideas/renamed.mp4")
    assert captured["path"].endswith("/Ideas/renamed.mp4")


def test_deliver_tolerates_existing_folder(monkeypatch, tmp_path):
    def raising_create(path):
        raise RuntimeError("405 folder exists")

    uploaded = {}
    monkeypatch.setattr("core.forge.delivery.create_folder", raising_create)
    monkeypatch.setattr(
        "core.forge.delivery.upload_file",
        lambda path, content, content_type=None: uploaded.setdefault("ok", True) or {"success": True},
    )
    from core.forge import delivery
    local = tmp_path / "f.mp4"
    local.write_bytes(b"x")
    remote = delivery.deliver(local, "Ideas")  # must not raise despite folder error
    assert uploaded["ok"] is True
    assert remote.endswith("/Ideas/f.mp4")


def test_deliver_raises_on_upload_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("core.forge.delivery.create_folder", lambda path: {"success": True})

    def raising_upload(path, content, content_type=None):
        raise RuntimeError("boom")

    monkeypatch.setattr("core.forge.delivery.upload_file", raising_upload)
    from core.forge import delivery
    local = tmp_path / "f.mp4"
    local.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        delivery.deliver(local, "Ideas")
