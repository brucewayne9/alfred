from pathlib import Path
import core.forge.delivery as delivery


def test_deliver_creates_each_path_segment(tmp_path, monkeypatch):
    created, uploaded = [], []
    monkeypatch.setattr(delivery, "create_folder", lambda p: created.append(p))
    monkeypatch.setattr(delivery, "upload_file", lambda remote, data: uploaded.append(remote))
    f = tmp_path / "clip.mp4"; f.write_bytes(b"vid")

    delivery.deliver(str(f), "Viral Music Verticals/Kinetic Lyric", filename="a.mp4")

    assert any(p.endswith("/Viral Music Verticals") for p in created)
    assert any(p.endswith("/Viral Music Verticals/Kinetic Lyric") for p in created)
    assert uploaded and uploaded[0].endswith("/Viral Music Verticals/Kinetic Lyric/a.mp4")
