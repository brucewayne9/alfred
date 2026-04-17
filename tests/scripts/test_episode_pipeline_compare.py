"""Smoke test for compare_rigs_for_clip — verifies it calls the render
function twice with the two composition ids. Uses monkeypatch, no real render."""
from pathlib import Path
from unittest.mock import MagicMock
import pytest

import scripts.rucktalk_episode_pipeline as pipeline


def test_compare_rigs_calls_both(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_render(raw_clip_path, output_path, *args, **kwargs):
        # Infer which rig was requested from the module-level EPISODE_RIG at call time
        calls.append(pipeline.EPISODE_RIG)
        output_path.write_bytes(b"fake mp4")
        return True

    monkeypatch.setattr(pipeline, "_render_branded_clip", fake_render)
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"x")
    out_dir = tmp_path / "out"

    old_path, new_path = pipeline.compare_rigs_for_clip(
        raw_clip_path=raw,
        output_dir=out_dir,
        episode_number=42,
        episode_title="Test",
        context_line="ctx",
        host_name="MIKE",
        guest_name=None,
        transcript={"segments": []},
        clip_start=0.0,
        clip_end=5.0,
        duration_frames=150,
    )

    assert calls == ["RuckTalkClip", "MagazineRig"]
    assert old_path.name == "ep42_rucktalkclip.mp4"
    assert new_path.name == "ep42_magazinerig.mp4"
    assert old_path.exists() and new_path.exists()


def test_compare_rigs_restores_flag(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pipeline, "_render_branded_clip",
                        lambda *a, **k: a[1].write_bytes(b"x") or True)
    pipeline.EPISODE_RIG = "RuckTalkClip"

    raw = tmp_path / "raw.mp4"; raw.write_bytes(b"x")
    pipeline.compare_rigs_for_clip(
        raw_clip_path=raw, output_dir=tmp_path / "out",
        episode_number=1, episode_title="T", context_line="c",
        host_name="M", guest_name=None, transcript={"segments": []},
        clip_start=0.0, clip_end=1.0, duration_frames=30,
    )

    assert pipeline.EPISODE_RIG == "RuckTalkClip"  # restored
