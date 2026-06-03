"""Tests for core.forge.renderers.multi_montage — hand-pick multi-source montage.

Pure-logic tests only (no ffmpeg): caption-event accumulation across sources,
source resolution errors, and duration-band enforcement on picks.
"""
from __future__ import annotations

import pytest

from core.forge.renderers.multi_montage import (
    build_multiclip_events,
    _resolve_source_path,
)
from core.forge.renderers.topic_clip import enforce_duration


def test_build_multiclip_events_accumulates_offset_across_sources():
    """Each pick's caption events are rebased onto the running output timeline."""
    picks = [
        {"source_id": "A", "start_s": 100.0, "end_s": 105.0, "text": "alpha"},
        {"source_id": "B", "start_s": 200.0, "end_s": 206.0, "text": "beta"},
    ]
    fine_by_source = {
        "A": [{"start_s": 100.0, "end_s": 105.0, "text": "alpha phrase"}],
        "B": [{"start_s": 200.0, "end_s": 206.0, "text": "beta phrase"}],
    }
    ev = build_multiclip_events(picks, fine_by_source)
    assert len(ev) == 2
    # First pick starts at output t=0
    assert ev[0][0] == 0.0
    assert ev[0][2] == "alpha phrase"
    # Second pick starts at output t=5 (length of first pick), NOT source t=200
    assert abs(ev[1][0] - 5.0) < 1e-6
    assert ev[1][2] == "beta phrase"


def test_build_multiclip_events_falls_back_to_pick_text_without_fine():
    picks = [{"source_id": "A", "start_s": 0.0, "end_s": 8.0, "text": "no fine data"}]
    ev = build_multiclip_events(picks, {})
    assert len(ev) == 1
    assert ev[0] == (0.0, 8.0, "no fine data")


def test_resolve_source_path_raises_on_unknown(monkeypatch):
    import core.forge.ingest as _ingest
    monkeypatch.setattr(_ingest, "get_source", lambda sid: None)
    with pytest.raises(RuntimeError, match="source not found"):
        _resolve_source_path("nope")


def test_resolve_source_path_raises_on_missing_file(monkeypatch, tmp_path):
    import core.forge.ingest as _ingest
    monkeypatch.setattr(
        _ingest, "get_source",
        lambda sid: {"id": sid, "file_path": str(tmp_path / "gone.mp4")},
    )
    with pytest.raises(RuntimeError, match="missing on disk"):
        _resolve_source_path("x")


def test_picks_obey_duration_band():
    """enforce_duration trims an over-long pick list to <=60s (same engine)."""
    picks = [
        {"source_id": "A", "start_s": 0.0, "end_s": 40.0, "text": "a"},
        {"source_id": "B", "start_s": 0.0, "end_s": 40.0, "text": "b"},
    ]
    out = enforce_duration(picks)
    total = sum(p["end_s"] - p["start_s"] for p in out)
    assert total <= 60.0
    # input not mutated
    assert picks[1]["end_s"] == 40.0


def test_resolve_bed_none_when_absent():
    from core.forge.renderers.multi_montage import _resolve_bed
    assert _resolve_bed({}) is None


def test_resolve_bed_path_missing_raises(tmp_path):
    from core.forge.renderers.multi_montage import _resolve_bed
    import pytest as _pt
    with _pt.raises(RuntimeError, match="bed audio path missing"):
        _resolve_bed({"bed_audio_path": str(tmp_path / "nope.mp3")})
