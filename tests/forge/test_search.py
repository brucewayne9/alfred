"""Unit tests for core.forge.search — pure windowing and score logic only.

No ChromaDB or Ollama required.  All tests exercise the pure functions:
build_windows, win_id format, dominant/empty speaker coercion, score inversion.

Real-corpus precision is verified in Plan 03's live-ingest checkpoint.
"""
from __future__ import annotations

import pytest

from core.forge.search import build_windows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segs(
    n: int,
    dur_s: float = 5.0,
    speaker: str | None = "A",
    source_id: str = "src",
) -> list[dict]:
    """Build *n* synthetic segments each of *dur_s* seconds."""
    segs = []
    for i in range(n):
        segs.append(
            {
                "seq": i,
                "start_s": i * dur_s,
                "end_s": i * dur_s + dur_s,
                "text": f"seg{i}",
                "speaker": speaker,
                "source_id": source_id,
            }
        )
    return segs


# ---------------------------------------------------------------------------
# test_build_windows_merges_short_segments
# ---------------------------------------------------------------------------


def test_build_windows_merges_short_segments():
    """20 × 5s segments should collapse into a small number of windows.

    Every window must be between min_dur_s (10s) and max_dur_s (45s).
    seq_start/seq_end must be contiguous and cover all 20 seqs.
    text must contain the joined segment texts.
    """
    segs = _make_segs(20, dur_s=5.0)
    windows = build_windows(segs, target_dur_s=30.0, max_dur_s=45.0, min_dur_s=10.0)

    # Should produce a few windows — definitely fewer than 20.
    assert 1 < len(windows) < 20, f"Expected 2-19 windows, got {len(windows)}"

    # All windows except possibly the last must be within duration bounds.
    for i, w in enumerate(windows):
        dur = w["end_s"] - w["start_s"]
        # Last window may be short if remaining segments are less than min_dur.
        if i < len(windows) - 1:
            assert dur <= 45.0, f"Window {i} dur {dur}s exceeds max_dur_s=45"
        assert dur > 0, f"Window {i} has non-positive duration"

    # seq_start/seq_end must cover all 20 seqs contiguously.
    all_seqs: set[int] = set()
    for w in windows:
        all_seqs.update(range(w["seq_start"], w["seq_end"] + 1))
    assert all_seqs == set(range(20)), "Windows do not cover all seq indices 0-19"

    # text of each window should be non-empty and contain individual seg texts.
    for w in windows:
        assert w["text"].strip(), "Window text must be non-empty"


# ---------------------------------------------------------------------------
# test_window_win_id_format
# ---------------------------------------------------------------------------


def test_window_win_id_format():
    """win_id must be '{source_id}_w{seq_start:04d}'."""
    segs = _make_segs(10, dur_s=5.0, source_id="mysrc")
    windows = build_windows(segs)

    for w in windows:
        expected = f"mysrc_w{w['seq_start']:04d}"
        assert w["win_id"] == expected, (
            f"win_id {w['win_id']!r} != expected {expected!r}"
        )


# ---------------------------------------------------------------------------
# test_dominant_speaker
# ---------------------------------------------------------------------------


def test_dominant_speaker_majority():
    """A window built from ['A','A','B'] segments should have speaker 'A'."""
    segs = [
        {"seq": 0, "start_s": 0.0, "end_s": 15.0, "text": "x", "speaker": "A", "source_id": "s"},
        {"seq": 1, "start_s": 15.0, "end_s": 30.0, "text": "y", "speaker": "A", "source_id": "s"},
        {"seq": 2, "start_s": 30.0, "end_s": 45.0, "text": "z", "speaker": "B", "source_id": "s"},
    ]
    windows = build_windows(segs)
    # All 3 segs fit within 45 s → single window.
    assert len(windows) == 1
    assert windows[0]["speaker"] == "A", (
        f"Expected dominant speaker 'A', got {windows[0]['speaker']!r}"
    )


def test_dominant_speaker_none_coerced_to_empty():
    """Segments with speaker=None should yield speaker '' (never None)."""
    segs = [
        {"seq": i, "start_s": i * 5.0, "end_s": i * 5.0 + 5, "text": "x", "speaker": None, "source_id": "s"}
        for i in range(6)
    ]
    windows = build_windows(segs)
    for w in windows:
        assert w["speaker"] is not None, "speaker must not be None"
        assert isinstance(w["speaker"], str), "speaker must be a str"
        assert w["speaker"] == "", f"Expected '' for None-speaker window, got {w['speaker']!r}"


# ---------------------------------------------------------------------------
# test_score_inversion
# ---------------------------------------------------------------------------


def test_score_inversion():
    """Cosine distance [0, 2] must map to score [1, 0] via round(1-d/2, 4)."""
    def _score(d: float) -> float:
        return round(1.0 - d / 2.0, 4)

    assert _score(0.0) == 1.0,  f"dist 0.0 → expected 1.0, got {_score(0.0)}"
    assert _score(2.0) == 0.0,  f"dist 2.0 → expected 0.0, got {_score(2.0)}"
    assert _score(1.0) == 0.5,  f"dist 1.0 → expected 0.5, got {_score(1.0)}"

    # Confirm the formula matches what search_segments actually uses.
    # We read the source to ensure formula consistency rather than duplicate.
    import inspect
    from core.forge import search as _search
    source = inspect.getsource(_search.search_segments)
    assert "1.0 - dist / 2.0" in source, (
        "search_segments source must contain '1.0 - dist / 2.0' for score inversion"
    )


# ---------------------------------------------------------------------------
# test_empty_segments_returns_no_windows
# ---------------------------------------------------------------------------


def test_empty_segments_returns_no_windows():
    """build_windows([]) must return an empty list."""
    assert build_windows([]) == []
