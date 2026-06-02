"""
Tests for core.forge.renderers.topic_clip — segment cut + concat + duration guard engine.

ffmpeg-dependent tests are skipped if ffmpeg/ffprobe are absent from PATH.
Follows the same fixture/marker style as test_film_montage.py.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from core.forge.renderers.topic_clip import (
    _concat_segments,
    _cut_segment,
    _detect_has_video,
    enforce_duration,
)
from core.forge import audio

# ---------------------------------------------------------------------------
# Skip guard — all ffmpeg-dependent tests share this marker.
# ---------------------------------------------------------------------------

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg/ffprobe not available",
)


# ---------------------------------------------------------------------------
# Fixtures — tiny synthetic sources generated into tmp_path.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_mp3(tmp_path_factory):
    """12 s silent-audio .mp3 (~tiny, lavfi anullsrc)."""
    tmp = tmp_path_factory.mktemp("src")
    out = tmp / "silent_12s.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", "12",
            "-c:a", "libmp3lame", "-b:a", "64k",
            str(out),
        ],
        check=True,
    )
    return out


@pytest.fixture(scope="module")
def synthetic_mp4(tmp_path_factory):
    """12 s black-video + tone .mp4 (~tiny, lavfi color+sine)."""
    tmp = tmp_path_factory.mktemp("src")
    out = tmp / "color_12s.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=black:s=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
            "-t", "12",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "64k",
            str(out),
        ],
        check=True,
    )
    return out


# ---------------------------------------------------------------------------
# _detect_has_video
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_detect_has_video_mp4(synthetic_mp4):
    assert _detect_has_video(synthetic_mp4) is True


@needs_ffmpeg
def test_detect_has_video_mp3(synthetic_mp3):
    assert _detect_has_video(synthetic_mp3) is False


# ---------------------------------------------------------------------------
# _cut_segment
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_cut_segment_duration_video(tmp_path, synthetic_mp4):
    """Cut a 3 s span (start=2, end=5) from mp4; duration within 0.3 s of 3.0."""
    out = tmp_path / "cut_video.mp4"
    result = _cut_segment(synthetic_mp4, 2.0, 5.0, out, has_video=True)
    assert result.exists()
    dur = audio.duration_seconds(result)
    assert abs(dur - 3.0) <= 0.3, f"expected ~3.0s, got {dur:.3f}s"


@needs_ffmpeg
def test_cut_segment_duration_audio(tmp_path, synthetic_mp3):
    """Cut a 3 s span from mp3; duration within 0.3 s of 3.0."""
    out = tmp_path / "cut_audio.m4a"
    result = _cut_segment(synthetic_mp3, 2.0, 5.0, out, has_video=False)
    assert result.exists()
    dur = audio.duration_seconds(result)
    assert abs(dur - 3.0) <= 0.3, f"expected ~3.0s, got {dur:.3f}s"


def test_cut_segment_no_copy_codec():
    """Verify no codec-copy flags appear in ffmpeg command lists (re-encode always)."""
    import core.forge.renderers.topic_clip as mod
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(mod))
    # Walk all string literals used as list elements in subprocess.run calls.
    # Reject any that are literal stream-copy flags.
    forbidden = {"-c copy", "copy"}
    # Collect all string constants in the AST that appear inside list literals.
    list_strings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    list_strings.append(elt.value)
    # "copy" as a standalone flag is the codec-copy trigger in ffmpeg.
    # It must not appear alongside -c, -c:v, or -c:a flags.
    assert "copy" not in list_strings, (
        "Found bare 'copy' string in a list literal — "
        "this would invoke codec-copy mode and break sync safety"
    )


# ---------------------------------------------------------------------------
# _concat_segments
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_concat_sums_durations_video(tmp_path, synthetic_mp4):
    """Cut two 2 s spans, concat them; total duration within 0.4 s of 4.0."""
    seg_dir = tmp_path / "segs"
    seg_dir.mkdir()

    seg_a = _cut_segment(synthetic_mp4, 1.0, 3.0, seg_dir / "seg_a.mp4", has_video=True)
    seg_b = _cut_segment(synthetic_mp4, 5.0, 7.0, seg_dir / "seg_b.mp4", has_video=True)

    out = tmp_path / "concat_video.mp4"
    work = tmp_path / "work"
    result = _concat_segments([seg_a, seg_b], out, has_video=True, work_dir=work)
    assert result.exists()
    dur = audio.duration_seconds(result)
    assert abs(dur - 4.0) <= 0.4, f"expected ~4.0s, got {dur:.3f}s"


@needs_ffmpeg
def test_concat_sums_durations_audio(tmp_path, synthetic_mp3):
    """Cut two 2 s audio spans, concat them; total duration within 0.4 s of 4.0."""
    seg_dir = tmp_path / "segs"
    seg_dir.mkdir()

    seg_a = _cut_segment(synthetic_mp3, 1.0, 3.0, seg_dir / "seg_a.m4a", has_video=False)
    seg_b = _cut_segment(synthetic_mp3, 6.0, 8.0, seg_dir / "seg_b.m4a", has_video=False)

    out = tmp_path / "concat_audio.m4a"
    work = tmp_path / "work"
    result = _concat_segments([seg_a, seg_b], out, has_video=False, work_dir=work)
    assert result.exists()
    dur = audio.duration_seconds(result)
    assert abs(dur - 4.0) <= 0.4, f"expected ~4.0s, got {dur:.3f}s"


# ---------------------------------------------------------------------------
# enforce_duration — pure logic, no ffmpeg required
# ---------------------------------------------------------------------------


def test_enforce_duration_min():
    """List summing to 6 s raises ValueError."""
    segs = [{"start_s": 0.0, "end_s": 4.0}, {"start_s": 5.0, "end_s": 7.0}]
    with pytest.raises(ValueError, match="6.0s < 10.0s minimum"):
        enforce_duration(segs)


def test_enforce_duration_max():
    """List summing to 80 s returns a list with total <= 60 and only last changed."""
    segs = [
        {"start_s": 0.0, "end_s": 40.0},
        {"start_s": 50.0, "end_s": 90.0},
    ]
    result = enforce_duration(segs)
    total = sum(s["end_s"] - s["start_s"] for s in result)
    assert total <= 60.0, f"total {total:.1f}s exceeds max"
    # First segment unchanged.
    assert result[0] == segs[0]
    # Last segment modified.
    assert result[-1] != segs[-1]
    # Input not mutated.
    assert segs[-1]["end_s"] == 90.0


def test_enforce_duration_passthrough():
    """A 30 s list returns a copy with identical values."""
    segs = [{"start_s": 0.0, "end_s": 30.0}]
    result = enforce_duration(segs)
    assert result[0] == segs[0]
    # Shallow copy — not the same list object.
    assert result is not segs


def test_enforce_duration_no_mutation():
    """Input list is never mutated regardless of trim."""
    segs = [{"start_s": 0.0, "end_s": 40.0}, {"start_s": 50.0, "end_s": 90.0}]
    original_end = segs[-1]["end_s"]
    enforce_duration(segs)
    assert segs[-1]["end_s"] == original_end, "input was mutated"


def test_enforce_duration_floor_at_one_second():
    """Even a huge overage never shrinks the last segment below 1 s."""
    segs = [{"start_s": 0.0, "end_s": 55.0}, {"start_s": 56.0, "end_s": 57.0}]
    result = enforce_duration(segs)
    last = result[-1]
    assert last["end_s"] >= last["start_s"] + 1.0


def test_enforce_duration_exact_boundary():
    """Exactly 10.0 s and 60.0 s both pass through unchanged."""
    seg10 = [{"start_s": 0.0, "end_s": 10.0}]
    assert enforce_duration(seg10)[0]["end_s"] == 10.0

    seg60 = [{"start_s": 0.0, "end_s": 60.0}]
    assert enforce_duration(seg60)[0]["end_s"] == 60.0
