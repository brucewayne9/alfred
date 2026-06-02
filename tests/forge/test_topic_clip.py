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
    CAPTION_POSITIONS,
    CAPTION_STYLES,
    _ass_escape,
    _build_caption_events,
    _concat_segments,
    _cut_segment,
    _detect_has_video,
    _format_ass_time,
    _group_karaoke_lines,
    _write_ass_file,
    _write_karaoke_ass,
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


# ---------------------------------------------------------------------------
# _safe_drawtext — pure string sanitiser, no ffmpeg required
# ---------------------------------------------------------------------------

from core.forge.renderers.topic_clip import (  # noqa: E402
    _safe_drawtext,
    _build_variant_assemblies,
)


def test_safe_drawtext_apostrophe():
    """Straight apostrophe is replaced with U+2019, never a raw '."""
    result = _safe_drawtext("it's a can't")
    assert "'" not in result, "raw straight apostrophe found in output"
    # U+2019 RIGHT SINGLE QUOTATION MARK should be present
    assert "’" in result, "U+2019 not present after apostrophe replacement"


def test_safe_drawtext_colon_escaped():
    """Colon is escaped to \\: for ffmpeg drawtext."""
    result = _safe_drawtext("time: 10:30")
    assert "\\:" in result, "colon not escaped"
    # No unescaped colons should remain (strip escaped ones, check for bare :)
    stripped = result.replace("\\:", "")
    assert ":" not in stripped, "unescaped colon remains"


def test_safe_drawtext_no_raw_apostrophe():
    """Combined: no raw apostrophes and colons are escaped."""
    text = "it's a can't: test"
    result = _safe_drawtext(text)
    assert "'" not in result
    assert "\\:" in result


def test_safe_drawtext_percent_escaped():
    """Percent sign is escaped to \\% for ffmpeg drawtext."""
    result = _safe_drawtext("100% done")
    assert "\\%" in result


def test_safe_drawtext_backslash_escaped():
    """Backslash is doubled (escaped) for ffmpeg drawtext."""
    result = _safe_drawtext("path\\to\\file")
    assert "\\\\" in result


# ---------------------------------------------------------------------------
# _build_variant_assemblies — pure logic, no ffmpeg required
# ---------------------------------------------------------------------------

def _make_segments(n: int, score_base: float = 0.7) -> list[dict]:
    """Build *n* synthetic 7.5-second segments (sum=7.5n; well over 10s min)."""
    segs = []
    for i in range(n):
        segs.append({
            "start_s": float(i * 10),
            "end_s": float(i * 10 + 7.5),
            "text": f"Segment {i} text.",
            "score": round(score_base + i * 0.01, 3),
        })
    return segs


def test_build_variant_assemblies_count():
    """variant_count=4 returns exactly 4 items with structurally distinct orderings."""
    segs = _make_segments(4)  # 4 * 7.5 s = 30 s, within 10-60 band
    variants = _build_variant_assemblies(segs, 4)
    assert len(variants) == 4

    # V0 original vs V1 reverse must differ
    v0_order = [s["start_s"] for s in variants[0]["segments"]]
    v1_order = [s["start_s"] for s in variants[1]["segments"]]
    assert v0_order != v1_order, "V1 (reverse) not distinct from V0"

    # V2 hook-first: first segment must be the highest-score one
    max_score_start = max(segs, key=lambda s: s["score"])["start_s"]
    assert variants[2]["segments"][0]["start_s"] == max_score_start, (
        "V2 hook-first: first segment is not the highest-score segment"
    )


def test_build_variant_assemblies_single():
    """variant_count=1 returns exactly [V0] (original order only)."""
    segs = _make_segments(4)
    variants = _build_variant_assemblies(segs, 1)
    assert len(variants) == 1
    assert variants[0]["label"] == "original"


def test_variant_segments_independent():
    """Mutating one variant's segments list does not affect another variant."""
    segs = _make_segments(4)
    variants = _build_variant_assemblies(segs, 3)

    # Mutate V0's first segment
    variants[0]["segments"][0]["end_s"] = 9999.0

    # V1 and V2 must be unaffected
    for i in (1, 2):
        for seg in variants[i]["segments"]:
            assert seg["end_s"] != 9999.0, (
                f"Variant {i} was mutated when Variant 0 was modified"
            )


def test_build_variant_assemblies_labels():
    """Each variant has a non-empty unique label string."""
    segs = _make_segments(4)
    variants = _build_variant_assemblies(segs, 4)
    labels = [v["label"] for v in variants]
    assert len(set(labels)) == 4, "duplicate labels in variants"
    for lbl in labels:
        assert isinstance(lbl, str) and lbl, "empty label"


# ---------------------------------------------------------------------------
# Timed rolling captions (CLIP-03 — follow the speaker's words)
# ---------------------------------------------------------------------------


def test_format_ass_time():
    assert _format_ass_time(0) == "0:00:00.00"
    assert _format_ass_time(65.37) == "0:01:05.37"
    assert _format_ass_time(3661.5) == "1:01:01.50"
    # negative clamps to zero
    assert _format_ass_time(-5) == "0:00:00.00"


def test_ass_escape_neutralises_overrides_and_wraps():
    out = _ass_escape("hello {world} this is a fairly long caption line here")
    assert "{" not in out and "}" not in out
    # wrapped into <=2 lines joined by ASS hard-break
    assert out.count("\\N") <= 1


def test_build_caption_events_rebases_to_output_time():
    # One window [100,110]; two fine phrases inside it.
    window = [{"start_s": 100.0, "end_s": 110.0, "text": "full window text"}]
    fine = [
        {"start_s": 100.0, "end_s": 103.0, "text": "first phrase"},
        {"start_s": 103.0, "end_s": 108.0, "text": "second phrase"},
    ]
    ev = _build_caption_events(window, fine)
    assert len(ev) == 2
    # first event starts at output t=0 (rebased), not source t=100
    assert ev[0][0] == 0.0
    assert abs(ev[0][1] - 3.0) < 1e-6
    assert ev[0][2] == "first phrase"
    assert abs(ev[1][0] - 3.0) < 1e-6


def test_build_caption_events_reorder_follows_variant_order():
    # Two windows; variant order is reversed. Output offsets must follow
    # the GIVEN order, not source time.
    windows = [
        {"start_s": 200.0, "end_s": 205.0, "text": "B"},
        {"start_s": 100.0, "end_s": 105.0, "text": "A"},
    ]
    fine = [
        {"start_s": 100.0, "end_s": 105.0, "text": "alpha"},
        {"start_s": 200.0, "end_s": 205.0, "text": "beta"},
    ]
    ev = _build_caption_events(windows, fine)
    # First output event is from the FIRST given window (200-205 -> "beta")
    assert ev[0][2] == "beta"
    assert ev[0][0] == 0.0
    # Second window starts at output t=5
    assert abs(ev[1][0] - 5.0) < 1e-6
    assert ev[1][2] == "alpha"


def test_build_caption_events_fallback_when_no_fine():
    window = [{"start_s": 0.0, "end_s": 10.0, "text": "whole window caption"}]
    ev = _build_caption_events(window, [])
    assert len(ev) == 1
    assert ev[0] == (0.0, 10.0, "whole window caption")


def test_write_ass_file_has_styled_events(tmp_path):
    events = [(0.0, 2.0, "first"), (2.0, 4.0, "second")]
    out = _write_ass_file(events, tmp_path / "c.ass")
    body = out.read_text()
    assert "[V4+ Styles]" in body
    assert "Style: Cap,DejaVu Sans" in body
    # Two Dialogue lines, one per event
    assert body.count("Dialogue:") == 2
    assert "0:00:00.00" in body and "0:00:02.00" in body


# ---------------------------------------------------------------------------
# Caption style presets + karaoke (style picker)
# ---------------------------------------------------------------------------


def test_caption_style_presets_exist():
    for key in ("clean", "bold", "karaoke"):
        assert key in CAPTION_STYLES
    for key in ("lower", "center", "upper"):
        assert key in CAPTION_POSITIONS
    # only karaoke flips the word-by-word switch
    assert CAPTION_STYLES["karaoke"]["karaoke"] is True
    assert CAPTION_STYLES["clean"]["karaoke"] is False


def test_write_ass_bold_uppercases(tmp_path):
    events = [(0.0, 2.0, "hello world")]
    out = _write_ass_file(events, tmp_path / "b.ass", style="bold")
    body = out.read_text()
    assert "HELLO WORLD" in body          # bold preset uppercases
    assert "Dialogue:" in body


def test_write_ass_position_changes_alignment(tmp_path):
    events = [(0.0, 2.0, "x")]
    lower = _write_ass_file(events, tmp_path / "l.ass", position="lower").read_text()
    upper = _write_ass_file(events, tmp_path / "u.ass", position="upper").read_text()
    # alignment is the 19th Style field; lower=2, upper=8 — files must differ
    assert lower != upper
    assert ",8,80,80,220," in upper       # upper alignment + marginV


def test_group_karaoke_lines_breaks_on_pause_and_count():
    words = [
        {"word": "a", "start": 0.0, "end": 0.3},
        {"word": "b", "start": 0.3, "end": 0.6},
        {"word": "c", "start": 2.0, "end": 2.3},   # >0.6s gap -> new line
    ]
    lines = _group_karaoke_lines(words, max_words=5, gap_break=0.6)
    assert len(lines) == 2
    assert [w["word"] for w in lines[0]] == ["a", "b"]
    assert [w["word"] for w in lines[1]] == ["c"]


def test_write_karaoke_ass_has_kf_tags(tmp_path):
    words = [
        {"word": "first", "start": 0.0, "end": 0.5},
        {"word": "second", "start": 0.5, "end": 1.1},
    ]
    out = _write_karaoke_ass(words, tmp_path / "k.ass")
    body = out.read_text()
    assert "\\kf" in body                 # libass karaoke fill tags
    assert "FIRST" in body and "SECOND" in body   # karaoke preset uppercases
    assert body.count("Dialogue:") == 1   # both words on one line
