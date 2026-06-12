from core.forge.renderers.film_montage import (
    plan_segments,
    assign_offsets,
    build_xfade_filter,
)


def test_xfade_filter_uses_requested_transition():
    filt, _final = build_xfade_filter([2.0, 2.0, 2.0], transition="fade", fade=0.18)
    assert all("transition=fade" in f for f in filt)


def test_directional_transition_alternates_left_right():
    # Whip/wipe/blur read better when successive cuts push opposite directions.
    filt, _final = build_xfade_filter(
        [2.0, 2.0, 2.0], transition="slideleft", fade=0.18, directional=True
    )
    assert "transition=slideleft" in filt[0]
    assert "transition=slideright" in filt[1]


def test_xfade_first_offset_accounts_for_fade_overlap():
    filt, final = build_xfade_filter([2.0, 2.0], transition="fade", fade=0.2)
    assert "offset=1.800" in filt[0]  # 2.0 - 0.2
    assert final == "vx1"


def test_covers_hook_with_enough_segments():
    segs = plan_segments(clip_count=3, hook_seconds=12.0, seg_seconds=2.5)
    assert sum(s["seconds"] for s in segs) >= 12.0
    assert all(0 <= s["clip_index"] < 3 for s in segs)


def test_cycles_through_clips_round_robin():
    segs = plan_segments(clip_count=2, hook_seconds=10.0, seg_seconds=2.5)
    idxs = [s["clip_index"] for s in segs]
    assert idxs[:4] == [0, 1, 0, 1]


def test_last_segment_trimmed_to_exact_length():
    segs = plan_segments(clip_count=1, hook_seconds=5.0, seg_seconds=2.0)
    assert abs(sum(s["seconds"] for s in segs) - 5.0) < 0.01


def test_raises_on_no_clips():
    import pytest
    with pytest.raises(ValueError):
        plan_segments(clip_count=0, hook_seconds=5.0, seg_seconds=2.5)


def test_single_long_clip_samples_distinct_moments():
    # One 60s clip reused across many 2.5s segments -> offsets must spread,
    # not all sit at the same spot (the bug that caused the loop).
    segs = plan_segments(clip_count=1, hook_seconds=15.0, seg_seconds=2.5)
    out = assign_offsets(segs, durations=[60.0])
    offs = [s["offset"] for s in out]
    assert len(set(round(o, 1) for o in offs)) > 1, "offsets should differ"
    assert offs == sorted(offs), "should walk forward through the video"
    assert max(offs) >= 30.0, "should reach deep into the 60s source"


def test_offsets_never_exceed_playable_span():
    segs = plan_segments(clip_count=1, hook_seconds=12.0, seg_seconds=2.5)
    out = assign_offsets(segs, durations=[10.0])  # short source
    for s in out:
        assert s["offset"] <= 10.0 - s["seconds"] + 1e-6
        assert s["offset"] >= 0.0


def test_multiple_clips_spread_independently():
    segs = plan_segments(clip_count=2, hook_seconds=20.0, seg_seconds=2.5)
    out = assign_offsets(segs, durations=[60.0, 30.0])
    by_clip = {0: [], 1: []}
    for s in out:
        by_clip[s["clip_index"]].append(s["offset"])
    assert len(set(round(o, 1) for o in by_clip[0])) > 1
    assert len(set(round(o, 1) for o in by_clip[1])) > 1
    assert max(by_clip[1]) <= 30.0
