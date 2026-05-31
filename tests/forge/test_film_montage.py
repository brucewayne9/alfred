from core.forge.renderers.film_montage import plan_segments


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
