"""Speaker-aware reframe — pure crop/segment/smoothing logic (no ffmpeg/ML)."""
from core.forge.renderers import reframe


# ---- crop_window: 9:16 window from a face bbox ----

def test_crop_window_landscape_is_correct_size_and_even():
    # 1920x1080 source -> max 9:16 region is height-limited: w = 1080*9/16 = 607.5 -> 608 (even)
    x, y, w, h = reframe.crop_window((910, 490, 100, 100), 1920, 1080)
    assert (w, h) == (608, 1080)
    assert w % 2 == 0 and h % 2 == 0


def test_crop_window_centers_horizontally_on_face():
    x, y, w, h = reframe.crop_window((910, 490, 100, 100), 1920, 1080)  # face center cx=960
    assert x <= 960 <= x + w
    assert abs((x + w / 2) - 960) < 2  # crop center ~ face center


def test_crop_window_clamps_to_left_edge():
    x, y, w, h = reframe.crop_window((0, 490, 100, 100), 1920, 1080)  # face at far left
    assert x == 0


def test_crop_window_clamps_to_right_edge():
    x, y, w, h = reframe.crop_window((1820, 490, 100, 100), 1920, 1080)  # face at far right
    assert x == 1920 - w


def test_crop_window_gives_headroom_face_above_center():
    # Tall source so vertical isn't clamped: face should sit ABOVE the crop midline.
    x, y, w, h = reframe.crop_window((500, 1450, 80, 100), 1080, 3000)  # face cy=1500
    assert y + h / 2 > 1500  # crop midline is below the face -> headroom above


def test_crop_window_always_within_bounds():
    for cx in (0, 200, 960, 1700, 1920):
        x, y, w, h = reframe.crop_window((cx, 500, 100, 100), 1920, 1080)
        assert 0 <= x and x + w <= 1920
        assert 0 <= y and y + h <= 1080


# ---- build_segments: per-frame speaker timeline -> stable cut segments ----

def test_build_segments_two_stable_speakers():
    timeline = [0] * 50 + [1] * 50  # 2s each at 25fps
    segs = reframe.build_segments(timeline, fps=25.0, min_dwell=1.2)
    assert [s[2] for s in segs] == [0, 1]
    assert abs(segs[0][0] - 0.0) < 1e-6 and abs(segs[0][1] - 2.0) < 1e-6
    assert abs(segs[1][1] - 4.0) < 1e-6


def test_build_segments_suppresses_short_flip():
    # 1.6s of A, 0.4s blip of B, 1.6s of A -> the blip is below min_dwell, absorbed.
    timeline = [0] * 40 + [1] * 10 + [0] * 40
    segs = reframe.build_segments(timeline, fps=25.0, min_dwell=1.2)
    assert len(segs) == 1
    assert segs[0][2] == 0


def test_build_segments_keeps_long_switch():
    timeline = [0] * 40 + [1] * 40  # both 1.6s > 1.2s min_dwell
    segs = reframe.build_segments(timeline, fps=25.0, min_dwell=1.2)
    assert len(segs) == 2


def test_build_segments_empty():
    assert reframe.build_segments([], fps=25.0) == []


# ---- smooth_path: EMA + dead-zone + velocity clamp ----

def test_smooth_path_ema_eases_not_jumps():
    out = reframe.smooth_path([0, 0, 0, 100, 100, 100], alpha=0.12)
    assert out[2] == 0
    assert abs(out[3] - 12.0) < 1e-6   # eased 12% toward 100, not snapped
    assert out[3] < out[4] < out[5]    # keeps approaching


def test_smooth_path_dead_zone_holds_on_small_drift():
    out = reframe.smooth_path([50, 50.5, 49.8, 50.3], alpha=0.5, dead_zone=2.0)
    assert all(abs(v - 50) < 1e-6 for v in out)  # micro-drift ignored


def test_smooth_path_velocity_clamp_caps_step():
    out = reframe.smooth_path([0, 1000], alpha=1.0, max_step=10.0)
    assert out[1] == 10.0  # would jump 1000 without the clamp


def test_smooth_path_empty():
    assert reframe.smooth_path([]) == []


# ---- plan_reframe: active-speaker windows -> per-sub-segment crop plan ----

def _win(s, e, bbox):
    return {"start_s": s, "end_s": e, "bbox": bbox}


def test_plan_reframe_one_window_per_speaker():
    windows = [_win(0.0, 2.0, (100, 400, 120, 120)), _win(2.0, 4.0, (1600, 400, 120, 120))]
    plan = reframe.plan_reframe(windows, 0.0, 4.0, 1920, 1080)
    assert len(plan) == 2
    # first speaker is on the left, second on the right -> different crop x
    assert plan[0]["crop"][0] < plan[1]["crop"][0]
    assert plan[0]["start_s"] == 0.0 and plan[1]["end_s"] == 4.0


def test_plan_reframe_clips_windows_to_span():
    windows = [_win(-1.0, 1.5, (900, 400, 120, 120))]  # starts before the clip
    plan = reframe.plan_reframe(windows, 0.0, 1.5, 1920, 1080)
    assert len(plan) == 1
    assert plan[0]["start_s"] == 0.0  # clipped to span start


def test_plan_reframe_drops_nonoverlapping_window():
    windows = [_win(10.0, 12.0, (900, 400, 120, 120))]
    plan = reframe.plan_reframe(windows, 0.0, 4.0, 1920, 1080)
    assert plan == []


def test_plan_reframe_empty_windows():
    assert reframe.plan_reframe([], 0.0, 4.0, 1920, 1080) == []
