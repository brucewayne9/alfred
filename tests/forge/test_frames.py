from pathlib import Path

from core.forge import frames


def test_frame_cache_path_is_deterministic_per_decisecond(monkeypatch, tmp_path):
    monkeypatch.setenv("FORGE_FRAMES_DIR", str(tmp_path))
    a = frames.frame_cache_path("src1", 12.34)
    b = frames.frame_cache_path("src1", 12.3)   # same decisecond bucket
    assert a == b
    assert a == tmp_path / "src1" / "123.jpg"


def test_frame_cache_path_separates_sources_and_times(monkeypatch, tmp_path):
    monkeypatch.setenv("FORGE_FRAMES_DIR", str(tmp_path))
    assert frames.frame_cache_path("a", 1.0) != frames.frame_cache_path("b", 1.0)
    assert frames.frame_cache_path("a", 1.0) != frames.frame_cache_path("a", 2.0)


def test_clamp_t_floors_at_zero():
    assert frames.clamp_t(-5.0, 100) == 0.0


def test_clamp_t_pulls_back_from_the_end():
    # 100s duration → never request exactly the last frame
    assert frames.clamp_t(100.0, 100.0) == 99.8


def test_clamp_t_passes_through_when_duration_unknown():
    assert frames.clamp_t(42.5, None) == 42.5


def test_extract_frame_returns_true_without_reextracting_a_cached_frame(tmp_path):
    out = tmp_path / "f.jpg"
    out.write_bytes(b"\xff\xd8\xff\xe0cached")   # pretend a real jpeg is already there
    # file_path is bogus — must be skipped because the cache hit short-circuits ffmpeg
    assert frames.extract_frame("/nonexistent/video.mp4", 5.0, out) is True
