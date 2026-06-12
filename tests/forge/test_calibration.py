import pytest


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db, intel, scorer
    db.init_db()
    intel.init_intel()
    return scorer, intel


# ── editorial signal (live now — no TikTok audit needed) ──────────────────

def test_calibration_editorial_shows_lift_when_editors_pick_high_scores(env):
    scorer, intel = env
    scorer.save_candidates("s1", [
        {"start_s": 0, "end_s": 5, "score": 90},
        {"start_s": 6, "end_s": 9, "score": 85},
        {"start_s": 10, "end_s": 14, "score": 40},
        {"start_s": 15, "end_s": 19, "score": 30},
    ], now=1)
    cands = scorer.get_candidates("s1")
    # editors cut the two high-scored clips, skip the two low ones
    for c in cands:
        if c["score"] >= 85:
            scorer.mark_rendered(c["id"])
    ed = intel.calibration()["editorial"]
    assert ed["has_data"] is True
    assert ed["scored"] == 4
    assert ed["rendered"] == 2
    assert ed["avg_score_rendered"] == 87.5   # (90+85)/2
    assert ed["avg_score_skipped"] == 35.0     # (40+30)/2
    assert ed["lift"] == 52.5                   # editors strongly prefer the scorer's picks


def test_calibration_editorial_empty_when_nothing_rendered(env):
    scorer, intel = env
    scorer.save_candidates("s1", [{"start_s": 0, "end_s": 5, "score": 90}], now=1)
    ed = intel.calibration()["editorial"]
    assert ed["has_data"] is False
    assert ed["rendered"] == 0


# ── engagement signal (gated — lights up when real views flow) ────────────

def test_calibration_engagement_buckets_views_by_score_band(env):
    scorer, intel = env
    # two posted clips with predicted scores and real view counts
    intel.record_video("v1", views=100000, likes=8000, comments=500, shares=1500,
                       predicted_score=92)
    intel.record_video("v2", views=2000, likes=80, comments=5, shares=10,
                       predicted_score=45)
    eng = intel.calibration()["engagement"]
    assert eng["has_data"] is True
    assert eng["tracked"] == 2
    bands = {b["band"]: b for b in eng["bands"]}
    assert bands["85-100"]["posts"] == 1
    assert bands["85-100"]["avg_views"] == 100000
    assert bands["0-49"]["posts"] == 1
    assert bands["0-49"]["avg_views"] == 2000


def test_calibration_engagement_empty_before_any_tracked_views(env):
    scorer, intel = env
    eng = intel.calibration()["engagement"]
    assert eng["has_data"] is False
    assert eng["tracked"] == 0


def test_record_video_persists_predicted_score(env):
    scorer, intel = env
    intel.record_video("v1", views=10, predicted_score=77)
    eng = intel.calibration()["engagement"]
    assert eng["tracked"] == 1
