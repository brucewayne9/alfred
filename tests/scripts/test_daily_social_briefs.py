"""Tests for AutoBrief builders — Phase 3 daily social migration."""
from scripts.daily_social_briefs import (
    build_monologue_brief,
    build_conversation_brief,
    derive_word_beats_from_script,
)


def test_derive_word_beats_basic():
    script = "Show up anyway. Comfort is the enemy."
    beats = derive_word_beats_from_script(script, audio_duration_s=6.0, fps=30)
    # 2 sentences → 2 beat groups minimum
    assert len(beats) >= 2
    for b in beats:
        assert "word" in b and "startFrame" in b and "endFrame" in b
        assert b["endFrame"] > b["startFrame"]
        assert b["startFrame"] >= 0
        assert b["endFrame"] <= 6 * 30
        assert b.get("variant") in {"single", "stacked", "scaleOnBeat"}


def test_derive_word_beats_preserves_order():
    beats = derive_word_beats_from_script(
        "First. Second. Third.", audio_duration_s=6.0, fps=30
    )
    # ending frames should be monotonically non-decreasing
    ends = [b["endFrame"] for b in beats]
    assert ends == sorted(ends)


def test_build_monologue_brief_shape():
    brief = build_monologue_brief(
        date="2026-04-20",
        rotation=2,
        script="Show up anyway. Comfort is the enemy.",
        bg_clip_public_name="daily_mono_20260420.mp4",
        audio_duration_s=6.0,
    )
    assert brief["brand"] == "rucktalk"
    assert brief["date"] == "2026-04-20"
    assert brief["rotation"] == 2
    assert brief["bgClip"] == "daily_mono_20260420.mp4"
    assert len(brief["wordBeats"]) >= 2


def test_build_conversation_brief_shape():
    brief = build_conversation_brief(
        date="2026-04-20",
        rotation=1,
        bg_clips=[
            {"src": "bg_a.mp4", "durationFrames": 240},
            {"src": "bg_b.mp4", "durationFrames": 240},
        ],
    )
    assert brief["brand"] == "rucktalk"
    # Must align with autoProps expectations for GritDocRig — clips list required
    assert len(brief["clips"]) == 2
    assert all("src" in c and "durationFrames" in c for c in brief["clips"])
    # rotation value must land on GritDocRig (index 1 of ["MagazineRig","GritDocRig","KineticTypeRig"])
    assert brief["rotation"] % 3 == 1
