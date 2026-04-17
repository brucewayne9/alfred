"""Tests for rig prop builders — Phase 2 migration."""
from scripts.rucktalk_rig_props import (
    build_rucktalkclip_props,
    build_magazinerig_props,
    CaptionPhrase,
)


def _sample_phrases() -> list[CaptionPhrase]:
    return [
        {"text": "HELLO WORLD", "startFrame": 0, "endFrame": 60},
        {"text": "TESTING", "startFrame": 70, "endFrame": 120},
    ]


def test_rucktalkclip_shape():
    p = build_rucktalkclip_props(
        clip_filename="clip_test.mp4",
        episode_number=42,
        episode_title="The Answer",
        context_line="Comfort is the enemy.",
        host_name="MIKE JOHNSON",
        guest_name=None,
        caption_phrases=_sample_phrases(),
    )
    assert p["videoSrc"] == "clip_test.mp4"
    assert p["episodeNumber"] == 42
    assert p["episodeTitle"] == "The Answer"
    assert p["contextLine"] == "Comfort is the enemy."
    assert p["hostName"] == "MIKE JOHNSON"
    assert p["guestName"] == ""
    assert p["captionPhrases"] == _sample_phrases()


def test_rucktalkclip_with_guest():
    p = build_rucktalkclip_props(
        clip_filename="c.mp4",
        episode_number=1,
        episode_title="T",
        context_line="",
        host_name="MIKE",
        guest_name="DR SMITH",
        caption_phrases=_sample_phrases(),
    )
    assert p["guestName"] == "DR SMITH"


def test_magazinerig_shape():
    p = build_magazinerig_props(
        clip_filename="clip_test.mp4",
        episode_number=42,
        episode_title="The Answer",
        host_name="MIKE JOHNSON",
        guest_name=None,
        caption_phrases=_sample_phrases(),
    )
    assert p["brand"] == "rucktalk"
    assert p["clipSrc"] == "clip_test.mp4"
    assert p["episodeNumber"] == 42
    assert p["episodeTitle"] == "The Answer"
    assert p["hostName"] == "MIKE JOHNSON"
    assert p["captionPhrases"] == _sample_phrases()
    # contextLine is dropped — must NOT be in output
    assert "contextLine" not in p
    # videoSrc is renamed — must NOT be in output
    assert "videoSrc" not in p


def test_magazinerig_with_guest():
    p = build_magazinerig_props(
        clip_filename="c.mp4",
        episode_number=1,
        episode_title="T",
        host_name="MIKE",
        guest_name="DR SMITH",
        caption_phrases=_sample_phrases(),
    )
    assert p["guestName"] == "DR SMITH"


def test_magazinerig_omits_none_guest():
    """When no guest, the key should either be absent or empty string — never None."""
    p = build_magazinerig_props(
        clip_filename="c.mp4",
        episode_number=1,
        episode_title="T",
        host_name="MIKE",
        guest_name=None,
        caption_phrases=_sample_phrases(),
    )
    assert p.get("guestName", "") == ""
