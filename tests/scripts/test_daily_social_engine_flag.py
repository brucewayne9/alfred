"""Smoke test for the DAILY_SOCIAL_ENGINE dispatcher — verifies the flag
routes calls to the correct producer without invoking the heavy
TTS/ComfyUI/Remotion pipelines."""
from unittest.mock import patch
import scripts.rucktalk_daily_social as ds


def test_monologue_dispatches_legacy_by_default(monkeypatch):
    monkeypatch.setattr(ds, "DAILY_SOCIAL_ENGINE", "legacy")
    with patch.object(ds, "_produce_monologue_video_legacy", return_value="/tmp/legacy.mp4") as legacy, \
         patch.object(ds, "_produce_monologue_video_remotion", return_value="/tmp/remotion.mp4") as remo:
        result = ds._produce_monologue_video({"script": "x"}, "pillar", "r1")
        assert result == "/tmp/legacy.mp4"
        assert legacy.called
        assert not remo.called


def test_monologue_dispatches_remotion_when_flagged(monkeypatch):
    monkeypatch.setattr(ds, "DAILY_SOCIAL_ENGINE", "remotion")
    with patch.object(ds, "_produce_monologue_video_legacy", return_value="/tmp/legacy.mp4") as legacy, \
         patch.object(ds, "_produce_monologue_video_remotion", return_value="/tmp/remotion.mp4") as remo:
        result = ds._produce_monologue_video({"script": "x"}, "pillar", "r1")
        assert result == "/tmp/remotion.mp4"
        assert not legacy.called
        assert remo.called


def test_conversation_dispatches_legacy_by_default(monkeypatch):
    monkeypatch.setattr(ds, "DAILY_SOCIAL_ENGINE", "legacy")
    with patch.object(ds, "_produce_conversation_video_legacy", return_value="/tmp/c_legacy.mp4") as legacy, \
         patch.object(ds, "_produce_conversation_video_remotion", return_value="/tmp/c_remotion.mp4") as remo:
        result = ds._produce_conversation_video({"topic": "x"}, "pillar", "r1")
        assert result == "/tmp/c_legacy.mp4"
        assert legacy.called
        assert not remo.called


def test_conversation_dispatches_remotion_when_flagged(monkeypatch):
    monkeypatch.setattr(ds, "DAILY_SOCIAL_ENGINE", "remotion")
    with patch.object(ds, "_produce_conversation_video_legacy", return_value="/tmp/c_legacy.mp4") as legacy, \
         patch.object(ds, "_produce_conversation_video_remotion", return_value="/tmp/c_remotion.mp4") as remo:
        result = ds._produce_conversation_video({"topic": "x"}, "pillar", "r1")
        assert result == "/tmp/c_remotion.mp4"
        assert not legacy.called
        assert remo.called
