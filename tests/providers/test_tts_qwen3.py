import pytest
from pathlib import Path
from scripts.providers.tts.qwen3 import Qwen3Tts
from scripts.providers.tts.base import TtsRequest

@pytest.mark.integration
def test_qwen3_lists_voices():
    """Integration: requires Qwen3-TTS service on localhost:7860."""
    p = Qwen3Tts()
    voices = p.list_voices()
    assert isinstance(voices, list)
    # Expect both cloned and designed voices merged:
    assert "MJ" in voices
    assert "Lois_Lane" in voices

@pytest.mark.integration
def test_qwen3_synthesizes_short_clip(tmp_path: Path):
    p = Qwen3Tts()
    out = tmp_path / "test.wav"
    result = p.synth(TtsRequest(text="Testing one two three.", voice="JAYDEE", output_path=out))
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 1000
    assert result.voice == "JAYDEE"
    assert result.duration_s > 0.5
