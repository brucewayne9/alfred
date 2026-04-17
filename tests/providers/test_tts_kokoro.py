import pytest
from pathlib import Path
from scripts.providers.tts.kokoro import KokoroTts
from scripts.providers.tts.base import TtsRequest

@pytest.mark.integration
def test_kokoro_lists_voices():
    """Integration: requires Kokoro service on localhost:8880."""
    p = KokoroTts()
    voices = p.list_voices()
    assert isinstance(voices, list)
    assert "am_adam" in voices
    assert "af_sarah" in voices

@pytest.mark.integration
def test_kokoro_synthesizes_short_clip(tmp_path: Path):
    p = KokoroTts()
    out = tmp_path / "test.wav"
    result = p.synth(TtsRequest(text="Testing one two three.", voice="am_adam", output_path=out))
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 1000
    assert result.voice == "am_adam"
    assert result.duration_s > 0.5
