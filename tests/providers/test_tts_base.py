import pytest
from pathlib import Path
from scripts.providers.tts.base import TtsProvider, TtsRequest, TtsResult

class FakeTts(TtsProvider):
    name = "fake"
    def list_voices(self): return ["v1", "v2"]
    def synth(self, req: TtsRequest) -> TtsResult:
        return TtsResult(audio_path=Path("/tmp/fake.wav"), duration_s=1.0, voice=req.voice)

def test_tts_provider_contract():
    p = FakeTts()
    assert p.name == "fake"
    assert p.list_voices() == ["v1", "v2"]
    result = p.synth(TtsRequest(text="hello", voice="v1"))
    assert result.voice == "v1"
    assert result.audio_path == Path("/tmp/fake.wav")

def test_tts_provider_is_abstract():
    with pytest.raises(TypeError):
        TtsProvider()  # abstract, cannot instantiate
