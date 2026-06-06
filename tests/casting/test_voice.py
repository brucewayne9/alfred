# tests/casting/test_voice.py
import os, tempfile, subprocess, importlib
import pytest

@pytest.fixture()
def voicemod(monkeypatch):
    tmp = tempfile.mkdtemp()
    from config.settings import settings
    monkeypatch.setattr(settings, "casting_voices_dir", os.path.join(tmp, "voices"), raising=False)
    import core.casting.voice as v
    importlib.reload(v)
    return v

def _make_wav(path, seconds=2):
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
         "-ar", "44100", "-ac", "2", "-y", path],
        check=True, capture_output=True,
    )

def test_validate_rejects_too_short(voicemod, tmp_path):
    p = str(tmp_path / "tiny.wav"); _make_wav(p, seconds=1)
    ok, reason = voicemod.validate_clip(p)
    assert ok is False and "short" in reason.lower()

def test_validate_accepts_good_clip(voicemod, tmp_path):
    p = str(tmp_path / "good.wav"); _make_wav(p, seconds=20)
    ok, reason = voicemod.validate_clip(p)
    assert ok is True, reason

def test_store_mood_normalizes_and_places(voicemod, tmp_path):
    src = str(tmp_path / "src.wav"); _make_wav(src, seconds=20)
    out = voicemod.store_mood(dj_id=7, mood="neutral", src_path=src)
    assert out.endswith("/7/neutral.wav")
    assert os.path.exists(out)
    # normalized to mono 24k
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=channels,sample_rate", "-of", "csv=p=0", out],
                           capture_output=True, text=True)
    assert "1" in probe.stdout and "24000" in probe.stdout
