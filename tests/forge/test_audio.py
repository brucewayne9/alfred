import subprocess
from pathlib import Path
import pytest
from core.forge import audio


@pytest.fixture
def tone(tmp_path):
    p = tmp_path / "tone.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=6", "-q:a", "4", str(p)], check=True)
    return p


def test_clip_audio_window(tone, tmp_path):
    out = audio.clip_audio(tone, 1.0, 3.0, tmp_path / "clip.mp3")
    assert out.exists()
    assert 1.8 <= audio.duration_seconds(out) <= 2.2


def test_loudness_db_detects_sound_vs_silence(tone, tmp_path):
    assert audio.mean_loudness_db(tone) > -40
    silent = tmp_path / "silent.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "anullsrc=r=44100:cl=stereo", "-t", "2", str(silent)], check=True)
    assert audio.mean_loudness_db(silent) < -80


def test_assert_audible_raises_on_silence(tmp_path):
    silent = tmp_path / "s.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "anullsrc=r=44100:cl=stereo", "-t", "1", "-f", "lavfi", "-i",
         "color=c=black:s=64x64:d=1", "-shortest", str(silent)], check=True)
    with pytest.raises(RuntimeError, match="silent"):
        audio.assert_audible(silent)
