"""Audio helpers for Forge renderers: clip, word-timing transcription, loudness guard."""
from __future__ import annotations
import re
import subprocess
from pathlib import Path


def duration_seconds(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def clip_audio(src: str | Path, start: float, end: float, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.1, float(end) - float(start))
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ss", str(start),
         "-t", str(dur), "-c:a", "libmp3lame", "-q:a", "2", str(out_path)], check=True)
    return out_path


def mean_loudness_db(path: str | Path) -> float:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", proc.stderr)
    return float(m.group(1)) if m else -120.0


def assert_audible(path: str | Path, floor_db: float = -80.0) -> None:
    """Raise if the file's audio is effectively silent — guards the mux bug."""
    db = mean_loudness_db(path)
    if db < floor_db:
        raise RuntimeError(f"output audio is silent (mean {db} dB < {floor_db} dB): {path}")


def transcribe_words(path: str | Path, model_size: str = "small") -> list[dict]:
    """Word-level timings: [{'word','start','end'}]. CPU int8 faster-whisper."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(path), word_timestamps=True, vad_filter=False, beam_size=5)
    words: list[dict] = []
    for seg in segments:
        for w in (seg.words or []):
            t = w.word.strip()
            if t:
                words.append({"word": t, "start": round(w.start, 3), "end": round(w.end, 3)})
    return words
