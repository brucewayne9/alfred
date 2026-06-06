# core/casting/voice.py
from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path
from config.settings import settings

MIN_SECONDS = 8.0      # below this is too short to be a usable reference
MAX_SECONDS = 90.0

def _voices_root() -> Path:
    return Path(settings.casting_voices_dir)

def _probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0

def validate_clip(path: str) -> tuple[bool, str]:
    if not Path(path).exists():
        return False, "file not found"
    dur = _probe_duration(path)
    if dur <= 0:
        return False, "could not read audio (corrupt or unsupported format)"
    if dur < MIN_SECONDS:
        return False, f"too short ({dur:.1f}s); need at least {MIN_SECONDS:.0f}s"
    if dur > MAX_SECONDS:
        return False, f"too long ({dur:.1f}s); keep it under {MAX_SECONDS:.0f}s"
    return True, "ok"

def store_mood(*, dj_id: int, mood: str, src_path: str) -> str:
    """Normalize to mono 24kHz, trim edge silence, loudnorm, write to the library."""
    dest_dir = _voices_root() / str(dj_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{mood}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path,
         "-af", "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,"
                "areverse,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,"
                "areverse,loudnorm=I=-18:TP=-1.5:LRA=11",
         "-ar", "24000", "-ac", "1", str(dest)],
        check=True, capture_output=True, timeout=120,
    )
    return str(dest)

def mood_path(dj_id: int, mood: str) -> str:
    return str(_voices_root() / str(dj_id) / f"{mood}.wav")

def engine_voice_name(dj_id: int, mood: str) -> str:
    """Collision-proof registered name (never clobbers live jocks like MJ_neutral)."""
    return f"cc{dj_id}_{mood}"

def register_to_engine(dj_id: int, moods: list[str]) -> dict[str, str]:
    """Copy each stored mood clip into the Qwen resources dir under the id-namespaced
    name `cc<dj_id>_<mood>.wav` so the 105:7860 server can resolve it by name.
    Returns {mood: registered_name}. Skips moods with no stored clip."""
    res_dir = Path(settings.qwen_resources_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    registered: dict[str, str] = {}
    for mood in moods:
        src = Path(mood_path(dj_id, mood))
        if not src.exists():
            continue
        dest = res_dir / f"cc{dj_id}_{mood}.wav"
        shutil.copyfile(str(src), str(dest))
        registered[mood] = engine_voice_name(dj_id, mood)
    return registered
