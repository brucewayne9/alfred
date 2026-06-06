# core/casting/preview.py
from __future__ import annotations
import requests
from pathlib import Path
from config.settings import settings

DEFAULT_LINE = (
    "You're listening to News Muse. I'm glad you're here. Let's get into it."
)

def render_preview(*, voice_wav: str, line: str = DEFAULT_LINE, out_path: str) -> str:
    """Render a short off-air sample using the DJ's neutral reference clip.
    voice_wav is the absolute path to the reference clip as the Qwen server sees it
    (the Qwen server shares the 105 filesystem with Alfred Labs)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        f"{settings.qwen_tts_url}/synthesize_speech/",
        params={"text": line, "voice": voice_wav, "speed": "1.0"},
        timeout=120,
    )
    resp.raise_for_status()
    with open(out_path, "wb") as fh:
        fh.write(resp.content)
    return out_path
