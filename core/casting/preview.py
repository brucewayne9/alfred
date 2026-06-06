# core/casting/preview.py
from __future__ import annotations
import requests
from pathlib import Path
from config.settings import settings

DEFAULT_LINE = (
    "You're listening to News Muse. I'm glad you're here. Let's get into it."
)

def render_preview(*, voice_name: str, line: str = DEFAULT_LINE, out_path: str) -> str:
    """Render a short off-air sample using the DJ's neutral reference clip.
    NOTE: the 105:7860 Qwen server resolves voices by NAME from its resources dir,
    NOT by file path (passing a path returns HTTP 400 — confirmed via live check).
    The clip must already be registered into the resources dir under voice_name
    (see voice.register_to_engine); voice_name is that registered name (no path,
    no .wav extension)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        f"{settings.qwen_tts_url}/synthesize_speech/",
        params={"text": line, "voice": voice_name, "speed": "1.0"},
        timeout=120,
    )
    resp.raise_for_status()
    with open(out_path, "wb") as fh:
        fh.write(resp.content)
    return out_path
