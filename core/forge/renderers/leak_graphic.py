"""Leak-Graphic renderer — a Rod-Wave-style cover image via ComfyUI Cloud.

The "leak graphic" is the single-image post format (fake album cover / engagement
bait). For v1 we generate the cover ART from the caption via ComfyUI Cloud
(cloud-first, local GPU fallback handled inside run_comfyui_cloud). Title/tracklist
text overlay is a later Remotion refinement.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent


def build_prompt(caption: str) -> str:
    """Turn the user's caption into a cover-art prompt (no literal text in the image)."""
    caption = (caption or "").strip()
    base = (
        "cinematic melancholic rap album cover, moody dramatic low-key lighting, "
        "deep shadows, film grain, 35mm, emotional portrait energy, lonely night mood, "
        "muted gold and charcoal palette, vertical poster composition, high detail, no text"
    )
    return f"{base}, themed around: {caption}" if caption else base


def render(params: dict, out_path: str | Path) -> Path:
    """Generate the leak-graphic cover art to `out_path` (PNG). Raises on failure."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from scripts.rucktalk_common import run_comfyui_cloud  # lazy: heavy import

    prompt = build_prompt(params.get("caption", ""))
    result = run_comfyui_cloud(prompt, width=768, height=1344)  # ~9:16 portrait
    if not result:
        raise RuntimeError("ComfyUI Cloud returned no image")
    src = Path(result)
    if not src.exists():
        raise RuntimeError(f"ComfyUI image missing on disk: {src}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != out_path.resolve():
        out_path.write_bytes(src.read_bytes())
    return out_path
