"""Forge image dispatcher — Mainstay Forge generates via Higgsfield (or user upload).

A single seam so renderers don't each hard-code a generator. ``source="higgsfield"``
(the DEFAULT for Forge) draws on the Mainstay Ultra subscription credits via the
higgsfield CLI bridge — Nano Banana Pro 4K by default.

ComfyUI — our own in-house service — is NOT a user-facing Forge option and is never
silently served to the Mainstay team. It is a dark break-glass fallback that activates
ONLY when Higgsfield is down/out-of-credits AND the emergency switch is on (see
``comfyui_emergency_enabled``). Otherwise a Higgsfield failure fails the render loudly
so we notice and top up / re-auth rather than quietly degrading the team's output.

Like the rest of Forge, heavy deps (rucktalk_common / the higgsfield SDK path) are
lazy-imported inside functions so this module stays importable in bare test envs.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent.parent

# Break-glass: ComfyUI fallback is OFF unless Mike flips it after confirming Higgsfield
# credits are exhausted. Toggle via env FORGE_EMERGENCY_COMFYUI=1 or by creating this
# flag file (live, no service restart needed).
EMERGENCY_FLAG_FILE = REPO / "data" / "forge_emergency_comfyui.flag"


def comfyui_emergency_enabled() -> bool:
    """True only when the ComfyUI break-glass fallback has been deliberately enabled."""
    if os.environ.get("FORGE_EMERGENCY_COMFYUI", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    try:
        return EMERGENCY_FLAG_FILE.exists()
    except OSError:
        return False

# Frontend sends "9x16" | "1x1" | "16x9"; Higgsfield wants colon form.
_ASPECT_MAP = {
    "9x16": "9:16",
    "1x1": "1:1",
    "16x9": "16:9",
}


def map_aspect(aspect: str | None) -> str | None:
    """Map a Forge aspect id ("9x16") to a Higgsfield aspect ("9:16"), or None."""
    if not aspect:
        return None
    return _ASPECT_MAP.get(aspect)


def infer_aspect(width: int, height: int) -> str:
    """Infer a Higgsfield aspect ("9:16"/"1:1"/"16:9") from pixel dimensions."""
    if not width or not height:
        return "9:16"
    if width == height:
        return "1:1"
    return "16:9" if width > height else "9:16"


def _run_comfyui(prompt: str, width: int, height: int) -> Path | None:
    """ComfyUI Cloud still — same lazy-import pattern the renderers already use."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from scripts.rucktalk_common import run_comfyui_cloud  # lazy: heavy import

    result = run_comfyui_cloud(prompt, width=width, height=height)
    if result and Path(result).exists():
        return Path(result)
    return None


def render_image(
    prompt: str,
    width: int,
    height: int,
    *,
    source: str = "higgsfield",
    model: str | None = None,
    reference_image: str | Path | None = None,
    aspect: str | None = "9x16",
) -> Path | None:
    """Render a still for Forge. Returns a local path, or None on a gated failure.

    ``source="higgsfield"`` (the DEFAULT) generates on the Ultra subscription
    (Nano Banana Pro 4K). On failure (CLI missing, auth, credits, None) it does NOT
    silently use ComfyUI: it only falls back when the emergency switch is enabled
    (``comfyui_emergency_enabled``); otherwise it logs loudly and returns None so the
    job fails visibly and we top up / re-auth.

    ``source="comfyui"`` is the explicit break-glass path (emergency/internal only).
    """
    source = (source or "higgsfield").lower()

    # Explicit ComfyUI request = the deliberate break-glass / internal path.
    if source == "comfyui":
        return _run_comfyui(prompt, width, height)

    # Default Forge path: Higgsfield only.
    from core.forge import higgsfield  # lazy: keeps SDK optional

    if higgsfield.is_available():
        out = Path(tempfile.gettempdir()) / f"forge_hf_{uuid.uuid4().hex}.png"
        ratio = map_aspect(aspect) or infer_aspect(width, height)
        result = higgsfield.generate_image(
            prompt,
            out,
            model=model or higgsfield.DEFAULT_IMAGE_MODEL,
            aspect_ratio=ratio,
            resolution="4k",
            reference_image=reference_image,
        )
        if result and Path(result).exists():
            return Path(result)
        logger.error("HIGGSFIELD image generation failed (auth/credits/empty result).")
    else:
        logger.error("HIGGSFIELD CLI unavailable — cannot generate image.")

    # Higgsfield is down — break-glass only.
    if comfyui_emergency_enabled():
        logger.warning("FORGE_EMERGENCY_COMFYUI active — serving ComfyUI fallback image.")
        return _run_comfyui(prompt, width, height)

    logger.error(
        "Higgsfield image unavailable and emergency ComfyUI is OFF — failing this render. "
        "Check `higgsfield account status`; if credits are exhausted, enable the emergency "
        "switch (touch data/forge_emergency_comfyui.flag) to bring ComfyUI back."
    )
    return None
