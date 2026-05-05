"""
Vision step: pick the best photo, describe the piece.

Uses qwen3-vl:235b-cloud via local Ollama relay. Output is a structured
dict containing a human-readable description plus taxonomy tags for
downstream pickers (color family, material class, style class, dominant hex).
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

OLLAMA_HOST = "http://localhost:11434"
VISION_MODEL = "qwen3-vl:235b-cloud"
TIMEOUT_SECONDS = 180
MAX_ATTEMPTS = 2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Closed-vocabulary constants — imported by Task 10 picker and tests
# ---------------------------------------------------------------------------
COLOR_FAMILIES: tuple[str, ...] = ("warm", "cool", "neutral", "mixed", "statement")
MATERIAL_CLASSES: tuple[str, ...] = (
    "beaded", "metal-chain", "leather", "mixed-media", "gemstone", "other"
)
STYLE_CLASSES: tuple[str, ...] = (
    "minimal", "bohemian", "statement", "layering", "classic"
)

_HEX_RE = re.compile(r'^#[0-9A-Fa-f]{6}$')

SYSTEM_PROMPT = (
    "You are a jewelry product photographer and writer for a high-end "
    "handmade jewelry studio called Roen. You are looking at photos of a "
    "single piece of handmade jewelry that the maker just finished.\n\n"
    "Return ONLY a JSON object with these fields and no other text:\n"
    "  - description: 5-7 sentences describing what you see — material, color, "
    "gemstones or beads, findings, approximate dimensions if visible, finish "
    "(matte/polished), and the general piece type (bracelet, necklace, earring, "
    "ring). Be specific and concrete. No marketing language. If the photo is "
    "blurry or you cannot see the piece clearly, say so.\n"
    '  - color_family: one of "warm", "cool", "neutral", "mixed", "statement"\n'
    '  - dominant_hex: a single hex color like "#C8794E" representing the piece\n'
    '  - material_class: one of "beaded", "metal-chain", "leather", "mixed-media", "gemstone", "other"\n'
    '  - style_class: one of "minimal", "bohemian", "statement", "layering", "classic"\n\n'
    "If a categorical field is unclear, choose the closest option. Never invent values."
)


def _encode_image(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _call_ollama(payload: dict) -> dict:
    """POST to Ollama /api/chat with retries. Returns the parsed response dict."""
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logger.info(
                "vision: calling %s (attempt %d/%d)",
                payload.get("model", VISION_MODEL),
                attempt,
                MAX_ATTEMPTS,
            )
            r = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=TIMEOUT_SECONDS,
            )
            r.raise_for_status()
            data = r.json()
            return data
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            logger.warning("vision timeout/conn error on attempt %d: %s", attempt, e)
            if attempt >= MAX_ATTEMPTS:
                raise
            time.sleep(2)
    raise RuntimeError(f"vision exhausted retries: {last_err}")


def _normalize_tags(raw: dict) -> dict:
    """Fill in safe defaults for any missing or out-of-vocab fields."""
    description = raw.get("description", "")

    color_family = raw.get("color_family", "")
    if color_family not in COLOR_FAMILIES:
        color_family = "mixed"

    material_class = raw.get("material_class", "")
    if material_class not in MATERIAL_CLASSES:
        material_class = "other"

    style_class = raw.get("style_class", "")
    if style_class not in STYLE_CLASSES:
        style_class = "classic"

    dominant_hex = raw.get("dominant_hex", "")
    if not _HEX_RE.match(str(dominant_hex)):
        dominant_hex = "#888888"

    return {
        "description": description,
        "color_family": color_family,
        "material_class": material_class,
        "style_class": style_class,
        "dominant_hex": dominant_hex,
    }


def describe_piece(photo_paths: List[Any]) -> Dict[str, Any]:
    """Return a structured dict describing the jewelry piece based on the photos.

    Keys: description, color_family, material_class, style_class, dominant_hex.
    Never raises on JSON-parse failure — falls back to defaulted dict.
    """
    if not photo_paths:
        raise ValueError("describe_piece needs at least one photo")

    # Use up to the first 3 photos — vision models cap on tokens and
    # 3 angles is plenty for description.
    images = [_encode_image(p) for p in photo_paths[:3]]

    payload = {
        "model": VISION_MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Describe this piece for our product catalog. Return JSON only.",
                "images": images,
            },
        ],
        "options": {"temperature": 0.2},
    }

    data = _call_ollama(payload)
    content = data.get("message", {}).get("content", "").strip()

    if not content:
        raise RuntimeError(f"vision returned empty: {data}")

    logger.info("vision: %d chars returned", len(content))

    # Parse JSON; fall back gracefully on failure.
    try:
        raw = json.loads(content)
        if not isinstance(raw, dict):
            raise ValueError(f"expected JSON object, got {type(raw)}")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("vision: JSON parse failed (%s) — using raw text as description", exc)
        raw = {"description": content}

    return _normalize_tags(raw)
