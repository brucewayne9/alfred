"""
Vision step: pick the best photo, describe the piece.

Uses qwen3-vl:235b-cloud via local Ollama relay. Output is a structured
description of materials, color, dimensions, gemstones, finish — input
to the copywriter step.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import requests

OLLAMA_HOST = "http://localhost:11434"
VISION_MODEL = "qwen3-vl:235b-cloud"
TIMEOUT_SECONDS = 180
MAX_ATTEMPTS = 2

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a jewelry product photographer and writer for a high-end "
    "handmade jewelry studio called Roen. You are looking at photos of a "
    "single piece of handmade jewelry that the maker just finished. "
    "Describe what you see — material, color, gemstones or beads, findings, "
    "approximate dimensions if visible, finish (matte/polished), and the "
    "general piece type (bracelet, necklace, earring, ring). Be specific "
    "and concrete. No marketing language, no 'beautiful' or 'stunning'. "
    "If the photo is blurry or you cannot see the piece clearly, say so. "
    "5-7 sentences. Plain text only — no markdown."
)


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def describe_piece(photo_paths: List[Path]) -> str:
    """Return a plain-text description of the jewelry piece based on the photos."""
    if not photo_paths:
        raise ValueError("describe_piece needs at least one photo")

    # Use up to the first 3 photos as input — vision models cap on tokens and
    # 3 angles is plenty for description.
    images = [_encode_image(p) for p in photo_paths[:3]]

    payload = {
        "model": VISION_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Describe this piece for our product catalog.",
                "images": images,
            },
        ],
        "options": {"temperature": 0.2},
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logger.info("vision: calling %s with %d image(s) (attempt %d/%d)", VISION_MODEL, len(images), attempt, MAX_ATTEMPTS)
            r = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=TIMEOUT_SECONDS,
            )
            r.raise_for_status()
            data = r.json()
            content = data.get("message", {}).get("content", "").strip()
            if not content:
                raise RuntimeError(f"vision returned empty: {data}")
            logger.info("vision: %d chars returned", len(content))
            return content
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            logger.warning("vision timeout/conn error on attempt %d: %s", attempt, e)
            if attempt >= MAX_ATTEMPTS:
                raise
            time.sleep(2)
    raise RuntimeError(f"vision exhausted retries: {last_err}")
