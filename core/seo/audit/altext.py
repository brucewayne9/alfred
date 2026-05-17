# core/seo/audit/altext.py
"""Alt-text backfill via local Ollama vision model (qwen3-vl:235b-cloud).

Used by the audit runner as an OPT-IN extra pass. For any image flagged
with `missing_alt_text`, the runner can call generate_alt_text(image_url)
to attach a `suggested_alt` field onto the issue's detail_payload. The
audit module does NOT push the alt back into WordPress — that's the
Page Optimizer's job.

Cost/time guard: caller (runner) is expected to cap how many images per
audit get processed. This module does the single-image work only.
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3-vl:235b-cloud"
DEFAULT_TIMEOUT = 60.0

ALT_TEXT_PROMPT = (
    "You are writing alt text for a jewelry e-commerce site. Look at this "
    "image and write 8-15 word alt text that describes what's shown, "
    "includes the product type if visible, and reads naturally. Output ONLY "
    "the alt text, nothing else."
)


def _fetch_image_bytes(image_url: str, timeout: float = 30.0) -> bytes:
    """GET the image URL; raise on HTTP error."""
    resp = httpx.get(image_url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def _ollama_generate(
    image_b64: str,
    *,
    model: str,
    prompt: str,
    ollama_url: str,
    timeout: float,
) -> str:
    """Call Ollama /api/generate with one image. Returns response string."""
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    resp = httpx.post(ollama_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # Some vision models return the answer in `response`. Reasoning-mode
    # variants may stash it under `reasoning` (see Kimi quirk in MEMORY).
    text = (data.get("response") or data.get("reasoning") or "").strip()
    return text


def generate_alt_text(
    image_source: str | bytes,
    *,
    model: str = DEFAULT_MODEL,
    prompt: str = ALT_TEXT_PROMPT,
    ollama_url: str = OLLAMA_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> Optional[str]:
    """Generate alt text for an image. Returns the alt string or None on failure.

    `image_source` is either a URL (str) or raw bytes.
    """
    try:
        if isinstance(image_source, (bytes, bytearray)):
            img_bytes = bytes(image_source)
        else:
            img_bytes = _fetch_image_bytes(image_source)
    except Exception:
        logger.exception("alt-text: failed to fetch image %r", image_source)
        return None

    image_b64 = base64.b64encode(img_bytes).decode("ascii")

    try:
        text = _ollama_generate(
            image_b64,
            model=model,
            prompt=prompt,
            ollama_url=ollama_url,
            timeout=timeout,
        )
    except Exception:
        logger.exception("alt-text: ollama call failed for %r", image_source)
        return None

    if not text:
        return None

    # Strip surrounding quotes if the model wrapped its output.
    text = text.strip().strip('"').strip("'").strip()
    return text or None
