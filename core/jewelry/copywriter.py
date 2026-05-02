"""
Copywriter step: turn vision description + price into product metadata.

Uses a fast cloud LLM (kimi-k2.6:cloud) via Ollama. Returns a structured
JSON dict with name, sku, short_description, long_description, tags, category.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

import requests

OLLAMA_HOST = "http://localhost:11434"
COPY_MODEL = "kimi-k2.6:cloud"
TIMEOUT_SECONDS = 180
MAX_ATTEMPTS = 2

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are the copywriter for Roen, a small handmade jewelry studio in Atlanta. "
    "Brand voice: minimal, modern, third person, no marketing fluff. Never use "
    "the words 'stunning', 'gorgeous', 'beautiful', 'unique', 'one of a kind', "
    "'perfect', or anything that sounds like a Shopify dropshipper. Calm, plain "
    "language. The maker is presented as the brand 'Roen', not as a person — "
    "never write 'I' or 'we crafted'. Write in third person about the piece. "
    "All copy lowercase except proper nouns and the start of full sentences."
)


USER_TEMPLATE = """Vision description of the piece:
---
{description}
---

Price: ${price_dollars}

Output a JSON object with these exact keys:
- name: 2 to 4 words, descriptive title-case (e.g., "Silver Moonstone Bracelet"). No filler words, no "handmade".
- sku: 6-10 character uppercase code, derived from the name (e.g., "SILMOONB"). Letters and digits only.
- short_description: 1-2 sentences, ~140 characters. For Meta Catalog and previews.
- long_description: 3-5 short sentences. For the WooCommerce product page. Include material, finish, and care if relevant. End with one sentence on how the piece is made (handmade in Atlanta).
- category: one of: bracelet, necklace, earring, ring, anklet, set, other. Lowercase singular.
- tags: a JSON array of 4 to 8 lowercase strings. Material words, gemstone names, style words. No hashtags, no spaces inside tags.

Output ONLY the JSON object. No prose before or after. No markdown fences."""


def _extract_json(text: str) -> dict:
    """Pull out the first JSON object from the LLM response."""
    text = text.strip()
    if text.startswith("```"):
        # strip triple-backtick fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # find the outermost { ... }
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    return json.loads(m.group(0))


def _validate(d: dict) -> None:
    required = ["name", "sku", "short_description", "long_description", "category", "tags"]
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"copywriter response missing keys: {missing}")
    if not isinstance(d["tags"], list) or not all(isinstance(t, str) for t in d["tags"]):
        raise ValueError("tags must be a list of strings")
    if not re.match(r"^[A-Z0-9]{4,12}$", d["sku"]):
        raise ValueError(f"invalid sku: {d['sku']!r}")
    if d["category"] not in {"bracelet", "necklace", "earring", "ring", "anklet", "set", "other"}:
        raise ValueError(f"invalid category: {d['category']!r}")


def write_copy(description: str, price_cents: int, feedback: Optional[str] = None) -> dict:
    """Return validated product metadata dict.

    feedback: optional steering note from the user when re-writing copy
    (e.g. "more poetic", "shorter", "emphasize that it's vintage-inspired").
    """
    price_dollars = f"{price_cents / 100:.2f}".rstrip("0").rstrip(".")
    user = USER_TEMPLATE.format(description=description, price_dollars=price_dollars)
    if feedback:
        user = f"User feedback for this rewrite: {feedback.strip()}\n\n" + user

    payload = {
        "model": COPY_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0.4 if not feedback else 0.6},
        "format": "json",
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logger.info("copywriter: calling %s (attempt %d/%d)", COPY_MODEL, attempt, MAX_ATTEMPTS)
            r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=TIMEOUT_SECONDS)
            r.raise_for_status()
            raw = r.json().get("message", {}).get("content", "")
            if not raw:
                raise RuntimeError("copywriter returned empty content")
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            logger.warning("copywriter timeout/conn error on attempt %d: %s", attempt, e)
            if attempt >= MAX_ATTEMPTS:
                raise
            time.sleep(2)
    else:
        raise RuntimeError(f"copywriter exhausted retries: {last_err}")

    parsed = _extract_json(raw)
    _validate(parsed)

    # Stable SKU: append a 4-char timestamp suffix so re-running on the same
    # piece doesn't collide.
    suffix = format(int(time.time()) % 10000, "04d")
    parsed["sku"] = f"{parsed['sku']}-{suffix}"

    logger.info("copywriter: name=%r sku=%s", parsed["name"], parsed["sku"])
    return parsed
