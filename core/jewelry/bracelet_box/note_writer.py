"""Note generator — kimi-k2.6:cloud, with prompt construction + constraint validation.

The model is asked to return a 60-90 word personal note in Roen's third-
person brand voice. Output is plain text, single paragraph, no emojis.

If the model misbehaves (wrong length, exclamation point, missing signoff),
we retry once. After two attempts we return whatever the model produced —
Sarah is the human-in-the-loop and can edit/reroll via Telegram.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

NOTE_MODEL = "kimi-k2.6:cloud"
OLLAMA_HOST = "http://localhost:11434"
TIMEOUT_SECONDS = 180
MAX_ATTEMPTS = 2

SIGNOFFS = ("with care, roen", "yours, roen", "from the studio")

SYSTEM_PROMPT = (
    "You write a short personal note from a small jewelry brand called Roen "
    "to a customer who just bought a curated 5-bracelet bundle.\n\n"
    "Voice rules:\n"
    "- Third-person brand voice. \"Roen chose...\" — never \"I\" or \"we\" or \"Sarah\".\n"
    "- 60-90 words, single paragraph.\n"
    "- Mention at least 2 of the 5 bracelets by name or visual detail.\n"
    "- State ONE reason why this set works as a set (color story, mood, contrast, etc).\n"
    "- No exclamation points. No emojis. No marketing hype.\n"
    "- End with one signoff, lowercase: \"with care, roen\" / \"yours, roen\" / \"from the studio\".\n\n"
    "Return ONLY the note text. No preamble, no explanation."
)


def build_prompt(picks: List[dict], first_name: Optional[str],
                 past_notes: List[str], order_count: int) -> str:
    """Construct the user-message body sent to kimi.

    `picks` is a list of 5 dicts with at least keys: name, short, color_family,
    material_class. `past_notes` is up to 3 prior note texts to avoid repeating.
    """
    pieces = "\n".join(
        f"  {i+1}. {p['name']} — {p.get('short', '')} "
        f"({p.get('color_family','')}, {p.get('material_class','')})"
        for i, p in enumerate(picks)
    )
    name_part = (
        f"Recipient first name: {first_name}" if first_name
        else "Recipient: anonymous"
    )

    if order_count == 1:
        order_part = "This is their first order — welcome them, briefly."
    elif order_count == 2:
        order_part = "This is their second order — note that Roen is glad they're back."
    else:
        order_part = f"This is order #{order_count} for them — warm but not gushy."

    avoid_part = ""
    if past_notes:
        snippets = "\n".join(f"  - {n[:120]}..." for n in past_notes[:3])
        avoid_part = (
            "\nIMPORTANT — avoid repeating these themes/openers from past "
            "notes to this customer:\n" + snippets
        )

    return (
        f"{name_part}\n{order_part}\n\n"
        f"The five bracelets in this box:\n{pieces}\n"
        f"{avoid_part}"
    )


def _call_kimi(prompt: str) -> str:
    """Send the system prompt + user prompt to kimi via Ollama. Returns text."""
    payload = {
        "model": NOTE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.85, "top_p": 0.9},
    }
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def is_valid_length(text: str) -> bool:
    """60-90 words inclusive."""
    n = len(text.split())
    return 60 <= n <= 90


def has_no_exclamations(text: str) -> bool:
    return "!" not in text


def has_signoff(text: str) -> bool:
    """One of the three approved signoff phrases appears (case-insensitive),
    typically in the last line/two lines of the note."""
    if not text:
        return False
    tail = text.strip().lower()
    # Check the last 80 chars to be lenient about formatting
    tail_window = tail[-80:]
    return any(s in tail_window for s in SIGNOFFS)


def generate(picks: List[dict], first_name: Optional[str],
             past_notes: List[str], order_count: int,
             max_attempts: int = MAX_ATTEMPTS) -> str:
    """Generate a note. Retries once on constraint violation; returns the
    last attempt regardless (Sarah will approve/edit on Telegram)."""
    prompt = build_prompt(picks, first_name, past_notes, order_count)
    last_text = ""
    for attempt in range(max_attempts):
        try:
            text = _call_kimi(prompt)
        except requests.RequestException as e:
            logger.warning("kimi call failed (attempt %d/%d): %s",
                           attempt + 1, max_attempts, e)
            continue
        last_text = text
        if (is_valid_length(text)
                and has_no_exclamations(text)
                and has_signoff(text)):
            return text
        logger.info("note failed validation (attempt %d), retrying", attempt + 1)
    return last_text  # may be empty if all kimi calls raised
