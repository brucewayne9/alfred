"""
Telegram edit dispatcher: parse Sarah's free-text edit instruction into a
structured action and apply it to an existing WooCommerce draft.

Supported actions:
    set_name <new name>
    set_price <dollars>
    set_short_description <text>
    set_description <text>
    rewrite_copy <feedback steering note>     -> re-runs copywriter
    delete                                    -> trash the draft
    publish                                   -> publish the draft
    unknown                                   -> bot asks for clarification
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

import requests

from core.jewelry import db, copywriter, woocommerce

logger = logging.getLogger(__name__)

OLLAMA_HOST = "http://localhost:11434"
INTENT_MODEL = "kimi-k2.6:cloud"
TIMEOUT_SECONDS = 60

INTENT_SYSTEM = (
    "You parse short free-text edit instructions for a jewelry shop owner "
    "editing a product draft. Return a JSON object with two keys: "
    "'action' (one of: set_name, set_price, set_short_description, "
    "set_description, rewrite_copy, delete, publish, unknown) and "
    "'value' (a string — for set_price use just the number in dollars; "
    "for delete/publish leave it empty; for rewrite_copy include the user's "
    "feedback verbatim). Nothing else. No prose."
)

INTENT_FEW_SHOT = [
    {"role": "user", "content": "change name to Sterling Moonstone Cuff"},
    {"role": "assistant", "content": '{"action":"set_name","value":"Sterling Moonstone Cuff"}'},
    {"role": "user", "content": "drop the price to $55"},
    {"role": "assistant", "content": '{"action":"set_price","value":"55"}'},
    {"role": "user", "content": "rewrite the description, more poetic, keep it short"},
    {"role": "assistant", "content": '{"action":"rewrite_copy","value":"more poetic, keep it short"}'},
    {"role": "user", "content": "trash this"},
    {"role": "assistant", "content": '{"action":"delete","value":""}'},
    {"role": "user", "content": "make this live"},
    {"role": "assistant", "content": '{"action":"publish","value":""}'},
    {"role": "user", "content": "make the short description: Hand-strung garnet on gold-fill chain."},
    {"role": "assistant", "content": '{"action":"set_short_description","value":"Hand-strung garnet on gold-fill chain."}'},
]

VALID_ACTIONS = {
    "set_name", "set_price", "set_short_description",
    "set_description", "rewrite_copy", "delete", "publish", "unknown",
}


# ---- post id extraction from a quoted bot message ----

POST_ID_PATTERNS = [
    re.compile(r"[?&]p=(\d+)"),                     # preview link
    re.compile(r"[?&]post=(\d+)"),                  # admin edit link
]


def extract_post_id(text: str) -> Optional[int]:
    """Pull a post_id out of a bot-formatted draft message."""
    if not text:
        return None
    for pat in POST_ID_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1))
    return None


# ---- intent parser ----

def parse_intent(text: str) -> dict:
    """Return {'action': str, 'value': str}. Raises on transport error."""
    text = (text or "").strip()
    if not text:
        return {"action": "unknown", "value": ""}

    # Cheap deterministic shortcuts before paying for an LLM call.
    lower = text.lower()
    if lower in ("delete", "trash", "trash this", "remove", "drop this"):
        return {"action": "delete", "value": ""}
    if lower in ("publish", "make live", "go live", "ship it"):
        return {"action": "publish", "value": ""}

    messages = [{"role": "system", "content": INTENT_SYSTEM}]
    messages.extend(INTENT_FEW_SHOT)
    messages.append({"role": "user", "content": text})

    payload = {
        "model": INTENT_MODEL,
        "stream": False,
        "messages": messages,
        "options": {"temperature": 0.0},
        "format": "json",
    }
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    raw = r.json().get("message", {}).get("content", "").strip()
    if not raw:
        return {"action": "unknown", "value": ""}
    # Strip optional fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return {"action": "unknown", "value": ""}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "unknown", "value": ""}
    action = d.get("action", "unknown")
    if action not in VALID_ACTIONS:
        action = "unknown"
    value = (d.get("value") or "").strip()
    return {"action": action, "value": value}


# ---- apply ----

def apply_edit(post_id: int, intent: dict) -> dict:
    """Execute the parsed intent. Returns a small result dict for the bot to relay."""
    action = intent["action"]
    value = intent["value"]

    intake_row = db.find_intake_by_post(post_id)
    intake_id = int(intake_row["id"]) if intake_row else None

    if action == "set_name":
        if not value:
            return {"ok": False, "msg": "I didn't catch the new name. Try: change name to <name>."}
        woocommerce.update_product_field(post_id, "name", value)
        if intake_id:
            db.set_copy(
                intake_id,
                seo_title=value,
                sku=intake_row["sku"] or "",
                short_description=intake_row["short_description"] or "",
                long_description=intake_row["long_description"] or "",
            )
        return {"ok": True, "msg": f"Name updated to <b>{value}</b>."}

    if action == "set_price":
        m = re.search(r"(\d+)(?:[.,](\d{2}))?", value)
        if not m:
            return {"ok": False, "msg": "I didn't catch a price. Try: price is $55."}
        dollars = m.group(1)
        cents = m.group(2) or "00"
        price_str = f"{dollars}.{cents}"
        woocommerce.update_product_field(post_id, "regular_price", price_str)
        return {"ok": True, "msg": f"Price updated to <b>${price_str}</b>."}

    if action == "set_short_description":
        if not value:
            return {"ok": False, "msg": "I didn't catch the new short description."}
        woocommerce.update_product_field(post_id, "short_description", value)
        if intake_id:
            db.set_copy(
                intake_id,
                seo_title=intake_row["seo_title"] or "",
                sku=intake_row["sku"] or "",
                short_description=value,
                long_description=intake_row["long_description"] or "",
            )
        return {"ok": True, "msg": "Short description updated."}

    if action == "set_description":
        if not value:
            return {"ok": False, "msg": "I didn't catch the new description."}
        woocommerce.update_product_field(post_id, "description", value)
        if intake_id:
            db.set_copy(
                intake_id,
                seo_title=intake_row["seo_title"] or "",
                sku=intake_row["sku"] or "",
                short_description=intake_row["short_description"] or "",
                long_description=value,
            )
        return {"ok": True, "msg": "Description updated."}

    if action == "rewrite_copy":
        if not intake_row or not intake_row["description"]:
            return {"ok": False, "msg": "Can't rewrite — original vision description not on file for this draft."}
        feedback = value or "tighten and clarify"
        new_copy = copywriter.write_copy(
            intake_row["description"],
            int(intake_row["price_cents"]),
            feedback=feedback,
        )
        # Apply name + descriptions back to the product.
        woocommerce.update_product_field(post_id, "name", new_copy["name"])
        woocommerce.update_product_field(post_id, "short_description", new_copy["short_description"])
        woocommerce.update_product_field(post_id, "description", new_copy["long_description"])
        if intake_id:
            db.set_copy(
                intake_id,
                seo_title=new_copy["name"],
                sku=intake_row["sku"] or new_copy["sku"],
                short_description=new_copy["short_description"],
                long_description=new_copy["long_description"],
            )
        return {"ok": True, "msg": f"Rewrote the listing.\n\n<b>{new_copy['name']}</b>\n{new_copy['short_description']}"}

    if action == "delete":
        woocommerce.trash_product(post_id)
        if intake_id:
            db.set_status(intake_id, "deleted")
        return {"ok": True, "msg": f"Draft #{post_id} moved to trash."}

    if action == "publish":
        woocommerce.publish_product(post_id)
        return {"ok": True, "msg": f"Published draft #{post_id}."}

    return {
        "ok": False,
        "msg": (
            "I didn't catch that edit. Try one of:\n"
            "  <code>change name to ...</code>\n"
            "  <code>price is $55</code>\n"
            "  <code>rewrite description, more poetic</code>\n"
            "  <code>delete</code>  /  <code>publish</code>"
        ),
    }
