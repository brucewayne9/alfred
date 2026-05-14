"""Post-publish hook: after a Roen product goes live on WooCommerce, sync it
to the Meta catalog and enqueue an FB Page draft for Mike's approval.

Called from the Telegram bot publish path. Failures are logged but do NOT
roll back the WC publish — Sarah's product is already live regardless.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any

import requests

from core.jewelry import social_queue

logger = logging.getLogger(__name__)

WC_STORE_PRODUCT_URL = "https://www.roenhandmade.com/wp-json/wc/store/v1/products/{wc_id}"

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = TAG_RE.sub(" ", s)
    s = html.unescape(s)
    s = WS_RE.sub(" ", s).strip()
    return s


def _fetch_wc_product(wc_id: int) -> dict | None:
    try:
        r = requests.get(WC_STORE_PRODUCT_URL.format(wc_id=wc_id), timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        logger.exception("WC product fetch failed for #%s", wc_id)
        return None


def _build_caption(name: str, description: str, price_display: str, url: str) -> str:
    """FB caption v1: lead with the product description, close with price + link."""
    desc = description.strip()
    name = name.strip()
    if desc and desc.lower() != name.lower():
        body = desc
    else:
        body = f"{name}."
    return f"{body}\n\n{name} — {price_display}\nHandmade in Atlanta · {url}"


def _wc_to_meta_item(wc: dict) -> dict | None:
    """Same shape as scripts/roen_meta_catalog_sync.py — keep in sync."""
    wc_id = wc.get("id")
    if not wc_id:
        return None
    images = wc.get("images") or []
    if not images:
        return None
    image_url = images[0].get("src") or images[0].get("thumbnail")
    if not image_url:
        return None
    prices = wc.get("prices") or {}
    price_minor = prices.get("price")
    currency = prices.get("currency_code") or "USD"
    if not price_minor or price_minor == "0":
        return None
    stock = wc.get("stock_availability") or {}
    in_stock = stock.get("availability") == "in-stock" or wc.get("is_in_stock", True)
    title = _strip_html(wc.get("name") or "")[:150]
    short = _strip_html(wc.get("short_description") or "")
    long = _strip_html(wc.get("description") or "")
    description = (short or long or title)[:9999]
    item: dict[str, Any] = {
        "retailer_id": f"roen-{wc_id}",
        "title": title,
        "description": description,
        "link": wc.get("permalink") or "",
        "image_link": image_url,
        "price": f"{int(price_minor) / 100:.2f} {currency}",
        "availability": "in stock" if in_stock else "out of stock",
        "condition": "new",
        "brand": "Roen",
        "google_product_category": "188",
        "fb_product_category": "188",
    }
    return item


def on_product_published(wc_id: int) -> dict:
    """Fan-out after a WC product is published.

    Returns a summary dict for logging — never raises. Errors land in
    summary['errors'] so the caller can surface them without rolling back.
    """
    summary: dict[str, Any] = {
        "wc_id": wc_id,
        "catalog_upserted": False,
        "fb_draft_id": None,
        "errors": [],
    }

    wc = _fetch_wc_product(wc_id)
    if wc is None:
        summary["errors"].append("could not fetch WC product")
        return summary

    item = _wc_to_meta_item(wc)
    if item is None:
        summary["errors"].append("product not catalog-able (missing image/price)")
    else:
        try:
            from integrations.meta_roen import client as meta
            meta.upsert_products([item])
            summary["catalog_upserted"] = True
        except Exception as e:
            logger.exception("meta catalog upsert failed for wc#%s", wc_id)
            summary["errors"].append(f"catalog: {e}")

    images = wc.get("images") or []
    image_url = images[0].get("src") if images else None
    if not image_url:
        summary["errors"].append("no image, skipping FB draft")
        return summary

    prices = wc.get("prices") or {}
    minor = prices.get("price") or "0"
    try:
        price_display = f"${int(minor) / 100:.2f}"
    except Exception:
        price_display = "$" + minor
    name = _strip_html(wc.get("name") or "")
    desc = _strip_html(wc.get("short_description") or wc.get("description") or "")
    url = wc.get("permalink") or ""
    caption = _build_caption(name, desc, price_display, url)

    try:
        draft = social_queue.enqueue_draft(
            wc_product_id=int(wc_id),
            product_name=name,
            product_price=price_display,
            product_url=url,
            image_url=image_url,
            caption=caption,
            source="bot_publish",
        )
        summary["fb_draft_id"] = draft["draft_id"]
    except Exception as e:
        logger.exception("enqueue_draft failed for wc#%s", wc_id)
        summary["errors"].append(f"fb_draft: {e}")

    return summary
