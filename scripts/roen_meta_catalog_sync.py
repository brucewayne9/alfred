#!/usr/bin/env python3
"""Sync roenhandmade.com WooCommerce products into the Roen Meta catalog.

Source: WC Store API (public, no auth) at https://www.roenhandmade.com/wp-json/wc/store/v1/products.
Target: Meta Commerce catalog 2341589166333978.

Idempotent — safe to re-run. Reconciles by retailer_id (`roen-<wc_product_id>`).

Flags:
    --dry-run         Print the plan, don't push anything.
    --wipe-stale      Delete items in the Meta catalog that are no longer on the site.
    --verbose         Per-product log lines.

Usage:
    python3 scripts/roen_meta_catalog_sync.py --wipe-stale
"""

from __future__ import annotations

import argparse
import html
import logging
import re
import sys
from typing import Any

import requests

sys.path.insert(0, "/home/aialfred/alfred")

from integrations.meta_roen import client as meta  # noqa: E402

logger = logging.getLogger("roen_meta_sync")

WC_PRODUCTS_URL = "https://www.roenhandmade.com/wp-json/wc/store/v1/products"
BRAND = "Roen"
RETAILER_ID_PREFIX = "roen-"
MAX_NAME = 150
MAX_DESC = 9999

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def strip_html(s: str) -> str:
    if not s:
        return ""
    s = TAG_RE.sub(" ", s)
    s = html.unescape(s)
    s = WS_RE.sub(" ", s).strip()
    return s


def fetch_wc_products() -> list[dict]:
    """Paginate the WC Store API and return every published product."""
    out: list[dict] = []
    page = 1
    while True:
        r = requests.get(
            WC_PRODUCTS_URL,
            params={"per_page": 100, "page": page},
            timeout=20,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 100:
            break
        page += 1
        if page > 50:  # hard safety stop
            logger.warning("WC pagination hit safety stop at page 50")
            break
    return out


def wc_to_meta_item(p: dict) -> dict | None:
    """Convert one WC product to the Meta catalog item shape. Returns None if not catalog-able."""
    wc_id = p.get("id")
    if not wc_id:
        return None

    images = p.get("images") or []
    if not images:
        # Meta requires image_url. Skip imageless products and log.
        logger.warning("skip wc#%s (%s): no images", wc_id, p.get("name", "?"))
        return None
    image_url = images[0].get("src") or images[0].get("thumbnail")
    if not image_url:
        logger.warning("skip wc#%s: image entry has no src", wc_id)
        return None

    # WC Store API returns price in MINOR units (cents) as a string.
    # Meta items_batch expects a DECIMAL-formatted string like "10.00 USD".
    prices = p.get("prices") or {}
    price_minor = prices.get("price")
    currency = prices.get("currency_code") or "USD"
    if not price_minor or price_minor == "0":
        logger.warning("skip wc#%s (%s): no price set", wc_id, p.get("name", "?"))
        return None
    price_str = f"{int(price_minor) / 100:.2f} {currency}"

    stock = p.get("stock_availability") or {}
    in_stock = stock.get("availability") == "in-stock" or p.get("is_in_stock", True)
    availability = "in stock" if in_stock else "out of stock"

    title = strip_html(p.get("name") or "")[:MAX_NAME]
    short = strip_html(p.get("short_description") or "")
    long = strip_html(p.get("description") or "")
    description = (short or long or title)[:MAX_DESC]

    link = p.get("permalink") or ""

    # NOTE: Meta items_batch uses Google-product-feed field names:
    # title (not name), link (not url), image_link (not image_url).
    item: dict[str, Any] = {
        "retailer_id": f"{RETAILER_ID_PREFIX}{wc_id}",
        "title": title,
        "description": description,
        "link": link,
        "image_link": image_url,
        "price": price_str,
        "availability": availability,
        "condition": "new",
        "brand": BRAND,
    }

    # Optional but improves discovery
    categories = p.get("categories") or []
    if categories:
        item["google_product_category"] = "188"  # Apparel & Accessories > Jewelry
        item["fb_product_category"] = "188"

    return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wipe-stale", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("fetching WC products from %s", WC_PRODUCTS_URL)
    wc = fetch_wc_products()
    logger.info("WC: %d products fetched", len(wc))

    items: list[dict] = []
    skipped = 0
    for p in wc:
        item = wc_to_meta_item(p)
        if item is None:
            skipped += 1
            continue
        items.append(item)
    logger.info("prepared %d catalog items (%d skipped)", len(items), skipped)

    current_rids = set(meta.list_all_catalog_retailer_ids())
    logger.info("Meta catalog currently has %d items", len(current_rids))

    desired_rids = {it["retailer_id"] for it in items}
    stale = current_rids - desired_rids
    new_or_updated = items
    logger.info(
        "plan: upsert %d items, %d stale in catalog (will %s)",
        len(new_or_updated),
        len(stale),
        "delete" if args.wipe_stale else "leave",
    )

    if args.dry_run:
        logger.info("DRY RUN — exiting without writes")
        for it in items[:3]:
            logger.info("sample: %s", it)
        return 0

    # Upsert in chunks of 100 (well under Meta's 5000 cap, easier to read errors)
    for i in range(0, len(new_or_updated), 100):
        chunk = new_or_updated[i : i + 100]
        resp = meta.upsert_products(chunk)
        logger.info("upsert chunk %d-%d: handles=%s", i, i + len(chunk), resp.get("handles", [])[:1])

    if args.wipe_stale and stale:
        for i in range(0, len(stale), 100):
            chunk = list(stale)[i : i + 100]
            resp = meta.delete_products(chunk)
            logger.info("delete chunk %d-%d: handles=%s", i, i + len(chunk), resp.get("handles", [])[:1])

    # Re-check
    final = meta.list_all_catalog_retailer_ids()
    logger.info("DONE — Meta catalog now has %d items", len(final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
