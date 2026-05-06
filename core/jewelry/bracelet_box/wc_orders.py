"""Poll WooCommerce for new paid orders containing the bracelet-box SKU.

The bot owns the polling loop (a daemon thread in scripts/roen_telegram_bot.py).
Cursor (last seen order id) is persisted at data/roen/last_box_order_id.txt
so we resume cleanly across bot restarts.

Helpers:
- iter_new_box_line_items(after_id) — yield {order_id, line_item_id,
  quantity, customer_email, customer_first_name} for box items in newer orders
- fetch_in_stock_bracelets() — current in-stock bracelets with their tags
- reserve_skus(ids) — atomically set stock=0 / outofstock on each (returns
  False if any was already unavailable)
- release_skus(ids) — restore stock=1 / instock on each (refund/cancel path)
- load_cursor() / save_cursor(path, id) — file-based cursor persistence
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterator, Optional

sys.path.insert(0, "/home/aialfred/alfred")

import requests

from config.settings import Settings

WC_BASE = "https://www.roenhandmade.com/wp-json/wc/v3"
BOX_SKU = "bracelet-box"
CURSOR_PATH = Path("/home/aialfred/alfred/data/roen/last_box_order_id.txt")

log = logging.getLogger(__name__)


def _settings() -> Settings:
    """Lazy-load settings so test patches can intercept easily."""
    return Settings()


def _auth():
    s = _settings()
    return (s.wc_roen_key, s.wc_roen_secret)


# --------------- order polling ---------------

def _fetch_orders_after(after_id: int) -> list:
    """Return paid/processing orders with id > after_id, oldest first.

    Date filter ('after' parameter) gives a hard floor so a stale cursor
    can't ask WC to return years of history.
    """
    r = requests.get(
        f"{WC_BASE}/orders",
        auth=_auth(),
        params={
            "status": "processing,completed",
            "orderby": "id",
            "order": "asc",
            "per_page": 50,
            "after": "2026-05-01T00:00:00",
        },
        timeout=30,
    )
    r.raise_for_status()
    return [o for o in r.json() if o['id'] > after_id]


def iter_new_box_line_items(after_id: int) -> Iterator[dict]:
    """Yield {order_id, line_item_id, quantity, customer_email,
    customer_first_name} for every line item with the box SKU in orders
    newer than after_id."""
    orders = _fetch_orders_after(after_id)
    for o in orders:
        for li in o.get('line_items', []):
            if li.get('sku') == BOX_SKU:
                billing = o.get('billing') or {}
                yield {
                    'order_id': o['id'],
                    'line_item_id': li['id'],
                    'quantity': int(li.get('quantity', 1)),
                    'customer_email': (billing.get('email') or '').strip().lower(),
                    'customer_first_name': (billing.get('first_name') or '').strip() or None,
                }


# --------------- cursor ---------------

def load_cursor(path: Path = CURSOR_PATH) -> int:
    if not path.exists():
        return 0
    try:
        return int(path.read_text().strip())
    except (ValueError, OSError):
        return 0


def save_cursor(path: Path, order_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(order_id))


# --------------- bracelet inventory ---------------

_BRACELETS_CAT_ID: Optional[int] = None


def _bracelets_cat_id() -> int:
    """Cache the 'bracelet' product_cat term id for one process lifetime.

    Note: actual slug in the WP catalog is 'bracelet' (singular).
    Function name kept plural for readability in callers.
    """
    global _BRACELETS_CAT_ID
    if _BRACELETS_CAT_ID is None:
        r = requests.get(
            f"{WC_BASE}/products/categories",
            auth=_auth(),
            params={"slug": "bracelet"},
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json()
        _BRACELETS_CAT_ID = rows[0]['id'] if rows else 0
    return _BRACELETS_CAT_ID


def fetch_in_stock_bracelets() -> list[dict]:
    """Return all in-stock published bracelets as candidate dicts the picker
    expects: {id, name, short, color_family, material_class, style_class,
    days_in_stock, image_url}."""
    cat = _bracelets_cat_id()
    if not cat:
        log.warning("bracelets category not found — returning empty list")
        return []

    out: list[dict] = []
    page = 1
    while True:
        r = requests.get(
            f"{WC_BASE}/products",
            auth=_auth(),
            params={
                "status": "publish",
                "stock_status": "instock",
                "category": cat,
                "per_page": 50,
                "page": page,
            },
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        for p in rows:
            meta = {m['key']: m['value'] for m in p.get('meta_data', [])}
            images = p.get('images') or []
            out.append({
                'id': p['id'],
                'name': p['name'],
                'short': p.get('short_description', ''),
                'color_family': meta.get('_roen_color_family', 'mixed'),
                'material_class': meta.get('_roen_material_class', 'other'),
                'style_class': meta.get('_roen_style_class', 'classic'),
                'days_in_stock': 7,  # placeholder; could derive from date_modified later
                'image_url': images[0]['src'] if images else '',
            })
        page += 1
    return out


# --------------- stock reservation ---------------

def reserve_skus(product_ids: list[int]) -> bool:
    """Atomically set stock_quantity=0 / status=outofstock on each id.

    Returns True on success, False if any product was already out of stock
    (in which case caller should re-pick — this is the concurrency guard).

    Note: there is a genuine TOCTOU window between the pre-check pass and
    the commit pass. Two concurrent pick sessions could both pass the
    pre-check for overlapping SKUs and then both attempt to commit. The
    commit is last-writer-wins at WC, so the second writer would silently
    double-commit. Mitigation: the polling loop should be single-threaded
    and sessions should be serialized; for extra safety, callers may verify
    the final stock_status after reserve_skus returns True.
    """
    # Pre-check pass: bail early if anything is already gone
    for pid in product_ids:
        r = requests.get(f"{WC_BASE}/products/{pid}", auth=_auth(), timeout=30)
        r.raise_for_status()
        prod = r.json()
        if prod.get('stock_status') != 'instock' or (prod.get('stock_quantity') or 0) < 1:
            return False
    # Commit pass
    for pid in product_ids:
        requests.put(
            f"{WC_BASE}/products/{pid}",
            auth=_auth(),
            json={'stock_quantity': 0, 'stock_status': 'outofstock'},
            timeout=30,
        ).raise_for_status()
    return True


def release_skus(product_ids: list[int]) -> None:
    """Restore stock_quantity=1 / status=instock on each (refund/cancel path)."""
    for pid in product_ids:
        requests.put(
            f"{WC_BASE}/products/{pid}",
            auth=_auth(),
            json={'stock_quantity': 1, 'stock_status': 'instock'},
            timeout=30,
        ).raise_for_status()
