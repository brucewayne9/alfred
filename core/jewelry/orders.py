"""
Roen order management helpers used by the Telegram bot's /orders flow.

Talks to WooCommerce via wp-cli (existing pattern) and exposes:
  - list_pending_orders()      — processing + on-hold orders for Sarah/Mike
  - get_order_details(id)      — full order incl. items, customer, address
  - get_paypal_transaction_id  — finds the WC PayPal Payments txn id meta
  - save_tracking(...)         — stores tracking + order note + (optionally) calls PayPal
  - complete_order(id)         — sets status processing -> completed (fires WC email)
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

SSH_HOST = "server-104"
CONTAINER = "roenhandmade-wp"
WP_PATH = "/var/www/html"


def _wp(args: List[str], timeout: int = 30) -> tuple[int, str, str]:
    """Shell out to wp-cli inside the roenhandmade-wp container via SSH."""
    inner = "wp " + " ".join(shlex.quote(a) for a in args) + f" --allow-root --path={WP_PATH}"
    cmd = ["ssh", SSH_HOST, f"timeout {timeout} docker exec {CONTAINER} {inner}"]
    res = subprocess.run(cmd, capture_output=True, timeout=timeout + 10)
    return res.returncode, res.stdout.decode("utf-8", "replace"), res.stderr.decode("utf-8", "replace")


@dataclass
class OrderLineItem:
    name: str
    product_id: int
    quantity: int
    total: str  # money string e.g. "35.00"


@dataclass
class OrderSummary:
    id: int
    status: str
    total: str
    currency: str
    customer_name: str
    customer_email: str
    items: List[OrderLineItem]
    date_created: str  # ISO-8601


@dataclass
class OrderDetails(OrderSummary):
    shipping_address_lines: List[str]
    billing_address_lines: List[str]
    payment_method: str
    payment_method_title: str
    transaction_id: str
    note_count: int


def _parse_iso(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


def _addr_lines(addr: dict) -> List[str]:
    lines = []
    name = " ".join(filter(None, [addr.get("first_name"), addr.get("last_name")])).strip()
    if name:
        lines.append(name)
    if addr.get("company"):
        lines.append(addr["company"])
    if addr.get("address_1"):
        lines.append(addr["address_1"])
    if addr.get("address_2"):
        lines.append(addr["address_2"])
    city_state_zip = " ".join(
        filter(None, [addr.get("city", "").rstrip(","), addr.get("state"), addr.get("postcode")])
    ).strip()
    if city_state_zip:
        lines.append(city_state_zip)
    if addr.get("country"):
        lines.append(addr["country"])
    return lines


def _to_summary(o: dict) -> OrderSummary:
    items = [
        OrderLineItem(
            name=li.get("name", "?"),
            product_id=int(li.get("product_id", 0)),
            quantity=int(li.get("quantity", 1)),
            total=str(li.get("total", "0.00")),
        )
        for li in (o.get("line_items") or [])
    ]
    billing = o.get("billing") or {}
    customer_name = " ".join(
        filter(None, [billing.get("first_name"), billing.get("last_name")])
    ).strip() or (billing.get("email") or f"order #{o.get('id')}")
    return OrderSummary(
        id=int(o["id"]),
        status=str(o.get("status", "")),
        total=str(o.get("total", "0.00")),
        currency=str(o.get("currency", "USD")),
        customer_name=customer_name,
        customer_email=str(billing.get("email", "")),
        items=items,
        date_created=str(o.get("date_created", "")),
    )


def list_pending_orders() -> List[OrderSummary]:
    """Return processing + on-hold orders, newest first."""
    # wp wc shop_order list accepts only a single --status value, so we run two
    # queries and merge. WC REST itself supports multi-status but the CLI param
    # validator is stricter.
    combined: List[dict] = []
    for status in ("processing", "on-hold"):
        rc, out, err = _wp(
            [
                "wc", "shop_order", "list",
                "--user=1",
                f"--status={status}",
                "--format=json",
                "--per_page=50",
                "--orderby=date",
                "--order=desc",
            ],
            timeout=45,
        )
        if rc != 0:
            raise RuntimeError(f"wp wc shop_order list ({status}) failed: {err.strip()[:200]}")
        try:
            batch = json.loads(out) if out.strip() else []
        except json.JSONDecodeError as e:
            raise RuntimeError(f"could not parse wp wc shop_order ({status}): {e}; head={out[:200]!r}")
        combined.extend(batch)
    combined.sort(key=lambda o: o.get("date_created", ""), reverse=True)
    return [_to_summary(o) for o in combined]


def get_order_details(order_id: int) -> OrderDetails:
    rc, out, err = _wp(
        ["wc", "shop_order", "get", str(order_id), "--user=1", "--format=json"],
        timeout=30,
    )
    if rc != 0 or not out.strip():
        raise RuntimeError(f"wp wc shop_order get {order_id} failed: {err.strip()[:200]}")
    o = json.loads(out)
    summary = _to_summary(o)
    txn_id = _extract_paypal_capture_id(o) or ""
    return OrderDetails(
        **summary.__dict__,
        shipping_address_lines=_addr_lines(o.get("shipping") or {}),
        billing_address_lines=_addr_lines(o.get("billing") or {}),
        payment_method=str(o.get("payment_method", "")),
        payment_method_title=str(o.get("payment_method_title", "")),
        transaction_id=txn_id,
        note_count=int(o.get("customer_note") and 1 or 0),
    )


def _extract_paypal_capture_id(o: dict) -> Optional[str]:
    """
    Find the PayPal capture/transaction id from a WC order JSON response.

    WC 10 stores it in two places (HPOS — meta lives on wp_wc_orders_meta,
    not wp_postmeta):
      1. top-level `transaction_id`  — the PayPal CAPTURE id (preferred)
      2. `meta_data[].key == '_ppcp_paypal_order_id'` — PayPal ORDER id (fallback)

    PayPal's Tracking API accepts the capture id, so prefer (1).
    """
    txn = o.get("transaction_id")
    if txn:
        return str(txn).strip()
    for m in o.get("meta_data") or []:
        if m.get("key") == "_ppcp_paypal_order_id" and m.get("value"):
            return str(m["value"]).strip()
    return None


def get_paypal_transaction_id(order_id: int) -> Optional[str]:
    """Fetch the PayPal capture id for a WC order. Returns None if missing."""
    rc, out, err = _wp(
        ["wc", "shop_order", "get", str(order_id), "--user=1", "--format=json"],
        timeout=30,
    )
    if rc != 0 or not out.strip():
        return None
    try:
        o = json.loads(out)
    except json.JSONDecodeError:
        return None
    return _extract_paypal_capture_id(o)


def save_tracking(order_id: int, tracking: str, carrier: str) -> None:
    """
    Store tracking on the WC order as post_meta + an order note.

    Note text is set with customer_note=true so the buyer also sees it on
    the order-status page.
    """
    # 1. post meta — our own keys + the woocommerce-shipping plugin's expected keys
    for key, value in [
        ("_roen_tracking_number", tracking),
        ("_roen_tracking_carrier", carrier),
    ]:
        rc, _, err = _wp(["post", "meta", "update", str(order_id), key, value], timeout=15)
        if rc != 0:
            raise RuntimeError(f"failed to set {key} on order {order_id}: {err.strip()[:120]}")

    # 2. customer-visible order note
    note = f"Shipped via {carrier}. Tracking: {tracking}"
    rc, _, err = _wp(
        [
            "wc", "shop_order_note", "create",
            "--user=1",
            "--order_id=" + str(order_id),
            "--note=" + note,
            "--customer_note=true",
            "--porcelain",
        ],
        timeout=20,
    )
    if rc != 0:
        # Note failure is non-fatal — meta still applied. Log and continue.
        logger.warning("order note create failed for %s: %s", order_id, err.strip()[:120])


def complete_order(order_id: int) -> None:
    """Set order status from processing → completed (fires customer email)."""
    rc, _, err = _wp(
        [
            "wc", "shop_order", "update", str(order_id),
            "--user=1",
            "--status=completed",
        ],
        timeout=20,
    )
    if rc != 0:
        raise RuntimeError(f"failed to mark order {order_id} completed: {err.strip()[:200]}")
