"""Klaviyo client for the Rod Wave DLD tour — list/segment counts ("collected").

Account: "Rod Wave Merch" (public XYKnGf). Private key from config/.env
(KLAVIYO_PRIVATE_KEY). Used by Tour Pulse to show how many fans we've actually
collected via the pop-up + waitlist, and per-market segment breakdowns.

Klaviyo quirks handled here:
  - profile_count is only returned by the SINGLE list/segment GET (not the plural
    collection endpoints), via additional-fields[list|segment]=profile_count.
  - tight rate limits -> we throttle between calls and back off on 429.
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path

import requests

logger = logging.getLogger("klaviyo")

BASE = "https://a.klaviyo.com/api"
REVISION = "2024-10-15"
_ENV = Path("/home/aialfred/alfred/config/.env")

# Rod Wave lists that represent "collected" fans.
COLLECT_LISTS = {
    "VBLb4N": "Pop-up leads",
    "V75mRt": "DLD waitlist",
}


def _load_key() -> str:
    key = os.environ.get("KLAVIYO_PRIVATE_KEY", "")
    if key:
        return key.strip()
    if _ENV.exists():
        for line in _ENV.read_text().splitlines():
            if line.startswith("KLAVIYO_PRIVATE_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


_KEY = _load_key()


def _headers() -> dict:
    return {
        "Authorization": f"Klaviyo-API-Key {_KEY}",
        "revision": REVISION,
        "accept": "application/vnd.api+json",
    }


def _get(path: str, params: dict | None = None, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE}{path}", headers=_headers(), params=params, timeout=15)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 1.5)) + 0.5
                time.sleep(min(wait, 6))
                continue
            if r.status_code != 200:
                logger.warning("klaviyo %s -> %s %s", path, r.status_code, r.text[:120])
                return None
            return r.json()
        except Exception as e:  # pragma: no cover
            logger.warning("klaviyo %s error: %s", path, e)
            time.sleep(1)
    return None


def _count(kind: str, oid: str) -> int | None:
    """kind = 'lists' or 'segments'. Returns profile_count or None."""
    field = "list" if kind == "lists" else "segment"
    data = _get(f"/{kind}/{oid}/", {f"additional-fields[{field}]": "profile_count"})
    if not data:
        return None
    return data.get("data", {}).get("attributes", {}).get("profile_count")


def get_overview() -> dict:
    """Collected totals + per-market segment breakdown for the DLD tour."""
    if not _KEY:
        return {"error": "no_key"}

    # 1) collected lists (pop-up + waitlist)
    lists = []
    total_collected = 0
    for lid, label in COLLECT_LISTS.items():
        c = _count("lists", lid)
        time.sleep(0.4)
        if c is not None:
            lists.append({"id": lid, "label": label, "count": c})
            total_collected += c

    # 2) segments — keep only the DLD tour market segments
    seg_index = _get("/segments/") or {}
    segments = []
    for s in seg_index.get("data", []):
        name = s["attributes"]["name"]
        if not name.lower().startswith("dld tour"):
            continue
        c = _count("segments", s["id"])
        time.sleep(0.4)
        market = name.split("·")[-1].strip() if "·" in name else name
        segments.append({"id": s["id"], "market": market, "name": name, "count": c or 0})
    segments.sort(key=lambda x: x["count"], reverse=True)

    return {
        "total_collected": total_collected,
        "lists": lists,
        "segments": segments,
        "ts": int(time.time()),
    }


# --------------------------------------------------------------------------- #
# Events — fire a Klaviyo metric to trigger a Flow (e.g. SMS verification code)
# --------------------------------------------------------------------------- #

def track_event(metric_name: str, *, email: str | None = None,
                phone: str | None = None, properties: dict | None = None) -> bool:
    """Record a Klaviyo event (creates/updates the profile + the metric).

    A Flow triggered by ``metric_name`` can then send an SMS using the event's
    properties (e.g. ``{{ event.code }}``). Returns True on apparent success;
    never raises so it can't break the signup flow. Phone must be E.164 (+1...).
    """
    if not _KEY:
        logger.warning("klaviyo track_event: no API key configured")
        return False
    attrs: dict = {}
    if email:
        attrs["email"] = email
    if phone:
        attrs["phone_number"] = phone
    if not attrs:
        logger.warning("klaviyo track_event: need email or phone")
        return False
    payload = {
        "data": {
            "type": "event",
            "attributes": {
                "metric": {"data": {"type": "metric",
                                    "attributes": {"name": metric_name}}},
                "profile": {"data": {"type": "profile", "attributes": attrs}},
                "properties": properties or {},
            },
        }
    }
    try:
        r = requests.post(
            f"{BASE}/events/",
            headers={**_headers(), "content-type": "application/vnd.api+json"},
            json=payload, timeout=15,
        )
        if r.status_code in (200, 201, 202):
            return True
        logger.warning("klaviyo track_event -> %s %s", r.status_code, r.text[:200])
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning("klaviyo track_event error: %s", e)
        return False
