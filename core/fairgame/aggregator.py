"""Fair Game / Fans First — the "everything else" aggregator (camouflage tab).

This is the SPIKE for the second half of Fans First: the tab that makes the
site look like a general ticket marketplace (sports, concerts, theatre, the
World Cup) instead of a Rod-Wave-only store. We do NOT hold or sell this
inventory. We index public listings from a real feed, show the fan the best
verified option, and on click redirect OUT to the originating marketplace,
carrying our affiliate sub-ID. The fan pays a tiny "verified ticket" service
fee to us; the affiliate commission rides the redirect on the back end.

Model mirrors SeatSearch / "Listed": a neutral aggregation layer over partner
feeds. Plug new feeds in behind `search()` (TicketNetwork Direct, SeatGeek,
Vivid Seats affiliate) without touching the API or UI.

Live feed: Ticketmaster Discovery API (instant free key). Set the key in the
environment and the tab pulls live; with no key it serves a clearly-labelled
DEMO sample so the UI is reviewable before the credential lands. Nothing here
ever fabricates a "live" result — the response always reports `live: true|false`.

Env:
  FAIRGAME_DISCOVERY_KEY   Ticketmaster Discovery API key (developer.ticketmaster.com)
  FAIRGAME_AFFIL_SUBID     affiliate sub-ID appended to every outbound buy link
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger("fairgame.aggregator")

_DISCOVERY_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
_HTTP_TIMEOUT = 8  # seconds — the tab must stay snappy; fail to demo on timeout

# The flat "verified ticket" service fee we charge for sourcing a real listing.
# This is the $1 "insurance" Dharmic described — small, framed as fraud protection.
SERVICE_FEE_CENTS = 100

# Discovery segment names we expose as the tab's category chips.
SEGMENTS = ["Sports", "Music", "Arts & Theatre", "Film", "Miscellaneous"]


def _api_key() -> str:
    return (
        os.environ.get("FAIRGAME_DISCOVERY_KEY")
        or os.environ.get("TM_DISCOVERY_API_KEY")
        or ""
    ).strip()


def _affil_subid() -> str:
    return os.environ.get("FAIRGAME_AFFIL_SUBID", "fansfirst").strip()


def is_live() -> bool:
    """True when a real feed credential is configured."""
    return bool(_api_key())


# --------------------------------------------------------------------------- #
# Outbound link — carry our affiliate sub-ID to the originating marketplace
# --------------------------------------------------------------------------- #

def affiliate_url(raw_url: str, source: str) -> str:
    """Append our affiliate sub-ID to a partner buy URL.

    The real commission rail (TM runs affiliates through Impact/Partnerize) is
    wired by swapping this for the network's deep-link wrapper. For the spike we
    tag the URL with our sub-ID so click attribution is provable end-to-end.
    """
    if not raw_url:
        return raw_url
    sub = _affil_subid()
    parts = list(urllib.parse.urlsplit(raw_url))
    q = dict(urllib.parse.parse_qsl(parts[3]))
    q.setdefault("camref", sub)        # Impact-style sub-ID slot
    q.setdefault("utm_source", "fansfirst")
    q.setdefault("utm_medium", "aggregator")
    parts[3] = urllib.parse.urlencode(q)
    return urllib.parse.urlunsplit(parts)


# --------------------------------------------------------------------------- #
# Normalisation — one shape regardless of which feed produced the row
# --------------------------------------------------------------------------- #

def _normalize_tm_event(ev: dict[str, Any]) -> dict[str, Any] | None:
    try:
        emb = ev.get("_embedded", {})
        venues = emb.get("venues", []) or []
        venue = venues[0] if venues else {}
        dates = ev.get("dates", {}).get("start", {})
        prices = ev.get("priceRanges", []) or []
        price = prices[0] if prices else {}
        classes = ev.get("classifications", []) or []
        seg = (classes[0].get("segment", {}) if classes else {}).get("name")
        imgs = ev.get("images", []) or []
        # Prefer a wide 16:9-ish image for the card.
        img = next((i["url"] for i in imgs if i.get("ratio") == "16_9" and i.get("width", 0) >= 640), None)
        if not img and imgs:
            img = imgs[0].get("url")
        return {
            "id": ev.get("id"),
            "name": ev.get("name"),
            "segment": seg,
            "date": dates.get("localDate"),
            "time": dates.get("localTime"),
            "venue": venue.get("name"),
            "city": (venue.get("city") or {}).get("name"),
            "state": (venue.get("state") or {}).get("stateCode"),
            "min_price": price.get("min"),
            "max_price": price.get("max"),
            "currency": price.get("currency") or "USD",
            "source": "Ticketmaster",
            "buy_url": ev.get("url"),
            "image": img,
        }
    except Exception as e:  # noqa: BLE001 - one bad row must not kill the feed
        logger.warning("aggregator: skipped malformed TM event: %s", e)
        return None


def _fetch_tm(query: str, segment: str | None, city: str | None, size: int) -> list[dict]:
    params = {
        "apikey": _api_key(),
        "size": str(max(1, min(size, 40))),
        "sort": "relevance,desc",
        "locale": "*",
    }
    if query:
        params["keyword"] = query
    if segment:
        params["segmentName"] = segment
    if city:
        params["city"] = city
    url = _DISCOVERY_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    events = (data.get("_embedded", {}) or {}).get("events", []) or []
    out = [_normalize_tm_event(e) for e in events]
    return [r for r in out if r]


# --------------------------------------------------------------------------- #
# SeatGeek — second price source (free Platform API, client_id read access).
# Returns aggregate stats.lowest_price (null until an event has live listings).
# --------------------------------------------------------------------------- #

_SEATGEEK_URL = "https://api.seatgeek.com/2/events"

# SeatGeek event `type` -> our segment chips.
_SG_SEGMENT = {
    "concert": "Music", "music_festival": "Music",
    "nba": "Sports", "nfl": "Sports", "mlb": "Sports", "nhl": "Sports", "mls": "Sports",
    "ncaa_basketball": "Sports", "ncaa_football": "Sports", "soccer": "Sports", "sports": "Sports",
    "theater": "Arts & Theatre", "broadway_tickets_national": "Arts & Theatre",
    "comedy": "Arts & Theatre", "dance_performance_tour": "Arts & Theatre", "classical": "Arts & Theatre",
}


def _seatgeek_id() -> str:
    return os.environ.get("FAIRGAME_SEATGEEK_CLIENT_ID", "").strip()


def _normalize_sg_event(ev: dict[str, Any]) -> dict[str, Any] | None:
    try:
        venue = ev.get("venue", {}) or {}
        stats = ev.get("stats", {}) or {}
        perfs = ev.get("performers", []) or []
        img = next((p["image"] for p in perfs if p.get("image")), None)
        dt = ev.get("datetime_local") or ""        # "2026-11-14T20:30:00"
        return {
            "id": "sg_" + str(ev.get("id")),
            "name": ev.get("short_title") or ev.get("title"),
            "segment": _SG_SEGMENT.get(ev.get("type") or "", "Miscellaneous"),
            "date": (dt[:10] or None),
            "time": (dt[11:16] or None),
            "venue": venue.get("name"),
            "city": venue.get("city"),
            "state": venue.get("state"),
            "min_price": stats.get("lowest_price"),
            "max_price": stats.get("highest_price"),
            "currency": "USD",
            "source": "SeatGeek",
            "buy_url": ev.get("url"),
            "image": img,
        }
    except Exception as e:  # noqa: BLE001 - one bad row must not kill the feed
        logger.warning("aggregator: skipped malformed SeatGeek event: %s", e)
        return None


def _fetch_sg(query: str, city: str | None, size: int) -> list[dict]:
    cid = _seatgeek_id()
    if not cid:
        return []
    params = {"client_id": cid, "per_page": str(max(1, min(size, 40)))}
    if query:
        params["q"] = query
    if city:
        params["venue.city"] = city
    url = _SEATGEEK_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = [_normalize_sg_event(e) for e in (data.get("events", []) or [])]
    return [r for r in out if r]


def _evt_key(e: dict) -> tuple | None:
    """Coarse identity for matching the same event across feeds: squashed name + date."""
    import re
    n = re.sub(r"[^a-z0-9]", "", (e.get("name") or "").lower())[:18]
    return (n, e.get("date") or "") if n else None


def _merge_lowest(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Blend two feeds. The same event across both collapses to one card that
    headlines the LOWEST verified price and keeps every source under `sources`
    so the page can show who has the best deal."""
    out: list[dict] = []
    index: dict[tuple, dict] = {}
    for e in list(primary) + list(secondary):
        k = _evt_key(e)
        if k and k in index:
            base = index[k]
            base["sources"].append({"source": e.get("source"), "price": e.get("min_price"), "buy_url": e.get("buy_url")})
            priced = [s for s in base["sources"] if s.get("price") is not None]
            if priced:
                best = min(priced, key=lambda s: s["price"])
                base["min_price"], base["source"], base["buy_url"] = best["price"], best["source"], best["buy_url"]
            if not base.get("image") and e.get("image"):
                base["image"] = e["image"]
        else:
            row = dict(e)
            row["sources"] = [{"source": e.get("source"), "price": e.get("min_price"), "buy_url": e.get("buy_url")}]
            out.append(row)
            if k:
                index[k] = row
    return out


# --------------------------------------------------------------------------- #
# Demo fallback — clearly labelled, never passed off as live
# --------------------------------------------------------------------------- #

_DEMO: list[dict[str, Any]] = [
    {"id": "demo-1", "name": "New York Knicks vs. Boston Celtics", "segment": "Sports",
     "date": "2026-11-14", "time": "19:30:00", "venue": "Madison Square Garden",
     "city": "New York", "state": "NY", "min_price": 89.0, "max_price": 1450.0,
     "currency": "USD", "source": "Ticketmaster",
     "buy_url": "https://www.ticketmaster.com/", "image": None},
    {"id": "demo-2", "name": "FIFA World Cup 2026 — Group Stage", "segment": "Sports",
     "date": "2026-06-28", "time": "16:00:00", "venue": "MetLife Stadium",
     "city": "East Rutherford", "state": "NJ", "min_price": 145.0, "max_price": 980.0,
     "currency": "USD", "source": "Ticketmaster",
     "buy_url": "https://www.ticketmaster.com/", "image": None},
    {"id": "demo-3", "name": "Hamilton", "segment": "Arts & Theatre",
     "date": "2026-07-09", "time": "20:00:00", "venue": "Richard Rodgers Theatre",
     "city": "New York", "state": "NY", "min_price": 179.0, "max_price": 699.0,
     "currency": "USD", "source": "Ticketmaster",
     "buy_url": "https://www.ticketmaster.com/", "image": None},
    {"id": "demo-4", "name": "Atlanta Braves vs. New York Mets", "segment": "Sports",
     "date": "2026-08-02", "time": "13:35:00", "venue": "Truist Park",
     "city": "Atlanta", "state": "GA", "min_price": 22.0, "max_price": 410.0,
     "currency": "USD", "source": "Ticketmaster",
     "buy_url": "https://www.ticketmaster.com/", "image": None},
    {"id": "demo-5", "name": "NASCAR Cup Series — Coca-Cola 600", "segment": "Sports",
     "date": "2026-05-24", "time": "18:00:00", "venue": "Charlotte Motor Speedway",
     "city": "Concord", "state": "NC", "min_price": 59.0, "max_price": 320.0,
     "currency": "USD", "source": "Ticketmaster",
     "buy_url": "https://www.ticketmaster.com/", "image": None},
    {"id": "demo-6", "name": "Kendrick Lamar", "segment": "Music",
     "date": "2026-09-18", "time": "20:00:00", "venue": "Crypto.com Arena",
     "city": "Los Angeles", "state": "CA", "min_price": 95.0, "max_price": 760.0,
     "currency": "USD", "source": "Ticketmaster",
     "buy_url": "https://www.ticketmaster.com/", "image": None},
]


def _demo(query: str, segment: str | None) -> list[dict]:
    rows = _DEMO
    if segment:
        rows = [r for r in rows if r["segment"] == segment]
    if query:
        ql = query.lower()
        rows = [r for r in rows if ql in r["name"].lower() or ql in (r.get("city") or "").lower()]
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #

def search(query: str = "", segment: str | None = None, city: str | None = None,
           size: int = 24) -> dict[str, Any]:
    """Return normalized listings for the aggregator tab.

    Always returns a dict with `live` (was this a real feed call?), `results`,
    and `service_fee_cents`. Falls back to clearly-flagged demo data when no
    feed credential is set or the live call fails — never silently fakes live.
    """
    t0 = time.time()
    live = is_live()
    results: list[dict] = []
    note = None
    sources_used = []
    if live:
        try:
            results = _fetch_tm(query, segment, city, size)
            sources_used.append("Ticketmaster")
        except Exception as e:  # noqa: BLE001 - degrade to demo, surface the reason
            logger.warning("aggregator live feed failed, serving demo: %s", e)
            live = False
            note = "live feed unavailable — showing sample"
    # Second price source — SeatGeek. Augments the live feed; never blocks it.
    if live and _seatgeek_id():
        try:
            sg = _fetch_sg(query, city, size)
            if sg:
                results = _merge_lowest(results, sg)[:size]
                sources_used.append("SeatGeek")
        except Exception as e:  # noqa: BLE001 - a flaky 2nd feed must not break Discover
            logger.warning("aggregator: SeatGeek feed skipped: %s", e)
    if not live:
        results = _demo(query, segment)
    return {
        "live": live,
        "note": note,
        "source": (" + ".join(sources_used) if sources_used else "demo sample"),
        "sources": sources_used,
        "service_fee_cents": SERVICE_FEE_CENTS,
        "count": len(results),
        "took_ms": int((time.time() - t0) * 1000),
        "results": results,
    }
