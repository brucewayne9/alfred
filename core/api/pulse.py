"""Tour Pulse — real-time traffic command center for the Rod Wave "Don't Look Down" site.

A dead-simple, Rod-friendly live dashboard fed straight from GA4 (property 456717749)
via Alfred's service-account credential. No Google login, no Looker Studio.

Endpoints (all gated by a secret token `?t=` so a shared link "just works"):
  GET /pulse/api/live      -> realtime: active now, live signups, top cities (+latlng), countries, devices
  GET /pulse/api/overview  -> rolling stats: summary, sources, signup funnel, daily trend, devices

Served behind Caddy at /pulse/* (static UI) and /pulse/api/* (this app, localhost:8404).
"""
from __future__ import annotations

import os
import time
import logging

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse

from google.analytics.data_v1beta.types import (
    RunRealtimeReportRequest,
    RunReportRequest,
    DateRange,
    Dimension,
    Metric,
    OrderBy,
    Filter,
    FilterExpression,
)

from integrations.google_analytics.client import ga_client
from integrations import klaviyo_client

logger = logging.getLogger("pulse")

PROPERTY = "rodwave"
PROPERTY_ID = "456717749"
TOKEN = os.environ.get("PULSE_TOKEN", "63fda7c3a9f83238fcec5235")

app = FastAPI(title="Tour Pulse — Rod Wave Live")

# ---------------------------------------------------------------------------
# Geocoder: GA returns city/country names; the live map needs lat/lng.
# Built once at import from geonamescache (US cities + world cities + country
# centroids). Falls back gracefully so an unknown city is simply not plotted.
# ---------------------------------------------------------------------------

_US_CITIES: dict[str, tuple[float, float]] = {}
_WORLD_CITIES: dict[str, tuple[float, float]] = {}
_COUNTRY_CENTROID: dict[str, tuple[float, float]] = {}

# GA city names that differ from the gazetteer's canonical name.
_CITY_ALIASES = {
    "New York": "New York City",
    "Washington": "Washington, D.C.",
    "Saint Louis": "St. Louis",
    "Saint Paul": "St. Paul",
}


def _build_gazetteer() -> None:
    try:
        import geonamescache
    except Exception as e:  # pragma: no cover
        logger.warning("geonamescache unavailable, map will be empty: %s", e)
        return
    gc = geonamescache.GeonamesCache()
    for c in gc.get_cities().values():
        name = c["name"]
        lat, lng, pop = c["latitude"], c["longitude"], c.get("population", 0)
        if c["countrycode"] == "US":
            cur = _US_CITIES.get(name)
            if cur is None or pop > _US_CITIES_POP.get(name, 0):
                _US_CITIES[name] = (lat, lng)
                _US_CITIES_POP[name] = pop
        cur = _WORLD_CITIES.get(name)
        if cur is None or pop > _WORLD_CITIES_POP.get(name, 0):
            _WORLD_CITIES[name] = (lat, lng)
            _WORLD_CITIES_POP[name] = pop
    # rough country centroids from each country's most populous city
    for c in gc.get_cities().values():
        cc = c["countrycode"]
        pop = c.get("population", 0)
        if pop > _COUNTRY_POP.get(cc, 0):
            _COUNTRY_CENTROID[cc] = (c["latitude"], c["longitude"])
            _COUNTRY_POP[cc] = pop


_US_CITIES_POP: dict[str, int] = {}
_WORLD_CITIES_POP: dict[str, int] = {}
_COUNTRY_POP: dict[str, int] = {}
_build_gazetteer()


def _geocode_city(name: str, country: str = "United States") -> tuple[float, float] | None:
    if not name:
        return None
    canon = _CITY_ALIASES.get(name, name)
    if country in ("United States", "", None):
        hit = _US_CITIES.get(canon) or _WORLD_CITIES.get(canon)
    else:
        hit = _WORLD_CITIES.get(canon) or _US_CITIES.get(canon)
    return hit


# ---------------------------------------------------------------------------
# Tiny TTL cache — protects GA quota and keeps the UI snappy under auto-refresh.
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, dict]] = {}
_last_good: dict[str, dict] = {}
_LG_PATH = __import__("pathlib").Path(__file__).parent.parent.parent / "data" / "mainstay" / "pulse" / ".last_good.json"

# restore last-good across restarts so a quota blip never shows a blank/error screen
try:
    import json as _json
    if _LG_PATH.exists():
        _last_good.update(_json.loads(_LG_PATH.read_text()))
except Exception:
    pass


def _persist_last_good() -> None:
    try:
        import json as _json
        _LG_PATH.write_text(_json.dumps(_last_good))
    except Exception:
        pass


def _cached(key: str, ttl: float, producer):
    """TTL cache that degrades gracefully: on producer failure (e.g. GA quota),
    serve the last good value flagged stale rather than 500-ing the dashboard."""
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        val = producer()
        _cache[key] = (now, val)
        _last_good[key] = val
        _persist_last_good()
        return val
    except Exception as e:
        logger.warning("producer %s failed (%s) — serving last good", key, e)
        if key in _last_good:
            return {**_last_good[key], "stale": True}
        return {"warming": True}


def _check(t: str) -> None:
    if t != TOKEN:
        raise HTTPException(status_code=403, detail="invalid token")


# Circuit breaker: GA4 realtime has a per-property HOURLY token budget. If we keep
# retrying after a 429 we spend every token the moment it refills and stay pinned
# at "exhausted" forever. So on a quota error we stop calling realtime entirely for
# a cooldown, letting the bucket actually recover.
from google.api_core.exceptions import ResourceExhausted

_rt_cooldown_until = 0.0
_RT_COOLDOWN = 720  # 12 min — long enough for the hourly bucket to refill meaningfully


def _realtime(dimensions: list[str], metric: str, limit: int = 100):
    global _rt_cooldown_until
    if time.time() < _rt_cooldown_until:
        raise RuntimeError("realtime cooling down (quota)")
    client = ga_client._get_client()
    req = RunRealtimeReportRequest(
        property=f"properties/{PROPERTY_ID}",
        dimensions=[Dimension(name=d) for d in dimensions],
        metrics=[Metric(name=metric)],
        limit=limit,
    )
    try:
        return client.run_realtime_report(req)
    except ResourceExhausted:
        _rt_cooldown_until = time.time() + _RT_COOLDOWN
        logger.warning("realtime 429 — backing off %ss", _RT_COOLDOWN)
        raise


# ---------------------------------------------------------------------------
# Live (realtime) — cached 60s. Kept to TWO cheap realtime calls (each <=2 dims)
# to stay well inside the GA4 realtime hourly token budget. Devices come from the
# (cheaper, core-API) overview, not realtime — they barely move minute to minute.
# ---------------------------------------------------------------------------

def _build_live() -> dict:
    """True per-second realtime when GA allows; otherwise a real 'today so far'
    snapshot from the Core API (separate, healthy quota) so the dashboard always
    shows live, truthful data instead of an empty 'connecting' state."""
    try:
        return _build_live_realtime()
    except (ResourceExhausted, RuntimeError) as e:
        logger.info("realtime unavailable (%s) — Core 'today' fallback", str(e)[:60])
        return _build_live_today()


def _build_live_today() -> dict:
    """Core-API 'today so far' snapshot. Real data, refreshes every few minutes."""
    client = ga_client._get_client()
    summary = ga_client.get_traffic_summary(PROPERTY, "today")
    active = summary.get("active_users", 0)
    pageviews = summary.get("page_views", 0)

    geo = client.run_report(RunReportRequest(
        property=f"properties/{PROPERTY_ID}",
        date_ranges=[DateRange(start_date="today", end_date="today")],
        dimensions=[Dimension(name="city"), Dimension(name="country")],
        metrics=[Metric(name="activeUsers")],
        order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="activeUsers"), desc=True)],
        limit=120,
    ))
    cities, country_totals = [], {}
    for row in geo.rows:
        city = row.dimension_values[0].value
        country = row.dimension_values[1].value
        users = int(row.metric_values[0].value)
        country_totals[country] = country_totals.get(country, 0) + users
        if city:
            coords = _geocode_city(city, country)
            if coords:
                cities.append({"city": city, "country": country, "users": users,
                               "lat": coords[0], "lng": coords[1], "us": country == "United States"})
    cities.sort(key=lambda c: c["users"], reverse=True)
    countries = sorted(({"country": k or "(unknown)", "users": v} for k, v in country_totals.items()),
                       key=lambda c: c["users"], reverse=True)[:12]
    su = _signups_report(0)
    return {
        "mode": "today",
        "active_now": active,
        "us_now": country_totals.get("United States", 0),
        "intl_now": active - country_totals.get("United States", 0),
        "pageviews_today": pageviews,
        "signups_today": su.get("form_start", 0),
        "cities": cities[:60],
        "countries": countries,
        "ts": int(time.time()),
    }


def _build_live_realtime() -> dict:
    # Call 1: city + country (map, countries, intl split, total active users)
    geo_resp = _realtime(["city", "country"], "activeUsers", limit=250)
    city_totals: dict[tuple, int] = {}
    country_totals: dict[str, int] = {}
    total = 0
    for row in geo_resp.rows:
        city = row.dimension_values[0].value
        country = row.dimension_values[1].value
        users = int(row.metric_values[0].value)
        total += users
        country_totals[country] = country_totals.get(country, 0) + users
        if city:
            city_totals[(city, country)] = city_totals.get((city, country), 0) + users

    cities = []
    for (city, country), users in city_totals.items():
        coords = _geocode_city(city, country)
        if coords:
            cities.append({"city": city, "country": country, "users": users,
                           "lat": coords[0], "lng": coords[1], "us": country == "United States"})
    cities.sort(key=lambda c: c["users"], reverse=True)

    countries = sorted(
        ({"country": k or "(unknown)", "users": v} for k, v in country_totals.items()),
        key=lambda c: c["users"], reverse=True,
    )[:12]

    # Call 2: live events — signups (form_submit) etc. happening right now
    ev_resp = _realtime(["eventName"], "eventCount")
    events = {r.dimension_values[0].value: int(r.metric_values[0].value) for r in ev_resp.rows}

    return {
        "mode": "live",
        "active_now": total,
        "intl_now": total - country_totals.get("United States", 0),
        "us_now": country_totals.get("United States", 0),
        "signups_30m": events.get("form_submit", 0),
        "form_starts_30m": events.get("form_start", 0),
        "pageviews_30m": events.get("page_view", 0),
        "cities": cities[:60],
        "countries": countries,
        "ts": int(time.time()),
    }


@app.get("/pulse/api/live")
def live(t: str = Query("")):
    _check(t)
    return JSONResponse(_cached("live", 60.0, _build_live))


# ---------------------------------------------------------------------------
# Overview — cached 60s
# ---------------------------------------------------------------------------

def _signups_report(period_days: int) -> dict:
    """form_start / form_submit counts over the trailing window (today=0)."""
    client = ga_client._get_client()
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end if period_days == 0 else end - timedelta(days=period_days)
    resp = client.run_report(RunReportRequest(
        property=f"properties/{PROPERTY_ID}",
        date_ranges=[DateRange(start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"))],
        dimensions=[Dimension(name="eventName")],
        metrics=[Metric(name="eventCount")],
        dimension_filter=FilterExpression(filter=Filter(
            field_name="eventName",
            in_list_filter=Filter.InListFilter(values=["form_start", "form_submit"]),
        )),
    ))
    out = {"form_start": 0, "form_submit": 0}
    for row in resp.rows:
        out[row.dimension_values[0].value] = int(row.metric_values[0].value)
    return out


def _build_overview() -> dict:
    summary = ga_client.get_traffic_summary(PROPERTY, "last_7_days")
    today = ga_client.get_traffic_summary(PROPERTY, "today")
    sources = ga_client.get_traffic_sources(PROPERTY, "last_7_days", 8)
    daily = ga_client.get_daily_traffic(PROPERTY, "last_14_days")
    dev = ga_client.get_devices(PROPERTY, "last_7_days")
    devices = [{"device": d["device"], "pct": int(round(float(d["percentage"].rstrip("%"))))}
               for d in dev.get("devices", [])]
    # form_start is the reliable PROCESSED conversion signal (waitlist opens).
    # form_submit (completed signups) is currently realtime-only — surfaced live
    # via /pulse/api/live, not here, so we never show a misleading processed 0.
    su_today = _signups_report(0)
    su_7d = _signups_report(7)

    # normalize source labels into something Rod reads at a glance
    label_map = {
        "ig": "Instagram", "instagram": "Instagram", "l.instagram.com": "Instagram",
        "facebook": "Facebook", "facebook.com": "Facebook", "m.facebook.com": "Facebook",
        "lm.facebook.com": "Facebook", "l.facebook.com": "Facebook",
        "google": "Google", "(direct)": "Direct / link", "tiktok": "TikTok",
        "t.co": "X / Twitter", "youtube.com": "YouTube",
    }
    src = {}
    for s in sources.get("sources", []):
        name = label_map.get(s["source"].lower(), s["source"])
        src[name] = src.get(name, 0) + s["sessions"]
    sources_clean = sorted(({"source": k, "sessions": v} for k, v in src.items()),
                           key=lambda x: x["sessions"], reverse=True)[:7]

    return {
        "summary_7d": summary,
        "today": today,
        "sources": sources_clean,
        "devices": devices,
        "daily": daily.get("daily", []),
        "signups": {
            "today_starts": su_today["form_start"],
            "week_starts": su_7d["form_start"],
        },
        "ts": int(time.time()),
    }


@app.get("/pulse/api/overview")
def overview(t: str = Query("")):
    _check(t)
    return JSONResponse(_cached("overview", 60.0, _build_overview))


@app.get("/pulse/api/klaviyo")
def klaviyo(t: str = Query("")):
    """Collected fans (pop-up + waitlist) and per-market segments, 5-min cache."""
    _check(t)
    return JSONResponse(_cached("klaviyo", 300.0, klaviyo_client.get_overview))


@app.get("/pulse/api/health")
def health():
    return {"ok": True, "property": PROPERTY_ID, "cities_loaded": len(_US_CITIES)}
