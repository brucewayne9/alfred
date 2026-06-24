"""Fair Game seat maps — venue geometry served to web V1 and native V2.

The geometry is the OPEN layer of Ticketmaster's data: the per-venue seat
manifest (sections -> rows -> seat coordinates) published at
mapsapi.tmol.io/.../placeDetailNoKeys, ingested once per market and normalized
into a compact schematic map (see data/mainstay/fairgame/seatmaps/). No live
availability or pricing lives here -- that is the key-gated layer and overlays
on top of these coordinates when a Commerce/Inventory key lands.

Resolution chain:  show_<idx>  ->  show_event_map.json  ->  eventId  ->
normalized/<eventId>.json. Maps are deduped by venueConfigId upstream, so the
two nights of a market share one file.

The normalized payload is ~250-290 KB with every seat. `overview()` strips the
per-seat arrays (sections + bboxes + centroids only, a few KB) for the first
paint; `section()` lazy-loads one section's rows + seats on zoom. This keeps
mobile fast: draw section polygons first, fetch seats only where the fan looks.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_SEATMAP_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "mainstay" / "fairgame" / "seatmaps"
)
_NORMALIZED = _SEATMAP_DIR / "normalized"
_SHOW_MAP = _SEATMAP_DIR / "show_event_map.json"


def _show_idx(show_id: str) -> str:
    """`show_7` -> `7`. Returns the raw id back if it isn't the expected shape."""
    return show_id[5:] if show_id.startswith("show_") else show_id


@lru_cache(maxsize=1)
def _show_event_map() -> dict:
    if not _SHOW_MAP.exists():
        return {}
    return json.loads(_SHOW_MAP.read_text())


def event_id_for_show(show_id: str):
    """TM event id backing a show, or None (non-TM system or unmapped)."""
    entry = _show_event_map().get(_show_idx(show_id))
    return entry.get("eventId") if entry else None


def system_for_show(show_id: str) -> str:
    """Ticketing system for a show: ticketmaster | axs | seatgeek | unknown."""
    entry = _show_event_map().get(_show_idx(show_id))
    return entry.get("system", "unknown") if entry else "unknown"


@lru_cache(maxsize=64)
def _load_normalized(event_id: str):
    """Parsed normalized map for an event id, or None if not ingested yet."""
    path = _NORMALIZED / f"{event_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _resolve(show_id: str):
    """(event_id, normalized_map) for a show. Either may be None."""
    eid = event_id_for_show(show_id)
    if not eid:
        return None, None
    return eid, _load_normalized(eid)


def status(show_id: str) -> dict:
    """Lightweight availability flag for a show's map (no geometry payload).

    available=True means we have an ingested map. `reason` explains a miss so
    the client can show the right empty state (non-TM venue vs. map not yet
    published by Ticketmaster).
    """
    system = system_for_show(show_id)
    if system in ("axs", "seatgeek"):
        return {"available": False, "system": system, "reason": "non_ticketmaster_venue"}
    eid, m = _resolve(show_id)
    if not eid:
        return {"available": False, "system": system, "reason": "unmapped_show"}
    if m is None:
        return {"available": False, "system": system, "event_id": eid,
                "reason": "map_not_published"}
    return {"available": True, "system": system, "event_id": eid,
            "venue_config_id": m.get("venueConfigId"), "capacity": m.get("capacity")}


def overview(show_id: str):
    """Section-level map for first paint: sections, bboxes, centroids, seat
    counts, GA flags, canvas dims -- but NOT the per-seat arrays. A few KB.

    Returns None when no map is available (caller maps to 404).
    """
    eid, m = _resolve(show_id)
    if m is None:
        return None
    sections = [
        {"id": s["id"], "name": s["name"], "ga": s["ga"],
         "bbox": s["bbox"], "c": s["c"], "n": s["n"]}
        for s in m.get("sections", [])
    ]
    return {
        "event_id": eid,
        "venue_config_id": m.get("venueConfigId"),
        "capacity": m.get("capacity"),
        "canvas": m.get("canvas"),
        "section_count": len(sections),
        "sections": sections,
    }


def section(show_id: str, section_id: str):
    """One section's rows + seats (lazy-loaded on zoom). None if not found."""
    _eid, m = _resolve(show_id)
    if m is None:
        return None
    for s in m.get("sections", []):
        if s["id"] == section_id:
            return {"id": s["id"], "name": s["name"], "ga": s["ga"],
                    "bbox": s["bbox"], "n": s["n"], "rows": s.get("rows", [])}
    return None


def full(show_id: str):
    """The complete normalized map (sections + every seat). None if unavailable."""
    _eid, m = _resolve(show_id)
    return m
