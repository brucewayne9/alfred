"""RuckTalk Content Board — brain-dumps -> shoot-ready Reel cards + live IG stats.

Single JSON-blob state, last-write-wins. Mirrors core/api/rollout_board.py.
Gated at the edge by Caddy basic_auth on /rt-board/*. The Telegram bot POSTs
rough cards to /rt-board/api/card directly on localhost:8400 (bypasses Caddy).

Endpoints:
  GET  /rt-board/api/state          -> current board (seeds empty on first run)
  PUT  /rt-board/api/state          -> overwrite board (edits / column moves)
  POST /rt-board/api/card           -> append a rough card  {raw: str}
  POST /rt-board/api/refresh-stats  -> pull IG stats for posted cards w/ links
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, Request

logger = logging.getLogger(__name__)

STATE_PATH = Path(__file__).parent.parent.parent / "data" / "rucktalk" / "content_board" / "board_state.json"

RUCKTALK_IG_ID = "17841461784057534"
GRAPH = "https://graph.facebook.com/v21.0"

DEFAULT_STATE = {
    "columns": {"to_shoot": [], "shot": [], "posted": []},
    "updated": 0,
}


def new_card(raw: str) -> dict:
    raw = (raw or "").strip()
    words = raw.split()
    title = " ".join(words[:7]) if words else "Untitled reel"
    return {
        "id": uuid.uuid4().hex[:8],
        "created_at": int(time.time()),
        "title": title,
        "raw": raw,
        "hook": "",
        "shot": "",
        "script": "",
        "caption": "",
        "hashtags": "",
        "cta": "",
        "status": "to_shoot",
        "polished": False,
        "reel_url": "",
        "stats": None,
    }


_SHORTCODE_RE = re.compile(r"instagram\.com/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)")


def shortcode_from_url(url: str):
    if not url:
        return None
    m = _SHORTCODE_RE.search(url)
    return m.group(1) if m else None


def _load() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    _save(DEFAULT_STATE)
    return json.loads(json.dumps(DEFAULT_STATE))


def _save(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_PATH)


# ---------------------------------------------------------------------------
# Live Instagram stats (Meta Graph API)
# ---------------------------------------------------------------------------

def _token() -> str:
    from config.settings import settings
    return (getattr(settings, "meta_access_token", "")
            or getattr(settings, "META_ACCESS_TOKEN", "") or "")


def _media_index(token: str) -> dict:
    """Map shortcode -> media_id for the account's recent media (paged)."""
    index, url = {}, f"{GRAPH}/{RUCKTALK_IG_ID}/media"
    params = {"fields": "id,permalink", "limit": 50, "access_token": token}
    for _ in range(6):  # up to ~300 recent media
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        for m in data.get("data", []):
            sc = shortcode_from_url(m.get("permalink", ""))
            if sc:
                index[sc] = m["id"]
        nxt = data.get("paging", {}).get("next")
        if not nxt:
            break
        url, params = nxt, None
    return index


def _insights(media_id: str, token: str) -> dict:
    r = requests.get(
        f"{GRAPH}/{media_id}/insights",
        params={"metric": "reach,saved,shares", "access_token": token},
        timeout=20,
    )
    r.raise_for_status()
    out = {}
    for row in r.json().get("data", []):
        vals = row.get("values", [])
        out[row["name"]] = vals[0].get("value", 0) if vals else 0
    return {
        "reach": out.get("reach", 0),
        "saves": out.get("saved", 0),
        "shares": out.get("shares", 0),
        "fetched_at": int(time.time()),
    }


def pull_all_stats() -> dict:
    token = _token()
    if not token:
        return {"ok": False, "error": "no token"}
    state = _load()
    posted = [c for c in state["columns"].get("posted", []) if c.get("reel_url")]
    if not posted:
        return {"ok": True, "updated": 0, "note": "no posted cards with links"}
    try:
        index = _media_index(token)
    except Exception as e:
        logger.error("media index failed: %s", e)
        return {"ok": False, "error": "graph error"}
    updated = 0
    for c in posted:
        sc = shortcode_from_url(c["reel_url"])
        mid = index.get(sc) if sc else None
        if not mid:
            continue
        try:
            c["stats"] = _insights(mid, token)
            updated += 1
        except Exception as e:
            logger.error("insights failed for %s: %s", sc, e)
    if updated:
        state["updated"] = int(time.time())
        _save(state)
    return {"ok": True, "updated": updated}


def register(app: FastAPI) -> None:
    @app.get("/rt-board/api/state")
    def get_state():
        return _load()

    @app.put("/rt-board/api/state")
    async def put_state(request: Request):
        body = await request.json()
        if not isinstance(body, dict) or "columns" not in body:
            return {"ok": False, "error": "bad state"}
        body["updated"] = int(time.time())
        _save(body)
        return {"ok": True, "updated": body["updated"]}

    @app.post("/rt-board/api/card")
    async def add_card(request: Request):
        body = await request.json()
        raw = body.get("raw", "") if isinstance(body, dict) else ""
        if not raw.strip():
            return {"ok": False, "error": "empty"}
        state = _load()
        card = new_card(raw)
        state["columns"]["to_shoot"].insert(0, card)
        state["updated"] = int(time.time())
        _save(state)
        return {"ok": True, "id": card["id"]}

    @app.post("/rt-board/api/refresh-stats")
    def refresh_stats():
        return pull_all_stats()
