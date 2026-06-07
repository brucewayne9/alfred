# RuckTalk Content Board Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A phone-friendly Kanban board where RuckTalk social brain-dumps become shoot-ready Reel cards and posted reels show live Instagram stats.

**Architecture:** Mirror the existing Rollout-board pattern — a FastAPI router registered on the main app (`core/api/main.py`, localhost:8400) backed by a single JSON state file, plus a static HTML page served by Caddy under `/rt-board/` with basic_auth. The Telegram bot routes `social` brain-dumps to the board API as rough cards. A stats endpoint pulls reach/saves/shares from the Meta Graph API for posted cards that have a reel link.

**Tech Stack:** Python/FastAPI, vanilla HTML/JS (no build step), Caddy, Meta Graph API, python-telegram-bot.

---

## File Structure

- **Create** `core/api/rucktalk_content_board.py` — board state (load/save), card model, FastAPI routes (`/rt-board/api/*`), stats refresh via Meta API.
- **Create** `data/rucktalk/content_board/index.html` — the Kanban page (3 columns, inline edit, mobile-first, posted-card stats + link prompt).
- **Modify** `core/api/main.py` (~line 161-164 area) — register the new router like `rollout_board`.
- **Modify** `interfaces/telegram/bot.py` — make brain-dump capture lane-aware (`radio`/`social`/none) and POST social dumps to the board API.
- **Modify** `/etc/caddy/Caddyfile` — add `/rt-board/api/*` and `/rt-board/*` blocks (basic_auth → 8400 / static root).
- **Create** `tests/test_rucktalk_content_board.py` — lane extraction, card creation, reel-URL→shortcode parsing.

---

## Task 1: Board backend — state + card model + core routes

**Files:**
- Create: `core/api/rucktalk_content_board.py`
- Test: `tests/test_rucktalk_content_board.py`

- [ ] **Step 1: Write failing tests for card creation + shortcode parsing**

```python
# tests/test_rucktalk_content_board.py
import importlib
cb = importlib.import_module("core.api.rucktalk_content_board")

def test_new_card_defaults():
    c = cb.new_card("the walk is cheaper than therapy, reset before the day")
    assert c["status"] == "to_shoot"
    assert c["polished"] is False
    assert c["raw"].startswith("the walk")
    assert c["title"]  # non-empty short label
    assert c["reel_url"] == ""
    assert c["stats"] is None
    assert "id" in c

def test_shortcode_from_reel_url():
    assert cb.shortcode_from_url("https://www.instagram.com/reel/Cabc123XYZ/") == "Cabc123XYZ"
    assert cb.shortcode_from_url("https://instagram.com/p/DEF456/?igsh=x") == "DEF456"
    assert cb.shortcode_from_url("not a url") is None
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd /home/aialfred/alfred && venv/bin/python -m pytest tests/test_rucktalk_content_board.py -v`
Expected: FAIL (module not found / functions undefined)

- [ ] **Step 3: Implement the backend module**

```python
# core/api/rucktalk_content_board.py
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
import re
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request

STATE_PATH = Path(__file__).parent.parent.parent / "data" / "rucktalk" / "content_board" / "board_state.json"

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
        from core.api.rucktalk_content_board import pull_all_stats
        return pull_all_stats()
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd /home/aialfred/alfred && venv/bin/python -m pytest tests/test_rucktalk_content_board.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add core/api/rucktalk_content_board.py tests/test_rucktalk_content_board.py
git commit -m "feat(rucktalk): content board backend — state, card model, routes"
```

---

## Task 2: Live IG stats puller (Meta Graph API)

**Files:**
- Modify: `core/api/rucktalk_content_board.py` (add `pull_all_stats` + helpers)

- [ ] **Step 1: Implement stats functions**

Add to `core/api/rucktalk_content_board.py`. Uses `config.settings.settings` for the
token (never `os.environ` directly — house rule). RuckTalk IG id is fixed.

```python
import logging
import requests  # already a dep across integrations

logger = logging.getLogger(__name__)

RUCKTALK_IG_ID = "17841461784057534"
GRAPH = "https://graph.facebook.com/v21.0"


def _token() -> str:
    from config.settings import settings
    return getattr(settings, "META_ACCESS_TOKEN", "") or ""


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
```

- [ ] **Step 2: Smoke-test the puller (no posted cards yet → graceful)**

Run:
```bash
cd /home/aialfred/alfred && venv/bin/python -c "from core.api.rucktalk_content_board import pull_all_stats; print(pull_all_stats())"
```
Expected: `{'ok': True, 'updated': 0, 'note': 'no posted cards with links'}`

- [ ] **Step 3: Commit**

```bash
cd /home/aialfred/alfred
git add core/api/rucktalk_content_board.py
git commit -m "feat(rucktalk): live IG stats puller (reach/saves/shares via Graph API)"
```

---

## Task 3: Register the router on the main app

**Files:**
- Modify: `core/api/main.py` (next to the rollout_board registration, ~line 160-164)

- [ ] **Step 1: Add registration block**

Find the existing rollout_board registration:
```python
    from core.api.rollout_board import register as _register_rollout
    _register_rollout(app)
```
Immediately after its `try/except`, add:
```python
try:
    from core.api.rucktalk_content_board import register as _register_rtboard
    _register_rtboard(app)
    logger.info("rucktalk content board registered")
except Exception as _e:
    logger.exception("rucktalk_content_board register failed: %s", _e)
```

- [ ] **Step 2: Verify the app imports cleanly**

Run:
```bash
cd /home/aialfred/alfred && venv/bin/python -c "import core.api.main as m; print('routes:', [r.path for r in m.app.routes if 'rt-board' in getattr(r,'path','')])"
```
Expected: lists `/rt-board/api/state`, `/rt-board/api/card`, `/rt-board/api/refresh-stats`

- [ ] **Step 3: Commit**

```bash
cd /home/aialfred/alfred
git add core/api/main.py
git commit -m "feat(rucktalk): register content board router on main app"
```

---

## Task 4: Lane-aware brain-dump routing in the bot

**Files:**
- Modify: `interfaces/telegram/bot.py` (`_extract_brain_dump` → return lane; `handle_message` → route social to board)
- Test: `tests/test_rucktalk_content_board.py` (add lane tests, importing the bot helper)

- [ ] **Step 1: Write failing tests for lane extraction**

```python
def test_lane_extraction():
    import importlib
    bot = importlib.import_module("interfaces.telegram.bot")
    f = bot._extract_brain_dump_lane
    assert f("brain dump radio talk about emotion") == ("radio", "talk about emotion")
    assert f("brain dump social the walk is cheaper") == ("social", "the walk is cheaper")
    assert f("brain dump instagram studio reset") == ("social", "studio reset")
    assert f("brain dump just an idea") == ("none", "just an idea")
    assert f("/dump quick note") == ("none", "quick note")
    assert f("what is on my calendar") is None
```

- [ ] **Step 2: Run, verify fail**

Run: `cd /home/aialfred/alfred && venv/bin/python -m pytest tests/test_rucktalk_content_board.py::test_lane_extraction -v`
Expected: FAIL (`_extract_brain_dump_lane` undefined)

- [ ] **Step 3: Add the lane helper in `interfaces/telegram/bot.py`**

After the existing `_extract_brain_dump`, add:
```python
# Lanes recognized right after the dump trigger. Order matters (longest first).
_DUMP_LANES = (("radio", "radio"), ("social", "social"),
               ("instagram", "social"), ("ig", "social"))


def _extract_brain_dump_lane(message_text: str):
    """Return (lane, body) for a brain-dump message, else None.
    lane in {'radio','social','none'}. A leading lane word is stripped from body."""
    body = _extract_brain_dump(message_text)
    if body is None:
        return None
    low = body.lower()
    for word, lane in _DUMP_LANES:
        if low.startswith(word):
            rest = body[len(word):].lstrip(" :;-—\n")
            return (lane, rest.strip())
    return ("none", body)
```

- [ ] **Step 4: Run, verify pass**

Run: `cd /home/aialfred/alfred && venv/bin/python -m pytest tests/test_rucktalk_content_board.py::test_lane_extraction -v`
Expected: PASS

- [ ] **Step 5: Wire routing in `handle_message`**

Replace the current brain-dump block (the `dump_body = _extract_brain_dump(...)` block
added earlier) with a lane-aware version:
```python
        # RuckTalk brain-dump capture (lane-aware): radio -> show journal,
        # social/instagram -> content board card, untagged -> journal (Alfred triages).
        dump = _extract_brain_dump_lane(message_text)
        if dump is not None:
            lane, body = dump
            if not body:
                await update.message.reply_text(
                    "Ready when you are, sir — 'brain dump social <idea>' for a reel, "
                    "'brain dump radio <idea>' for the show."
                )
                return
            try:
                source = {"radio": "radio", "social": "social"}.get(lane, "text")
                path = await asyncio.to_thread(_save_brain_dump, body, source)
                wc = len(body.split())
                if lane == "social":
                    import requests as _rq
                    try:
                        _rq.post("http://localhost:8400/rt-board/api/card",
                                 json={"raw": body}, timeout=10)
                    except Exception as _e:
                        logger.error(f"board card post failed: {_e}")
                    await update.message.reply_text(
                        f"🎬 Card's on the board ({wc} words) — I'll polish it. "
                        "aialfred.groundrushcloud.com/rt-board/"
                    )
                else:
                    await update.message.reply_text(
                        f"🎙️ Brain-dump captured for today's RuckTalk show ({wc} words). "
                        "Keep them coming, sir."
                    )
                logger.info(f"Brain-dump [{lane}] captured ({wc} words) -> {path}")
            except Exception as e:
                logger.error(f"Brain-dump save failed: {e}")
                await update.message.reply_text(
                    "Sorry, I hit an error saving that brain-dump. Give it another shot?"
                )
            return
```

- [ ] **Step 6: Syntax check + restart the bot**

Run:
```bash
cd /home/aialfred/alfred && venv/bin/python -m py_compile interfaces/telegram/bot.py && echo OK
sudo systemctl restart telegram-bot.service && sleep 3 && systemctl is-active telegram-bot.service
```
Expected: `OK` then `active`

- [ ] **Step 7: Commit**

```bash
cd /home/aialfred/alfred
git add interfaces/telegram/bot.py tests/test_rucktalk_content_board.py
git commit -m "feat(rucktalk): lane-aware brain-dumps (radio/social) -> journal + board"
```

---

## Task 5: The Kanban page

**Files:**
- Create: `data/rucktalk/content_board/index.html`

- [ ] **Step 1: Write the page**

Mobile-first, 3 columns, tap-to-advance, tap-to-open editor, posted-card stats +
link prompt + "link needed" nag. Talks to `api/state` (GET/PUT) relative to
`/rt-board/`. (Full file written during execution — vanilla HTML/CSS/JS, dark
theme to match RuckTalk, no build step. Pattern mirrors data/mainstay/board/index.html.)

Key behaviors the page MUST implement:
- Load state via `fetch('api/state')`; render 3 columns from `state.columns`.
- Each card shows title + ✨ if `polished`. Tap → modal with all blueprint fields editable; Save → PUT whole state.
- Move buttons (◀ ▶) advance/retreat `status` between to_shoot/shot/posted; on entering `posted`, if `reel_url` empty, focus an inline "Paste reel link" input.
- Posted cards: if `stats`, show `reach · saves · shares` (🔥 when shares≥5); else if no `reel_url`, show yellow "⚠️ link needed".
- A "Refresh stats" button → `fetch('api/refresh-stats',{method:'POST'})` then reload.
- Debounced autosave after edits (PUT api/state).

- [ ] **Step 2: Commit**

```bash
cd /home/aialfred/alfred
git add data/rucktalk/content_board/index.html
git commit -m "feat(rucktalk): content board Kanban page (mobile, stats, link prompt)"
```

---

## Task 6: Caddy routing for /rt-board/

**Files:**
- Modify: `/etc/caddy/Caddyfile` (add two handle blocks inside the `aialfred.groundrushcloud.com {` site, near the `/board/` blocks)

- [ ] **Step 1: Back up the Caddyfile**

```bash
sudo cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak.rtboard.$(date +%s 2>/dev/null || echo manual)
```
(Note: `date +%s` may be unavailable in this harness; a literal suffix is fine.)

- [ ] **Step 2: Add the blocks after the `/board/*` handler**

```
    redir /rt-board /rt-board/ permanent
    handle /rt-board/api/* {
        basic_auth {
            mainstay $2a$14$eQYVtJJq5EZqmB25iyycA.tqe8yftiFOPmnFaQGuWxMLSuIecCmjm
        }
        reverse_proxy localhost:8400
    }
    handle /rt-board/* {
        basic_auth {
            mainstay $2a$14$eQYVtJJq5EZqmB25iyycA.tqe8yftiFOPmnFaQGuWxMLSuIecCmjm
        }
        uri strip_prefix /rt-board
        root * /home/aialfred/alfred/data/rucktalk/content_board
        file_server
    }
```
(Reuse the `mainstay` cred for v1 — same login Mike already uses for the Rollout
board. A dedicated cred can be minted later with `caddy hash-password`.)

- [ ] **Step 3: Validate + reload Caddy**

```bash
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile && sudo systemctl reload caddy && echo reloaded
```
Expected: `Valid configuration` then `reloaded`. If validate fails, restore the backup.

- [ ] **Step 4: End-to-end smoke test**

```bash
# API reachable through Caddy (expect 200 + JSON with columns)
curl -s -u mainstay:'<pw>' https://aialfred.groundrushcloud.com/rt-board/api/state | head -c 200
# Page loads (expect 200)
curl -s -o /dev/null -w "%{http_code}\n" -u mainstay:'<pw>' https://aialfred.groundrushcloud.com/rt-board/
# Card POST direct to 8400 (what the bot does) then confirm it lands
curl -s -X POST localhost:8400/rt-board/api/card -H 'Content-Type: application/json' -d '{"raw":"smoke test card"}'
curl -s localhost:8400/rt-board/api/state | python3 -c "import sys,json; print(len(json.load(sys.stdin)['columns']['to_shoot']),'card(s)')"
```
Expected: state JSON, `200`, `{"ok":true,...}`, `1 card(s)`. Then delete the smoke card via the page or a PUT.

---

## Task 7: Verify & hand off

- [ ] **Step 1: Send a real `brain dump social ...` from Telegram**, confirm the "🎬 Card's on the board" reply and that the card appears at `/rt-board/`.
- [ ] **Step 2: Polish that first card** in a working session (fill hook/shot/script/caption/hashtags/cta, set polished=true via the page).
- [ ] **Step 3: Update memory** `project_rucktalk_instagram_strategy.md` — board is LIVE, URL, the dump lanes, the link-paste step.
- [ ] **Step 4: Email Mike** the live URL + the three dump phrasings + the one manual step.

---

## Notes for the implementer

- **House rules:** pull secrets from `config.settings.settings`, never `os.environ` at import. Wrap any `docker exec` in `timeout` (n/a here). 105 is Alfred's own box — Caddy edits are in-scope but back up + `validate` before reload (it serves prod sites).
- **No tests exist for the sibling boards** — keep test surface to the logic-bearing helpers (card creation, shortcode parse, lane extraction); smoke-test endpoints rather than mocking FastAPI.
- **Stats wrinkle:** reel URL → media id is resolved by matching the pasted permalink's shortcode against the account's recent media list. If a reel is older than ~300 posts back it won't resolve — acceptable for a growth board working with fresh posts.
