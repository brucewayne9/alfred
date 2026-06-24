# Fans First — Seat-Map Page Implementation Plan (Plan 2 of 7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Reskin the show page (`show.html`) to the Fans First design system and add the full-arena **green (available) / red (sold) / grey (not ours)** seat-map overlay on the existing TM venue geometry.

**Architecture:** The page already renders venue geometry as SVG (`renderSeatMap`→`drawVenue`→`zoomSection`) from `GET /fairgame/api/shows/{id}/seatmap`. We add a **decoupled holdings overlay** — a new backend module `core/fairgame/holdings.py` + endpoint `GET /fairgame/api/shows/{id}/seatmap/holdings` that derives, deterministically from the geometry, which sections Rod holds and whether each is available or sold. The frontend colors section rects and seat dots from that overlay. This leaves the existing inventory/pricing/buy/orders flow (3-tier `Floor/Lower/Upper`) completely untouched.

**Tech Stack:** Python (FastAPI, stdlib only) backend; pytest for the backend logic; static HTML/CSS/vanilla JS frontend; served at `https://aialfred.groundrushcloud.com/fairgame/app/`.

## Global Constraints (binding — every task)
- Design = DESIGN.md (`data/mainstay/fairgame/DESIGN.md`). One gold accent (`--gold #e6bd72` / `--gold-deep #9c6f23` / buttons `--gold-btn #d8a84c` text `#1a1206`). **Seat-map green/red/grey are the only other colors, only in the map.** Figtree only. Buttons squared `11px` (no pills). NO emoji — line SVG icons only. Dark seat-map panel inside the light page (matches the homepage seat-map preview).
- Seat colors: available `--green #13bd57`, sold `--red #e23b3b`, not-ours grey `#322f2a` (on the dark panel). Section rects use translucent fills of these; seat dots use solid.
- Copy: the not-ours legend says "Not available here — we only sell what Rod gave us, and you can't get these seats anywhere else." Never overstate.
- Backend: stdlib only, deterministic (no `random`/time-based) so the overlay is stable across requests. Do NOT modify `events.py` inventory, the buy/access/orders flow, or their tests.
- 14/25 shows are mapped; the other 11 must keep gracefully hiding the seat-map card (existing behavior — do not regress).

## File Structure
- Create: `core/fairgame/holdings.py` — the deterministic overlay (geometry → held/available/sold).
- Create: `tests/fairgame/test_holdings.py` — unit tests for the overlay logic.
- Modify: `core/api/fairgame.py` — add `GET /fairgame/api/shows/{id}/seatmap/holdings` (≈6 lines, next to the existing seatmap routes ~line 356-380).
- Modify: `data/mainstay/fairgame/app/show.html` — reskin to DESIGN.md (self-contained style block) + color the map from the holdings overlay + legend.
- Unchanged: `events.py`, `orders.py`, `access.py`, `seatmap.py`, the homepage, other pages.

---

### Task 1: Holdings overlay module + endpoint (TDD)

**Files:**
- Create: `core/fairgame/holdings.py`
- Create: `tests/fairgame/test_holdings.py`
- Modify: `core/api/fairgame.py` (add one route)

**Interfaces:**
- Consumes: `seatmap.overview(show_id)` → `{sections:[{id,name,ga,bbox,c,n}], ...}` or `None`.
- Produces:
  - `holdings.classify_tier(name: str, ga: bool) -> str` → `"floor" | "lower" | "upper"`.
  - `holdings.section_status(name: str, ga: bool) -> str` → `"available" | "sold" | "not_ours"`.
  - `holdings.overlay(show_id: str) -> dict | None` → `None` if no map, else
    `{"sections": {<name:str>: <status:str>}, "held": int, "available": int, "sold": int}`.

- [ ] **Step 1: Write the failing tests** — `tests/fairgame/test_holdings.py`:

```python
from core.fairgame import holdings

def test_classify_tier_floor_lower_upper():
    assert holdings.classify_tier("C4W", False) == "floor"   # court/letter code
    assert holdings.classify_tier("FLOOR", True) == "floor"   # GA flag
    assert holdings.classify_tier("105", False) == "lower"    # 100s
    assert holdings.classify_tier("224", False) == "upper"    # 200s+

def test_section_status_upper_is_not_ours():
    # upper bowl is never held -> grey
    assert holdings.section_status("224", False) == "not_ours"

def test_section_status_held_is_available_or_sold_deterministic():
    # floor + lower are held; status is available or sold, and STABLE across calls
    s1 = holdings.section_status("105", False)
    s2 = holdings.section_status("105", False)
    assert s1 == s2
    assert s1 in ("available", "sold")
    assert holdings.section_status("C1", False) in ("available", "sold")

def test_overlay_none_when_unmapped(monkeypatch):
    monkeypatch.setattr(holdings.seatmap, "overview", lambda sid: None)
    assert holdings.overlay("show_999") is None

def test_overlay_counts_and_keys(monkeypatch):
    fake = {"sections": [
        {"id":"a","name":"101","ga":False,"bbox":[0,0,1,1],"c":[0,0],"n":10},
        {"id":"b","name":"224","ga":False,"bbox":[0,0,1,1],"c":[0,0],"n":10},
        {"id":"c","name":"C1","ga":False,"bbox":[0,0,1,1],"c":[0,0],"n":10},
    ]}
    monkeypatch.setattr(holdings.seatmap, "overview", lambda sid: fake)
    ov = holdings.overlay("show_1")
    assert set(ov["sections"].keys()) == {"101","224","C1"}
    assert ov["sections"]["224"] == "not_ours"          # upper = grey
    assert ov["sections"]["101"] in ("available","sold")
    assert ov["held"] == ov["available"] + ov["sold"]   # held = available + sold
    assert ov["held"] == 2                               # 101 + C1 held; 224 not
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_holdings.py -v`
Expected: FAIL (module `holdings` does not exist).

- [ ] **Step 3: Implement `core/fairgame/holdings.py`**

```python
"""Fans First seat-map holdings overlay.

Derives, deterministically from the OPEN venue geometry, which sections Rod
holds and whether each is available or sold -- so the seat map can paint the
arena green (ours, available) / red (ours, sold) / grey (not ours). This is a
DEMO/visual layer decoupled from the pricing inventory in events.py; when real
held-seat data lands it replaces section_status(). No randomness or clock use,
so the same section always paints the same colour across requests.
"""
from __future__ import annotations

from . import seatmap

# Rod holds the premium block: floor/court + the lower bowl. Upper bowl is grey.
_HELD_TIERS = ("floor", "lower")


def classify_tier(name: str, ga: bool) -> str:
    """Map a geometry section name to a tier: floor | lower | upper."""
    if ga:
        return "floor"
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:            # court/letter codes like C1, C4W -> floor
        return "floor"
    num = int(digits)
    if num < 200:
        return "lower"
    return "upper"


def _stable_hash(name: str) -> int:
    """Deterministic small int from a section name (no randomness)."""
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h


def section_status(name: str, ga: bool) -> str:
    """available | sold | not_ours for one section. Deterministic + stable."""
    if classify_tier(name, ga) not in _HELD_TIERS:
        return "not_ours"
    # ~1 in 6 held sections shown as sold, deterministically.
    return "sold" if _stable_hash(name) % 6 == 0 else "available"


def overlay(show_id: str) -> dict | None:
    """Per-section status map for a show, or None if no geometry is ingested."""
    ov = seatmap.overview(show_id)
    if ov is None:
        return None
    sections = {}
    available = sold = 0
    for s in ov.get("sections", []):
        st = section_status(s["name"], s.get("ga", False))
        sections[s["name"]] = st
        if st == "available":
            available += 1
        elif st == "sold":
            sold += 1
    return {"sections": sections, "held": available + sold,
            "available": available, "sold": sold}
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_holdings.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Add the endpoint** in `core/api/fairgame.py`, immediately after the existing `show_seatmap_section` handler (~line 380). Match the existing handlers' style:

```python
@app.get("/fairgame/api/shows/{show_id}/seatmap/holdings")
async def show_seatmap_holdings(show_id: str):
    """Green/red/grey overlay: which sections Rod holds + their status.
    404 when no map is ingested for this show."""
    if not events.get_show(show_id):
        raise HTTPException(status_code=404, detail="show not found")
    data = holdings.overlay(show_id)
    if data is None:
        raise HTTPException(status_code=404,
                            detail=seatmap.status(show_id).get("reason", "no seatmap"))
    return data
```

Also add `holdings` to the existing `from core.fairgame import ...` import block at the top of the file (it already imports `events`, `seatmap`, etc. — add `holdings` to that list).

- [ ] **Step 6: Verify endpoint live** (mapped show = `show_1` Philadelphia; unmapped check returns 404)

Run:
```bash
cd /home/aialfred/alfred
curl -s https://aialfred.groundrushcloud.com/fairgame/api/shows/show_1/seatmap/holdings | python -m json.tool | head -20
```
Expected: JSON with a `sections` object (section-name → status), and `held`/`available`/`sold` ints. (If the live server needs a restart to pick up the new route, note it; the route is correct in source.)

- [ ] **Step 7: Run the full fairgame test suite — nothing regressed**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/ -q`
Expected: all pass (existing tests + 5 new).

- [ ] **Step 8: Commit**

```bash
git add core/fairgame/holdings.py tests/fairgame/test_holdings.py core/api/fairgame.py
git commit -m "feat(fairgame): seat-map holdings overlay (green/red/grey) + endpoint"
```

---

### Task 2: Reskin `show.html` to the design system

**Files:**
- Modify: `data/mainstay/fairgame/app/show.html` (replace the `<head>` style approach + nav/header/inventory markup with the DESIGN.md system; KEEP all existing JS function names and the seat-map SVG structure)

**Interfaces:**
- Consumes: design tokens/components from the homepage (`index.html`) — reuse the same self-contained `<style>` token block, Figtree link, `nav`, `.btn`, `.btn-gold`, `.tag`, card styles.
- Produces: unchanged JS API (`renderSeatMap`, `drawVenue`, `zoomSection`, `loadShow`, etc.) — Task 3 edits their internals only.

- [ ] **Step 1: Port the design shell** — give `show.html` the same self-contained look as `index.html`: copy the `:root` token block, the Figtree `<link>`, the `body`/`nav`/`.btn`/`.btn-gold`/`.btn-line`/`.tag` rules from `index.html`'s `<style>`. Keep the `<link rel="stylesheet" href="fairgame.css">` ONLY if needed for layout classes still in use; prefer dropping it and porting the few card/section rules inline so the page is self-contained like the homepage. Light `--paper` body, dark seat-map panel.

- [ ] **Step 2: Restyle the page chrome** — announce bar + co-brand nav identical to the homepage; the show header (city, venue, date, "from $X") as a clean light section with gold accents; the inventory/buy list (`.inv-row`) as light cards with squared buttons. Replace any `fg-chip out`/"Gone" emoji-free; keep copy.

- [ ] **Step 3: Verify reskin serves and JS intact**

Run:
```bash
cd /home/aialfred/alfred
curl -s -o /dev/null -w "%{http_code}\n" "https://aialfred.groundrushcloud.com/fairgame/app/show.html?id=show_1"
grep -c 'function drawVenue\|function zoomSection\|function renderSeatMap' data/mainstay/fairgame/app/show.html
grep -P '[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}]' data/mainstay/fairgame/app/show.html
```
Expected: `200`; `3` (all three functions still present); emoji grep prints nothing. Open `show.html?id=show_1` — chrome matches the homepage; the venue map still draws (uncolored for now).

- [ ] **Step 4: Commit**

```bash
git add data/mainstay/fairgame/app/show.html
git commit -m "feat(fairgame): reskin show page to Fans First design system"
```

---

### Task 3: Wire the green/red/grey overlay into the map

**Files:**
- Modify: `data/mainstay/fairgame/app/show.html` (the seat-map JS: `renderSeatMap`/`drawVenue`/`zoomSection`, + a legend in the card header + CSS)

**Interfaces:**
- Consumes: `GET /fairgame/api/shows/{id}/seatmap/holdings` → `{sections:{name:status}, held, available, sold}` (Task 1).

- [ ] **Step 1: Add seat/section status CSS** to the page `<style>` (dark seat panel):

```css
.sec-shape.avail{fill:rgba(19,189,87,.22);stroke:rgba(19,189,87,.7)}
.sec-shape.sold{fill:rgba(226,59,59,.18);stroke:rgba(226,59,59,.55)}
.sec-shape.mine-na{fill:rgba(120,132,150,.10);stroke:rgba(120,132,150,.30)}
.seat-dot.avail{fill:#13bd57}.seat-dot.sold{fill:#e23b3b}.seat-dot.mine-na{fill:#3a3a3a}
.sm-legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:10px;font-size:12px;color:var(--ink-2)}
.sm-legend span{display:flex;align-items:center;gap:7px}
.sm-legend i{width:13px;height:13px;border-radius:4px;display:inline-block}
```

- [ ] **Step 2: Fetch the overlay in `renderSeatMap()`** — after the geometry overview loads and before/with `drawVenue()`, fetch holdings and stash a lookup:

```javascript
let HOLD = {};   // section name -> 'available' | 'sold' | 'not_ours'
async function loadHoldings(){
  try{
    const r = await fetch(API + '/shows/' + encodeURIComponent(SHOW_ID) + '/seatmap/holdings');
    HOLD = r.ok ? ((await r.json()).sections || {}) : {};
  }catch(e){ HOLD = {}; }
}
function statusClass(name){
  const s = HOLD[name];
  return s === 'available' ? 'avail' : s === 'sold' ? 'sold' : 'mine-na';
}
```
Call `await loadHoldings();` inside `renderSeatMap()` right before it calls `drawVenue()`.

- [ ] **Step 2b: Color section rects in `drawVenue()`** — where each section `<rect>` is created, set its class from status instead of the current hardcoded GA/non-GA fill:

```javascript
// when building the section rect (replace the hardcoded fill attr):
rect.setAttribute('class', 'sec-shape ' + statusClass(s.name));
```
Remove the old inline `fill` attribute on the rect so the CSS class governs color.

- [ ] **Step 2c: Color seat dots in `zoomSection(s)`** — the section's status applies to all its seats:

```javascript
const cls = 'seat-dot ' + statusClass(s.name);
// when creating each seat circle: circle.setAttribute('class', cls);
```

- [ ] **Step 3: Add the legend + copy** under the seat-map card header (replace the `#seatmapHint` placeholder text "Live availability lights up here once it's connected."):

```html
<div class="sm-legend">
  <span><i style="background:#13bd57"></i> Available — Rod-held</span>
  <span><i style="background:#e23b3b"></i> Sold</span>
  <span><i style="background:#3a3a3a;border:1px solid #4a463f"></i> Not available here</span>
</div>
<p class="fg-muted" style="margin-top:8px;font-size:13px">We only sell what Rod gave us — you can't get the greyed seats anywhere else.</p>
```

- [ ] **Step 4: Verify the overlay renders**

Run:
```bash
cd /home/aialfred/alfred
curl -s "https://aialfred.groundrushcloud.com/fairgame/api/shows/show_1/seatmap/holdings" | python -c "import sys,json;d=json.load(sys.stdin);print('available',d['available'],'sold',d['sold'],'total',len(d['sections']))"
grep -c 'sec-shape ' data/mainstay/fairgame/app/show.html
```
Expected: counts print (available>0, sold>0); grep ≥1. Open `show.html?id=show_1`: lower/floor sections render green with some red; upper bowl grey. Tap a green section → seats are green; tap a grey section → grey seats. Legend visible.

- [ ] **Step 5: Commit**

```bash
git add data/mainstay/fairgame/app/show.html
git commit -m "feat(fairgame): paint seat map green/red/grey from holdings overlay + legend"
```

---

### Task 4: Responsive, no-emoji, graceful-empty, parity

**Files:**
- Modify: `data/mainstay/fairgame/app/show.html` (mobile CSS only if gaps)

- [ ] **Step 1: Emoji gate**

Run: `cd /home/aialfred/alfred && grep -P '[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}]' data/mainstay/fairgame/app/show.html`
Expected: no output.

- [ ] **Step 2: Unmapped-show graceful state** — confirm a non-mapped show still hides the map cleanly (no JS error from the holdings 404):

Run:
```bash
cd /home/aialfred/alfred
curl -s -o /dev/null -w "%{http_code}\n" "https://aialfred.groundrushcloud.com/fairgame/api/shows/show_5/seatmap/holdings"
```
Expected: `404` (and the page must not throw — `loadHoldings` swallows it; the card stays hidden via the existing `SHOW.seatmap.available` check). Open an unmapped show (e.g. an AXS/SeatGeek one) and confirm no console error and the map card is hidden.

- [ ] **Step 3: Responsive check** — load `show.html?id=show_1` at 1280px, 768px, 380px: seat-map SVG scales (max-height cap holds), legend wraps, buy list stacks, no horizontal overflow, touch targets ≥44px.

- [ ] **Step 4: Parity** — chrome (announce/nav/buttons/fonts/gold) matches the homepage; dark seat panel matches the homepage's seat-map preview.

- [ ] **Step 5: Commit**

```bash
git add data/mainstay/fairgame/app/show.html
git commit -m "feat(fairgame): show-page responsive polish + no-emoji + graceful empty (Plan 2 done)"
```

---

## Self-Review
- **Spec coverage:** full-arena green/red/grey overlay ✓ (T1 backend, T3 frontend), only-our-seats colored + rest grey ✓ (`not_ours`), legend + "can't get these anywhere else" copy ✓ (T3), design-system reskin ✓ (T2), graceful empty for the 11 unmapped shows ✓ (T4), no buy/inventory/orders regression ✓ (decoupled overlay, T1 suite gate). Out of plan (later): real held-seat data replacing the deterministic overlay; per-seat (not per-section) sold status; click-to-buy from the map.
- **Placeholders:** none — backend code, tests, endpoint, CSS, and JS deltas are all complete. Frontend reskin (T2) references the homepage's already-written primitives (DRY) rather than restating them.
- **Type consistency:** `overlay()` returns `{sections,held,available,sold}`; the endpoint returns it verbatim; the frontend reads `.sections[name]` → `statusClass()` → `avail|sold|mine-na`, matching the CSS classes added in T3. `classify_tier`/`section_status`/`overlay` signatures match between module, tests, and endpoint.

## Next plans
3. Checkout (TM email + triple no-refund) · 4. Discover $1 paywall · 5. Accounts / My Tickets · 6. Admin CMS · 7. Delivery assist tool.
