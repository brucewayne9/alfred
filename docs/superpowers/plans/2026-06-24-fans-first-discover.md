# Fans First — Discover + $1 Paywall Implementation Plan (Plan 4 of 7)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Reskin the Discover page to the Fans First design system, show each event's **image** on the result cards (real TM image, with a branded fallback), and gate the verified prices + buy-links behind a one-time **$1 unlock**.

**Architecture:** The aggregator (`core/fairgame/aggregator.py`) already returns normalized results including `image`, `min_price`, `currency`, `source`, and `buy_url`, via `GET /fairgame/api/discover`; outbound clicks go through `GET /fairgame/api/discover/out` (affiliate redirect). We add one backend endpoint `POST /fairgame/api/discover/unlock` (the $1 charge — sim-instant in dev, Stripe in live) and rebuild `discover.html`: self-contained design, image-led cards, and a blur→pay-$1→reveal gate.

**Tech Stack:** FastAPI + `stripe_connect` (sim by default); static HTML/CSS/JS; pytest for the endpoint.

## Global Constraints
- DESIGN.md: one gold accent, Figtree, squared 11px buttons, **NO emoji** (line SVG icons only — the existing 🛡️ on discover.html line ~94 must go). Image-led editorial cards on the light page.
- Honest copy: "the **lowest verified price** we can find" / "verified-ticket guarantee" — never "lowest anywhere."
- The $1 is a **one-time unlock per search session** (not per event). Once unlocked, all cards in the current results reveal. Persist the unlocked state client-side (`localStorage`).
- Stripe stays in **sim mode** by default (`stripe_connect.is_sim()` true when no key) — the unlock endpoint must work with no keys (instant sim success) and leave a clear real-Stripe branch.
- Do not break the existing `/discover` or `/discover/out` endpoints or their behavior.

## File Structure
- Modify: `core/api/fairgame.py` — add `POST /fairgame/api/discover/unlock`.
- Create: `tests/fairgame/test_discover_unlock.py`.
- Rewrite: `data/mainstay/fairgame/app/discover.html` — self-contained reskin, image cards, $1 gate.
- Optional create: `data/mainstay/fairgame/app/img/cat-*.jpg` — generated category fallback art (concert/sports/theater/comedy) — done out-of-band by the controller; the page references them as fallbacks if present.

---

### Task 1: `POST /discover/unlock` endpoint (TDD)

**Files:** Modify `core/api/fairgame.py`; Create `tests/fairgame/test_discover_unlock.py`.

**Interfaces:** `POST /fairgame/api/discover/unlock` body `{}` → in sim mode returns
`{"unlocked": true, "sim": true, "amount_cents": 100}`; in live mode returns
`{"unlocked": false, "checkout_url": "<stripe url>", "amount_cents": 100}`.

- [ ] **Step 1: Failing test** — `tests/fairgame/test_discover_unlock.py`:

```python
from fastapi.testclient import TestClient
from core.api.fairgame import app

def test_discover_unlock_sim_returns_unlocked(monkeypatch):
    monkeypatch.delenv("FAIRGAME_STRIPE_KEY", raising=False)
    monkeypatch.setenv("FAIRGAME_STRIPE_SIM", "1")
    c = TestClient(app)
    r = c.post("/fairgame/api/discover/unlock", json={})
    assert r.status_code == 200
    d = r.json()
    assert d["unlocked"] is True and d["sim"] is True and d["amount_cents"] == 100
```

- [ ] **Step 2: Run — fails** (`python -m pytest tests/fairgame/test_discover_unlock.py -v`).

- [ ] **Step 3: Implement** — add to `core/api/fairgame.py` near the other discover routes:

```python
@app.post("/fairgame/api/discover/unlock")
async def discover_unlock(req: Request):
    """The $1 Discover unlock. Sim mode (no Stripe key) grants instantly so the
    product is demoable; live mode returns a Stripe checkout URL to complete."""
    amount = aggregator.SERVICE_FEE_CENTS  # 100 = $1
    if stripe_connect.is_sim():
        return {"unlocked": True, "sim": True, "amount_cents": amount}
    # Live: create a $1 checkout. Buyer completes payment, returns unlocked.
    url = stripe_connect.create_unlock_checkout(amount)  # returns hosted checkout url
    return {"unlocked": False, "checkout_url": url, "amount_cents": amount}
```
If `stripe_connect` has no `create_unlock_checkout`, add a minimal one that builds a Checkout Session for a $1 line item (only reached in live mode); guard it so the sim path never calls Stripe. Ensure `aggregator` and `stripe_connect` are imported in the file (aggregator already is for `/discover`).

- [ ] **Step 4: Run — green.** Then full suite `python -m pytest tests/fairgame/ -q` — no regression.

- [ ] **Step 5: Commit** (`git add core/api/fairgame.py tests/fairgame/test_discover_unlock.py && git commit -m "feat(fairgame): $1 Discover unlock endpoint (sim-instant / Stripe live)"`).

---

### Task 2: Rebuild `discover.html` — reskin + image cards + $1 gate

**Files:** Rewrite `data/mainstay/fairgame/app/discover.html`.

- [ ] **Step 1: Self-contained reskin** — adopt the homepage design system (copy `index.html`'s `<style>` tokens, Figtree link, `body`, `nav`/`.scrolled`, announce bar, `.btn`/`.btn-gold`/`.tag`). Drop `fairgame.css`. **Remove the 🛡️ emoji** (replace with a line-SVG shield or nothing).

- [ ] **Step 2: Image-led result cards** — each event card leads with its image. In the card renderer, use `e.image`; when absent, fall back to a category image (`img/cat-<segment>.jpg` if present) else a branded gold-on-dark gradient placeholder showing the segment name. Card structure:

```javascript
function catFallback(seg){
  const s=(seg||'').toLowerCase();
  const map={music:'cat-concert',sports:'cat-sports',arts:'cat-theater','arts & theatre':'cat-theater',theatre:'cat-theater',comedy:'cat-comedy'};
  return map[s] ? ('img/'+map[s]+'.jpg') : '';
}
function eventCard(e, unlocked){
  const img = e.image || catFallback(e.segment);
  const header = img
    ? '<div class="ev-img" style="background-image:url('+JSON.stringify(img)+')"></div>'
    : '<div class="ev-img ev-img-ph"><span>'+esc(e.segment||'Live event')+'</span></div>';
  const price = money(e.min_price, e.currency);
  const priceHtml = unlocked
    ? '<div class="price">'+(price||'<small>TBA</small>')+'</div>'
    : '<div class="price blur">'+(price||'$00')+'</div>';
  const out = API + '/discover/out?src=' + encodeURIComponent(e.source||'partner') + '&url=' + encodeURIComponent(e.buy_url||'#');
  const cta = unlocked
    ? '<a class="btn btn-gold btn-sm" href="'+out+'" target="_blank" rel="noopener">Get verified ticket</a>'
    : '<button class="btn btn-line btn-sm" onclick="openUnlock()">Unlock for $1</button>';
  return '<article class="ev">'+header+
    '<div class="ev-body"><div class="ev-name">'+esc(e.name)+'</div>'+
    '<div class="ev-meta">'+esc(e.date||'')+' · '+esc(e.city||'')+(e.venue?' · '+esc(e.venue):'')+'</div>'+
    '<div class="ev-foot"><div><div class="price-l">Lowest verified</div>'+priceHtml+
    '<div class="src">via '+esc(e.source||'partner')+' · verified</div></div>'+cta+'</div></div></article>';
}
```
CSS (add): `.ev{background:var(--card);border:1px solid var(--line-l);border-radius:var(--r-card);overflow:hidden;box-shadow:var(--sh-l);display:flex;flex-direction:column}` · `.ev-img{aspect-ratio:16/9;background-size:cover;background-position:center;background-color:#15110c}` · `.ev-img-ph{display:grid;place-items:center;background:linear-gradient(135deg,#1a1206,#2a1f0c)}` · `.ev-img-ph span{font-family:var(--font-display);font-weight:800;color:var(--gold);letter-spacing:.04em;text-transform:uppercase}` · `.ev-body{padding:16px 18px;display:flex;flex-direction:column;gap:6px}` · `.ev-name{font-weight:800;font-size:1.05rem;letter-spacing:-.01em}` · `.ev-meta{font-size:13px;color:var(--ink-2)}` · `.ev-foot{display:flex;justify-content:space-between;align-items:flex-end;margin-top:10px;gap:12px}` · `.price{font-weight:800;font-size:1.5rem;color:var(--gold-deep)}` · `.price.blur{filter:blur(7px);user-select:none}` · `.price-l{font-size:.66rem;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)}` · `.src{font-size:12px;color:var(--ink-3)}` · results grid `.ev-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:18px}`.

- [ ] **Step 3: The $1 gate** — track unlocked state in `localStorage` (`ff_discover_unlocked`). On load, `const UNLOCKED = localStorage.getItem('ff_discover_unlocked')==='1';` render cards with it. Add:

```javascript
async function openUnlock(){
  try{
    const r = await fetch(API + '/discover/unlock', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    const d = await r.json();
    if(d.unlocked){ localStorage.setItem('ff_discover_unlocked','1'); rerender(); toast('Unlocked — every verified link is yours.'); }
    else if(d.checkout_url){ location.href = d.checkout_url; }
  }catch(e){ toast('Could not start the unlock. Try again.'); }
}
```
`rerender()` re-runs the card render with `UNLOCKED` re-read from `localStorage`. Add a clear value line near the results: "One dollar. One time. Unlocks every verified price and direct buy-link in your search." and an "Unlock all for $1" primary button that calls `openUnlock()`.

- [ ] **Step 4: Verify**

```bash
cd /home/aialfred/alfred
F=data/mainstay/fairgame/app/discover.html
curl -s -o /dev/null -w "%{http_code}\n" https://aialfred.groundrushcloud.com/fairgame/app/discover.html
curl -s -X POST https://aialfred.groundrushcloud.com/fairgame/api/discover/unlock -H 'Content-Type: application/json' -d '{}'
grep -c 'ev-img\|openUnlock\|ff_discover_unlocked\|price blur' $F
grep -c 'fairgame.css' $F
grep -P '[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}]' $F
```
Expected: `200`; unlock returns `{"unlocked":true,"sim":true,"amount_cents":100}`; first grep ≥3; `fairgame.css`=0; emoji grep empty. Open `discover.html`: search returns image-led cards; prices blurred + "Unlock for $1"; clicking unlock reveals prices + "Get verified ticket" links.

- [ ] **Step 5: Commit** (`git add data/mainstay/fairgame/app/discover.html && git commit -m "feat(fairgame): Discover reskin — image-led cards + $1 unlock gate"`).

---

### Task 3: Verify + responsive + parity

- [ ] Restart `fairgame-api.service`; confirm `/discover/unlock` live (200, sim true). Emoji gate clean. Responsive at 380/1280 (image cards reflow, grid collapses to 1 col). Parity with homepage chrome. Commit any polish; mark Plan 4 done.

---

## Self-Review
- **Spec coverage:** reskin ✓ (T2), event images on cards + fallback ✓ (T2 `eventCard`/`catFallback`), $1 paywall blur→pay→reveal ✓ (T1 endpoint + T2 gate), one-time-per-session unlock ✓ (`localStorage`), no emoji ✓ (T2), honest "verified" copy ✓. Out of plan: real Stripe webhook settlement (sim now; live branch returns checkout_url); per-event unlock (intentionally one-time per session).
- **Placeholders:** none — endpoint, test, card renderer, gate JS, and CSS are complete. Category fallback images are optional and degrade to a branded placeholder if absent.
- **Type consistency:** `eventCard(e, unlocked)` reads aggregator fields (`image,name,segment,date,city,venue,min_price,currency,source,buy_url`) that match `_normalize_tm_event`; unlock endpoint returns `unlocked/sim/amount_cents` consumed by `openUnlock`.

## Next plans
5. Accounts / My Tickets · 6. Admin CMS · 7. Delivery assist tool.
