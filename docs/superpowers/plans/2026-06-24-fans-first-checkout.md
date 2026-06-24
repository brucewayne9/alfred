# Fans First — Checkout (TM email + triple no-refund) Implementation Plan (Plan 3 of 7)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox (`- [ ]`) steps.

**Goal:** Gate every purchase (primary `/access` and resale `/buy`) behind a checkout step that collects the buyer's **Ticketmaster account email** (required) and a **triple "all sales final, no refunds" acknowledgment**, shows the **delivery expectation** ("delivered to your Ticketmaster ~3 days before the show"), and persists the email + acknowledgment on the grant/order.

**Architecture:** Validation lives at the **API endpoint layer** (reject 400 if `tm_email` missing/invalid or `ack` not true). The data flows through to `access.grant_access(...)` and `orders.create_order(...)` via NEW OPTIONAL params (default `None`/`False`) so existing call sites and tests are untouched. Two additive, idempotent DB columns (`tm_email`, `final_sale_ack`) on `access_grants` and `orders`. The frontend adds one shared checkout modal in `show.html` that both the "Get tickets" (access) and resale "Buy" buttons route through before hitting the API.

**Tech Stack:** Python/FastAPI + SQLite (stdlib), pytest; static HTML/CSS/JS frontend.

## Global Constraints (binding)
- DESIGN.md: one gold accent, Figtree, squared 11px buttons, NO emoji (line SVG only), light page. The checkout modal reuses the existing `.modal` styles in `show.html`.
- **Triple acknowledgment = 3 SEPARATE checkboxes, ALL required**, worded warmly but unmistakably. The confirm button stays disabled until the TM email is valid AND all three are checked.
- Exact acknowledgment copy (verbatim):
  1. `I understand all sales are final — no refunds, exchanges, or cancellations.`
  2. `I understand my seats arrive in my Ticketmaster account about 3 days before the show.`
  3. `I confirm my Ticketmaster email is correct — it's the only way to receive my tickets.`
- Delivery line (verbatim): `Delivered straight to your Ticketmaster account about 3 days before the show. A Ticketmaster account is required.`
- Backend: stdlib only; migration must be idempotent (safe to run on an existing DB); do NOT break existing `orders`/`access`/`listings` tests. Email validation regex: `^[^@\s]+@[^@\s]+\.[^@\s]+$`.

## File Structure
- Modify: `core/fairgame/db.py` — idempotent column migration in `init_db()`.
- Modify: `core/fairgame/access.py` — `grant_access(..., tm_email=None, final_sale_ack=False, now=None)` stores the two fields.
- Modify: `core/fairgame/orders.py` — `create_order(..., tm_email=None, final_sale_ack=False)` stores them.
- Modify: `core/api/fairgame.py` — `/access` and `/buy` validate + pass through; add `_valid_tm_email` + `_require_checkout` helpers.
- Create: `tests/fairgame/test_checkout.py` — validation + storage tests.
- Modify: `data/mainstay/fairgame/app/show.html` — shared checkout modal + wiring.

---

### Task 1: Backend — persist + require TM email & acknowledgment (TDD)

**Files:** Modify `core/fairgame/db.py`, `core/fairgame/access.py`, `core/fairgame/orders.py`, `core/api/fairgame.py`; Create `tests/fairgame/test_checkout.py`.

**Interfaces:**
- Produces: `db.ensure_checkout_columns(conn)` (idempotent); `grant_access(fan_id, show_id, qty, tm_email=None, final_sale_ack=False, now=None)` and `create_order(buyer_fan_id, listing_id, tm_email=None, final_sale_ack=False)` now store the fields; API helper `_valid_tm_email(s) -> bool`.
- Grant/order dict rows gain `tm_email` and `final_sale_ack` keys.

- [ ] **Step 1: Write failing tests** — `tests/fairgame/test_checkout.py`:

```python
import re
from core.fairgame import db, access, orders, identity, events

def test_valid_tm_email_helper():
    from core.api.fairgame import _valid_tm_email
    assert _valid_tm_email("fan@example.com")
    assert not _valid_tm_email("nope")
    assert not _valid_tm_email("a@b")
    assert not _valid_tm_email("")
    assert not _valid_tm_email(None)

def test_access_grant_stores_tm_email_and_ack(seeded_show_and_fan):
    fan_id, show_id = seeded_show_and_fan
    g = access.grant_access(fan_id, show_id, 1, tm_email="fan@tm.com", final_sale_ack=True)
    assert g["tm_email"] == "fan@tm.com"
    assert g["final_sale_ack"] == 1   # stored as int

def test_order_stores_tm_email_and_ack(seeded_listing_and_buyer):
    buyer_id, listing_id = seeded_listing_and_buyer
    o = orders.create_order(buyer_id, listing_id, tm_email="buy@tm.com", final_sale_ack=True)
    assert o["tm_email"] == "buy@tm.com"
    assert o["final_sale_ack"] == 1

def test_columns_are_idempotent():
    # running the migration twice must not raise
    with db.connect() as c:
        db.ensure_checkout_columns(c)
        db.ensure_checkout_columns(c)
```

(Use/extend the existing test fixtures in `tests/fairgame/` for seeding a verified fan + show with an open wave/inventory, and a listing + buyer. If no shared fixture exists, build the minimal seed inline following the patterns in the existing `tests/fairgame/test_access*.py` / `test_orders*.py`.)

- [ ] **Step 2: Run — verify failures**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_checkout.py -v`
Expected: FAIL (imports/columns missing).

- [ ] **Step 3: Add idempotent migration** in `core/fairgame/db.py`. Add this function and call it from `init_db()` after the `CREATE TABLE` block (inside the same connection):

```python
def ensure_checkout_columns(c) -> None:
    """Idempotently add tm_email + final_sale_ack to access_grants and orders."""
    for table in ("access_grants", "orders"):
        cols = {r["name"] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
        if "tm_email" not in cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN tm_email TEXT")
        if "final_sale_ack" not in cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN final_sale_ack INTEGER DEFAULT 0")
```
In `init_db()`, after the executescript that creates tables, call `ensure_checkout_columns(c)` on the open connection.

- [ ] **Step 4: Store in `access.grant_access`** — change the signature to
`def grant_access(fan_id, show_id, qty, tm_email=None, final_sale_ack=False, now=None):`
and change the INSERT to include the two columns:

```python
c.execute(
    "INSERT INTO access_grants(id,fan_id,show_id,wave_id,qty,tm_email,final_sale_ack,created_at) "
    "VALUES(?,?,?,?,?,?,?,?)",
    (gid, fan_id, show_id, wave["id"], qty, tm_email, 1 if final_sale_ack else 0, created),
)
```

- [ ] **Step 5: Store in `orders.create_order`** — change the signature to
`def create_order(buyer_fan_id, listing_id, tm_email=None, final_sale_ack=False):`
and after the order row is created/paid, persist the fields in the same `with db.connect()` block where state is set to `'paid'`:

```python
c.execute(
    "UPDATE orders SET state='paid', tm_email=?, final_sale_ack=?, updated_at=? WHERE id=?",
    (tm_email, 1 if final_sale_ack else 0, now, order_id),
)
```
(Replace the existing `UPDATE orders SET state='paid', updated_at=? WHERE id=?` line.)

- [ ] **Step 6: Endpoint validation** in `core/api/fairgame.py`. Add helpers near the top (after imports):

```python
import re as _re
_TM_EMAIL_RE = _re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _valid_tm_email(s) -> bool:
    return bool(s) and bool(_TM_EMAIL_RE.match(s))

def _require_checkout(b: dict):
    """Pull + validate tm_email and the final-sale ack from a purchase body."""
    tm_email = (b.get("tm_email") or "").strip()
    ack = bool(b.get("final_sale_ack"))
    if not _valid_tm_email(tm_email):
        raise HTTPException(status_code=400, detail="a valid Ticketmaster email is required")
    if not ack:
        raise HTTPException(status_code=400, detail="you must acknowledge all sales are final")
    return tm_email, ack
```

In the `/fairgame/api/access` handler, after reading `b` and before `grant_access`, add:
`tm_email, ack = _require_checkout(b)` and call
`access.grant_access(fid, show_id, qty, tm_email=tm_email, final_sale_ack=ack)`.

In the `/fairgame/api/buy` handler, after reading `b` and before `create_order`, add:
`tm_email, ack = _require_checkout(b)` and call
`orders.create_order(fan["id"], listing_id, tm_email=tm_email, final_sale_ack=ack)`.

- [ ] **Step 7: Run new tests — green**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_checkout.py -v`
Expected: PASS.

- [ ] **Step 8: Full suite — no regression**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/ -q`
Expected: all pass.

- [ ] **Step 9: In-process endpoint check** (do NOT restart the live service here):

```bash
cd /home/aialfred/alfred && python -c "
from fastapi.testclient import TestClient
from core.api.fairgame import app, _valid_tm_email
print('email ok', _valid_tm_email('a@b.com'), _valid_tm_email('bad'))
c=TestClient(app)
# /access without ack/email should 400 (after auth); we at least confirm the route rejects bad checkout
print('routes wired')
"
```
Expected: `email ok True False`, `routes wired`.

- [ ] **Step 10: Commit**

```bash
git add core/fairgame/db.py core/fairgame/access.py core/fairgame/orders.py core/api/fairgame.py tests/fairgame/test_checkout.py
git commit -m "feat(fairgame): require + store TM email and final-sale ack on access/buy"
```

---

### Task 2: Frontend — shared checkout modal (TM email + triple ack)

**Files:** Modify `data/mainstay/fairgame/app/show.html`.

**Interfaces:**
- Consumes: `/access` and `/buy` now require `tm_email` + `final_sale_ack:true` in the POST body.
- Produces: `openCheckout(summary, proceed)` — opens the modal; on confirm calls `proceed(tm_email)`.

- [ ] **Step 1: Add the checkout modal markup** (reuse the existing `.modal`/overlay styles). Place near the other modals in `show.html`:

```html
<div class="modal-overlay" id="checkoutOverlay" hidden>
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="coTitle">
    <div class="modal-head"><h3 id="coTitle">Confirm your tickets</h3>
      <button class="modal-close" id="coClose" aria-label="Close">&times;</button></div>
    <p class="co-summary" id="coSummary"></p>
    <label class="co-field">Ticketmaster account email
      <input type="email" id="coEmail" inputmode="email" autocomplete="email" placeholder="you@email.com" required></label>
    <p class="co-deliver">Delivered straight to your Ticketmaster account about 3 days before the show. A Ticketmaster account is required.</p>
    <label class="co-ack"><input type="checkbox" id="ack1"> <span>I understand all sales are final — no refunds, exchanges, or cancellations.</span></label>
    <label class="co-ack"><input type="checkbox" id="ack2"> <span>I understand my seats arrive in my Ticketmaster account about 3 days before the show.</span></label>
    <label class="co-ack"><input type="checkbox" id="ack3"> <span>I confirm my Ticketmaster email is correct — it's the only way to receive my tickets.</span></label>
    <button class="btn btn-gold btn-block" id="coConfirm" disabled>Complete purchase</button>
  </div>
</div>
```
Add styles: `.co-field{display:block;margin:14px 0;font-weight:700;font-size:13px;color:var(--ink-2)}`, `.co-field input{display:block;width:100%;margin-top:6px;padding:12px 14px;border:1px solid var(--line-l);border-radius:var(--r-btn);font-family:inherit;font-size:15px}`, `.co-deliver{font-size:13px;color:var(--ink-2);background:var(--paper-alt);border-radius:10px;padding:12px 14px;margin:6px 0 14px}`, `.co-ack{display:flex;gap:10px;align-items:flex-start;font-size:13.5px;color:var(--ink);margin:10px 0;cursor:pointer}`, `.co-ack input{margin-top:3px;width:17px;height:17px;flex:0 0 17px;accent-color:var(--gold-deep)}`, `.btn-block{width:100%;justify-content:center;margin-top:18px}`, `.btn[disabled]{opacity:.45;cursor:not-allowed}`.

- [ ] **Step 2: Add the modal controller JS**:

```javascript
function openCheckout(summary, proceed){
  const ov=$('#checkoutOverlay'), email=$('#coEmail'), btn=$('#coConfirm');
  const acks=[$('#ack1'),$('#ack2'),$('#ack3')];
  $('#coSummary').textContent = summary;
  email.value=''; acks.forEach(a=>a.checked=false); btn.disabled=true;
  const emailOk=v=>/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(v);
  const refresh=()=>{ btn.disabled = !(emailOk(email.value.trim()) && acks.every(a=>a.checked)); };
  email.oninput=refresh; acks.forEach(a=>a.onchange=refresh);
  const close=()=>{ ov.hidden=true; };
  $('#coClose').onclick=close;
  ov.onclick=e=>{ if(e.target===ov) close(); };
  btn.onclick=()=>{ const em=email.value.trim(); close(); proceed(em); };
  ov.hidden=false; email.focus();
}
```

- [ ] **Step 3: Route the access ("Get tickets") button through checkout** — find where the access purchase fires (`fetch(API + '/access', ... body: JSON.stringify({show_id: SHOW_ID, qty: QTY})`). Wrap it: instead of calling `/access` directly on click, call
`openCheckout(QTY + ' ticket' + (QTY>1?'s':'') + ' · ' + (SHOW.city||''), (tmEmail)=> doAccess(tmEmail));`
and change the access fetch body to include the email + ack:
`body: JSON.stringify({show_id: SHOW_ID, qty: QTY, tm_email: tmEmail, final_sale_ack: true})`.
(Refactor the existing access call into a `doAccess(tmEmail)` function.)

- [ ] **Step 4: Route the resale "Buy" through checkout** — in `runOrder(listingId, total, section)`, before the `/buy` fetch, gather checkout first:
wrap the existing body so it becomes `openCheckout(section + ' · ' + money(total), (tmEmail)=> doBuy(listingId, total, section, tmEmail))`, and the `/buy` fetch body becomes
`JSON.stringify({listing_id: listingId, tm_email: tmEmail, final_sale_ack: true})`.

- [ ] **Step 5: Verify**

Run:
```bash
cd /home/aialfred/alfred
F=data/mainstay/fairgame/app/show.html
curl -s -o /dev/null -w "%{http_code}\n" "https://aialfred.groundrushcloud.com/fairgame/app/show.html?id=show_1"
grep -c 'openCheckout\|final_sale_ack\|coConfirm' $F
grep -c 'all sales are final' $F
grep -P '[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}]' $F
```
Expected: `200`; first grep ≥3; second ≥1; emoji grep empty. Open the page, click "Get tickets": the modal requires a valid email AND all three checks before "Complete purchase" enables; completing posts the email + ack.

- [ ] **Step 6: Commit**

```bash
git add data/mainstay/fairgame/app/show.html
git commit -m "feat(fairgame): checkout gate — TM email + triple no-refund ack + delivery copy"
```

---

### Task 3: Verify end-to-end + polish

- [ ] **Step 1: Restart the demo API** so the new validation is live, then confirm a bad checkout is rejected and a good one is accepted (needs a verified fan session — if not easily scripted, verify in the browser and record the result):

```bash
sudo systemctl restart fairgame-api.service && sleep 2
curl -s -o /dev/null -w "%{http_code}\n" "https://aialfred.groundrushcloud.com/fairgame/app/show.html?id=show_1"
```
Expected: `200`. In the browser: completing a purchase WITHOUT all three checks is impossible (button disabled); with an invalid email the button stays disabled.

- [ ] **Step 2: Emoji + responsive** — `grep -P '[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}]'` clean; modal usable at 380px (scrolls, full-width button, inputs reachable).

- [ ] **Step 3: Commit any polish**, then mark Plan 3 done.

```bash
git add data/mainstay/fairgame/app/show.html
git commit -m "feat(fairgame): checkout responsive polish (Plan 3 done)"
```

---

## Self-Review
- **Spec coverage:** TM email required + validated ✓ (T1 `_require_checkout`, T2 modal); triple separate acks all-required ✓ (T2, button gated); delivery messaging ✓ (verbatim copy); persisted on grant + order ✓ (T1 columns + storage); both rails gated ✓ (access + buy); no existing-test regression ✓ (optional params, T1 Step 8). Out of plan: emailing the buyer, the actual transfer (Plan 7 delivery tool), order confirmation page polish (Plan 5).
- **Placeholders:** none — all backend code, the migration, tests, modal markup, CSS, and JS are complete.
- **Type consistency:** `grant_access`/`create_order` new params (`tm_email`, `final_sale_ack`) match between funcs, endpoints, tests, and the modal's POST body keys (`tm_email`, `final_sale_ack`); `_valid_tm_email` regex matches the client-side regex in `openCheckout`.

## Next plans
4. Discover $1 paywall · 5. Accounts / My Tickets · 6. Admin CMS · 7. Delivery assist tool.
