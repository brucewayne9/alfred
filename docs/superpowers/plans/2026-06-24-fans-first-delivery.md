# Fans First — Delivery Operator Queue Plan (Plan 6 of 7)

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** An operator **delivery queue**: every purchase needing a Ticketmaster transfer, with the buyer's TM email, the show, what they bought, and a "Mark delivered" action that records the transfer — the human-in-the-loop half of the supervised delivery model.

**Architecture:** New `core/fairgame/delivery.py` assembles the queue from paid **orders** (resale, escrow via the existing `transfers` state machine) and **access grants** (primary Rod-held buys), each joined to the show + the buyer's `tm_email` (captured at checkout). `mark_delivered` drives `tm_transfer.initiate`+`confirm` for orders and stamps a new `delivered_at` on grants. Two admin endpoints + a Delivery panel added to the (already reskinned) `admin.html`. Real TM automation stays out — this is the supervised queue an operator works; the actual transfer happens in TM, then they mark it here.

## Global Constraints
- Admin-token gated (`_require_admin`, `x-fairgame-admin: RodWaveAdmin2026`). DESIGN.md system in the UI (gold, Figtree, 11px buttons, no emoji). Integer cents. Don't break existing flows/tests.
- Honest UI copy: transfers typically open ~3 days before the show; this queue is where the operator delivers when the window is open. No fake live-TM automation.

## File Structure
- Create: `core/fairgame/delivery.py`, `tests/fairgame/test_delivery.py`.
- Modify: `core/fairgame/db.py` — add idempotent `delivered_at` column to `access_grants` (extend the existing `ensure_checkout_columns` or add `ensure_delivery_columns`).
- Modify: `core/api/fairgame.py` — `GET /admin/delivery`, `POST /admin/delivery/mark`.
- Modify: `data/mainstay/fairgame/app/admin.html` — Delivery panel.

---

### Task 1: Delivery backend (TDD)

**Interfaces:**
- `delivery.queue() -> list[dict]` items: `{kind:'order'|'grant', id, buyer_tm_email, show_id, city, show_date, detail, state}` where `state` ∈ `'delivered'|'pending'`. For orders, `delivered` iff a `transfers` row is `'confirmed'`; for grants, `delivered` iff `delivered_at` is set. `detail` = section/seat for orders, `"{qty} ticket(s)"` for grants.
- `delivery.mark_delivered(kind, item_id) -> dict` → for `'order'`: `tm_transfer.initiate(item_id)` then `tm_transfer.confirm(item_id)`, return `{kind,id,state:'delivered'}`; for `'grant'`: `UPDATE access_grants SET delivered_at=? WHERE id=?` (set now), return same. Raises `ValueError` on unknown kind / missing id.

- [ ] **Step 1: Failing tests** `tests/fairgame/test_delivery.py` (seed like `test_orders.py`/`test_access.py`): a paid order appears in the queue as pending then `delivered` after `mark_delivered('order', id)`; a grant appears and flips to `delivered` after `mark_delivered('grant', id)`; `delivered_at` migration is idempotent; unknown kind raises.

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Migration** — add to `db.py` (idempotent, same PRAGMA pattern): ensure `access_grants` has `delivered_at INTEGER`. Call it from `init_db()`.

- [ ] **Step 4: Implement `delivery.py`** — `queue()` selects paid/released orders (join `listings` for section, `transfers` for confirmed state, and the order's `tm_email`) and all access grants (with their `tm_email`, `delivered_at`, joined to `shows` for city/date); normalize both into the item shape. `mark_delivered` as specified (import `tm_transfer`, `db`).

- [ ] **Step 5: Endpoints** in `core/api/fairgame.py` (admin-gated): `GET /fairgame/api/admin/delivery` → `{"queue": delivery.queue()}`; `POST /fairgame/api/admin/delivery/mark` body `{kind, id}` → `{"item": delivery.mark_delivered(kind, id)}` (400 on bad kind/missing id).

- [ ] **Step 6: New tests green; full suite green.**

- [ ] **Step 7: Commit** `feat(fairgame): delivery operator queue (orders + grants) + endpoints`.

---

### Task 2: Delivery panel in admin.html

- [ ] **Step 1: Add a "Delivery" panel** to `admin.html` (same design system, no emoji): a table from `GET /admin/delivery` — Buyer TM email · Show (city + date) · What · State badge (Delivered green / Pending gold) · a "Mark delivered" button for pending rows that POSTs `{kind, id}` to `/admin/delivery/mark`, then refreshes the row. A short honest note: "Transfers usually open ~3 days before each show. Do the transfer in Ticketmaster, then mark it delivered here." Reuse the page's `adminHeaders()` + toast.
- [ ] **Step 2: Verify**
```bash
cd /home/aialfred/alfred
F=data/mainstay/fairgame/app/admin.html
curl -s -o /dev/null -w "%{http_code}\n" https://aialfred.groundrushcloud.com/fairgame/app/admin.html
curl -s -H 'x-fairgame-admin: RodWaveAdmin2026' https://aialfred.groundrushcloud.com/fairgame/api/admin/delivery | head -c 160
grep -c '/admin/delivery' $F
grep -P '[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}]' $F
```
Expected: 200; queue JSON; grep ≥1; no emoji.
- [ ] **Step 3: Commit** `feat(fairgame): admin delivery panel — mark transfers delivered`.

---

## Self-Review
- **Coverage:** queue of all deliverable purchases (orders + grants) ✓, buyer TM email shown ✓ (captured at checkout), mark-delivered with tracking ✓ (transfers state machine for orders, `delivered_at` for grants), admin-gated ✓, honest supervised-not-automated framing ✓. Out of plan: real TM Partner automation (intentionally — supervised only); per-window auto-eligibility (informational note only); buyer-side My Tickets countdown (separate page).
- **Placeholders:** none — `queue`/`mark_delivered`, migration, endpoints, and the panel flow are specified.
- **Types:** queue item keys (`kind,id,buyer_tm_email,show_id,city,show_date,detail,state`) match between `delivery.py`, the endpoint, and the panel render; `mark_delivered(kind,id)` body matches the POST.

## Next plans
7. Accounts / My Tickets reskin (buyer-side delivery countdown) — the last surface.
