# Fans First — Admin CMS (price/inventory control) Plan (Plan 5 of 7)

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Give the operator a back-office to **edit ticket prices and quantities** per show/section (and add sections), reskinned to the Fans First design system. Ships with the seeded defaults; the operator edits them in the back.

**Architecture:** `admin.py` is read-only today. Add `events.update_inventory(...)` plus three admin-token-gated endpoints (list/update/add inventory), and rebuild `admin.html` with a "Prices & inventory" manager: pick a show → editable table of sections (price + qty) → Save (PATCH changed rows) → Add-section form. Admin auth = existing `x-fairgame-admin` header (`ADMIN_TOKEN`, token `RodWaveAdmin2026`).

## Global Constraints
- DESIGN.md system (gold, Figtree, 11px squared buttons, no emoji, light page). Self-contained reskin like the homepage; drop `fairgame.css`.
- All write endpoints require a valid admin token (reuse `_require_admin`). Validate inputs (price/qty are non-negative ints).
- Money is integer cents end-to-end. Don't break the existing read endpoints or the buy/access flow.

## File Structure
- Modify: `core/fairgame/events.py` — `update_inventory(...)`, and make `get_inventory` rows expose `id` (they already do).
- Modify: `core/api/fairgame.py` — 3 admin endpoints.
- Create: `tests/fairgame/test_admin_cms.py`.
- Rewrite: `data/mainstay/fairgame/app/admin.html`.

---

### Task 1: Inventory edit backend (TDD)

**Interfaces:**
- `events.update_inventory(inv_id, *, face_price_cents=None, qty_available=None, qty_total=None) -> dict | None` — updates only the provided fields; returns the updated row or None if not found.
- `GET /fairgame/api/admin/shows/{show_id}/inventory` (admin) → `{"inventory":[{id,section,qty_total,qty_available,face_price_cents}]}`.
- `PATCH /fairgame/api/admin/inventory/{inv_id}` (admin) body `{face_price_cents?,qty_available?,qty_total?}` → `{"inventory": <row>}`; 404 if missing; 400 on negative/non-int.
- `POST /fairgame/api/admin/shows/{show_id}/inventory` (admin) body `{section, qty, face_price_cents}` → `{"inventory": <row>}` (wraps `events.add_inventory`).

- [ ] **Step 1: Failing tests** `tests/fairgame/test_admin_cms.py` — model seeding on `tests/fairgame/test_events.py` / `test_admin.py`:

```python
from core.fairgame import events, db

def test_update_inventory_changes_price_and_qty(seeded_inventory):
    inv_id = seeded_inventory          # an existing inventory row id
    row = events.update_inventory(inv_id, face_price_cents=12345, qty_available=7)
    assert row["face_price_cents"] == 12345 and row["qty_available"] == 7

def test_update_inventory_partial_only_touches_given(seeded_inventory):
    inv_id = seeded_inventory
    before = events.update_inventory(inv_id, qty_total=999)
    assert before["qty_total"] == 999  # price unchanged from seed

def test_update_inventory_missing_returns_none():
    assert events.update_inventory("inv_does_not_exist", face_price_cents=100) is None
```
Plus admin-endpoint tests via TestClient: PATCH with the admin header updates; PATCH without the admin header → 401/403; PATCH with a negative price → 400. (Copy the admin-header + seeding pattern from `tests/fairgame/test_admin.py`.)

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement `events.update_inventory`**:

```python
def update_inventory(inv_id, *, face_price_cents=None, qty_available=None, qty_total=None):
    """Patch an inventory row's price/quantities. Returns the row or None."""
    sets, vals = [], []
    for col, v in (("face_price_cents", face_price_cents),
                   ("qty_available", qty_available), ("qty_total", qty_total)):
        if v is not None:
            sets.append(f"{col}=?"); vals.append(int(v))
    with db.connect() as c:
        if sets:
            vals.append(inv_id)
            cur = c.execute(f"UPDATE inventory SET {', '.join(sets)} WHERE id=?", vals)
            if cur.rowcount == 0:
                return None
        row = c.execute("SELECT * FROM inventory WHERE id=?", (inv_id,)).fetchone()
    return dict(row) if row else None
```

- [ ] **Step 4: Add the 3 endpoints** in `core/api/fairgame.py` (near the other `/admin` routes). Each begins `_require_admin(x_fairgame_admin)`. PATCH validates each provided field is an int ≥ 0 (else `HTTPException(400)`), calls `events.update_inventory`, 404 if None. POST validates section/qty/price and calls `events.add_inventory`. GET returns `{"inventory": events.get_inventory(show_id)}`.

- [ ] **Step 5: Run new tests green; full suite `python -m pytest tests/fairgame/ -q` green.**

- [ ] **Step 6: Commit** `feat(fairgame): admin inventory price/qty edit endpoints`.

---

### Task 2: Reskin admin.html + price manager

- [ ] **Step 1: Self-contained reskin** to the design system (copy homepage tokens/nav/buttons; drop `fairgame.css`; no emoji). Keep the existing KPI dashboard + orders table, restyled.
- [ ] **Step 2: "Prices & inventory" panel** — a show `<select>` (from `GET /fairgame/api/shows`); on pick, `GET /admin/shows/{id}/inventory` renders a table: Section | Price ($) input | Qty avail input | Save. "Save" sends `PATCH /admin/inventory/{inv_id}` with the changed fields (convert dollars→cents). Show a success toast. An "Add section" row posts to `POST /admin/shows/{id}/inventory`. The admin token is entered once (existing `adminHeaders()` pattern) and sent on every write.
- [ ] **Step 3: Verify**
```bash
cd /home/aialfred/alfred
F=data/mainstay/fairgame/app/admin.html
curl -s -o /dev/null -w "%{http_code}\n" https://aialfred.groundrushcloud.com/fairgame/app/admin.html
curl -s -H 'x-fairgame-admin: RodWaveAdmin2026' https://aialfred.groundrushcloud.com/fairgame/api/admin/shows/show_1/inventory | head -c 200
grep -c 'admin/inventory\|/admin/shows\|fairgame.css' $F
grep -P '[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}]' $F
```
Expected: 200; inventory JSON; grep shows the edit calls present and `fairgame.css`=0; no emoji. In the browser (with the admin token) editing a price + Save persists (reload shows the new value).
- [ ] **Step 4: Commit** `feat(fairgame): admin CMS — reskin + editable price/inventory manager`.

---

## Self-Review
- **Coverage:** edit prices ✓ (T1 PATCH + T2 manager), edit qty ✓, add sections ✓, admin-gated ✓ (`_require_admin`), reskin ✓, ships-with-defaults ✓ (seeded inventory editable). Out of plan: editing the Discover $1 fee / resale fee constants (separate later); audit log of edits.
- **Placeholders:** none — `update_inventory`, endpoints, tests, and the manager flow are specified with code.
- **Types:** `update_inventory` keyword-only price/qty params match the PATCH body keys; rows expose `id` used by the manager's PATCH URL.

## Next plans
6. (was 5 → renumber) Accounts / My Tickets reskin · 7. Delivery assist tool.
