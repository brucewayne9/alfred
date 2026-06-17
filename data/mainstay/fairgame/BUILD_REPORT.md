# Fair Game — Morning Build Report

**Date:** 2026-06-17
**For:** Mike (owner) + Rod (client)
**Branch:** `feat/fairgame`
**Status:** Backend + frontend built, tested, and smoke-passed locally. Ready for the operator to deploy.

---

## 1. What Fair Game Is

Fair Game is Rod Wave's own fan-first, anti-scalper ticket marketplace — a SeatGeek/StubHub
competitor built specifically for Rod's "Don't Look Down" tour. Scalpers and resale sites
buy up Rod's tickets, flip them at 4–6× face, and fans blame Rod for the gouging. Fair Game
takes that back: fair prices, verified real fans, and the resale fees + fan data (that
scalpers and SeatGeek capture today) become Rod's.

The mechanism is **fair transfer, never lock-down.** A fan who can't attend can still gift
their ticket free or resell it — but only through Rod's booth, hard-capped at face + $15. The
seller is always made whole on face and pockets $10; Rod earns a light $5; the buyer pays a
fair price. Tickets stay 100% transferable — the only thing removed is the freedom to gouge.

It rides Ticketmaster's rails: TM still issues the barcode and handles entry. Fair Game owns
the fan, the verification gate, the price cap, and the resale fee.

## 2. What Got Built Tonight

This went well past the original "M1 foundation" plan (fan identity only) — the full
fan-to-fan resale loop is now standing.

**Backend modules (`core/fairgame/`):**
- `db.py` — sqlite (WAL) persistence; all tables created here, honors `FAIRGAME_DB_PATH`.
- `identity.py` — fan registration, one-identity-one-account dedupe (salted hash on email
  AND phone), device/IP fingerprint logging, DLD-waitlist priority flag.
- `verify.py` — 6-digit verification codes, SMS then email, with resend cooldown, attempt
  limits, and expiry (the anti-bot "medium strength" gate).
- `sessions.py` — opaque Bearer session tokens for verified fans.
- `waitlist.py` — Klaviyo DLD-waitlist priority check (local seed file in v1; clean seam to
  swap in the live Klaviyo pull later).
- `events.py` — seeds Rod's tour shows + per-section inventory; cap is computed off the TRUE
  primary face, not a seller-declared number.
- `access.py` — the presale gate: access waves, priority-only gating, max-qty-per-fan caps,
  inventory decrement.
- `listings.py` — capped resale listings; fixed $10 seller / $5 Rod split; anti-gouge
  (cannot list above true face).
- `tm_transfer.py` — Ticketmaster transfer state machine (SIMULATED in v1; clean seam).
- `stripe_connect.py` — Stripe Connect Express onboarding + held-payment escrow (simulator
  by default; real test-mode Stripe behind an env key).
- `orders.py` — the resale escrow state machine: hold buyer funds → confirm transfer →
  release to seller, or fail → refund buyer (the fraud wall). Seat is claimed atomically so
  two buyers can never both pay for one seat.
- `admin.py` — operator rollups (KPIs, fans, orders, broker flagging).

**API (`core/api/fairgame.py`):** one FastAPI sub-app, all endpoints under `/fairgame/api/*`,
binds 127.0.0.1, mirrors the existing `arena_portal.py` pattern. Self-bootstraps schema +
shows + inventory on import. Per-IP rate limiting on register/verify. Endpoints: register,
verify (SMS), verify-email, me, shows, show detail, access grant, exchange, listings, buy,
order confirm/fail/get, and token-gated admin stats/fans/orders.

**Frontend (`data/mainstay/fairgame/app/`):** Rod-branded black/orange storefront —
`index.html` (shows), `show.html` (show detail + buy/resale), `account.html` (register →
SMS → email → verified), `sell.html` (capped resale listing), `admin.html` (tour console),
`fairgame.css` (shared design system).

## 3. What Is REAL vs SANDBOX / SIMULATED

**Real and working end-to-end (no external services needed):**
- Fan registration, dedupe, device fingerprint logging — REAL (sqlite).
- The verification gate's full logic — code generation, cooldown, attempt limits, expiry,
  one-time consumption — REAL. (The actual SMS/email *delivery* is the sandbox part below.)
- Sessions / Bearer auth — REAL.
- Tour shows + inventory, access waves, priority/qty gating — REAL.
- Capped resale: listing, anti-gouge cap enforcement, the full $10/$5 split math — REAL.
- The resale escrow state machine — hold → confirm → release, or fail → refund, atomic seat
  claim, idempotent transitions — REAL.
- Admin rollups + token gating — REAL.

**Sandbox / simulated (honest list — these need real creds/integration for production):**
- **Stripe = TEST/SIM mode.** Defaults to a built-in simulator (`FAIRGAME_STRIPE_SIM=1`,
  auto-on when no key). NOTHING hits Stripe's network — the escrow money flow is modelled
  against sqlite so the resale loop runs and demos with zero keys. To go live: set a
  **test-mode** Stripe key (`FAIRGAME_STRIPE_KEY`) + `FAIRGAME_STRIPE_SIM=0`; the live branch
  uses manual-capture PaymentIntents (hold), capture (release), and cancel/refund. No live key
  is wired anywhere.
- **Ticketmaster transfer = SIMULATED seam.** `tm_transfer.py` models the
  initiate → confirm state machine locally. No real TM Partner API call is made. The seam is
  clean: when TM access is granted, only the bodies of `initiate`/`confirm` change — the
  contract and every caller stay the same.
- **SMS + email = no-op without creds.** Sends are wrapped so a missing Twilio/email
  credential just logs and continues — it never breaks the flow. A `FAIRGAME_DEV_ECHO=1` path
  returns the verification code in the API response (tests + local demo only — NEVER set in
  prod).
- **Waitlist priority = local seed.** Priority comes from a local seed file
  (`data/mainstay/fairgame/waitlist_emails.txt`). That file is not present yet, so every fan
  is currently non-priority. The live Klaviyo DLD-waitlist pull (list `V75mRt`, account
  `XYKnGf`) is a documented seam to swap in later. Drop emails into that file to flag priority
  fans immediately.
- **Tour data:** seeded from the canonical `data/mainstay/tour/arena_folder_links.json` →
  **25 arenas** (the spec referenced "35 shows"; the canonical tour file currently holds 25 —
  worth reconciling with the real on-sale count). Inventory is realistic *demo* sizing
  ($150 floor / $95 lower / $55 upper), not Rod's actual artist-hold counts.

## 4. Test Results

- **Automated suite: 131 tests, 131 passed** (`venv/bin/python -m pytest tests/fairgame/`).
  Summary line: `131 passed in 59.95s`.
- **Live smoke test: PASS.** Ran a real uvicorn server on 127.0.0.1:8402 and walked the full
  loop against live HTTP:
  - 25 shows served; first show 1,200 tickets remaining.
  - Seller registered → SMS verified → email verified → session issued.
  - Listed a $95 (9,500¢) Lower seat → buyer total **$110**, seller proceeds **$105**, Rod
    fee **$5** (cap math correct).
  - Anti-gouge: listing above true face rejected with **HTTP 400**.
  - Buyer registered + verified → bought the listing → escrow **held** ($110, sim payment).
  - Buyer confirmed transfer → order **released**, TM transfer ref recorded.
  - Admin stats without token → **HTTP 401**; with token → correct rollup
    (`gross_platform_fees_cents: 500`, i.e. Rod's $5 booked).
  - `/me` unauthenticated → **HTTP 401**.

## 5. How to Demo It Locally

From `/home/aialfred/alfred`, start the API (dev echo on so you can complete verification
without real SMS/email; admin token set so the console works):

```bash
FAIRGAME_DEV_ECHO=1 FAIRGAME_ADMIN_TOKEN=demo \
  venv/bin/python -m uvicorn core.api.fairgame:app --host 127.0.0.1 --port 8402
```

Then open the pages (served from `data/mainstay/fairgame/app/`):
- `/fairgame/app/index.html` — the storefront (Rod's 25 shows)
- `/fairgame/app/show.html` — a show's detail + buy / resale exchange
- `/fairgame/app/account.html` — register → SMS code → email code → verified
- `/fairgame/app/sell.html` — list a seat for capped resale
- `/fairgame/app/admin.html` — the tour console (needs the admin token)

(In local dev the verification codes echo in the API responses; in production leave
`FAIRGAME_DEV_ECHO` unset and use real SMS/email.)

## 6. Deploy Steps the OPERATOR Still Needs to Run

*(Not done here by design — no deploy, no sudo, no Caddy/systemd edits were made.)*

1. **Caddy** — add, BEFORE the existing static `/fairgame/*` block:
   - static serving for `/fairgame/` and `/fairgame/app/*` (from
     `data/mainstay/fairgame/app/`), and
   - `reverse_proxy /fairgame/api/* -> 127.0.0.1:8402`.
   Then `caddy validate` and reload.
2. **systemd** — install + start a unit that runs:
   ```
   /home/aialfred/alfred/venv/bin/python -m uvicorn core.api.fairgame:app --port 8402
   ```
   (WorkingDirectory `/home/aialfred/alfred`, `PYTHONPATH=/home/aialfred/alfred`, bind
   127.0.0.1, `Restart=on-failure`).
3. **Set production env** on the service: `FAIRGAME_ADMIN_TOKEN` (real secret, from
   config/.env — the admin API is hard-401'd until this is set), and DO NOT set
   `FAIRGAME_DEV_ECHO`. Point `FAIRGAME_DB_PATH` at a persistent location.
4. Smoke the live URL: `POST /fairgame/api/register` should return 200, and
   `/fairgame/app/` should load and walk the flow with real SMS/email.

## 7. Open Items / What M2+ Still Needs

- **Real Stripe keys** (test-mode first, then live) + Connect Express seller onboarding URLs
  + the `account.updated` / `payment_intent` webhooks to drive payout/refund from Stripe's
  source of truth.
- **Real Ticketmaster transfer integration** — swap the `tm_transfer.initiate/confirm`
  bodies for the TM Partner API + transfer-status webhook.
- **Live Klaviyo waitlist pull** to replace the local priority seed (list `V75mRt`).
- **SMS/email creds** wired on the service so verification actually delivers.
- **Reconcile show count** — 25 seeded vs the "35 shows" in the spec; confirm the real
  on-sale list, and replace demo inventory with Rod's actual artist-hold / fan-club counts
  per show.
- **Legal: per-state resale-cap config** (resale-cap and data rules vary by state — e.g. NY,
  CO) before public launch.
- **PII encryption at rest** (M1 stores raw email/phone; flagged for hardening).
- **On-sale spike load test** — the verification + access gate must survive a presale rush.
- **Stripe 1099-K / tax reporting** + chargeback handling for the marketplace.

---

*Built on the existing FastAPI + sqlite Alfred/Mainstay stack. Modules are clean enough to
later extract into a standalone multi-artist product once the thesis proves out on Rod's tour.*
