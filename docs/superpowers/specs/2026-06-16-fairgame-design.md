# Fair Game — Rod Wave Fan-First Ticket Platform (Design Spec)

**Date:** 2026-06-16
**Client/Partner:** Rod Wave (via Mainstay Tech partnership)
**Owner:** Ground Rush / Alfred
**Status:** Draft for Rod + Mike review

---

## 1. The Problem (Rod's words)

Scalpers and resale marketplaces (SeatGeek, StubHub) buy up Rod's tickets and flip
them at 4–6× face. Fans pay the inflated price and **blame Rod**, assuming he set it.
Rod wants to take back control: fair prices, real fans, and the revenue currently
captured by scalpers and resale sites.

## 2. The Honest Strategic Frame (read this first)

- **Operating "like SeatGeek" — an independent capped marketplace — does NOT by itself
  stop scalpers from listing Rod's tickets at 5× on SeatGeek.** SeatGeek *is* where
  scalpers sell; competition alone doesn't close their doors.
- The blunt lever that fully kills the 5× flip is **non-transferable** tickets — but
  **Rod has rejected that on principle (2026-06-16):** it punishes real fans (illness,
  gifting). We do NOT use non-transferable.
- **The mechanism we use instead is FAIR TRANSFER:** tickets stay fully transferable —
  a fan can **gift free or resell capped (face + small markup) through Rod's booth** —
  the only freedom removed is the freedom to gouge. This is more pro-fan AND is the
  anti-scalper tool, because it funnels resale through Rod's capped channel.
- **This spec is the no-negotiation version.** It does NOT claim to eliminate SeatGeek
  outright. It captures the fair-fan segment, the resale margin, and the fan data on the
  inventory Rod controls, and applies downward price pressure — without ever locking a
  fan down.

### What this version does / does not do
- ✅ Rod owns the fan relationship, data, brand, and resale fees on his shows
- ✅ Fair-priced, verified-gated supply pressures his shows' prices down
- ✅ Ships now, zero LN/TM negotiation required
- ❌ Does NOT fully eliminate SeatGeek's 5× listings (some scalpers stay there)
- ➡️ Upgrade path: non-transferable issuance (requires LN deal) to go from *dent* to *kill*

## 3. Strategy: Compete on Supply + Trust, Not on Restriction

Fair Game is Rod's own verified, price-capped marketplace, **seeded with inventory Rod
already controls** (no TM permission needed):

- **Artist holds** (every tour reserves an artist-controlled block)
- **Fan-club / presale allocation** (Rod decides who gets these)
- **VIP / bundle inventory** Rod carries

Real fans who can't attend are pulled to resell on Fair Game (instead of SeatGeek) via
**carrots, not sticks**:

- **Zero seller fee** (SeatGeek takes ~10–15%)
- **Instant payout** (SeatGeek holds funds until after the event)
- **"Rod Official" trust badge** — buyers trust it, sells faster
- **Loyalty hooks** — buy/sell on Fair Game unlocks presale access, merch discounts, M&G
- **All of Rod's marketing traffic** (socials + Klaviyo DLD waitlist) points only here

Carrots convert **good-faith resellers** (real fans), not profit-scalpers. That is the
intended segment.

## 4. Fulfillment Model — Ride Ticketmaster Rails

Fair Game is Rod's **brand + identity-verification + price-cap layer** in front of
Ticketmaster. **TM issues the barcode and handles entry**; Fair Game owns the fan, the
queue, the cap, and the resale fee. This avoids the venue-exclusivity contract fight.

## 5. Architecture (Approach A — extend the Mainstay/Alfred stack)

Built on existing FastAPI + React stack on server 105, reusing auth, Klaviyo
integration, and the "Rollout" board pattern. Modules built clean so the whole platform
can later be **extracted into a standalone multi-artist product** (Approach B) once the
thesis is proven on Rod's tour.

### Modules
1. **Fan Identity & Verification** — registration; SMS + email verification; device/IP
   fingerprint; one-identity-one-account dedupe. Seeds from the Klaviyo DLD waitlist
   with a `priority` flag (existing fans jump the queue). *Medium strength: blocks ~90%
   of bot/broker volume, low friction, no government ID.*
2. **Presale Access Engine** — runs the access waves; issues unique single-use presale
   access to verified fans; enforces max-qty-per-identity. The gate.
3. **Branded Fan Portal (React)** — Rod-branded: register → verify → access window →
   buy (capped) → TM handoff. Also hosts the resale exchange UI.
4. **Capped Resale Exchange** — verified fan-to-fan resale, **hard-capped at face
   value**, settles via Stripe with hold-until-transfer-confirmed escrow.
5. **Tour Admin Console** — extends the "Rollout" board: per-show inventory, cap
   settings, verification stats, broker-flagging, fan CRM.
6. **Integrations layer** — Ticketmaster (event handoff / transfer), Klaviyo (waitlist
   import + fan comms), Stripe Connect (resale settlement + fees).

## 6. Revenue Model (built in)

| Stream | Mechanism | Notes |
|---|---|---|
| **Resale markup split (DEFAULT, Rod's call 2026-06-16)** | Resale cap = **face + $15 markup**: seller keeps **$10**, Rod takes **$5**. Buyer pays face + $15. | Fan-generous (seller profits a little), Rod earns on every resale, still far below scalper 5×. |
| **Owned fan data** | Every txn writes a verified contact to Klaviyo/CRM | The long-term asset (500K+ verified superfans across 35 shows) |
| **VIP / merch bundles** | Artist-controlled inventory sold direct, uncapped | Rod's margin, his call |
| **Future: licensing** | Re-skin Fair Game for other artists | Approach B extraction |

**Primary tickets are NOT a new revenue stream** — Rod already earns on those via the
tour deal, and capping forgoes the dynamic-pricing upside by choice. Primary = brand/loyalty play.

## 7. Resale Settlement Mechanics (Stripe Connect, marketplace mode)

The ticket moves on TM's transfer rails; the money moves on Stripe. They never touch —
Fair Game **holds the buyer's money until the TM transfer is confirmed**, then pays out.

**Worked example — $60 face seat, cap = face + $15 markup, buyer pays $75:**
- Buyer pays **$75** ($60 seat + $10 seller markup + $5 Rod cut)
- Seller receives **$70** ($60 back + $10 in their pocket — fan profits a little)
- Fair Game/Rod keeps **$5** (less ≈$2.50 Stripe processing → ≈$2.50 net)

Note: whether the $10/$5 split is flat per ticket or scales with face price is an open
tuning question (Section 12). Flat is simplest for v1.

**Flow:**
1. Seller onboards once to a Stripe **Connect Express** account (Stripe runs KYC).
2. Buyer pays full amount → captured into Fair Game's Stripe balance (held).
3. Seller transfers the actual ticket via Ticketmaster.
4. Transfer confirmed (buyer confirms receipt / TM transfer status verified).
5. Only then: Fair Game releases seat price to seller, keeps the fee.
6. If transfer never completes → **auto-refund buyer, seller gets nothing** (fraud wall).

**Cap lever (Rod's call 2026-06-16):** resale cap = **face + $15 markup, split seller $10
/ Rod $5.** The fan profits a little; Rod earns a light cut on every resale. Still
radically below scalper 5×, and never locks a fan down — tickets stay transferable (gift
free, or resell capped through Rod's booth).

## 8. Data Flows

**Primary (seeded inventory):**
`Fan → register → verify → identity record (dedupe + waitlist priority) → presale wave →
capped purchase / TM handoff → fan CRM updated`

**Resale:**
`Verified holder lists (cap enforced ≤ face) → verified buyer pays (Stripe hold) →
TM transfer confirmed → funds released → both records updated`

## 9. v1 Scope Boundaries (YAGNI)

- ❌ No own barcode / entry system (TM rails)
- ❌ No native mobile app (responsive web)
- ❌ No multi-artist theming yet (single-tenant; modules built clean for later extraction)
- ❌ No general event catalog (Rod's 35 DLD shows only)
- ❌ Fair Game does not touch primary money (flows through TM); Stripe only for resale
- ❌ No non-transferable issuance — ever (rejected on principle). Fair capped transfer instead.

## 10. Risks & Dependencies

1. **Ceiling without lock-down (HIGH, accepted by choice):** because tickets stay
   transferable, some scalped tickets can still surface on SeatGeek. Accepted — Rod
   chose fan-fairness over a hard lock. Mitigation: carrots + supply seeding + capped
   fair-transfer through Rod's booth + price pressure.
2. **Inventory volume (HIGH):** impact scales with how much inventory Rod controls. Need
   to confirm the size of artist holds + fan-club allocation per show.
3. **Carrots don't convert profit-scalpers (MEDIUM, by design):** they convert
   good-faith resellers only. Accepted.
4. **Payments / compliance (MEDIUM):** Stripe Connect marketplace = KYC on sellers, tax
   reporting (1099-K thresholds), chargeback handling. Must be designed in, not bolted on.
5. **Legal (MEDIUM):** resale price caps and account/data rules vary by state; review
   ticket-resale law (e.g. NY, CO) before launch.
6. **On-sale load spike (MEDIUM):** verification + access engine must survive a presale
   rush. Load-test required.

## 11. Testing Focus

- Verification dedupe / anti-bot effectiveness
- Cap enforcement (cannot list or buy above cap)
- Single-use / expiry on presale access
- Resale escrow: hold, release-on-confirm, auto-refund-on-failure
- TM transfer-handoff integration
- On-sale spike load test

## 12. Open Questions for Rod / Mike

1. How much inventory does Rod actually control per show (artist holds + fan club)?
2. ✓ DECIDED: cap = face + $15, seller keeps $10, Rod takes $5. Flat per ticket, or scale with face price?
3. ✓ DECIDED: fans keep the upside ($10); Rod's cut is a light $5.
4. Brand name: keep "Fair Game" or Rod-branded (e.g. "Rod Wave Tickets")?
5. Is the non-transferable LN negotiation truly off the table, or revisit after v1 proves out?
