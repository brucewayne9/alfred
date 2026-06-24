# Fans First — Redesign & Product Spec

**Date:** 2026-06-24 · **Branch:** `feat/fairgame` · **Owner:** Mike Johnson (Rod Wave partnership, Mainstay Tech)
**Status:** Design language APPROVED. Product workstreams scoped, pending build plan.

## 1. What this is
Fans First is Rod Wave's **fan-first, anti-scalper resale platform** for the Don't Look Down Tour
2026. Two products under one roof:
1. **Rod-held seats** — premium inventory Rod pulled from Ticketmaster, sold to fans at face value
   + one fair fee. *"The best seats Rod ever held. Only here."*
2. **Discover** — search any event; we surface the **lowest *verified* price, guaranteed**, and (for
   a **$1 unlock**) reveal direct buy-links across marketplaces. Camouflage/aggregator play.

Web-only this phase, fully responsive. **No native iOS/Android app yet.**

## 2. Design language — APPROVED
Locked in `data/mainstay/fairgame/DESIGN.md`. Summary: Spotify-inspired but **lightened** —
cinematic dark hero/footer, warm light editorial body, **one gold accent** pulled from Rod's tour
art, `Figtree` type, **squared icon-driven buttons**, **line SVG icons (no emoji)**, Rod's
photography as the color. Reference mock (homepage, approved):
https://aialfred.groundrushcloud.com/drafts/fansfirst-mock/
Decisions reached iteratively: dark→too dark→**dark hero + light body**; bright multi-color→**rejected
as slop**→single gold; pill→**squared** buttons; emoji→**banned**; **flood with Rod photography**;
add **album-announcement** promo to sell the partnership.

## 3. Pages / workstreams
1. **Home / hero** — DONE in mock. Co-brand lockup, hero (ad-mat), photo gallery, tour rail,
   seat-map preview, Fans First promise, album promo, Discover teaser, footer.
2. **Album announcement** — promote **"Don't Look Down," out Aug 28, 2026** (Rod's 7th, Alamo).
   Official trailer **embedded** (lightbox): https://www.youtube.com/watch?v=VpMAwDs3oI0 +
   "Notify me" email capture. RESOLVED.
3. **Event / seat map (full page)** — full arena via existing real TM seat maps
   (`core/fairgame/seatmap.py`, 15 markets / 258K seats). Green = ours/available → Red = ours/sold;
   everything else **greyed = "not available here"** (no click action this phase). Legend + scarcity.
4. **Checkout (trust-armored)** — TM account **required** (delivery + tracking). **Triple no-refund
   acknowledgment**, worded warmly ("all sales final — make sure this is the one"). Sets expectation:
   *delivered to your Ticketmaster ~3 days before the show.*
5. **Discover + $1 paywall** — search → results blurred → $1 Stripe unlock → verified listings +
   direct buy-links. Reuses the proven SpotGate paywall mechanic. Honest claim: **"lowest verified
   price, guaranteed,"** never "lowest anywhere."
6. **Accounts / My Tickets** — sign up/login, order history, delivery countdown, "my tickets."
7. **Admin CMS (Mike's ask, CONFIRMED)** — back-office to **change prices, fees, and inventory** per
   event/seat block ourselves, toggle availability, manage Discover sources, view orders + transfer
   status. **Ships with editable defaults; Mike sets prices/fees in the back.** Grows from existing
   `data/mainstay/fairgame/app/admin.html`.

## 4. Supply & fulfillment (from research — `scratchpad/RESEARCH_tm_transfer.md`)
- Rod's seats live in a **standard consumer Ticketmaster account** we have access to.
- **SafeTix = transfer-only.** No PDF/Wallet export. TM has **no transfer API**; TradeDesk (the only
  API path) was shut down by the FTC (Oct 2025).
- **Transfers usually locked until ~72h before each show** → product is **"buy now, delivered to
  your TM account a few days before the show."**
- **Fulfillment = Option B (Mike's call): a supervised assist tool**, NOT pure-manual. A human
  approves each batch; the tool drives the TM transfer flow but **throttled + human-in-the-loop**,
  **never headless-at-scale** (high volume on one account risks a flag that cancels Rod's tickets).
  Required safeguards: per-batch human approval, pacing/jitter between transfers, pre-check each
  show's transfer-window eligibility before firing, secure the account's 2FA, ingest TM "accepted"
  emails to confirm delivery. (Manual queue = the fallback if the tool ever gets risky.)
- **Tracking model:** order_id · event · seat · buyer_tm_email · transfer_sent_at · accepted_status.
- **Checkout must collect** the buyer's Ticketmaster account email (+ phone/consent).

## 5. Discover data (from research — `scratchpad/RESEARCH_aggregator_apis.md`)
- "Lowest anywhere" is **not defensible from any single source**; SeatGeek's API **bans** displaying
  rival listings. Honest framing = **"lowest verified price, guaranteed."**
- **DECISION (Mike):** **TicketsData $499/mo = KILLED.** No subscription costs. Discover runs on
  **free/affiliate sources only — Ticketmaster Discovery (have it) + TickPick affiliate** (+ Impact/
  Partnerize/CJ as the outbound redirect/monetization layer).
- **The only Discover economics = the $1 unlock per transaction.** One dollar, one time, per reveal.

## 6. Revenue
1. Rod-held seats: face value + fair fee (set/controlled via the admin CMS).
2. Discover: **$1 unlock per reveal** (the only Discover charge — no feed subscription) + affiliate
   commission on outbound buy-links.

## 7. Out of scope (this phase)
Native mobile apps · P2P fan-to-fan resale (killed → later phase) · buyer identity verification ·
click-through for non-inventory (grey) seats.

## 8. Open items
- [x] ~~YouTube album link + name~~ → "Don't Look Down," Aug 28, trailer embedded.
- [x] ~~TicketsData $499/mo~~ → KILLED. $1-unlock only; TM + TickPick affiliate sources.
- [x] ~~Confirm gold accent~~ → approved.
- [x] ~~Fee/price points~~ → handled in the CMS (Mike sets them; ships with editable defaults).
- [x] ~~Ticket delivery~~ → **Option B: supervised assist tool** (throttled, human-in-the-loop),
  manual queue as fallback. See §4.

**All open items resolved → ready for build planning.**

## 9. Next step
Take the approved homepage design into the real app (`data/mainstay/fairgame/app/`), then build the
remaining screens to `DESIGN.md` in this order: **seat-map page → checkout (no-refund + TM email) →
Discover $1 paywall → accounts/My Tickets → admin CMS**. Proceed via the writing-plans skill.
