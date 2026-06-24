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
2. **Album announcement** — promote Rod's upcoming album. Embed the YouTube announce video +
   "Notify me" capture. *(OPEN: need YouTube link + album name from Mike.)*
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
7. **Admin CMS (NEW — Mike's ask)** — back-office to **change prices, fees, and inventory** per
   event/seat block ourselves, toggle availability, manage Discover sources, view orders + transfer
   status. Grows from existing `data/mainstay/fairgame/app/admin.html`.

## 4. Supply & fulfillment (from research — `scratchpad/RESEARCH_tm_transfer.md`)
- Rod's seats live in a **standard consumer Ticketmaster account** we have access to.
- **SafeTix = transfer-only.** No PDF/Wallet export. TM has **no transfer API**; TradeDesk (the only
  API path) was shut down by the FTC (Oct 2025).
- **Transfers usually locked until ~72h before each show** → product is **"buy now, delivered to
  your TM account a few days before the show."**
- **MVP fulfillment:** operator transfers each order via the TM web UI in batches when the window
  opens; ingest TM "accepted" emails to confirm delivery. Keep it **human-paced** — high volume on
  one account risks a flag that cancels Rod's tickets.
- **Tracking model:** order_id · event · seat · buyer_tm_email · transfer_sent_at · accepted_status.
- **Checkout must collect** the buyer's Ticketmaster account email (+ phone/consent).

## 5. Discover data (from research — `scratchpad/RESEARCH_aggregator_apis.md`)
- "Lowest anywhere" is **not defensible from any single source**; SeatGeek's API **bans** displaying
  rival listings. Honest framing = **"lowest verified price as of now" / guaranteed-fulfillment.**
- **Recommended stack:** Ticketmaster Discovery (have it) + **TicketsData (~$499/mo, 10-market
  compare feed)** as the cross-market core; **TickPick** partner API for the cleanest all-in
  "verifiable" price; affiliate networks (Impact/Partnerize/CJ) as the redirect/monetization layer.
- **Decision pending:** budget the $499/mo TicketsData feed, or launch with TM + TickPick affiliate.

## 6. Revenue
1. Rod-held seats: face value + fair fee (seller/Rod split per existing fairgame economics).
2. Discover **$1 unlock** per reveal + affiliate commission on outbound buy-links.

## 7. Out of scope (this phase)
Native mobile apps · P2P fan-to-fan resale (killed → later phase) · buyer identity verification ·
click-through for non-inventory (grey) seats.

## 8. Open items (need Mike / decisions)
- [ ] YouTube album-announcement link + album name (placeholder wired in mock).
- [ ] TicketsData $499/mo — approve, or launch on TM + TickPick only.
- [ ] Confirm gold as the final accent (approved in mock).
- [ ] Fee structure / price points per market for the CMS to manage.

## 9. Next step
Take the approved homepage design into the real app (`data/mainstay/fairgame/app/`), then build the
remaining screens to `DESIGN.md` in this order: **seat-map page → checkout (no-refund + TM email) →
Discover $1 paywall → accounts/My Tickets → admin CMS**. Proceed via the writing-plans skill.
