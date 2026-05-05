# Roen's Bracelet Box — Mystery 5-Piece Bundle Spec

**Date**: 2026-05-05
**Owner**: Mike Johnson
**Status**: Approved design, ready for implementation plan
**Sister spec**: `docs/superpowers/specs/2026-05-02-roen-handmade-website-design.md`

## Context

Sarah (the maker behind Roen Handmade) wants a "can't decide" path for customers who don't want to comb the catalog. The product is a curated **mystery 5-bracelet bundle for $25**: Sarah hand-picks five pieces, the customer doesn't know what they'll be until the package arrives, and a personalized printed card explains why she chose those specific five together.

Repeat customers must get a **deliberately different** experience the second time around — different style mix, different note. Customer identity for dedup is the email on the order (case-insensitive); guest checkout is fine because PayPal returns email.

This spec assumes the existing Roen Handmade infrastructure: WordPress + WooCommerce on `roenhandmade-wp` (server-104), `roen-minimal` Storefront child theme, PayPal Payments plugin live, `scripts/roen_telegram_bot.py` running on 105 with a Telegram intake/draft-approval flow Sarah uses daily. **This work is additive — none of that gets refactored.**

This is **not** routed through Oracle. Same isolation rationale as the photo-intake bot: Sarah has zero tolerance for flakiness, and Oracle's freeze risk would compromise that.

## Locked Decisions

These were chosen by Mike during a brainstorming session on 2026-05-05. Recorded as decided, not as proposals.

### 1. Pick workflow — hybrid (system suggests, Sarah swaps)

When a paid order arrives, the system runs the pick algorithm to propose 5 in-stock bracelets and pings Sarah on Telegram with a photo grid + draft note + inline buttons:

- ✅ **Approve** — accept the 5 picks and the draft note as-is
- ✏️ **Swap one** — replies "swap 3"; bot offers a different pick for slot 3, repeat as needed
- 📝 **Edit note** — quote-reply with rewritten text; bot uses Sarah's text verbatim on the card
- 🔄 **Reroll all 5** — picker re-runs with the same constraints

Stock is reserved (set to 0) only when Sarah taps ✅. This avoids holding inventory hostage during her approval window.

### 2. Repeat-customer dedup — different note + style profile (B)

Email is the dedup key (lowercased, exact match, no fuzzy matching). On a repeat order:

- The pick algorithm queries past pick records for that email and **deliberately tilts** the suggested 5 toward color families and style classes the customer has not received before
- The note generator is given the customer's prior notes (last 3) as "themes/openers to avoid" — never repeating greeting language, never explaining the same color story twice
- Sarah always retains final swap; dedup is a soft weighting, not a hard exclusion

### 3. Note delivery — printed card only, Sarah approves before print (A)

No automatic emails. The note exists only as a printed card slipped into the package. Sarah approves the AI-generated note via Telegram before any PDF is rendered. After approval, the bot sends her the print-ready PDF as a Telegram document; she opens it in her phone's PDF viewer and AirPrints to her home printer.

### 4. Inventory eligibility — bracelets only, hide-when-low (A)

The bundle is strictly bracelets. The `/pick` landing page (and the hidden cart product behind it) is visible only when **at least 5 in-stock bracelets** exist in the catalog. If fewer than 5 are eligible, the page renders a "back soon" state and the product cannot be added to cart.

No per-product "eligible for bundle" flag. Every published bracelet is eligible by default.

### 5. Site placement — dedicated landing page at `/pick`, hidden WC product behind it (B)

A custom theme template (`page-pick.php`) renders a storytelling landing page for the bundle. The "Reserve your box" button on that page adds a **hidden WooCommerce product** (slug: `bracelet-box`, $25, excluded from shop archives and search) to the cart. From there, checkout is the existing PayPal flow — no parallel transactional codepath.

The hidden product owns: stock quantity, taxes, order emails, PayPal handoff. The landing page owns: hero copy, photography, FAQ, "how it works" strip.

### 6. Cart limit — multiple bundles per cart, each gets its own pick + note + card (B)

Customers can add the bundle multiple times in one cart (qty 2 → 10 bracelets, two notes, two cards). Each unit becomes its own pick session with the bundle index (1..qty). The dedup logic applies both **across orders for the same email** and **across bundles within the same order** — bundle 2 of an order is deliberately different from bundle 1.

Cart maximum quantity is bounded live by `floor(eligible_in_stock_bracelets / 5)`.

### 7. Naming — "Roen's Bracelet Box"

The product name and the page H1 are "Roen's Bracelet Box." URL stays `/pick` for shortness. Possessive form is intentional — slightly warmer than the locked third-person voice but still brand-name single (no "Sarah," no "we").

### 8. SLA — none, daily nudge (A)

There is no auto-approval. The order sits in `awaiting_sarah` state until she taps ✅. The landing-page copy promises **"ships within 5 business days"** to set the customer expectation. If a pick session has been pending for >24h, the bot sends Sarah one daily reminder ("3 picks waiting on you"). Beyond that, no escalation.

## Architecture

```
                                    ┌──────────────────────────┐
   /pick landing page               │  WooCommerce             │
   (custom template,                │  hidden product:         │
   roen-minimal theme)              │   "Roen's Bracelet Box"  │
        │                           │   $25, stock = floor(    │
        ▼                           │   eligible bracelets / 5)│
   Add to cart  ──────────────────►│  qty up to that floor    │
        │                           └──────────────────────────┘
        ▼
   PayPal checkout (existing flow)
        │
        ▼
   Order placed → wp_action hook fires
        │
        ▼
   For each box-qty in line item:
     1. Pick algorithm queries in-stock bracelets
        with variety + dedup constraints (per email
        history) and proposes 5
     2. roen-bot pings Sarah on Telegram with
        photo grid + draft note + ✅/✏️ buttons
     3. Sarah confirms or swaps; bot loops until ✅
     4. On ✅, system reserves those 5 SKUs
        (sets stock=0), writes pick record to DB,
        renders card PDF, opens it on her phone
     5. She prints, packs, marks WC order shipped
```

Three new components. Everything else (PayPal, order emails, WC cart/stock, the bot's existing draft-approval pattern) is reused as-is.

## Components

### A. Landing page — `services/roen-minimal/page-pick.php`

A custom theme page template, registered as a WordPress Page (slug `pick`). Storefront chrome stays — the page sits inside the existing header/footer.

Sections, top to bottom:
1. **Hero strip** — H1: "Can't decide? Roen will." Sub: "Five hand-picked bracelets. One curated note. $25, shipped within five business days."
2. **Photography** — three-up of mismatched-color bracelet stacks (placeholder marble shots until Sarah produces the real ones; swap when ready)
3. **How it works** — 3-step strip: "you reserve a box → Roen hand-picks five pieces → it ships in five business days with a personal card"
4. **CTA** — "Reserve your box — $25" button (terracotta, the existing brand accent). Adds the hidden WC product to the cart, qty 1, redirects to cart.
5. **Live availability ribbon** — "3 boxes available right now" driven by the cached stock floor; flips to "Roen is restocking — back soon" when 0
6. **FAQ** — four short questions: sizing, returns, gift-friendly?, allergy disclosures
7. **Footer rule** — terracotta hairline, then default footer below

### B. Hidden WooCommerce product — slug `bracelet-box`

Standard WC product with these traits:
- Title: "Roen's Bracelet Box"
- Price: `$25`
- Manage stock: yes; `stock_quantity = floor(eligible_in_stock_bracelets / 5)`
- Catalog visibility: **hidden** (excluded from shop archives, category pages, and search)
- Featured image: a marble-shot bundle photo (placeholder for v1)
- Cart max-qty filter (`woocommerce_quantity_input_args`) bounds the qty selector to current stock

The stock quantity recomputes on:
- A wp-cron every 15 minutes (`roen_box_stock_recompute_cron`)
- The `woocommerce_product_set_stock` hook (instant when an individual bracelet sells)
- The `transition_post_status` hook for bracelet products (instant when Sarah publishes/trashes a piece)

Stock changes use a transient cache (`roen_box_eligible_count`, 60s TTL) to avoid hammering the database on every page render.

### C. roen-bot extension — additive on `scripts/roen_telegram_bot.py`

New handler chain triggered when a WC order pays for the `bracelet-box` SKU. Reuses the existing bot's Telegram allowlist, model wiring, and SQLite db.

New states in the bot's state machine:
- `pick_session_pending` — picks computed, waiting for Sarah to acknowledge
- `pick_session_swapping` — Sarah is mid-swap, slot index known
- `pick_session_note_review` — picks confirmed, note pending Sarah's approval
- `pick_session_approved` — locked, PDF rendered

New Telegram message types:
- **Pick proposal** — media-group of 5 thumbnails (with captioned slot index), then a follow-up text message with the draft note + inline buttons (✅ / ✏️ / 📝 / 🔄)
- **Single-slot swap** — bot replies to "swap 3" with a single new candidate photo + caption "swap to this? yes / try another"
- **PDF delivery** — final document with caption "tap to print"

Daily nudge cron (cron job at 9am ET): `SELECT count(*) FROM roen_bracelet_box_picks WHERE status IN ('suggested','awaiting_sarah') AND created_at < NOW() - INTERVAL 24 HOUR` → if > 0, single Telegram message: "{n} picks waiting on you."

### D. Card-PDF renderer — `services/roen-minimal/templates/card.html` + Python WeasyPrint

Format:
- A6 (105 × 148 mm), single-sided, prints two-up on US Letter cardstock for Sarah
- Cream background (`#FAF9F6`, the existing `--roen-bg-secondary` token)
- Crop marks for clean trimming if Sarah prefers

Layout:
1. **Header strip (centered)**:
   - **Rowan-branch SVG** — single-stroke monoline illustration: curving stem, small pinnate leaves, cluster of 3–4 berries at one end. ~50mm × 25mm, 1pt stroke, terracotta (`#B85C3D`), no fills, no shadows. Authored by hand, lives at `services/roen-minimal/assets/svg/rowan-mark.svg` so the theme can reuse it elsewhere (favicon backup, IG covers, hangtags). The mark is a quiet self-reference to the brand etymology (Sarah's maiden name Rowan → the rowan tree); no customer needs to recognize it for the design to read intentional
   - ~10mm whitespace
   - `roen` wordmark, Inter 200, lowercase, 28pt, dark text, centered
   - Centered hairline rule (~80% width, `#EEEEEE`)
2. **Body (left-aligned)**:
   - Recipient line: "for {first_name}," (lowercase, dark text)
   - Note body: 60–90 words, Inter 400, 11pt, generous line height
   - Piece list: 5 piece names, Inter italic 9pt, indented 1ch
   - Signoff: "with care, roen" or model-chosen variant, Inter 400 11pt
3. **Footer**:
   - Right-aligned: `roenhandmade.com`, 8pt, secondary text color

No QR code, no marketing CTA, no email-capture pitch. The card is the gift, not a billboard.

### E. Pick history table

New SQLite table colocated with the bot's existing intake db (path TBD during plan-writing — likely `data/roen/intake.db` if that's where the existing draft state lives):

```sql
CREATE TABLE roen_bracelet_box_picks (
  id INTEGER PRIMARY KEY,
  order_id INTEGER NOT NULL,           -- WC order id
  line_item_id INTEGER NOT NULL,       -- WC order item id
  bundle_index INTEGER NOT NULL,       -- 1..qty for fan-out within an order
  customer_email TEXT NOT NULL,        -- lowercased, dedup key
  customer_first_name TEXT,
  picked_skus TEXT NOT NULL,           -- JSON list of 5 product ids
  color_tags TEXT,                     -- JSON list, frozen at approval time
  style_tags TEXT,                     -- JSON list
  note_text TEXT NOT NULL,
  status TEXT NOT NULL,                -- suggested | awaiting_sarah | approved | shipped | cancelled
  created_at DATETIME,
  approved_at DATETIME,
  shipped_at DATETIME,
  UNIQUE(order_id, line_item_id, bundle_index)
);
CREATE INDEX idx_email_status ON roen_bracelet_box_picks (customer_email, status);
```

A small WC admin column on the orders list shows pick-session status per order (`suggested` / `awaiting_sarah` / `approved` / `shipped`) so Mike can sanity-check the pipeline without opening Telegram.

## Pick algorithm

### Step 1 — candidate set

```sql
SELECT post_id FROM wp_posts p
JOIN wp_term_relationships tr ON tr.object_id = p.ID
JOIN wp_term_taxonomy tt ON tt.term_taxonomy_id = tr.term_taxonomy_id
WHERE p.post_type = 'product'
  AND p.post_status = 'publish'
  AND tt.taxonomy = 'product_cat'
  AND tt.term_id IN (<bracelets-cat-id>)
  AND p.ID IN (<stock_status=instock AND stock_quantity >= 1>);
```

### Step 2 — extract style fingerprints

The existing vision step in `scripts/roen_telegram_bot.py` runs `qwen3-vl:235b-cloud` over each product photo when Sarah uploads it. Extend that prompt to also emit structured tags:
- `color_family` ∈ {`warm`, `cool`, `neutral`, `mixed`, `statement`}
- `dominant_hex` (one representative color, optional)
- `material_class` ∈ {`beaded`, `metal-chain`, `leather`, `mixed-media`, `gemstone`, `other`}
- `style_class` ∈ {`minimal`, `bohemian`, `statement`, `layering`, `classic`}

These are written as WC product meta (`_roen_color_family`, `_roen_material_class`, `_roen_style_class`, `_roen_dominant_hex`) on first draft. Sarah never sees them; they're for the picker.

**Backfill plan**: a one-time wp-cli script (`scripts/roen_retag_bracelets.py`) re-runs vision on every published bracelet that lacks tags. Sarah does nothing. Runs once before launch and again any time the tag schema changes.

### Step 3 — apply dedup against history

```python
past_picks = db.fetch_all(
    "SELECT color_tags, style_tags, note_text FROM roen_bracelet_box_picks "
    "WHERE customer_email = ? AND status IN ('approved','shipped') "
    "ORDER BY created_at DESC LIMIT 5",
    customer_email.lower()
)
served_color_families = Counter(c for p in past_picks for c in p.color_tags)
served_style_classes = Counter(s for p in past_picks for s in p.style_tags)
```

These counters become the dedup penalty in step 4.

### Step 4 — score & pick 5

Random sample of 5 from the candidate set, weighted by:
- **60% — variety within the 5**: no two picks share both `color_family` AND `material_class`
- **25% — dedup against past orders**: penalize picks whose `color_family` or `style_class` appears in `served_*` counters (deeper penalty for higher counts)
- **15% — fresh inventory bias**: slight nudge toward longer-in-stock pieces so older inventory rotates

If candidate set is exactly 5 (the minimum for the bundle to even be available), all 5 are taken without scoring. If <5, the picker raises `InsufficientStock` — but that case is gated upstream by the "hide button when <5" rule, so it should never fire in practice; if it does, the bot logs a warning and pings Mike (not Sarah).

### Step 5 — package for Sarah

The bot sends:
1. A Telegram media-group of the 5 product photos with captions ("1. {name} — ${price}", "2. {name}…")
2. A follow-up text message: the draft note in plaintext + inline buttons ✅/✏️/📝/🔄

## Note generation

### Model & runtime

- Primary: `kimi-k2.6:cloud` via Oracle's existing copywriter pipeline (same model that already writes Roen's product descriptions, so voice stays consistent)
- Timeout 180s, 1 retry on failure
- Failover: if both attempts fail, the bot sends Sarah a placeholder note ("[note generation failed — please write or tap reroll]") and lets her proceed with her own text

### Prompt inputs

- The 5 picked bracelets: `name`, `short_description`, `color_family`, `material_class`
- Customer's first name (parsed from PayPal billing name; fallback "you")
- Customer's past pick notes for this email, last 3 max — given as "themes/openers to avoid"
- Customer's order count for this email (1, 2, 3+) — drives subtle voice shift

### Output constraints

- **Length**: 60–90 words, single paragraph
- **Voice**: third-person brand voice — "Roen chose…", never "I chose" or "we chose" or "Sarah chose"
- **Specificity**: must mention at least 2 of the 5 bracelets by name or visual detail
- **Throughline**: must state ONE reason this set works as a set — color story, mood, season, contrast, etc.
- **No exclamation points** (per existing Roen voice rules)
- **No emojis** in the printed text
- **Signoff**: model picks one of three variants — "with care, roen" / "yours, roen" / "from the studio"

### Sarah's approval gate

After picks are confirmed, the note appears as plain text under the photo grid with these inline buttons:
- ✅ **Approve** — locks the note as-is
- 📝 **Edit** — Sarah quote-replies with her own rewrite; bot uses her text verbatim on the card
- 🔄 **Reroll** — regenerates with the same inputs (helpful when the model produces something off-tone)

No automatic send. Card PDF only renders after Sarah taps ✅.

## Edge cases

| Case | Behavior |
|---|---|
| Two simultaneous orders draining stock | Picker reserves only on Sarah's ✅; transactional `SELECT … FOR UPDATE` on the 5 SKUs at approval. If one order can't lock, bot sends "stock changed, here are 5 fresh suggestions." |
| Stock drops between order and approval | On Sarah's ✅, system rechecks each suggested SKU; swaps in fresh suggestions for any that became unavailable; asks her to re-confirm. |
| Sarah edits an already-approved note | One-way lock after approval. Re-do requires admin command `/redo <order_id>` (Mike chat-id-gated). |
| Customer cancels or refunds | WC `cancelled`/`refunded` order transition releases the 5 reserved SKUs back to in-stock; pick record marked `cancelled` so it doesn't count toward dedup on future orders. |
| Guest checkout, no email match | First-time customer → no dedup history; picker runs variety-only; new history starts. |
| Customer uses a different email next order | Treated as a new identity. Out of scope to merge — not worth solving. |
| AI note generator times out | Bot sends placeholder text and Sarah hand-writes; no shipment delay. |
| Vision-tagging mis-categorizes a piece | Sarah can override tags via a `/tag <product_id> color=cool style=minimal` admin command in the bot. Tags are advisory; she's still the human-in-the-loop. |

## Testing

### Unit (pick algorithm)
- Fixture of 30 fake bracelets across known tag combinations
- Variety check: no two picks share color_family AND material_class
- Dedup check: second order for same email penalizes prior color families
- Boundary: candidate set == 5 returns all 5 with no scoring
- Boundary: candidate set < 5 raises `InsufficientStock`
- Determinism: fixed RNG seed → snapshot comparison

### Integration (concurrency)
- Two simultaneous "approve" calls on the same SKU pool with stock=5 → exactly one succeeds, the other gets a "stock changed" message
- Crash recovery: kill the bot mid-approval, restart, verify pick session resumes with the same suggested 5 (state in the DB, not in process memory)

### Stock recompute
- Publish a 5th eligible bracelet → assert `bracelet-box` flips from out-of-stock to stock=1 within 15 min (cron) or instantly (hook)
- Sell a bracelet such that eligible drops to 4 → assert box flips to out-of-stock and `/pick` shows "back soon"

### Card PDF
- Snapshot: render PDF from a known data dict, compare PNG of page 1 against a committed reference image (pixel diff < threshold)
- Visual regression on `rowan-mark.svg` in isolation
- Run on every PR touching the template

### End-to-end smoke (manual, once before launch)
- Test order on the live site via PayPal sandbox
- Walk through Sarah's full Telegram flow on a test chat-id
- Print the actual card on actual cardstock and verify it doesn't look cheap
- Confirm WC order email, PayPal receipt, and stock decrement all fire

### Out of scope
- AI note quality — Sarah is the human-in-the-loop
- Vision-tagging accuracy — Sarah can override tags as needed

## Out of scope (explicitly)

- **No shipping integration** — Sarah continues marking orders shipped manually in the WC admin
- **No Meta Catalog / IG sync of the box product** — the box is hidden from the shop and shouldn't surface in the catalog
- **No subscription / recurring delivery** — single-purchase only for v1
- **No gift-message field** — possible follow-up; not in v1
- **No discount codes / multi-box bulk pricing** — flat $25 per box
- **No abandoned-cart recovery** — out of scope; handled by general WC plumbing if at all

## Why this matters

The whole Roen pipeline is built so Sarah does two things per piece: snap photos and name a price. Roen's Bracelet Box is the **first time** that pipeline produces something more than a single product page — it's a curated experience that justifies a $25 price point on what would otherwise be a rummage-sale-priced collection.

The card is the moment that makes the box feel like a gift instead of a discount bundle. The dedup logic is the moment that makes a repeat purchase feel like Sarah remembered you. Neither of those moments can happen without the picker, the AI note, and the printed card all working together — which is why this is a single integrated spec rather than three feature requests.
