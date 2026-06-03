# Roen Handmade Bot — Friends & Family Coupon Flow

**Date:** 2026-06-03
**Status:** Approved design, pending build
**Surface:** `scripts/roen_telegram_bot.py` (Telegram intake bot for Roen Handmade)

## Purpose

Let Sarah create WooCommerce discount codes from the Telegram bot — for friends,
family, or anyone she wants to treat — without touching the WooCommerce admin.
She names the code (usually after the person), picks the discount, and the bot
creates it live and hands her a confirmation she can forward.

## Who can use it

Sarah and Mike only — the bot's existing `ROEN_INTAKE_ALLOWED_CHAT_IDS`
allowlist already enforces this. No additional gating needed; coupon commands
are simply available to any allowed chat.

## Interaction model

Natural-language, conversational. Sarah can give everything at once or in pieces.

**All-at-once (fast path):**
- `create a coupon 20% off Brittany` → code **BRITTANY**, 20% off
- `create a coupon $100 off Mom` → code **MOM**, $100 off
- `create a coupon free shipping Kelly` → code **KELLY**, free shipping
- `create a free coupon Sarah` / `100% off Sarah` → code **SARAH**, fully comped

**Two-step (if the name is missing):**
- Sarah: `create a 20% off coupon`
- Bot: *"What do you want to call it?"* (sets a short-lived pending-coupon state,
  same pattern as `_pending_edit` / `_pending_ship`)
- Sarah: `Brittany` → bot creates **BRITTANY**

**Optional modifiers** appended in plain text:
- `one time` / `single use` → single redemption (otherwise unlimited)
- `good for a month` / `expires June 30` / `30 days` → sets an expiry date
  (otherwise no expiry)

## Discount types supported

| Phrase | WooCommerce `discount_type` | Notes |
|---|---|---|
| `20% off`, `20 percent` | `percent` | amount = 20 |
| `$15 off`, `15 dollars off` | `fixed_cart` | amount = 15 |
| `free`, `100% off`, `comp` | `percent` | amount = 100 |
| `free shipping` | (any) + `free_shipping=true` | amount 0; can combine, but default standalone |

## Defaults (Mike-approved)

- **Usage limit:** unlimited (reusable). It's a friends & family tool — that's the
  common case. `one time` / `single use` keyword flips it to `usage_limit=1`.
- **Expiry:** none unless she specifies one.
- **Per-user limit:** none (a friend reusing their own code is the point).
- **Individual use:** `true` (can't be stacked with other coupons — keeps the
  discount predictable).

## Code naming

- Code = the name she gives, normalized: uppercased, trimmed, spaces → hyphens,
  strip anything that isn't `A–Z 0–9 - _`. (`my sister` → `MY-SISTER`.)
- If the resulting code already exists in WooCommerce, the bot tells her and asks
  for a different name (WooCommerce coupon codes must be unique) rather than
  silently clobbering an existing code.
- If she gives no name and the two-step prompt also comes back empty/unusable,
  the bot keeps asking rather than auto-generating — she's naming these for people.

## Commands

- `create a coupon ...` / `make a coupon ...` / `/coupon ...` → create flow
- `my coupons` / `/coupons` → list active codes (code, discount, uses left,
  expiry), each with a 🗑️ **Delete** inline button
- Delete button → `cpn:del:<code>` callback → removes the coupon in WooCommerce

## Implementation notes

- New module `core/jewelry/coupons.py`:
  - `parse_coupon_request(text) -> CouponSpec | None` — pure parser, unit-testable,
    no I/O. Returns discount_type, amount, code (maybe None), usage_limit,
    free_shipping, expiry_date.
  - `create_coupon(spec) -> CouponResult` — shells to wp-cli
    (`wp wc shop_coupon create ...`) via the same SSH+docker `_wp` helper pattern
    already in `orders.py`. Reuse/share that helper rather than duplicating it.
  - `list_coupons()` / `delete_coupon(code)` — `wp wc shop_coupon list/delete`.
  - `coupon_exists(code) -> bool` — uniqueness check before create.
- Bot wiring in `roen_telegram_bot.py`:
  - `handle_text`: recognize create/list phrasing; manage `_pending_coupon` state
    (mirror `_pending_edit`: dict + lock + TTL ~5 min, consumed by next text).
  - `handle_callback_query`: handle `cpn:del:<code>`.
  - Add coupon lines to `/help`.
- Confirmation message format:
  ```
  ✅ Coupon created, Sarah
  Code: BRITTANY
  20% off · unlimited use · no expiry
  Share it with whoever you like.
  ```
  (Single-use / expiry reflected when set.)

## Parsing strategy

Deterministic regex/keyword parsing — NOT an LLM call. The phrase space is small
and bounded (percent, dollar, free, free shipping, name, one-time, expiry), so
regex is faster, free, and predictable. Order of extraction:
1. Detect `free shipping` → free_shipping flag.
2. Detect `%`/`percent` → percent + amount; or `free`/`100%`/`comp` → percent 100.
3. Else detect `$`/`dollar` → fixed_cart + amount.
4. Strip recognized modifiers (`one time`, expiry phrases) and discount tokens;
   the remaining words are the candidate code name.
5. If no discount token found → treat as malformed, ask what discount she wants.

## Testing

- Unit tests for `parse_coupon_request` covering every example phrase above plus
  edge cases (name with spaces, missing name, missing discount, `one time`,
  each expiry phrasing, free shipping, 100%/free).
- Manual smoke on the live bot: create one of each type, confirm in WooCommerce
  admin, redeem one on a test order, delete via `my coupons`.

## Out of scope (YAGNI)

- Product-/category-specific coupons (cart-wide only).
- Minimum-spend thresholds.
- Scheduling future start dates.
- Editing an existing coupon (delete + recreate instead).
