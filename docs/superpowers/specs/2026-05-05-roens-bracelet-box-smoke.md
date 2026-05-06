# Roen's Bracelet Box — Pre-launch Smoke Test

**Run this before announcing the bundle. All steps must pass.**

This is the only manual step the implementation plan leaves to you. Everything else has been implemented and committed; smoke is the final gate before customers can hit `/pick`.

---

## Prerequisites — one-time setup

These must be done by Mike before the smoke test:

- [ ] **Generate WooCommerce REST API keys** at `WP Admin → WooCommerce → Settings → Advanced → REST API → Add key`. Permission: **Read/Write**, owner: any admin user.
- [ ] **Add to `config/.env` on server-105:**
  ```
  WC_ROEN_KEY=ck_<consumer_key>
  WC_ROEN_SECRET=cs_<consumer_secret>
  ```
- [ ] **Run the bootstrap script (creates the hidden `bracelet-box` product):**
  ```bash
  python3 scripts/roen_create_box_product.py
  ```
  Expected: `CREATED: bracelet-box (id=NNN)` (first run) or `OK: bracelet-box already exists`.
- [ ] **Run the backfill script (re-tag existing bracelets if any):**
  ```bash
  python3 scripts/roen_retag_bracelets.py --dry-run
  python3 scripts/roen_retag_bracelets.py
  ```
- [ ] **Deploy theme changes to server-104:**
  ```bash
  bash services/roen-minimal/deploy.sh
  ```
- [ ] **Trigger the WP cron once to set initial box stock:**
  ```bash
  ssh root@75.43.156.104 'cd /var/www/html && wp cron event run roen_box_stock_recompute_cron --allow-root'
  ```
- [ ] **Restart the bot to pick up the new schema and poll thread:**
  ```bash
  sudo systemctl restart roen-bot
  journalctl -u roen-bot -f -n 50
  ```
  Expected log line: `bracelet-box poll thread started (60s interval)`.
- [ ] **Install the daily nudge cron** at `/etc/cron.d/roen-box-nudge`:
  ```bash
  sudo tee /etc/cron.d/roen-box-nudge <<'EOF'
  0 13 * * * aialfred /usr/bin/python3 /home/aialfred/alfred/scripts/roen_box_nudge.py >> /home/aialfred/alfred/data/roen/box_nudge.log 2>&1
  EOF
  ```
  (13:00 UTC = 9am ET. Adjust if Mike or Sarah are elsewhere.)
- [ ] **Create the WP Page** at `WP Admin → Pages → Add New → Title "Pick" → Page Attributes → Template "Roen's Bracelet Box (/pick)" → URL slug `pick`** so `/pick` resolves.

---

## Setup verification

- [ ] At least 5 published bracelets exist in product_cat=bracelets, all with `_roen_color_family`/`_roen_material_class`/`_roen_style_class`/`_roen_dominant_hex` meta keys
- [ ] WC admin → Products → "Roen's Bracelet Box" exists, hidden, $25, stock = floor(eligible/5)
- [ ] `/pick` page renders with the rowan mark, hero copy, FAQ, and CTA
- [ ] `Reserve your box — $25` button is enabled (stock > 0)

## Order placement

- [ ] Visit `/pick`, click Reserve, get redirected to `/cart`
- [ ] Cart shows "Roen's Bracelet Box", $25, qty 1
- [ ] Checkout via PayPal completes
- [ ] WC order is created with status=processing
- [ ] PayPal receipt arrives in customer email

## Bot side

- [ ] Within 60s, Sarah receives a Telegram message with 5 thumbnails + draft note + 4 inline buttons (✅/✏️/📝/🔄)
- [ ] `roen_bracelet_box_picks` row exists in `data/jewelry.db` with status `awaiting_sarah`:
  ```bash
  sqlite3 /home/aialfred/alfred/data/jewelry.db \
    "SELECT id, order_id, status FROM roen_bracelet_box_picks ORDER BY id DESC LIMIT 5;"
  ```

### Approval flow buttons

- [ ] Tap **✏️ Swap one** → reply `swap 3` → bot offers a different slot 3 piece + new keyboard
- [ ] Tap **📝 Edit note** → quote-reply with custom text → bot confirms note saved + new keyboard
- [ ] Tap **🔄 Reroll** → 5 fresh suggestions arrive, note regenerates
- [ ] Tap **✅ Approve** → bot reserves the 5 SKUs (their stock_status flips to `outofstock` in WC), pick row → `approved`, PDF document arrives in Telegram

### Card quality

- [ ] PDF opens cleanly on Sarah's phone
- [ ] Rowan mark renders crisply, terracotta
- [ ] Wordmark centered, Inter 200
- [ ] Body left-aligned, 60–90 words
- [ ] Piece list italicized
- [ ] Footer URL right-aligned, faint
- [ ] Print on actual cardstock looks good

## Stock floor

- [ ] After approval, eligible bracelet count drops by 5 → box stock decrements by 1 within 15min cron (or instantly via the woocommerce_product_set_stock hook)
- [ ] If eligible count drops below 5 after approval, `/pick` button disables and shows "Roen is restocking"

## Repeat-customer

- [ ] Place a second order with the same email
- [ ] Verify the picker's suggestions skew away from color families used in the prior order
- [ ] Verify the new note doesn't repeat the prior note's opener or theme

## Cancellation

- [ ] Cancel the test WC order from WP admin → box stock recomputes upward, an admin note is added on the order

## Daily nudge

- [ ] Manually run `python3 scripts/roen_box_nudge.py` while a pick is pending >24h → Sarah receives a "X picks waiting on you" message
- [ ] Run it again with no pending → no message sent (silent success)

## WC admin column

- [ ] WC admin → Orders list shows a 📦 column with the qty for box-containing orders, `—` for everything else

---

## If something fails

| Symptom | First thing to check |
|---|---|
| Bot doesn't message Sarah after order | `journalctl -u roen-bot -n 100` — look for `box poll iteration crashed` or `open_pick_session failed` |
| `wc_get_product_id_by_sku('bracelet-box')` returns 0 | Did the bootstrap script run? Was the SKU `bracelet-box` (no spaces, no caps)? |
| WC API 401s | `WC_ROEN_KEY`/`WC_ROEN_SECRET` missing from `config/.env` or the WP user owning the keys was deactivated |
| PDF renders without the rowan mark | `services/roen-minimal/assets/svg/rowan-mark.svg` not deployed to 105? Check `ls -la` |
| Tag values look uniform across all bracelets | Vision step is returning fallback defaults — check Ollama relay status, run `--dry-run` retag to see what model says |
| Pick suggestions never vary | Catalog might have <30 bracelets and the 30/30 fixture in the picker test was permissive — that's fine for v1 |

---

## After smoke passes

- [ ] Mike posts about the bundle (existing social tooling)
- [ ] Watch `journalctl -u roen-bot -f` for the first real customer order
- [ ] Watch `data/roen/box_nudge.log` daily for the first week to confirm cron is firing
