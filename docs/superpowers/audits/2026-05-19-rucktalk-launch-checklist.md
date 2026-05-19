# RuckTalk Phase 1A — Launch Checklist & Handoff

**Date prepared:** 2026-05-19
**Phase 0 spec:** [`docs/superpowers/specs/2026-05-19-rucktalk-rebuild-phase-0.md`](../specs/2026-05-19-rucktalk-rebuild-phase-0.md)
**Design language:** [`docs/superpowers/specs/2026-05-19-rucktalk-design-language.md`](../specs/2026-05-19-rucktalk-design-language.md)
**Plan:** [`docs/superpowers/plans/2026-05-19-rucktalk-phase-1a-site-redesign.md`](../plans/2026-05-19-rucktalk-phase-1a-site-redesign.md)
**Branch at handoff:** `feat/lucius-hermes-on-111`

This doc is Mike's launch runbook. Every checkbox is either DONE (✅), BLOCKED ON MIKE, or a T3 action Mike personally executes. Alfred has done everything local-side; the wire from staging → live is Mike's call.

---

## Pre-launch — DONE (✅)

Local-side build for Plan 1A is complete on `feat/lucius-hermes-on-111`. All items below are committed and the theme is deployed (inactive) to `rt-wordpress` on `server-100`.

### Assets landed
- ✅ Real RuckTalk **logo** + **logo-icon** at `services/rucktalk-minimal/assets/img/{logo.png,logo-icon.png,logo-square.png}` — mirrored to container
- ✅ Mike's **hero photo** at `services/rucktalk-minimal/assets/img/mike-hero.jpg`
- ✅ **PDF cover** at `services/rucktalk-minimal/assets/img/pdf-cover.jpg`
- ✅ **Fit as Ruck brand logo** + **8-week-plan PDF** at `services/rucktalk-minimal/assets/pdf/fitasruck-8week.pdf`
- ✅ Theme files deployed (inactive) to `rt-wordpress` — 38 files via `services/rucktalk-minimal/deploy.sh`

### Content seeded
- ✅ Pillar snippet pool seeded: 5 pillars × 4 lines = 20 entries — verify via:
  ```bash
  ssh server-100 "docker exec rt-wordpress wp option get rt_pillar_snippets --allow-root"
  ```

### Integration wiring (WP options set on rt-wordpress)
- ✅ **LoovaCast** — stream URL `https://studiob.loovacast.com/public/news_mews`, station ID `22`
- ✅ **LumaBot** — script URL `https://chatui-dev.groundrushlabs.com/...`, tenant + base URL set ⚠️ DEV subdomain — see "Pre-launch BLOCKED" below
- ✅ **Brevo** — RuckTalk list (id `6`) + double-opt-in template (id `2`) created in Mike's GroundRush Brevo account; API key + list ID + template ID options set

### Code shipped (commits on `feat/lucius-hermes-on-111`)
- ✅ Wave 1: Sonaar audit
- ✅ Wave 2: child theme scaffold (`rucktalk-minimal`)
- ✅ Wave 3: header / footer / shortcodes / pillar snippets / partials / front-page / CSS port
- ✅ Wave 4: training landing + free-PDF gate + REST signup (Brevo double-opt-in) + signup.js
- ✅ Wave 5: real assets + LumaBot wiring + AIROI auto-tagger (`inc/airoi-tagger.php`)
- ✅ Wave 6: defensive `rucktalk-redirects` mu-plugin + ecosystem strip + post-launch smoke script

---

## Pre-launch — BLOCKED ON MIKE

These are NOT Alfred-buildable — they need Mike's call before launch is safe.

- [ ] **WC product for $29 8-Week Plan** (Plan 1A Task 16) — needs Mike to confirm the exact price + create the product so `rt_training_product_id` WP option can be set. CLI:
  ```bash
  ssh server-100 "docker exec rt-wordpress wp wc product create \
    --name='8-Week RuckTalk Plan' \
    --type=simple \
    --regular_price=29 \
    --downloadable=true \
    --virtual=true \
    --status=publish \
    --user=1 \
    --allow-root"
  ssh server-100 "docker exec rt-wordpress wp option update rt_training_product_id <ID> --allow-root"
  ```
- [ ] **WP pages — `/training` + `/training/free`** — need `wp post create` with the page templates the Wave 4 work shipped (`page-training.php`, `page-training-free.php`):
  ```bash
  ssh server-100 "docker exec rt-wordpress wp post create \
    --post_type=page --post_status=publish \
    --post_title='Training' --post_name=training \
    --page_template=page-training.php --allow-root"
  ssh server-100 "docker exec rt-wordpress wp post create \
    --post_type=page --post_status=publish \
    --post_title='Free 8-Week Plan' --post_name=free \
    --post_parent=<training-page-id> \
    --page_template=page-training-free.php --allow-root"
  ```
- [ ] **LumaBot URL is `chatui-dev.groundrushlabs.com`** — DEV subdomain. Flip to production URL (`chatui.groundrushlabs.com` or whatever Mike picks) before public traffic hits the page. Update via:
  ```bash
  ssh server-100 "docker exec rt-wordpress wp option update rt_lumabot_script_url 'https://chatui.groundrushlabs.com/...' --allow-root"
  ```

---

## Launch (T3 — Mike approves each)

Each block is one explicit Mike approval. Run in order.

### L1. Activate the theme
```bash
ssh server-100 "docker exec rt-wordpress wp theme activate rucktalk-minimal --allow-root"
```
Verify: visit https://rucktalk.com/ — see new editorial layout.

### L2. Deploy the legacy redirects mu-plugin
```bash
# from /home/aialfred/alfred on 105
scp services/rucktalk-redirects/rucktalk-redirects.php server-100:/tmp/
ssh server-100 "docker exec rt-wordpress mkdir -p /var/www/html/wp-content/mu-plugins"
ssh server-100 "tar -C /tmp -cf - rucktalk-redirects.php | docker exec -i rt-wordpress tar -C /var/www/html/wp-content/mu-plugins -xf -"
ssh server-100 "docker exec rt-wordpress chown www-data:www-data /var/www/html/wp-content/mu-plugins/rucktalk-redirects.php"
```
Smoke:
```bash
curl -sI https://rucktalk.com/8-week-plan/ | head -5
# expect: HTTP/2 301  +  location: https://rucktalk.com/training/8-week-plan/
```

### L3. Cloudflare 301 — fitasruck.com → rucktalk.com/training
Cloudflare dashboard → fitasruck.com zone → Rules → Page Rules:
- URL pattern: `*fitasruck.com/*`
- Action: Forwarding URL → **301 Permanent Redirect** → `https://rucktalk.com/training/$2`

(Plan 1A Task 23 Strategy A — preferred. Strategy B mu-plugin fallback documented in plan if fitasruck.com is not Cloudflare-fronted.)

### L4. Fix the 525 on www.rucktalk.com
Re-issue the Let's Encrypt cert with a `www` SAN so the apex + www both serve cleanly, then add the www → apex 301. Exact steps per Plan 1A Task 24 — depends on whether nginx-proxy / Traefik / Caddy is fronting `rt-wordpress` on `server-100`. Verify with:
```bash
curl -sI https://www.rucktalk.com/ | head -5     # expect 200 or 301 (not 525)
```

### L5. Search Console — Change of Address for fitasruck.com
Search Console → property `https://fitasruck.com/` → Settings → Change of address → new site `https://rucktalk.com/` → Submit.

### L6. Re-submit rucktalk.com sitemap in Search Console
```bash
curl -s https://rucktalk.com/sitemap.xml | head -10
# verify it includes /training/, /training/free/, /training/8-week-plan/
```
Then GSC → Sitemaps → Submit.

---

## Post-launch (within 1 hour)

- [ ] **Smoke script** — all green:
  ```bash
  cd /home/aialfred/alfred && venv/bin/python scripts/rucktalk_redesign_smoke.py
  ```
  (Use `--skip-redirects` if you're running this before L3/L4 finish propagating.)
- [ ] **Visual eyeball in incognito** — homepage hero, sticky radio bar, popup fires on scroll-depth 50%, footer ecosystem strip renders 5 logos in greyscale
- [ ] **Real test signup** — submit your own email → confirmation arrives → click verify → PDF lands → contact appears in Brevo `RuckTalk` list as `confirmed`
- [ ] **$29 test order via Stripe test mode** — confirm WC creates the order + Stripe charges + customer email fires
- [ ] **`/sitemap.xml` 200** — confirms Sonaar/RankMath sitemap is generating with the new pages
- [ ] **`www.rucktalk.com` 200** — no 525
- [ ] **`fitasruck.com` 301** — `curl -sIL https://fitasruck.com/ | head -10` → ends at `rucktalk.com/training/`

## Post-launch (within 48 hours)

- [ ] **GA4** — pageviews flowing on `/`, `/training/`, `/training/free/`, `/blog/`, `/podcast/` (Realtime → Pages report)
- [ ] **Brevo signup-receive** — contacts arriving with `SIGNUP_PLACEMENT` attribute populated (hero / footer / popup / inline)
- [ ] **n8n webhook firing** — workflow `o9cIjGWj8z9pwknY` at `https://automate.groundrushlabs.com/workflow/o9cIjGWj8z9pwknY` should show new executions when Brevo confirmation webhook fires
- [ ] **rt-wordpress error log** — no PHP fatals:
  ```bash
  ssh server-100 "docker exec rt-wordpress tail -100 /var/log/apache2/error.log"
  ```
- [ ] **Search Console fitasruck.com Change of Address** — status reads "Processing" or "Successfully verified"

---

## Rollback path

If launch goes sideways, revert in two commands:

```bash
# Revert to Sonaar (parent theme)
ssh server-100 "docker exec rt-wordpress wp theme activate sonaar-child --allow-root"

# Remove the mu-plugin
ssh server-100 "docker exec rt-wordpress rm -f /var/www/html/wp-content/mu-plugins/rucktalk-redirects.php"
```

Cloudflare 301 for fitasruck.com is reversible via the same Page Rules UI used in L3 (toggle the rule off or delete it).

Brevo + LumaBot + LoovaCast WP options stay set — no rollback needed; they only do anything when the theme is active.

---

## Open follow-ups (moved to later plans)

These are intentionally NOT in Phase 1A scope:

- **Phase 1B** — Encode RuckTalk brand profile in Alfred SEO, run DataForSEO keyword discovery on `rucktalk.com`, kill Rank Math (we're standardized on Alfred SEO per Roen pattern), wire RuckTalk into the weekly SEO blog engine, post-redesign category cleanup
- **Phase 1C** — Podcast distribution audit per Phase 0 §10.5 (Spotify for Podcasters / Apple Podcasts Connect / YouTube Music for Podcasts feed reconciliation, orphan-show cleanup, weekly Sunday cron to validate each platform's "last episode date" matches rucktalk.com)
- **Phase 2** — Real shop product photography + add real SKUs (current homepage shop teaser is empty until products exist)
- **Phase 3** — Newsletter sequence copy + segment rules + A/B variants per Phase 0 §5f
- **Phase 4** — Social repurposing pipeline (NotebookLM → reels)
- **Phase 5** — YouTube long-form pipeline (NextCloud watcher extension)

---

## Handoff signal

When all of pre-launch BLOCKED ON MIKE is cleared + Mike says "go," execute L1 → L6 in order, then work the post-launch checklists. If anything red on the smoke script, fix or roll back per the rollback path — don't leave it half-cut-over.
