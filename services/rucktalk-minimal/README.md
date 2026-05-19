# rucktalk-minimal

Sonaar child theme for `rucktalk.com`. Personal-brand media business — podcast, blog, training, shop, ecosystem cross-promo.

## Specs of record

- **Phase 0 (strategy + IA):** [`docs/superpowers/specs/2026-05-19-rucktalk-rebuild-phase-0.md`](../../docs/superpowers/specs/2026-05-19-rucktalk-rebuild-phase-0.md)
- **Design language (LOCKED 2026-05-19):** [`docs/superpowers/specs/2026-05-19-rucktalk-design-language.md`](../../docs/superpowers/specs/2026-05-19-rucktalk-design-language.md)
- **Implementation plan:** [`docs/superpowers/plans/2026-05-19-rucktalk-phase-1a-site-redesign.md`](../../docs/superpowers/plans/2026-05-19-rucktalk-phase-1a-site-redesign.md)
- **Audit (active theme, parent surface):** [`docs/superpowers/audits/2026-05-19-sonaar-theme-audit.md`](../../docs/superpowers/audits/2026-05-19-sonaar-theme-audit.md)
- **Visual reference (live mockup):** https://aialfred.groundrushcloud.com/static/drafts/rucktalk-homepage-mockup.html

## Aesthetic

**"Editorial Workman."** Magazine-style typography with industrial fonts. Warm paper cream `#F4F0E6`, deep warm ink `#1B1815`, terracotta accent `#B85432`, rare forest-green accent `#2D6741`.

- **Display:** Archivo Black (one weight, hero + section showpieces)
- **Display secondary / italic:** Archivo (regular weights 400-700, italics + sub-heads)
- **Body:** Bricolage Grotesque (300-700, paragraphs + UI)

## Structure

```
rucktalk-minimal/
├── style.css                  Theme header + design tokens (this is what WP sees)
├── functions.php              Bootstrap, asset enqueues, module loader
├── deploy.sh                  T3: rsync + tar-pipe deploy to server-100/rt-wordpress
├── header.php · footer.php    Site chrome (Wave 3)
├── front-page.php             Homepage (Wave 3)
├── page-training.php          /training landing (Wave 4)
├── page-training-free.php     /training/free PDF email-gate (Wave 4)
├── page-about.php             Mike's about page (Wave 3)
├── inc/
│   ├── theme-supports.php     add_theme_support calls (WC + html5)
│   ├── menu-locations.php     Re-uses parent's main-menu (Sonaar already registers)
│   ├── shortcodes.php         [rt_signup] + [rt_ecosystem_strip] + [rt_pillars]
│   ├── rest-signup.php        Brevo double-opt-in signup endpoint
│   ├── airoi-tagger.php       Auto-tag posts that mention AI/business → render AIROI block
│   ├── sonaar-overrides.php   Surgical filter additions; NEVER touch sonaar_feed_*
│   └── pillar-snippets.php    Dynamic "today's take" snippets (Task 13b)
├── templates/parts/           Section partials (Wave 3)
└── assets/{css,js,img,pdf}/   Front-end assets
```

## Critical preservation rules

- Sonaar parent **must stay active** — the `podcast` CPT lives in the parent's bundled `sonaar-music` suite. Swapping parents = no more podcast posts.
- **NEVER touch** these Sonaar filters — they back the podcast RSS feed currently subscribed to by Spotify / Apple / YouTube Music:
  - `sonaar_feed_slug`
  - `sonaar_helper_feed_home_url`
  - `sonaar_podcast_feed_query_args`
- Existing `Main Menu` (term ID 6) on `main-menu` + `responsive-menu` locations stays. This child re-uses it, doesn't compete.

## Deploy

```bash
./deploy.sh
```

T3 per `CLAUDE.md` — every deploy must be triggered by Mike. Script rsyncs `rucktalk-minimal/` to `server-100:/tmp/rucktalk-minimal/`, then tar-pipes into `rt-wordpress:/var/www/html/wp-content/themes/rucktalk-minimal/` and `chown`s to `www-data:www-data`.

Activate after first deploy:
```bash
ssh server-100 "docker exec rt-wordpress wp theme activate rucktalk-minimal --allow-root"
```

Rollback (the existing `sonaar-child` is empty boilerplate — safe to flip back):
```bash
ssh server-100 "docker exec rt-wordpress wp theme activate sonaar-child --allow-root"
```
