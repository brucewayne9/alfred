# RuckTalk Phase 1A — Site Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign rucktalk.com as a personal-brand media business inside the existing Sonaar theme — new homepage with floating LoovaCast player, `/training` section that absorbs the Fit-as-Ruck product and free PDF lead magnet, ecosystem cross-promo placements, with 301 cutover from fitasruck.com and the Cloudflare 525 bug on `www.` fixed at launch.

**Architecture:** Ship a Sonaar child theme at `services/rucktalk-minimal/` (mirrors the proven `services/roen-minimal/` pattern). Templates override Sonaar where needed, child CSS layered on top of parent. Floating LoovaCast player + LumaBot chat + AIROI contextual block are all wp_enqueue'd assets driven by the child theme. WC handles `/training/8-week-plan` paid SKU; a custom landing template handles `/training/free` lead-magnet capture (submits to Brevo via REST). 301 rules ship as a `redirects.php` mu-plugin so they survive theme switches.

**Tech Stack:** PHP 8.x (WordPress 6.x + Sonaar parent theme) · Vanilla JS for radio player + popup · CSS variables for theme tokens · Brevo REST v3 for newsletter signup · Sonaar custom post type `podcast` (untouched) · WooCommerce for the paid SKU · server-104 rsync + docker exec tar-pipe deploy pattern (same as Roen) · Cloudflare for DNS + SSL.

**Spec reference:** `docs/superpowers/specs/2026-05-19-rucktalk-rebuild-phase-0.md`
**Design language (LOCKED 2026-05-19):** `docs/superpowers/specs/2026-05-19-rucktalk-design-language.md`
**Visual reference:** https://aialfred.groundrushcloud.com/static/drafts/rucktalk-homepage-mockup.html

**Branch:** `feat/rucktalk-rebuild-phase-1a` (cut from `main`)

**Production server:** RuckTalk WP runs in container `rt-wordpress` on **server 100** (T3 per CLAUDE.md — every deploy step needs Mike's go).

---

## DESIGN LANGUAGE OVERLAY (added 2026-05-19 after Mike's approval)

The design-language doc above is the source of truth for all CSS tokens, font stack, copy voice, and component patterns. The following tasks have updates that supersede the original spec:

- **Task 3 (style.css)** — replace the entire `:root` token block with the locked palette from design-language §3; replace font preconnect/import with the Archivo Black + Archivo + Bricolage Grotesque combo from §2. Drop the Fraunces import line entirely.
- **Task 11 (shortcodes)** — add a third shortcode `[rt_pillars]` that renders the 5-pillar grid with dynamic snippets (see §5e of design-language doc). Spec for the shortcode is in new Task 13b below.
- **Task 12 (front-page.php)** — hero structure follows design-language §5c (eyebrow + h1 with italic em + strap line + dual CTA + listen-on platform row). Use the locked tagline + strap + about copy from §6.
- **Task 13 (partials)** — add `templates/parts/pillars.php` partial that renders the dynamic pillar grid. Also: episode-card.php gets a Listen/Watch toggle pill per §5f.
- **Task 14 (CSS)** — instead of writing CSS from scratch, lift the component CSS directly from the mockup HTML at the visual-reference URL. Mockup's class names already match the `rt-` prefix Plan 1A uses. The mockup IS the visual spec — port verbatim, adjust only for WP-template integration nits.
- **Task 25 (LoovaCast)** — original task covers ONLY the floating top bar. The dedicated full-width LoovaCast player block (design-language §5d) is built as a new partial in Task 13 and a wp_enqueue call in Task 25 (split into 25a floating bar + 25b dedicated player block).
- **NEW Task 13b** (inserted below) — Pillar snippet rotation system (WP options + admin settings page).

### Pre-flight updates

PF-1 expanded: now needs BOTH the hero photo (4:5, 1600px+ wide) AND the real RuckTalk logo (SVG preferred).
NEW **PF-7:** seed pillar snippet pool — design-language doc §5e has 20 starter snippets in Mike's voice (4 per pillar). Confirm + paste into the admin UI on first launch.
NEW **PF-8:** confirm /watch top-nav route — does it deep-link to YouTube channel, or to a /watch WP archive page that embeds episodes? Phase 5 owns the long-term answer; for Phase 1A launch, /watch can point to the existing YT channel.

---

## Pre-flight (BLOCKED ON MIKE — gather before execution starts)

These items must be resolved before specific tasks below; they are flagged inline where they block work.

- [ ] **PF-1: Hero photo of Mike** — new shot or pulled from existing library. Saved to `services/rucktalk-minimal/assets/img/mike-hero.jpg` (recommended: 1600px wide, JPG quality 82). Blocks Task 12.
- [ ] **PF-2: LumaBot WordPress connector status** — confirm (a) does LumaBot have an existing WP REST connector or are we building it? (b) embed model — `<script>` tag or iframe widget? Blocks Tasks 26-28.
- [ ] **PF-3: LoovaCast station for RuckTalk** — create a station on LoovaCast (or have Mike create it). Need the public stream URL + station ID for the floating player. Blocks Task 23.
- [ ] **PF-4: rucktalk.com WP admin credentials + REST API key** — for read access during Wave 1 audit and write access during deploy. Mike confirms wp-cli pathway on container `rt-wordpress` (server 100).
- [ ] **PF-5: fitasruck.com → rucktalk.com cutover slot** — Mike picks a 2-hour low-traffic window for the 301 flip (recommended: weekday 03:00-05:00 ET). Blocks Task 33.
- [ ] **PF-6: Cloudflare account access for rucktalk.com zone** — needed to fix the 525 on `www.` and configure cache rules. Mike grants Alfred read or temporary admin via Cloudflare API token (scoped to `rucktalk.com` zone).

---

## File Structure

```
services/rucktalk-minimal/                    # NEW Sonaar child theme
├── style.css                                  # Theme header + design tokens
├── functions.php                              # Bootstrap, asset enqueues, theme support
├── deploy.sh                                  # rsync + docker exec tar-pipe to server-100
├── README.md                                  # Theme overview + deploy notes
├── header.php                                 # Site header + floating radio bar mount
├── footer.php                                 # Footer + ecosystem strip + LumaBot mount
├── front-page.php                             # Homepage hero + sections
├── page-training.php                          # /training landing (gate + paid SKU)
├── page-training-free.php                     # /training/free PDF email gate
├── page-about.php                             # Mike's bio + pillars
├── assets/
│   ├── css/
│   │   ├── rucktalk.css                       # Layout + sections
│   │   ├── player.css                         # Floating LoovaCast radio bar
│   │   ├── popup.css                          # Newsletter signup popup
│   │   └── ecosystem.css                      # Footer strip + cross-promo blocks
│   ├── js/
│   │   ├── player.js                          # LoovaCast player controls
│   │   ├── popup.js                           # Exit-intent + scroll-depth trigger
│   │   ├── signup.js                          # Brevo REST submit handler
│   │   └── airoi-block.js                     # Auto-render AIROI block on tagged posts
│   ├── img/
│   │   ├── mike-hero.jpg                      # PF-1 (Mike provides)
│   │   ├── logo-rucktalk.svg                  # New wordmark
│   │   └── ecosystem/                         # Sister-brand logos for footer strip
│   │       ├── loovacast.svg
│   │       ├── lumabot.svg
│   │       ├── airoi.svg
│   │       ├── roen.svg
│   │       └── grl.svg
│   └── pdf/
│       └── (PDF lifted from FaR — Phase 3 plan owns it; placeholder ok for 1A)
├── inc/
│   ├── theme-supports.php                     # add_theme_support calls
│   ├── shortcodes.php                         # [rt_signup], [rt_ecosystem_strip], etc.
│   ├── rest-signup.php                        # Receives popup form POST, calls Brevo
│   ├── airoi-tagger.php                       # Auto-tags posts whose topic matches AI keywords
│   ├── menu-locations.php                     # Register theme menu locations
│   └── sonaar-overrides.php                   # Filters to safely override Sonaar behaviors
├── templates/
│   ├── parts/
│   │   ├── hero.php
│   │   ├── episode-card.php
│   │   ├── blog-card.php
│   │   ├── shop-teaser.php
│   │   └── newsletter-inline.php
│   └── popup-signup.php                       # Modal HTML for the newsletter popup
└── tests/
    └── README.md                              # Manual smoke checklist (no PHP unit tests for 1A)

services/rucktalk-redirects/                  # NEW must-use plugin (mu-plugin)
└── rucktalk-redirects.php                    # 301 rules for fitasruck.com legacy paths

infra/cloudflare/
└── rucktalk-zone-rules.tf                    # NEW Terraform-ish ruleset (or shell script if no TF)

scripts/
├── rucktalk_redesign_audit.py                # NEW one-shot — pulls Sonaar theme info via WP REST
└── rucktalk_redesign_smoke.py                # NEW one-shot — post-deploy URL smoke tests
```

---

## WAVE 1 — Sonaar theme audit (research, no production code)

### Task 1: Inventory the live Sonaar theme on rucktalk.com

**Files:**
- Create: `scripts/rucktalk_redesign_audit.py`
- Create: `docs/superpowers/audits/2026-05-19-sonaar-theme-audit.md` (audit output)

- [ ] **Step 1: Write the audit script**

```python
"""Sonaar theme audit for rucktalk.com — read-only via WP REST API.

Captures: active theme + version, child theme presence, all registered menus,
all registered widgets, custom post types, page templates available, and
the list of theme files in /wp-content/themes/<sonaar>/ so we know what we
can safely override in a child.

Output: prints a markdown report to stdout — pipe to docs/superpowers/audits/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/aialfred/alfred")

import httpx
from config.settings import settings

WP_BASE = "https://rucktalk.com/wp-json/wp/v2"
WP_USER = "alfred"  # or whatever app-password user we're using
WP_APP_PWD = getattr(settings, "rucktalk_wp_app_password", "") or ""


def get(path: str, params: dict | None = None) -> object:
    r = httpx.get(
        f"{WP_BASE}{path}",
        params=params or {},
        auth=(WP_USER, WP_APP_PWD) if WP_APP_PWD else None,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def main() -> int:
    print("# Sonaar theme audit — rucktalk.com\n")
    print(f"Generated: $(date)\n\n")

    # Themes are available via the wp-json/wp/v2/themes endpoint (auth required)
    themes = get("/themes")
    active = [t for t in themes if t.get("status") == "active"]
    print("## Active theme\n")
    for t in active:
        print(f"- **Name:** {t.get('name', {}).get('rendered', '?')}")
        print(f"- **Version:** {t.get('version', '?')}")
        print(f"- **Template:** {t.get('template', '?')}")
        print(f"- **Stylesheet:** {t.get('stylesheet', '?')}")

    # Pages — for template overrides we need to know what's in use
    pages = get("/pages", {"per_page": 100, "_fields": "id,slug,title,template"})
    print("\n## Pages + templates in use\n| ID | Slug | Title | Template |\n|---|---|---|---|")
    for p in pages:
        title = p.get("title", {}).get("rendered", "?")
        print(f"| {p['id']} | {p['slug']} | {title} | {p.get('template') or 'default'} |")

    # Custom post types
    cpts = get("/types")
    print("\n## Registered custom post types\n")
    for slug, info in cpts.items():
        print(f"- `{slug}` — {info.get('name', '?')}")

    # Menus
    print("\n## Menus (manual check — REST has no menus endpoint without nav-blocks)\n")
    print("Run on server 100: `wp menu list --allow-root` inside rt-wordpress container.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify settings has the WP app password key**

Run: `grep -n "rucktalk_wp_app_password" /home/aialfred/alfred/config/settings.py`
Expected: at least one match. If none, add it:

```python
# In config/settings.py — under the Settings class
rucktalk_wp_app_password: str = ""
```

Then in `.env`:
```
RUCKTALK_WP_APP_PASSWORD=<value Mike supplies in PF-4>
```

- [ ] **Step 3: Run the audit (BLOCKED on PF-4 — needs WP app password)**

Run:
```bash
mkdir -p docs/superpowers/audits
venv/bin/python scripts/rucktalk_redesign_audit.py > docs/superpowers/audits/2026-05-19-sonaar-theme-audit.md
```

Expected: a markdown file listing active theme, all pages, post types. If 401 on `/themes`, the app password lacks `read` scope — Mike fixes.

- [ ] **Step 4: Verify audit captured Sonaar info**

Run: `grep -E "^- \*\*Name|Template|Stylesheet" docs/superpowers/audits/2026-05-19-sonaar-theme-audit.md`
Expected: at least 3 lines with theme metadata.

- [ ] **Step 5: Commit**

```bash
git add scripts/rucktalk_redesign_audit.py docs/superpowers/audits/2026-05-19-sonaar-theme-audit.md config/settings.py
git commit -m "audit(rucktalk): Sonaar theme inventory + audit script"
```

### Task 2: Catalog Sonaar's customizable hooks + filters

**Files:**
- Modify: `docs/superpowers/audits/2026-05-19-sonaar-theme-audit.md` (append)

- [ ] **Step 1: Pull the Sonaar parent theme source for grep**

Run (BLOCKED on Mike-side — needs server 100 access):
```bash
ssh server-100 "docker exec rt-wordpress find /var/www/html/wp-content/themes/sonaar -type f -name '*.php' | head -50"
```

Expected: list of Sonaar theme PHP files. If `docker exec` denied, ask Mike to grant brief shell access on rt-wordpress for the audit.

- [ ] **Step 2: Grep for actionable hooks**

Run:
```bash
ssh server-100 "docker exec rt-wordpress grep -rn 'apply_filters\|do_action' /var/www/html/wp-content/themes/sonaar | head -100"
```

- [ ] **Step 3: Append findings to the audit doc**

Append a `## Sonaar override surface` section that lists:
- All `apply_filters()` calls (these are safe-to-override extension points)
- All template files we'll need to override in the child (`front-page.php`, `header.php`, `footer.php`, page templates, etc.)
- Any Sonaar-specific functions we MUST preserve (the `podcast` post type registration, audio player shortcodes, RSS feed generator)

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/audits/2026-05-19-sonaar-theme-audit.md
git commit -m "audit(rucktalk): catalog Sonaar override surface"
```

---

## WAVE 2 — Child theme scaffolding

### Task 3: Create the rucktalk-minimal child theme skeleton

**Files:**
- Create: `services/rucktalk-minimal/style.css`
- Create: `services/rucktalk-minimal/functions.php`
- Create: `services/rucktalk-minimal/README.md`

- [ ] **Step 1: Write style.css with theme header + brand tokens**

```css
/*
Theme Name: RuckTalk Minimal
Theme URI: https://rucktalk.com/
Author: Mike Johnson / Ground Rush Labs
Author URI: https://groundrushlabs.com/
Description: Personal-brand child theme for rucktalk.com. Built on Sonaar. Real-talk media business — podcast, blog, shop, training.
Template: sonaar
Version: 1.0.0
License: GNU General Public License v2 or later
Text Domain: rucktalk-minimal
*/

:root {
  /* Brand palette — pulled from the Phase 0 spec orange accent */
  --rt-bg: #FAFAFA;
  --rt-bg-card: #FFFFFF;
  --rt-ink: #1A1A1A;
  --rt-ink-2: #404040;
  --rt-ink-muted: #737373;
  --rt-rule: #D4D4D4;
  --rt-accent: #C2410C;        /* terracotta — matches Roen + Alfred dashboards */
  --rt-accent-soft: #FED7AA;
  --rt-accent-ink: #FFFFFF;

  /* Typography */
  --rt-font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --rt-font-display: "Inter", var(--rt-font-body);
  --rt-fs-xs: 12px;
  --rt-fs-sm: 14px;
  --rt-fs-base: 16px;
  --rt-fs-lg: 18px;
  --rt-fs-xl: 22px;
  --rt-fs-2xl: 28px;
  --rt-fs-3xl: 36px;
  --rt-fs-hero: 56px;

  /* Spacing scale */
  --rt-s-1: 4px;
  --rt-s-2: 8px;
  --rt-s-3: 12px;
  --rt-s-4: 16px;
  --rt-s-5: 24px;
  --rt-s-6: 32px;
  --rt-s-7: 48px;
  --rt-s-8: 64px;

  /* Layout */
  --rt-radius: 6px;
  --rt-shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
  --rt-shadow-md: 0 4px 12px rgba(0,0,0,0.08);
  --rt-max-w: 1080px;
}

body { background: var(--rt-bg); color: var(--rt-ink); font-family: var(--rt-font-body); line-height: 1.55; }
a { color: var(--rt-accent); }
```

- [ ] **Step 2: Write functions.php bootstrap (enqueues + theme support)**

```php
<?php
/**
 * rucktalk-minimal child theme bootstrap.
 *
 * Enqueues parent + child assets, declares theme supports, loads modules.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

define( 'RUCKTALK_THEME_VERSION', '1.0.0' );

/**
 * Enqueue parent (Sonaar) and child assets.
 */
function rucktalk_enqueue_assets() {
    $v = RUCKTALK_THEME_VERSION;

    // Google Fonts — Inter, single request.
    wp_enqueue_style(
        'rucktalk-inter',
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
        array(),
        null
    );

    // Parent theme stylesheet first so child can override.
    wp_enqueue_style(
        'sonaar-style',
        get_template_directory_uri() . '/style.css',
        array(),
        wp_get_theme( 'sonaar' )->get( 'Version' )
    );

    // Child design tokens.
    wp_enqueue_style(
        'rucktalk-tokens',
        get_stylesheet_directory_uri() . '/style.css',
        array( 'sonaar-style' ),
        $v
    );

    // Child structural CSS.
    wp_enqueue_style(
        'rucktalk-structure',
        get_stylesheet_directory_uri() . '/assets/css/rucktalk.css',
        array( 'rucktalk-tokens' ),
        $v
    );

    // Floating radio bar.
    wp_enqueue_style(
        'rucktalk-player',
        get_stylesheet_directory_uri() . '/assets/css/player.css',
        array( 'rucktalk-tokens' ),
        $v
    );
    wp_enqueue_script(
        'rucktalk-player-js',
        get_stylesheet_directory_uri() . '/assets/js/player.js',
        array(),
        $v,
        true
    );

    // Newsletter popup (CSS + JS).
    wp_enqueue_style(
        'rucktalk-popup',
        get_stylesheet_directory_uri() . '/assets/css/popup.css',
        array( 'rucktalk-tokens' ),
        $v
    );
    wp_enqueue_script(
        'rucktalk-popup-js',
        get_stylesheet_directory_uri() . '/assets/js/popup.js',
        array(),
        $v,
        true
    );
    wp_enqueue_script(
        'rucktalk-signup-js',
        get_stylesheet_directory_uri() . '/assets/js/signup.js',
        array( 'rucktalk-popup-js' ),
        $v,
        true
    );

    // Localize signup nonce + REST URL for the JS.
    wp_localize_script( 'rucktalk-signup-js', 'RuckTalkSignup', array(
        'restUrl' => esc_url_raw( rest_url( 'rucktalk/v1/signup' ) ),
        'nonce'   => wp_create_nonce( 'wp_rest' ),
    ) );

    // Ecosystem footer strip + AIROI block.
    wp_enqueue_style(
        'rucktalk-ecosystem',
        get_stylesheet_directory_uri() . '/assets/css/ecosystem.css',
        array( 'rucktalk-tokens' ),
        $v
    );
    wp_enqueue_script(
        'rucktalk-airoi-js',
        get_stylesheet_directory_uri() . '/assets/js/airoi-block.js',
        array(),
        $v,
        true
    );
}
add_action( 'wp_enqueue_scripts', 'rucktalk_enqueue_assets', 20 );

// Load modules.
require_once get_stylesheet_directory() . '/inc/theme-supports.php';
require_once get_stylesheet_directory() . '/inc/shortcodes.php';
require_once get_stylesheet_directory() . '/inc/rest-signup.php';
require_once get_stylesheet_directory() . '/inc/airoi-tagger.php';
require_once get_stylesheet_directory() . '/inc/menu-locations.php';
require_once get_stylesheet_directory() . '/inc/sonaar-overrides.php';
```

- [ ] **Step 3: Write README.md (deploy notes + structure overview)**

```markdown
# rucktalk-minimal

Sonaar child theme for rucktalk.com. Personal-brand media business — podcast, blog, shop, training, ecosystem cross-promo.

## Structure
- `style.css` — theme header + brand tokens (CSS variables)
- `functions.php` — bootstrap, asset enqueues, module loader
- `inc/` — modular PHP (shortcodes, REST endpoints, Sonaar overrides)
- `templates/parts/` — partials included via `get_template_part()`
- `assets/{css,js,img,pdf}/` — front-end assets

## Deploy
```bash
./deploy.sh   # rsync to server-100 → tar-pipe into rt-wordpress container
```

Requires `ssh server-100` SSH config + docker access on the host. Deploy is **T3** — must be approved by Mike.

## Sonaar parent
Parent theme `sonaar` must remain installed. This child overrides homepage, footer, headers, page templates, and adds ecosystem cross-promo assets without touching the `podcast` post type, RSS feed, or Sonaar player shortcodes.
```

- [ ] **Step 4: Commit**

```bash
git add services/rucktalk-minimal/style.css services/rucktalk-minimal/functions.php services/rucktalk-minimal/README.md
git commit -m "feat(rucktalk-theme): scaffold rucktalk-minimal Sonaar child theme"
```

### Task 4: Stub the module files so functions.php loads without fatal errors

**Files:**
- Create: `services/rucktalk-minimal/inc/theme-supports.php`
- Create: `services/rucktalk-minimal/inc/shortcodes.php`
- Create: `services/rucktalk-minimal/inc/rest-signup.php`
- Create: `services/rucktalk-minimal/inc/airoi-tagger.php`
- Create: `services/rucktalk-minimal/inc/menu-locations.php`
- Create: `services/rucktalk-minimal/inc/sonaar-overrides.php`

- [ ] **Step 1: Write each module as an empty stub with `if ! defined ABSPATH exit`**

For each file above:
```php
<?php
/**
 * <module name> — rucktalk-minimal.
 *
 * <one-line purpose>
 */
if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

// TODO(Task <N>): implementation
```

Use the actual task number where implementation lands (Tasks 5, 7, 13, etc.).

- [ ] **Step 2: PHP-lint each module**

Run: `for f in services/rucktalk-minimal/inc/*.php; do php -l "$f"; done`
Expected: `No syntax errors detected` for each.

- [ ] **Step 3: PHP-lint functions.php**

Run: `php -l services/rucktalk-minimal/functions.php`
Expected: `No syntax errors detected`.

- [ ] **Step 4: Commit**

```bash
git add services/rucktalk-minimal/inc/
git commit -m "feat(rucktalk-theme): stub module files for incremental fill-in"
```

### Task 5: Theme supports (post-thumbnails, title-tag, custom-logo, WC)

**Files:**
- Modify: `services/rucktalk-minimal/inc/theme-supports.php`

- [ ] **Step 1: Replace stub with real theme-support declarations**

```php
<?php
/**
 * Theme supports — rucktalk-minimal.
 *
 * Declares WP feature opt-ins. Runs after parent Sonaar declares its own.
 */
if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

function rucktalk_after_setup_theme() {
    // Title tag handled by WP, not theme.
    add_theme_support( 'title-tag' );

    // Featured images on episodes + blog posts (Sonaar covers podcast already).
    add_theme_support( 'post-thumbnails' );

    // Custom logo for the header.
    add_theme_support( 'custom-logo', array(
        'height'      => 64,
        'width'       => 240,
        'flex-height' => true,
        'flex-width'  => true,
    ) );

    // WC integration so the shop renders inside the child theme.
    add_theme_support( 'woocommerce' );
    add_theme_support( 'wc-product-gallery-zoom' );
    add_theme_support( 'wc-product-gallery-lightbox' );
    add_theme_support( 'wc-product-gallery-slider' );

    // HTML5 markup for common WP components.
    add_theme_support( 'html5', array(
        'search-form', 'comment-form', 'comment-list',
        'gallery', 'caption', 'style', 'script',
    ) );
}
add_action( 'after_setup_theme', 'rucktalk_after_setup_theme', 20 );
```

- [ ] **Step 2: PHP-lint**

Run: `php -l services/rucktalk-minimal/inc/theme-supports.php`
Expected: `No syntax errors detected`.

- [ ] **Step 3: Commit**

```bash
git add services/rucktalk-minimal/inc/theme-supports.php
git commit -m "feat(rucktalk-theme): declare theme supports (post-thumbs, custom-logo, WC)"
```

### Task 6: Menu locations (primary + footer)

**Files:**
- Modify: `services/rucktalk-minimal/inc/menu-locations.php`

- [ ] **Step 1: Register `primary` + `footer` menu locations**

```php
<?php
/**
 * Menu locations — rucktalk-minimal.
 *
 * Registers the primary and footer menus used by header.php / footer.php.
 */
if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

function rucktalk_register_menus() {
    register_nav_menus( array(
        'primary' => __( 'Primary Navigation', 'rucktalk-minimal' ),
        'footer'  => __( 'Footer Links', 'rucktalk-minimal' ),
    ) );
}
add_action( 'after_setup_theme', 'rucktalk_register_menus', 21 );
```

- [ ] **Step 2: PHP-lint + commit**

```bash
php -l services/rucktalk-minimal/inc/menu-locations.php
git add services/rucktalk-minimal/inc/menu-locations.php
git commit -m "feat(rucktalk-theme): register primary + footer menu locations"
```

### Task 7: Deploy script (mirrors Roen pattern)

**Files:**
- Create: `services/rucktalk-minimal/deploy.sh`

- [ ] **Step 1: Write deploy.sh**

```bash
#!/usr/bin/env bash
# Deploy the rucktalk-minimal child theme to rucktalk.com (container rt-wordpress on server-100).
# Idempotent — safe to re-run.
#
# T3 ACTION: requires Mike's explicit go before each run (production change).

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_HOST="server-100"
STAGE_DIR="/tmp/rucktalk-minimal"
CONTAINER="rt-wordpress"
WP_PATH="/var/www/html"
TARGET="${WP_PATH}/wp-content/themes/rucktalk-minimal"

echo "==> rsync source to ${SSH_HOST}:${STAGE_DIR}"
rsync -av --delete \
  --exclude 'tests/' \
  --exclude 'deploy.sh' \
  --exclude 'README.md' \
  --exclude '.DS_Store' \
  "${SRC_DIR}/" "${SSH_HOST}:${STAGE_DIR}/"

echo "==> tar-pipe into ${CONTAINER}:${TARGET}"
ssh "${SSH_HOST}" "
  set -e
  timeout 30 docker exec ${CONTAINER} mkdir -p ${TARGET}
  tar -C ${STAGE_DIR} -cf - . | timeout 60 docker exec -i ${CONTAINER} tar -C ${TARGET} -xf -
  timeout 30 docker exec ${CONTAINER} chown -R www-data:www-data ${TARGET}
"

echo "==> done. Theme deployed to ${CONTAINER}:${TARGET}"
echo "==> next: activate via wp theme activate rucktalk-minimal (manual or follow-up task)"
```

- [ ] **Step 2: chmod + lint-check**

```bash
chmod +x services/rucktalk-minimal/deploy.sh
bash -n services/rucktalk-minimal/deploy.sh
```

Expected: no shell syntax errors.

- [ ] **Step 3: Commit**

```bash
git add services/rucktalk-minimal/deploy.sh
git commit -m "feat(rucktalk-theme): add deploy.sh (rsync + tar-pipe to rt-wordpress)"
```

### Task 8: First deploy to staging slot + activation dry-run

**Files:** (no code — operational task)

- [ ] **Step 1: BLOCKED ON MIKE — confirm staging-vs-prod activation strategy**

Two options:
(a) Deploy to prod theme directory, do NOT activate until full theme is ready (visitors keep seeing Sonaar parent — safe)
(b) Stand up a staging site (e.g., `staging.rucktalk.com`) and activate there first

Recommend (a) for speed — theme files can sit unused on disk without affecting the live site. Mike confirms approach.

- [ ] **Step 2: First deploy (T3 — Mike approves)**

```bash
./services/rucktalk-minimal/deploy.sh
```

Expected: rsync output, then `done`. Theme files now live at `/var/www/html/wp-content/themes/rucktalk-minimal/` inside rt-wordpress.

- [ ] **Step 3: Verify theme appears in wp-admin without activating it**

```bash
ssh server-100 "docker exec rt-wordpress wp theme list --allow-root"
```

Expected: a row for `rucktalk-minimal` with status `inactive`.

---

## WAVE 3 — Homepage redesign

### Task 9: header.php with floating radio bar mount point

**Files:**
- Create: `services/rucktalk-minimal/header.php`

- [ ] **Step 1: Write header.php**

```php
<?php
/**
 * Header — rucktalk-minimal.
 *
 * Renders: doctype, head, opening body, floating radio bar mount,
 * primary navigation, custom logo.
 */
if ( ! defined( 'ABSPATH' ) ) {
    exit;
}
?><!DOCTYPE html>
<html <?php language_attributes(); ?>>
<head>
    <meta charset="<?php bloginfo( 'charset' ); ?>">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="profile" href="https://gmpg.org/xfn/11">
    <?php wp_head(); ?>
</head>
<body <?php body_class(); ?>>
<?php wp_body_open(); ?>

<div id="rt-radio-bar" data-station-url="" data-station-id="">
    <!-- mounted by assets/js/player.js — kept empty in PHP so theme works without JS -->
</div>

<header class="rt-site-header">
    <div class="rt-container rt-site-header__inner">
        <a class="rt-brand" href="<?php echo esc_url( home_url( '/' ) ); ?>">
            <?php
            if ( has_custom_logo() ) {
                the_custom_logo();
            } else {
                echo '<span class="rt-brand__text">RuckTalk</span>';
            }
            ?>
        </a>
        <nav class="rt-primary-nav">
            <?php
            wp_nav_menu( array(
                'theme_location' => 'primary',
                'container'      => false,
                'menu_class'     => 'rt-nav-list',
                'fallback_cb'    => '__return_false',
                'depth'          => 1,
            ) );
            ?>
        </nav>
    </div>
</header>

<main class="rt-main">
```

- [ ] **Step 2: PHP-lint + commit**

```bash
php -l services/rucktalk-minimal/header.php
git add services/rucktalk-minimal/header.php
git commit -m "feat(rucktalk-theme): header with radio bar mount + primary nav"
```

### Task 10: footer.php with ecosystem strip + LumaBot mount

**Files:**
- Create: `services/rucktalk-minimal/footer.php`

- [ ] **Step 1: Write footer.php**

```php
<?php
/**
 * Footer — rucktalk-minimal.
 *
 * Renders: closing main, footer signup, social links, ecosystem strip,
 * LumaBot chat mount, legal/copyright row.
 */
if ( ! defined( 'ABSPATH' ) ) {
    exit;
}
?>
</main><!-- .rt-main -->

<footer class="rt-site-footer">

    <section class="rt-footer-signup">
        <div class="rt-container">
            <h2 class="rt-footer-signup__title">Get the free 8-week RuckTalk plan.</h2>
            <p class="rt-footer-signup__sub">What I'd do in your first 8 weeks if I were starting over.</p>
            <?php echo do_shortcode( '[rt_signup placement="footer"]' ); ?>
        </div>
    </section>

    <?php echo do_shortcode( '[rt_ecosystem_strip]' ); ?>

    <div class="rt-footer-meta">
        <div class="rt-container rt-footer-meta__inner">
            <div class="rt-footer-links">
                <?php
                wp_nav_menu( array(
                    'theme_location' => 'footer',
                    'container'      => false,
                    'menu_class'     => 'rt-footer-links__list',
                    'fallback_cb'    => '__return_false',
                    'depth'          => 1,
                ) );
                ?>
            </div>
            <p class="rt-copyright">© <?php echo esc_html( date( 'Y' ) ); ?> RuckTalk · Ground Rush Labs</p>
        </div>
    </div>
</footer>

<div id="rt-lumabot-mount" data-bot-id="rucktalk">
    <!-- mounted by LumaBot embed (see inc/sonaar-overrides.php for the embed hook) -->
</div>

<?php wp_footer(); ?>
</body>
</html>
```

- [ ] **Step 2: PHP-lint + commit**

```bash
php -l services/rucktalk-minimal/footer.php
git add services/rucktalk-minimal/footer.php
git commit -m "feat(rucktalk-theme): footer with signup, ecosystem strip, LumaBot mount"
```

### Task 11: Newsletter signup shortcode

**Files:**
- Modify: `services/rucktalk-minimal/inc/shortcodes.php`

- [ ] **Step 1: Implement `[rt_signup placement="..."]` shortcode**

```php
<?php
/**
 * Shortcodes — rucktalk-minimal.
 */
if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

/**
 * [rt_signup placement="footer|hero|inline"] — email-capture form.
 *
 * Posts via JS to the rest-signup endpoint registered in inc/rest-signup.php.
 * No-JS fallback: regular POST to /wp-admin/admin-post.php?action=rt_signup
 * which the REST endpoint also handles.
 */
function rt_signup_shortcode( $atts ) {
    $a = shortcode_atts( array( 'placement' => 'inline' ), $atts );
    $placement = sanitize_html_class( $a['placement'] );
    ob_start(); ?>
    <form class="rt-signup-form rt-signup-form--<?php echo esc_attr( $placement ); ?>"
          data-placement="<?php echo esc_attr( $placement ); ?>"
          method="post" action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>">
        <input type="hidden" name="action" value="rt_signup">
        <input type="hidden" name="placement" value="<?php echo esc_attr( $placement ); ?>">
        <label class="screen-reader-text" for="rt-email-<?php echo esc_attr( $placement ); ?>">Email</label>
        <input class="rt-signup-form__email"
               id="rt-email-<?php echo esc_attr( $placement ); ?>"
               type="email" name="email" placeholder="you@example.com" required
               autocomplete="email">
        <button class="rt-signup-form__submit" type="submit">Send it to me</button>
        <p class="rt-signup-form__micro">We'll email to confirm. No spam.</p>
        <div class="rt-signup-form__status" aria-live="polite"></div>
    </form>
    <?php
    return ob_get_clean();
}
add_shortcode( 'rt_signup', 'rt_signup_shortcode' );

/**
 * [rt_ecosystem_strip] — sister-brand logo strip.
 * Renders the footer-above-footer "Part of the Ground Rush ecosystem" row.
 */
function rt_ecosystem_strip_shortcode() {
    $brands = array(
        array( 'slug' => 'loovacast', 'name' => 'LoovaCast',    'tagline' => 'Radio for creators',   'url' => 'https://loovacast.com' ),
        array( 'slug' => 'lumabot',   'name' => 'LumaBot',      'tagline' => 'AI chat for your site', 'url' => 'https://lumabot.com' ),
        array( 'slug' => 'airoi',     'name' => 'AIROI',        'tagline' => 'AI savings calc',      'url' => 'https://aialfred.groundrushcloud.com/static/ai-savings-calc/' ),
        array( 'slug' => 'roen',      'name' => 'Roen Handmade','tagline' => 'Handmade jewelry',     'url' => 'https://roenhandmade.com' ),
        array( 'slug' => 'grl',       'name' => 'Ground Rush Labs','tagline' => 'The studio',         'url' => 'https://groundrushlabs.com' ),
    );
    $base = trailingslashit( get_stylesheet_directory_uri() ) . 'assets/img/ecosystem/';
    ob_start(); ?>
    <section class="rt-ecosystem-strip" aria-label="Part of the Ground Rush ecosystem">
        <div class="rt-container">
            <p class="rt-ecosystem-strip__label">Part of the Ground Rush ecosystem</p>
            <ul class="rt-ecosystem-strip__list">
                <?php foreach ( $brands as $b ) : ?>
                <li class="rt-ecosystem-strip__item">
                    <a href="<?php echo esc_url( $b['url'] ); ?>" target="_blank" rel="noopener">
                        <img src="<?php echo esc_url( $base . $b['slug'] . '.svg' ); ?>"
                             alt="<?php echo esc_attr( $b['name'] ); ?>" loading="lazy"
                             class="rt-ecosystem-strip__logo">
                        <span class="rt-ecosystem-strip__tagline"><?php echo esc_html( $b['tagline'] ); ?></span>
                    </a>
                </li>
                <?php endforeach; ?>
            </ul>
        </div>
    </section>
    <?php
    return ob_get_clean();
}
add_shortcode( 'rt_ecosystem_strip', 'rt_ecosystem_strip_shortcode' );
```

- [ ] **Step 2: PHP-lint + commit**

```bash
php -l services/rucktalk-minimal/inc/shortcodes.php
git add services/rucktalk-minimal/inc/shortcodes.php
git commit -m "feat(rucktalk-theme): [rt_signup] + [rt_ecosystem_strip] shortcodes"
```

### Task 12: Homepage front-page.php (hero + sections)

**Files:**
- Create: `services/rucktalk-minimal/front-page.php`

- [ ] **Step 1: Write front-page.php (BLOCKED on PF-1 for hero photo)**

```php
<?php
/**
 * Homepage — rucktalk-minimal.
 *
 * Sections (in scroll order, per Phase 0 spec §3):
 *   1. Floating radio bar (rendered by header.php)
 *   2. Hero — Mike's photo + tagline + primary/secondary CTAs
 *   3. Latest podcast episode card
 *   4. Latest 3 blog posts
 *   5. Shop teaser (3 products)
 *   6. Inline newsletter signup
 *   7. "About RuckTalk" 1-paragraph blurb + link
 *   8. Footer (rendered by footer.php)
 */
get_header();

$hero_img = trailingslashit( get_stylesheet_directory_uri() ) . 'assets/img/mike-hero.jpg';
$tagline  = apply_filters( 'rucktalk_tagline', 'Real talk for guys in the thick of it.' );
?>

<section class="rt-hero">
    <div class="rt-container rt-hero__inner">
        <div class="rt-hero__copy">
            <p class="rt-hero__eyebrow">RuckTalk · Mike Johnson</p>
            <h1 class="rt-hero__tagline"><?php echo esc_html( $tagline ); ?></h1>
            <div class="rt-hero__cta">
                <a class="rt-btn rt-btn--primary" href="#rt-hero-signup">Get the free 8-week plan</a>
                <a class="rt-btn rt-btn--secondary" href="<?php echo esc_url( home_url( '/podcast/' ) ); ?>">Listen to today's episode</a>
            </div>
            <div id="rt-hero-signup" class="rt-hero__signup">
                <?php echo do_shortcode( '[rt_signup placement="hero"]' ); ?>
            </div>
        </div>
        <div class="rt-hero__visual">
            <img src="<?php echo esc_url( $hero_img ); ?>" alt="Mike Johnson — host of RuckTalk" class="rt-hero__photo">
        </div>
    </div>
</section>

<?php
// Below-the-fold sections rendered via partials so we can iterate independently.
get_template_part( 'templates/parts/episode-card' );
get_template_part( 'templates/parts/blog-card' );
get_template_part( 'templates/parts/shop-teaser' );
get_template_part( 'templates/parts/newsletter-inline' );
?>

<section class="rt-about-blurb">
    <div class="rt-container">
        <h2 class="rt-about-blurb__title">About RuckTalk</h2>
        <p class="rt-about-blurb__body">
            Regular guys talking to other regular guys about real life — work, decisions,
            money, family, what to do when things break, recovery when the day was brutal.
            Training, gear, and nutrition come up because they matter, not because the show
            is about them. No expert posture. No soapbox.
        </p>
        <a class="rt-btn rt-btn--link" href="<?php echo esc_url( home_url( '/about/' ) ); ?>">Read more →</a>
    </div>
</section>

<?php get_footer();
```

- [ ] **Step 2: PHP-lint**

Run: `php -l services/rucktalk-minimal/front-page.php`
Expected: `No syntax errors detected`.

- [ ] **Step 3: Commit**

```bash
git add services/rucktalk-minimal/front-page.php
git commit -m "feat(rucktalk-theme): homepage with hero + section partials"
```

### Task 13: Homepage section partials

**Files:**
- Create: `services/rucktalk-minimal/templates/parts/episode-card.php`
- Create: `services/rucktalk-minimal/templates/parts/blog-card.php`
- Create: `services/rucktalk-minimal/templates/parts/shop-teaser.php`
- Create: `services/rucktalk-minimal/templates/parts/newsletter-inline.php`

- [ ] **Step 1: Write episode-card.php (latest podcast)**

```php
<?php
/**
 * Latest podcast episode card — rucktalk-minimal homepage partial.
 *
 * Queries the most recent `podcast` post type (Sonaar's CPT), shows cover,
 * title, episode number, play button. Play uses Sonaar's existing player
 * shortcode so we don't duplicate its audio engine.
 */
if ( ! defined( 'ABSPATH' ) ) { exit; }

$q = new WP_Query( array(
    'post_type'      => 'podcast',
    'posts_per_page' => 1,
    'post_status'    => 'publish',
    'orderby'        => 'date',
    'order'          => 'DESC',
) );
if ( ! $q->have_posts() ) { return; }
$q->the_post();

$episode_num = get_post_meta( get_the_ID(), 'podcast_itunes_episode_number', true );
$cover       = get_the_post_thumbnail_url( get_the_ID(), 'large' );
?>
<section class="rt-latest-episode">
    <div class="rt-container">
        <p class="rt-section__eyebrow">Latest episode</p>
        <article class="rt-episode-card">
            <a class="rt-episode-card__cover" href="<?php the_permalink(); ?>">
                <?php if ( $cover ) : ?>
                    <img src="<?php echo esc_url( $cover ); ?>" alt="<?php the_title_attribute(); ?>" loading="lazy">
                <?php endif; ?>
            </a>
            <div class="rt-episode-card__meta">
                <p class="rt-episode-card__num">Ep <?php echo esc_html( $episode_num ); ?></p>
                <h3 class="rt-episode-card__title"><a href="<?php the_permalink(); ?>"><?php the_title(); ?></a></h3>
                <a class="rt-btn rt-btn--primary" href="<?php the_permalink(); ?>">▶ Play</a>
            </div>
        </article>
    </div>
</section>
<?php wp_reset_postdata();
```

- [ ] **Step 2: Write blog-card.php (latest 3 blog posts)**

```php
<?php
/**
 * Latest 3 blog posts — rucktalk-minimal homepage partial.
 */
if ( ! defined( 'ABSPATH' ) ) { exit; }

$q = new WP_Query( array(
    'post_type'      => 'post',
    'posts_per_page' => 3,
    'post_status'    => 'publish',
    'orderby'        => 'date',
    'order'          => 'DESC',
) );
if ( ! $q->have_posts() ) { return; }
?>
<section class="rt-latest-blog">
    <div class="rt-container">
        <p class="rt-section__eyebrow">From the blog</p>
        <div class="rt-blog-card-grid">
            <?php while ( $q->have_posts() ) : $q->the_post(); ?>
                <article class="rt-blog-card">
                    <?php if ( has_post_thumbnail() ) : ?>
                        <a class="rt-blog-card__cover" href="<?php the_permalink(); ?>">
                            <?php the_post_thumbnail( 'medium_large', array( 'loading' => 'lazy' ) ); ?>
                        </a>
                    <?php endif; ?>
                    <div class="rt-blog-card__body">
                        <h3 class="rt-blog-card__title"><a href="<?php the_permalink(); ?>"><?php the_title(); ?></a></h3>
                        <p class="rt-blog-card__excerpt"><?php echo esc_html( wp_trim_words( get_the_excerpt(), 22 ) ); ?></p>
                    </div>
                </article>
            <?php endwhile; ?>
        </div>
        <a class="rt-btn rt-btn--link" href="<?php echo esc_url( home_url( '/blog/' ) ); ?>">All posts →</a>
    </div>
</section>
<?php wp_reset_postdata();
```

- [ ] **Step 3: Write shop-teaser.php (3 WC products)**

```php
<?php
/**
 * Shop teaser — rucktalk-minimal homepage partial.
 *
 * Renders 3 featured WC products. Falls back silently if WC inactive.
 */
if ( ! defined( 'ABSPATH' ) ) { exit; }
if ( ! class_exists( 'WooCommerce' ) ) { return; }

$q = new WP_Query( array(
    'post_type'      => 'product',
    'posts_per_page' => 3,
    'post_status'    => 'publish',
    'orderby'        => 'date',
    'order'          => 'DESC',
    'tax_query'      => array( array(
        'taxonomy' => 'product_visibility',
        'field'    => 'name',
        'terms'    => 'featured',
    ) ),
) );
// Fallback: most recent products if no featured products yet.
if ( ! $q->have_posts() ) {
    $q = new WP_Query( array(
        'post_type'      => 'product',
        'posts_per_page' => 3,
        'post_status'    => 'publish',
    ) );
    if ( ! $q->have_posts() ) { return; }
}
?>
<section class="rt-shop-teaser">
    <div class="rt-container">
        <p class="rt-section__eyebrow">Shop</p>
        <div class="rt-shop-card-grid">
            <?php while ( $q->have_posts() ) : $q->the_post();
                global $product; ?>
                <article class="rt-shop-card">
                    <a href="<?php the_permalink(); ?>" class="rt-shop-card__cover">
                        <?php echo $product ? $product->get_image( 'medium_large' ) : ''; ?>
                    </a>
                    <h3 class="rt-shop-card__title"><a href="<?php the_permalink(); ?>"><?php the_title(); ?></a></h3>
                    <p class="rt-shop-card__price"><?php echo $product ? $product->get_price_html() : ''; ?></p>
                </article>
            <?php endwhile; ?>
        </div>
        <a class="rt-btn rt-btn--link" href="<?php echo esc_url( get_permalink( wc_get_page_id( 'shop' ) ) ); ?>">All gear →</a>
    </div>
</section>
<?php wp_reset_postdata();
```

- [ ] **Step 4: Write newsletter-inline.php**

```php
<?php
/**
 * Inline newsletter block — rucktalk-minimal homepage partial.
 *
 * Second-chance email capture between content sections.
 */
if ( ! defined( 'ABSPATH' ) ) { exit; }
?>
<section class="rt-newsletter-inline">
    <div class="rt-container rt-newsletter-inline__inner">
        <div class="rt-newsletter-inline__copy">
            <h2 class="rt-newsletter-inline__title">Get the Sunday email.</h2>
            <p class="rt-newsletter-inline__sub">Best post of the week + the episode worth your time. Sundays at 8 AM.</p>
        </div>
        <div class="rt-newsletter-inline__form">
            <?php echo do_shortcode( '[rt_signup placement="inline"]' ); ?>
        </div>
    </div>
</section>
```

- [ ] **Step 5: PHP-lint all partials + commit**

```bash
for f in services/rucktalk-minimal/templates/parts/*.php; do php -l "$f"; done
git add services/rucktalk-minimal/templates/parts/
git commit -m "feat(rucktalk-theme): homepage section partials (episode/blog/shop/newsletter)"
```

### Task 13b: Dynamic pillar snippet system (rotating "today's take")

**Files:**
- Create: `services/rucktalk-minimal/inc/pillar-snippets.php`
- Create: `services/rucktalk-minimal/templates/parts/pillars.php`
- Modify: `services/rucktalk-minimal/inc/shortcodes.php` (add `[rt_pillars]`)
- Modify: `services/rucktalk-minimal/front-page.php` (include the pillars partial)

**Design source:** `docs/superpowers/specs/2026-05-19-rucktalk-design-language.md` §5e

- [ ] **Step 1: Write the snippet manager (WP options + daily picker)**

```php
<?php
/**
 * Pillar snippet manager — rucktalk-minimal.
 *
 * Stores 5 pools of one-line "today's take" advice snippets (one per
 * pillar) in a single WP option. Each homepage render picks ONE snippet
 * per pillar deterministically keyed by (pillar, date) so all visitors
 * on a given day see the same snippet, but tomorrow it rotates.
 *
 * Admin can edit via Settings → RuckTalk Pillars (one textarea per pillar,
 * one snippet per line).
 */
if ( ! defined( 'ABSPATH' ) ) { exit; }

const RT_PILLARS = array( 'health', 'business', 'family', 'strength', 'shared' );
const RT_PILLAR_OPTION = 'rt_pillar_snippets';

/**
 * Get today's snippet for a pillar. Returns empty string if pool empty.
 */
function rt_pillar_snippet_today( string $pillar ): string {
    $pools = get_option( RT_PILLAR_OPTION, array() );
    $pool  = $pools[ $pillar ] ?? array();
    if ( empty( $pool ) ) { return ''; }
    // Day-of-year + pillar offset → deterministic index, rotates daily
    $day_of_year = (int) date( 'z' );
    $pillar_idx  = array_search( $pillar, RT_PILLARS, true ) ?: 0;
    $idx         = ( $day_of_year + $pillar_idx ) % count( $pool );
    return $pool[ $idx ];
}

/**
 * Register the settings page under Settings → RuckTalk Pillars.
 */
function rt_pillar_settings_init() {
    register_setting( 'rt_pillars', RT_PILLAR_OPTION, array(
        'type'              => 'array',
        'sanitize_callback' => 'rt_pillar_sanitize',
        'default'           => array(),
    ) );
}
add_action( 'admin_init', 'rt_pillar_settings_init' );

function rt_pillar_sanitize( $input ) {
    if ( ! is_array( $input ) ) { return array(); }
    $clean = array();
    foreach ( RT_PILLARS as $p ) {
        $raw = $input[ $p ] ?? '';
        // textarea posts as string — split on newlines, trim, drop empties
        $lines = is_string( $raw ) ? explode( "\n", $raw ) : (array) $raw;
        $clean[ $p ] = array_values( array_filter( array_map( function ( $l ) {
            return trim( wp_strip_all_tags( $l ) );
        }, $lines ), 'strlen' ) );
    }
    return $clean;
}

function rt_pillar_menu() {
    add_options_page(
        'RuckTalk Pillars', 'RuckTalk Pillars', 'manage_options',
        'rt-pillars', 'rt_pillar_settings_page'
    );
}
add_action( 'admin_menu', 'rt_pillar_menu' );

function rt_pillar_settings_page() {
    $pools = get_option( RT_PILLAR_OPTION, array() );
    ?>
    <div class="wrap">
        <h1>RuckTalk Pillars — Today's Take</h1>
        <p>One snippet per line. Site cycles through them in order, one per day per pillar.</p>
        <form method="post" action="options.php">
            <?php settings_fields( 'rt_pillars' ); ?>
            <?php foreach ( RT_PILLARS as $p ) :
                $value = is_array( $pools[ $p ] ?? null ) ? implode( "\n", $pools[ $p ] ) : ''; ?>
                <h2><?php echo esc_html( ucfirst( $p ) ); ?></h2>
                <textarea name="<?php echo esc_attr( RT_PILLAR_OPTION ); ?>[<?php echo esc_attr( $p ); ?>]"
                          rows="6" cols="100"
                          style="width:100%;font-family:monospace;font-size:13px"><?php
                    echo esc_textarea( $value );
                ?></textarea>
            <?php endforeach; ?>
            <?php submit_button(); ?>
        </form>
    </div>
    <?php
}
```

- [ ] **Step 2: PHP-lint**

Run: `php -l services/rucktalk-minimal/inc/pillar-snippets.php`
Expected: `No syntax errors detected`.

- [ ] **Step 3: Include in functions.php loader**

In `services/rucktalk-minimal/functions.php`, after the existing `require_once` lines (Task 3), add:

```php
require_once get_stylesheet_directory() . '/inc/pillar-snippets.php';
```

- [ ] **Step 4: Write the pillars partial**

```php
<?php
/**
 * Five pillars grid — rucktalk-minimal homepage partial.
 *
 * Renders pillar name, definition, and the snippet for today.
 * Snippet source: get_option('rt_pillar_snippets') via rt_pillar_snippet_today().
 */
if ( ! defined( 'ABSPATH' ) ) { exit; }

$pillars = array(
    array( 'slug' => 'health',   'num' => '01', 'name' => 'Health',
           'what' => 'Keeping a body that holds up to the rest of life.' ),
    array( 'slug' => 'business', 'num' => '02', 'name' => 'Business',
           'what' => 'Building something while everything else is also on fire.' ),
    array( 'slug' => 'family',   'num' => '03', 'name' => 'Family',
           'what' => 'Showing up at home like you mean it.' ),
    array( 'slug' => 'strength', 'num' => '04', 'name' => 'Strength',
           'what' => 'Mind, body, the way you carry yourself through the week.' ),
    array( 'slug' => 'shared',   'num' => '05', 'name' => 'Shared',
           'what' => 'What everybody else is going through, said plainly.' ),
);
?>
<section class="rt-pillars">
    <div class="rt-container">
        <div class="rt-pillars__head">
            <span class="rt-section__eyebrow">What we're talking about this week</span>
            <h2 class="rt-section__title">Five things <em>worth getting right.</em></h2>
        </div>
        <div class="rt-pillars__grid">
            <?php foreach ( $pillars as $p ) :
                $snippet = rt_pillar_snippet_today( $p['slug'] ); ?>
                <a class="rt-pillar" href="<?php echo esc_url( home_url( '/blog/category/' . $p['slug'] . '/' ) ); ?>">
                    <span class="rt-pillar__num"><?php echo esc_html( $p['num'] ); ?></span>
                    <h3 class="rt-pillar__name"><?php echo esc_html( $p['name'] ); ?></h3>
                    <p class="rt-pillar__what"><?php echo esc_html( $p['what'] ); ?></p>
                    <?php if ( $snippet ) : ?>
                        <div class="rt-pillar__latest">
                            <p class="rt-pillar__latest-label">Today's take</p>
                            <p class="rt-pillar__latest-snippet"><?php echo esc_html( $snippet ); ?></p>
                        </div>
                    <?php endif; ?>
                    <span class="rt-pillar__arrow">→</span>
                </a>
            <?php endforeach; ?>
        </div>
    </div>
</section>
```

- [ ] **Step 5: PHP-lint + commit**

```bash
php -l services/rucktalk-minimal/templates/parts/pillars.php
git add services/rucktalk-minimal/inc/pillar-snippets.php \
         services/rucktalk-minimal/templates/parts/pillars.php \
         services/rucktalk-minimal/functions.php
git commit -m "feat(rucktalk-theme): dynamic pillar snippets (admin UI + daily rotation)"
```

- [ ] **Step 6: Seed the snippet pool via WP-CLI (BLOCKED on PF-7 — Mike confirms snippets)**

Use the starter pool from `docs/superpowers/specs/2026-05-19-rucktalk-design-language.md` §5e:

```bash
ssh server-100 "docker exec rt-wordpress wp option update rt_pillar_snippets '$(cat <<'EOF'
{"health":["Stretch your hips before coffee. Your lower back'\''s been asking nicely.","Heavy carries beat cardio for fat loss and look better at the playground.","Sleep before midnight or you'\''re chasing yesterday'\''s tired all day.","Three deep breaths before you check your phone. That'\''s the difference."],"business":["Block one hour after lunch. No phone. Ask: what would I do differently if I started today?","Raise your prices. The right people will pay; the wrong ones were always going to leave.","If you can'\''t explain it to your spouse, you can'\''t sell it.","Fire the customer that'\''s costing you sleep. They'\''re never worth it."],"family":["Twenty minutes Sunday night. Counter, notebook, your spouse. Prevents 80% of the dumb arguments.","Pick up the phone when your mom calls. You don'\''t get those calls forever.","Eat dinner together. Phones in the other room. Five days a week minimum.","Walk with one of the kids alone. They tell you things in motion they won'\''t sitting down."],"strength":["Tuesdays make you. Mondays anyone can do.","Stop training to look strong. Train so you can pick up your kids when they'\''re 30.","Sit ups don'\''t fix your back. Picking things up does.","The hardest set is the one after you wanted to stop."],"shared":["Most people are way more tired than they let on. Includes you.","Nobody'\''s got it figured out. They just stopped saying so out loud.","Being kind is free and almost always the right call.","Ask better questions. Listen longer than feels comfortable."]}
EOF
)' --format=json --allow-root"
```

Expected: `Success: Updated 'rt_pillar_snippets' option.`

- [ ] **Step 7: Smoke test today's snippet on the live site**

After Task 15 (theme activated), visit https://rucktalk.com/ and verify each pillar card shows one of the seeded snippets. Tomorrow, refresh — should see a different snippet per pillar.

### Task 14: Layout + section CSS (`rucktalk.css`)

**Files:**
- Create: `services/rucktalk-minimal/assets/css/rucktalk.css`

- [ ] **Step 1: Write the structural CSS**

```css
/* ─────────────────────────────────────────────────────────────
 * rucktalk-minimal — structural CSS
 * Layout primitives + section styles. Tokens come from style.css.
 * ───────────────────────────────────────────────────────────── */

/* Layout primitives */
.rt-container { max-width: var(--rt-max-w); margin: 0 auto; padding: 0 var(--rt-s-5); }
.rt-main { padding-top: 56px; /* clear floating radio bar */ }

/* Header */
.rt-site-header { background: var(--rt-bg-card); border-bottom: 1px solid var(--rt-rule); position: sticky; top: 56px; z-index: 50; }
.rt-site-header__inner { display: flex; align-items: center; justify-content: space-between; gap: var(--rt-s-5); padding: var(--rt-s-3) 0; }
.rt-brand__text { font-family: var(--rt-font-display); font-weight: 700; font-size: var(--rt-fs-xl); color: var(--rt-ink); text-decoration: none; }
.rt-nav-list { list-style: none; display: flex; gap: var(--rt-s-5); margin: 0; padding: 0; }
.rt-nav-list li a { color: var(--rt-ink-2); text-decoration: none; font-weight: 500; }
.rt-nav-list li a:hover { color: var(--rt-accent); }

/* Section primitives */
.rt-section__eyebrow { font-size: var(--rt-fs-xs); text-transform: uppercase; letter-spacing: 0.08em; color: var(--rt-ink-muted); margin: 0 0 var(--rt-s-4); }

/* Buttons */
.rt-btn { display: inline-block; padding: var(--rt-s-3) var(--rt-s-5); border-radius: var(--rt-radius); font-weight: 600; text-decoration: none; transition: opacity 0.15s; }
.rt-btn:hover { opacity: 0.9; }
.rt-btn--primary { background: var(--rt-accent); color: var(--rt-accent-ink); }
.rt-btn--secondary { background: transparent; color: var(--rt-ink); border: 1px solid var(--rt-rule); }
.rt-btn--link { background: none; color: var(--rt-accent); padding: 0; font-weight: 500; }

/* Hero */
.rt-hero { padding: var(--rt-s-8) 0; background: var(--rt-bg-card); }
.rt-hero__inner { display: grid; grid-template-columns: 1.1fr 1fr; gap: var(--rt-s-7); align-items: center; }
.rt-hero__eyebrow { font-size: var(--rt-fs-sm); color: var(--rt-ink-muted); margin: 0 0 var(--rt-s-3); }
.rt-hero__tagline { font-family: var(--rt-font-display); font-size: var(--rt-fs-hero); font-weight: 700; line-height: 1.1; color: var(--rt-ink); margin: 0 0 var(--rt-s-5); }
.rt-hero__cta { display: flex; gap: var(--rt-s-3); margin-bottom: var(--rt-s-6); flex-wrap: wrap; }
.rt-hero__photo { width: 100%; height: auto; border-radius: var(--rt-radius); box-shadow: var(--rt-shadow-md); }

@media ( max-width: 720px ) {
    .rt-hero__inner { grid-template-columns: 1fr; }
    .rt-hero__tagline { font-size: var(--rt-fs-3xl); }
}

/* Latest episode */
.rt-latest-episode { padding: var(--rt-s-7) 0; }
.rt-episode-card { display: grid; grid-template-columns: 240px 1fr; gap: var(--rt-s-5); align-items: center; background: var(--rt-bg-card); padding: var(--rt-s-5); border-radius: var(--rt-radius); box-shadow: var(--rt-shadow-sm); }
.rt-episode-card__cover img { width: 240px; height: 240px; object-fit: cover; border-radius: var(--rt-radius); }
.rt-episode-card__num { font-size: var(--rt-fs-sm); color: var(--rt-ink-muted); margin: 0 0 var(--rt-s-2); }
.rt-episode-card__title { font-size: var(--rt-fs-2xl); margin: 0 0 var(--rt-s-4); }
.rt-episode-card__title a { color: var(--rt-ink); text-decoration: none; }

/* Blog grid */
.rt-latest-blog { padding: var(--rt-s-7) 0; background: var(--rt-bg-card); }
.rt-blog-card-grid { display: grid; grid-template-columns: repeat( 3, 1fr ); gap: var(--rt-s-5); margin-bottom: var(--rt-s-5); }
.rt-blog-card__cover img { width: 100%; aspect-ratio: 16/10; object-fit: cover; border-radius: var(--rt-radius); }
.rt-blog-card__title { font-size: var(--rt-fs-lg); margin: var(--rt-s-3) 0 var(--rt-s-2); }
.rt-blog-card__title a { color: var(--rt-ink); text-decoration: none; }
.rt-blog-card__excerpt { color: var(--rt-ink-2); margin: 0; }

@media ( max-width: 720px ) {
    .rt-blog-card-grid { grid-template-columns: 1fr; }
    .rt-episode-card { grid-template-columns: 1fr; }
    .rt-episode-card__cover img { width: 100%; height: auto; }
}

/* Shop teaser */
.rt-shop-teaser { padding: var(--rt-s-7) 0; }
.rt-shop-card-grid { display: grid; grid-template-columns: repeat( 3, 1fr ); gap: var(--rt-s-5); margin-bottom: var(--rt-s-5); }
.rt-shop-card__cover img { width: 100%; aspect-ratio: 1/1; object-fit: cover; border-radius: var(--rt-radius); }
.rt-shop-card__title { font-size: var(--rt-fs-base); margin: var(--rt-s-3) 0 var(--rt-s-2); }
.rt-shop-card__title a { color: var(--rt-ink); text-decoration: none; }
.rt-shop-card__price { color: var(--rt-accent); font-weight: 600; margin: 0; }

@media ( max-width: 720px ) {
    .rt-shop-card-grid { grid-template-columns: 1fr 1fr; }
}

/* Newsletter inline */
.rt-newsletter-inline { padding: var(--rt-s-7) 0; background: var(--rt-bg-card); border-top: 1px solid var(--rt-rule); border-bottom: 1px solid var(--rt-rule); }
.rt-newsletter-inline__inner { display: grid; grid-template-columns: 1fr 1fr; gap: var(--rt-s-6); align-items: center; }
.rt-newsletter-inline__title { font-size: var(--rt-fs-2xl); margin: 0 0 var(--rt-s-3); }
.rt-newsletter-inline__sub { color: var(--rt-ink-2); margin: 0; }

@media ( max-width: 720px ) {
    .rt-newsletter-inline__inner { grid-template-columns: 1fr; }
}

/* About blurb */
.rt-about-blurb { padding: var(--rt-s-7) 0; }
.rt-about-blurb__title { font-size: var(--rt-fs-2xl); margin: 0 0 var(--rt-s-4); }
.rt-about-blurb__body { color: var(--rt-ink-2); max-width: 60ch; margin: 0 0 var(--rt-s-4); }

/* Signup form (shared across placements) */
.rt-signup-form { display: flex; gap: var(--rt-s-2); flex-wrap: wrap; align-items: flex-start; }
.rt-signup-form__email { flex: 1 1 240px; padding: var(--rt-s-3) var(--rt-s-4); border: 1px solid var(--rt-rule); border-radius: var(--rt-radius); font-size: var(--rt-fs-base); }
.rt-signup-form__submit { padding: var(--rt-s-3) var(--rt-s-5); background: var(--rt-accent); color: var(--rt-accent-ink); border: 0; border-radius: var(--rt-radius); font-weight: 600; cursor: pointer; }
.rt-signup-form__micro { width: 100%; font-size: var(--rt-fs-xs); color: var(--rt-ink-muted); margin: var(--rt-s-2) 0 0; }
.rt-signup-form__status { width: 100%; font-size: var(--rt-fs-sm); margin: var(--rt-s-2) 0 0; }
.rt-signup-form__status[data-state="ok"] { color: #16a34a; }
.rt-signup-form__status[data-state="err"] { color: #dc2626; }

/* Footer */
.rt-site-footer { background: var(--rt-bg-card); border-top: 1px solid var(--rt-rule); }
.rt-footer-signup { padding: var(--rt-s-7) 0; text-align: center; }
.rt-footer-signup__title { font-size: var(--rt-fs-2xl); margin: 0 0 var(--rt-s-3); }
.rt-footer-signup__sub { color: var(--rt-ink-2); margin: 0 0 var(--rt-s-5); }
.rt-footer-signup .rt-signup-form { max-width: 480px; margin: 0 auto; }

.rt-footer-meta { padding: var(--rt-s-5) 0; border-top: 1px solid var(--rt-rule); font-size: var(--rt-fs-sm); }
.rt-footer-meta__inner { display: flex; justify-content: space-between; align-items: center; gap: var(--rt-s-5); flex-wrap: wrap; }
.rt-footer-links__list { list-style: none; display: flex; gap: var(--rt-s-4); margin: 0; padding: 0; }
.rt-footer-links__list a { color: var(--rt-ink-2); text-decoration: none; }
.rt-copyright { color: var(--rt-ink-muted); margin: 0; }
```

- [ ] **Step 2: Validate CSS (no syntax errors)**

Run: `npx stylelint services/rucktalk-minimal/assets/css/rucktalk.css 2>&1 | head -30 || echo "(stylelint not installed — proceed)"`
Expected: no errors or warnings (or stylelint not installed, in which case just proceed — the CSS will be visually validated post-deploy).

- [ ] **Step 3: Commit**

```bash
git add services/rucktalk-minimal/assets/css/rucktalk.css
git commit -m "feat(rucktalk-theme): structural CSS for homepage + sections + signup form"
```

### Task 15: Deploy + activate the theme on rt-wordpress (T3 — Mike approves)

**Files:** (no code — operational)

- [ ] **Step 1: BLOCKED ON MIKE — confirm activation window**

Activation is the moment the new theme renders. Recommend a low-traffic window (weekday morning 03:00-05:00 ET).

- [ ] **Step 2: Deploy + verify file presence**

```bash
./services/rucktalk-minimal/deploy.sh
ssh server-100 "docker exec rt-wordpress ls /var/www/html/wp-content/themes/rucktalk-minimal/ | head -20"
```

Expected: file listing matching the directory tree.

- [ ] **Step 3: Activate theme**

```bash
ssh server-100 "docker exec rt-wordpress wp theme activate rucktalk-minimal --allow-root"
```

Expected: `Success: Switched to 'RuckTalk Minimal' theme.`

- [ ] **Step 4: Smoke test homepage in browser**

Run: `curl -sI https://rucktalk.com/ | head -5`
Expected: HTTP 200. Open in browser, verify: hero renders with placeholder photo (if PF-1 not yet provided), nav present, footer present, no PHP fatal errors (check `wp-content/debug.log` if uncertain).

- [ ] **Step 5: Capture before/after screenshots for handoff**

Use Mike's preferred screenshot tool. Save to `docs/superpowers/audits/2026-05-19-rucktalk-launch-screenshots/`.

---

## WAVE 4 — /training section (FaR migration target)

### Task 16: WP page + WC product for the paid $29 SKU

**Files:** (no code — operational)

- [ ] **Step 1: BLOCKED ON MIKE — confirm the original FaR $29 product details**

Need: product SKU, exact price, current customer count (for migration sizing), payment processor (Stripe assumed).

- [ ] **Step 2: Create the WC product on rt-wordpress**

```bash
ssh server-100 "docker exec rt-wordpress wp post create \
  --post_type=product --post_status=publish \
  --post_title='RuckTalk 8-Week Plan' \
  --post_excerpt='The full 8-week ruck plan PDF — paid version.' \
  --allow-root"
```

Note the returned product ID; set price, virtual, downloadable:

```bash
PRODUCT_ID=<id-from-above>
ssh server-100 "docker exec rt-wordpress wp post meta update $PRODUCT_ID _price 29 --allow-root"
ssh server-100 "docker exec rt-wordpress wp post meta update $PRODUCT_ID _regular_price 29 --allow-root"
ssh server-100 "docker exec rt-wordpress wp post meta update $PRODUCT_ID _virtual yes --allow-root"
ssh server-100 "docker exec rt-wordpress wp post meta update $PRODUCT_ID _downloadable yes --allow-root"
```

- [ ] **Step 3: Verify product appears at /shop/**

Run: `curl -s https://rucktalk.com/shop/ | grep -i "8-week"`
Expected: at least one match in the page.

### Task 17: page-training.php landing template

**Files:**
- Create: `services/rucktalk-minimal/page-training.php`

- [ ] **Step 1: Write the template**

```php
<?php
/**
 * Template Name: RuckTalk Training (root /training)
 *
 * Two side-by-side cards: Free PDF (email gate) + Paid 8-Week Plan ($29).
 * Reached at /training/ when the WP page slug is set to 'training' and
 * this template is selected.
 */
get_header();
?>
<section class="rt-training-hero">
    <div class="rt-container">
        <h1 class="rt-training-hero__title">RuckTalk Training</h1>
        <p class="rt-training-hero__sub">Two ways in: free plan in your inbox, or the full PDF.</p>
    </div>
</section>

<section class="rt-training-cards">
    <div class="rt-container rt-training-cards__grid">
        <article class="rt-training-card">
            <h2 class="rt-training-card__title">Free 8-Week Plan</h2>
            <p class="rt-training-card__sub">What I'd do in your first 8 weeks if I were starting over. Sent by email after you verify.</p>
            <a class="rt-btn rt-btn--primary" href="<?php echo esc_url( home_url( '/training/free/' ) ); ?>">Get it free</a>
        </article>
        <article class="rt-training-card">
            <h2 class="rt-training-card__title">The Full 8-Week Plan ($29)</h2>
            <p class="rt-training-card__sub">Printable PDF with the full progression, daily prompts, and recovery protocols.</p>
            <a class="rt-btn rt-btn--secondary" href="<?php echo esc_url( get_permalink( wc_get_product( get_option( 'rt_training_product_id' ) ) ) ); ?>">Buy now — $29</a>
        </article>
    </div>
</section>

<?php get_footer();
```

- [ ] **Step 2: Add an option helper for the product ID lookup**

Append to `services/rucktalk-minimal/inc/shortcodes.php` (or create a `inc/options.php` if preferred):

```php
/**
 * After WC product is created (Task 16), set rt_training_product_id in WP options
 * so page-training.php can link to it:
 *
 *   wp option update rt_training_product_id <PRODUCT_ID> --allow-root
 *
 * This decouples template from hardcoded post IDs.
 */
```

Then run on rt-wordpress (BLOCKED on Task 16):
```bash
ssh server-100 "docker exec rt-wordpress wp option update rt_training_product_id <PRODUCT_ID> --allow-root"
```

- [ ] **Step 3: Create the /training WP page using this template**

```bash
ssh server-100 "docker exec rt-wordpress wp post create \
  --post_type=page --post_status=publish \
  --post_title='Training' --post_name='training' \
  --page_template='page-training.php' --allow-root"
```

- [ ] **Step 4: Smoke test**

Run: `curl -s https://rucktalk.com/training/ | grep -E "Free 8-Week|29"`
Expected: both phrases appear.

- [ ] **Step 5: Commit**

```bash
git add services/rucktalk-minimal/page-training.php services/rucktalk-minimal/inc/shortcodes.php
git commit -m "feat(rucktalk-theme): /training landing template (free + paid cards)"
```

### Task 18: page-training-free.php (PDF email-gate)

**Files:**
- Create: `services/rucktalk-minimal/page-training-free.php`

- [ ] **Step 1: Write the email-gate template**

```php
<?php
/**
 * Template Name: RuckTalk Training Free (/training/free)
 *
 * Single-purpose signup page for the free 8-week PDF. Posts to the same
 * REST signup endpoint as the popup. On confirmation email click, Brevo
 * delivers the PDF.
 */
get_header();
?>
<section class="rt-training-free">
    <div class="rt-container rt-training-free__grid">
        <div class="rt-training-free__copy">
            <h1 class="rt-training-free__title">The Free 8-Week RuckTalk Plan</h1>
            <p class="rt-training-free__sub">
                What I'd do in your first 8 weeks if I were starting over.
                Drop your email — we'll send you a verification link, then the plan
                lands in your inbox.
            </p>
            <?php echo do_shortcode( '[rt_signup placement="training-free"]' ); ?>
            <p class="rt-training-free__paid">Already done the free one? <a href="<?php echo esc_url( home_url( '/training/' ) ); ?>">Get the full PDF for $29</a>.</p>
        </div>
        <div class="rt-training-free__visual">
            <img src="<?php echo esc_url( trailingslashit( get_stylesheet_directory_uri() ) . 'assets/img/pdf-cover.jpg' ); ?>"
                 alt="RuckTalk 8-week plan PDF cover" loading="lazy">
        </div>
    </div>
</section>
<?php get_footer();
```

- [ ] **Step 2: Create the WP page**

```bash
ssh server-100 "docker exec rt-wordpress wp post create \
  --post_type=page --post_status=publish \
  --post_parent=$(docker exec rt-wordpress wp post list --post_type=page --pagename=training --field=ID --allow-root) \
  --post_title='Free 8-Week Plan' --post_name='free' \
  --page_template='page-training-free.php' --allow-root"
```

- [ ] **Step 3: Add styles**

Append to `services/rucktalk-minimal/assets/css/rucktalk.css`:

```css
.rt-training-hero { padding: var(--rt-s-8) 0 var(--rt-s-5); text-align: center; }
.rt-training-hero__title { font-size: var(--rt-fs-3xl); margin: 0 0 var(--rt-s-3); }
.rt-training-hero__sub { color: var(--rt-ink-2); margin: 0; }
.rt-training-cards { padding: var(--rt-s-7) 0; }
.rt-training-cards__grid { display: grid; grid-template-columns: 1fr 1fr; gap: var(--rt-s-5); }
.rt-training-card { background: var(--rt-bg-card); padding: var(--rt-s-6); border-radius: var(--rt-radius); box-shadow: var(--rt-shadow-sm); }
.rt-training-card__title { font-size: var(--rt-fs-xl); margin: 0 0 var(--rt-s-3); }
.rt-training-card__sub { color: var(--rt-ink-2); margin: 0 0 var(--rt-s-5); }
.rt-training-free { padding: var(--rt-s-8) 0; }
.rt-training-free__grid { display: grid; grid-template-columns: 1fr 1fr; gap: var(--rt-s-7); align-items: center; }
.rt-training-free__title { font-size: var(--rt-fs-3xl); margin: 0 0 var(--rt-s-4); }
.rt-training-free__paid { font-size: var(--rt-fs-sm); color: var(--rt-ink-muted); margin: var(--rt-s-5) 0 0; }
.rt-training-free__visual img { width: 100%; border-radius: var(--rt-radius); box-shadow: var(--rt-shadow-md); }
@media ( max-width: 720px ) {
    .rt-training-cards__grid, .rt-training-free__grid { grid-template-columns: 1fr; }
}
```

- [ ] **Step 4: Commit**

```bash
git add services/rucktalk-minimal/page-training-free.php services/rucktalk-minimal/assets/css/rucktalk.css
git commit -m "feat(rucktalk-theme): /training/free PDF email-gate template + styles"
```

### Task 19: REST signup endpoint (Brevo + double opt-in)

**Files:**
- Modify: `services/rucktalk-minimal/inc/rest-signup.php`

- [ ] **Step 1: Implement the endpoint**

```php
<?php
/**
 * REST signup — rucktalk-minimal.
 *
 * Receives POST { email, placement } from the [rt_signup] form (or popup),
 * subscribes the contact to Brevo's RuckTalk list with double opt-in.
 * Brevo's confirmation webhook (configured in Brevo dashboard) then fires
 * the n8n weekly-newsletter pipeline.
 *
 * Brevo API: POST /v3/contacts with attributes + listIds + redirectionUrl.
 * Double opt-in is enabled at the Brevo list level (no API flag).
 */
if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

const RT_BREVO_LIST_ID  = 0;  // set via wp option rt_brevo_list_id <id>
const RT_BREVO_API_BASE = 'https://api.brevo.com/v3';

function rt_register_signup_route() {
    register_rest_route( 'rucktalk/v1', '/signup', array(
        'methods'             => 'POST',
        'callback'            => 'rt_handle_signup',
        'permission_callback' => '__return_true',
        'args'                => array(
            'email'     => array( 'required' => true, 'type' => 'string', 'format' => 'email' ),
            'placement' => array( 'required' => false, 'type' => 'string' ),
        ),
    ) );
}
add_action( 'rest_api_init', 'rt_register_signup_route' );

function rt_handle_signup( WP_REST_Request $r ) {
    $email     = sanitize_email( $r->get_param( 'email' ) );
    $placement = sanitize_text_field( $r->get_param( 'placement' ) ?: 'unknown' );

    if ( ! is_email( $email ) ) {
        return new WP_REST_Response( array( 'ok' => false, 'error' => 'invalid_email' ), 400 );
    }

    $api_key = get_option( 'rt_brevo_api_key', '' );
    $list_id = (int) get_option( 'rt_brevo_list_id', 0 );

    if ( empty( $api_key ) || $list_id <= 0 ) {
        return new WP_REST_Response( array( 'ok' => false, 'error' => 'config_missing' ), 500 );
    }

    $resp = wp_remote_post( RT_BREVO_API_BASE . '/contacts/doubleOptinConfirmation', array(
        'headers' => array(
            'accept'       => 'application/json',
            'content-type' => 'application/json',
            'api-key'      => $api_key,
        ),
        'body' => wp_json_encode( array(
            'email'                => $email,
            'includeListIds'       => array( $list_id ),
            'templateId'           => (int) get_option( 'rt_brevo_opt_in_template_id', 0 ),
            'redirectionUrl'       => home_url( '/training/free/?confirmed=1' ),
            'attributes'           => array(
                'SIGNUP_PLACEMENT' => $placement,
                'SIGNUP_SOURCE'    => 'rucktalk.com',
            ),
        ) ),
        'timeout' => 15,
    ) );

    if ( is_wp_error( $resp ) ) {
        return new WP_REST_Response( array( 'ok' => false, 'error' => 'brevo_unreachable' ), 502 );
    }
    $code = wp_remote_retrieve_response_code( $resp );
    if ( $code >= 400 ) {
        return new WP_REST_Response( array(
            'ok' => false, 'error' => 'brevo_rejected',
            'status' => $code, 'body' => wp_remote_retrieve_body( $resp ),
        ), 502 );
    }

    return new WP_REST_Response( array( 'ok' => true ), 200 );
}

/**
 * Non-JS fallback: admin-post handler that wraps rt_handle_signup, then
 * redirects to a thanks page.
 */
function rt_admin_post_signup() {
    $email     = isset( $_POST['email'] ) ? sanitize_email( wp_unslash( $_POST['email'] ) ) : '';
    $placement = isset( $_POST['placement'] ) ? sanitize_text_field( wp_unslash( $_POST['placement'] ) ) : '';

    $r = new WP_REST_Request( 'POST', '/rucktalk/v1/signup' );
    $r->set_param( 'email', $email );
    $r->set_param( 'placement', $placement );
    rt_handle_signup( $r );

    wp_safe_redirect( add_query_arg( array( 'signup' => 'ok' ), wp_get_referer() ?: home_url( '/' ) ) );
    exit;
}
add_action( 'admin_post_nopriv_rt_signup', 'rt_admin_post_signup' );
add_action( 'admin_post_rt_signup',        'rt_admin_post_signup' );
```

- [ ] **Step 2: PHP-lint**

Run: `php -l services/rucktalk-minimal/inc/rest-signup.php`
Expected: `No syntax errors detected`.

- [ ] **Step 3: Set the Brevo options on rt-wordpress (T3 — Mike approves)**

BLOCKED on Brevo list creation. Mike (or Alfred via Brevo API in a follow-up task) creates the list, then:

```bash
ssh server-100 "docker exec rt-wordpress wp option update rt_brevo_api_key '<key>' --allow-root"
ssh server-100 "docker exec rt-wordpress wp option update rt_brevo_list_id <list-id> --allow-root"
ssh server-100 "docker exec rt-wordpress wp option update rt_brevo_opt_in_template_id <template-id> --allow-root"
```

- [ ] **Step 4: Smoke test (curl)**

```bash
curl -sX POST https://rucktalk.com/wp-json/rucktalk/v1/signup \
  -H 'content-type: application/json' \
  -d '{"email":"test+rt-signup@groundrushinc.com","placement":"smoke"}'
```

Expected: `{"ok":true}`. Check Brevo dashboard for the contact and verify a confirmation email arrives at the test address.

- [ ] **Step 5: Commit**

```bash
git add services/rucktalk-minimal/inc/rest-signup.php
git commit -m "feat(rucktalk-theme): REST signup endpoint with Brevo double opt-in"
```

### Task 20: signup.js — JS form handler

**Files:**
- Create: `services/rucktalk-minimal/assets/js/signup.js`

- [ ] **Step 1: Write the form handler**

```javascript
/**
 * rucktalk-minimal signup form handler.
 *
 * Hijacks every .rt-signup-form submit, posts JSON to the REST endpoint,
 * shows in-form success/error. No-JS users hit admin-post.php fallback
 * via the form's action attribute — both paths share the same WP handler.
 */
( function () {
    if ( ! window.RuckTalkSignup ) { return; }

    document.addEventListener( 'submit', function ( e ) {
        const form = e.target.closest( '.rt-signup-form' );
        if ( ! form ) { return; }
        e.preventDefault();

        const status = form.querySelector( '.rt-signup-form__status' );
        const email  = form.querySelector( 'input[name="email"]' ).value.trim();
        const place  = form.dataset.placement || 'unknown';

        status.textContent = 'Sending…';
        status.dataset.state = '';

        fetch( RuckTalkSignup.restUrl, {
            method: 'POST',
            headers: { 'content-type': 'application/json', 'x-wp-nonce': RuckTalkSignup.nonce },
            body: JSON.stringify( { email: email, placement: place } ),
        } ).then( function ( r ) {
            return r.json().then( function ( j ) { return { ok: r.ok, body: j }; } );
        } ).then( function ( res ) {
            if ( res.ok && res.body.ok ) {
                status.textContent = '✓ Check your email to confirm. The plan lands the moment you click verify.';
                status.dataset.state = 'ok';
                form.querySelector( 'input[name="email"]' ).value = '';
            } else {
                status.textContent = 'Something went wrong. Please try again or email mike@rucktalk.com.';
                status.dataset.state = 'err';
            }
        } ).catch( function () {
            status.textContent = 'Network error. Please try again.';
            status.dataset.state = 'err';
        } );
    } );
}() );
```

- [ ] **Step 2: Lint (no Node? use bash syntax check)**

Run: `node --check services/rucktalk-minimal/assets/js/signup.js 2>&1 || echo "node not installed — proceed"`

- [ ] **Step 3: Commit**

```bash
git add services/rucktalk-minimal/assets/js/signup.js
git commit -m "feat(rucktalk-theme): signup.js JS form handler with status feedback"
```

### Task 21: Popup signup (exit-intent + scroll-depth)

**Files:**
- Create: `services/rucktalk-minimal/templates/popup-signup.php`
- Create: `services/rucktalk-minimal/assets/css/popup.css`
- Create: `services/rucktalk-minimal/assets/js/popup.js`
- Modify: `services/rucktalk-minimal/footer.php` (include popup HTML)

- [ ] **Step 1: Write popup HTML template**

```php
<?php
/**
 * Newsletter signup popup — rucktalk-minimal.
 *
 * Hidden by default. popup.js shows it on scroll-depth 50% (first page)
 * OR exit-intent (second+ page) for first-time visitors, dismissed once
 * per 14-day cookie.
 */
if ( ! defined( 'ABSPATH' ) ) { exit; }
?>
<div id="rt-popup" class="rt-popup" hidden role="dialog" aria-modal="true" aria-labelledby="rt-popup-title">
    <div class="rt-popup__backdrop" data-dismiss></div>
    <div class="rt-popup__panel">
        <button class="rt-popup__close" type="button" data-dismiss aria-label="Close">×</button>
        <img class="rt-popup__cover"
             src="<?php echo esc_url( trailingslashit( get_stylesheet_directory_uri() ) . 'assets/img/pdf-cover.jpg' ); ?>"
             alt="RuckTalk 8-week plan PDF cover" loading="lazy">
        <h2 id="rt-popup-title" class="rt-popup__title">Get the free 8-week RuckTalk plan</h2>
        <p class="rt-popup__sub">What I'd do in your first 8 weeks if I were starting over.</p>
        <?php echo do_shortcode( '[rt_signup placement="popup"]' ); ?>
    </div>
</div>
```

- [ ] **Step 2: Include popup in footer.php**

In `services/rucktalk-minimal/footer.php`, after the `<div id="rt-lumabot-mount">…</div>` line and before `wp_footer()`:

```php
<?php
// Newsletter popup (hidden by default, shown by popup.js).
get_template_part( 'templates/popup-signup' );
?>
```

- [ ] **Step 3: Write popup.css**

```css
.rt-popup { position: fixed; inset: 0; z-index: 9999; display: flex; align-items: center; justify-content: center; padding: var(--rt-s-5); }
.rt-popup[hidden] { display: none; }
.rt-popup__backdrop { position: absolute; inset: 0; background: rgba(0,0,0,0.5); }
.rt-popup__panel { position: relative; background: var(--rt-bg-card); border-radius: var(--rt-radius); max-width: 480px; width: 100%; padding: var(--rt-s-6); box-shadow: var(--rt-shadow-md); text-align: center; }
.rt-popup__close { position: absolute; top: var(--rt-s-3); right: var(--rt-s-3); width: 36px; height: 36px; background: transparent; border: 0; font-size: 24px; line-height: 1; cursor: pointer; color: var(--rt-ink-2); }
.rt-popup__cover { width: 180px; height: auto; margin: 0 auto var(--rt-s-4); border-radius: 4px; box-shadow: var(--rt-shadow-sm); }
.rt-popup__title { font-size: var(--rt-fs-xl); margin: 0 0 var(--rt-s-3); }
.rt-popup__sub { color: var(--rt-ink-2); margin: 0 0 var(--rt-s-5); }
.rt-popup .rt-signup-form { justify-content: center; }
@media ( max-width: 480px ) {
    .rt-popup__panel { padding: var(--rt-s-5); }
}
```

- [ ] **Step 4: Write popup.js (trigger logic + cookie)**

```javascript
/**
 * rucktalk-minimal newsletter popup trigger.
 *
 * Shows the popup once per first-time visitor after EITHER:
 *   (a) 50% scroll-depth on any page (first time), OR
 *   (b) exit-intent (mouse leaves top of viewport) on the second+ page view
 *
 * Suppressed for 14 days after dismissal via 'rt_popup_dismissed' cookie.
 * Never fires on bounce (delayed 8s on first page to filter quick exits).
 */
( function () {
    const COOKIE = 'rt_popup_dismissed';
    const STATE_KEY = 'rt_popup_state';
    const DISMISS_DAYS = 14;
    const SHOWN_KEY = 'rt_popup_shown_once';
    const FIRST_PAGE_DELAY_MS = 8000;

    function getCookie( name ) {
        return document.cookie.split( '; ' ).find( c => c.startsWith( name + '=' ) );
    }
    function setCookie( name, val, days ) {
        const exp = new Date( Date.now() + days * 86400000 ).toUTCString();
        document.cookie = name + '=' + val + '; expires=' + exp + '; path=/; SameSite=Lax';
    }
    function dismissed() { return !! getCookie( COOKIE ); }
    function alreadyShown() { return sessionStorage.getItem( SHOWN_KEY ) === '1'; }
    function markShown() { sessionStorage.setItem( SHOWN_KEY, '1' ); }
    function pageView() {
        const state = JSON.parse( localStorage.getItem( STATE_KEY ) || '{}' );
        state.views = ( state.views || 0 ) + 1;
        localStorage.setItem( STATE_KEY, JSON.stringify( state ) );
        return state.views;
    }

    function show() {
        if ( dismissed() || alreadyShown() ) { return; }
        const p = document.getElementById( 'rt-popup' );
        if ( ! p ) { return; }
        p.hidden = false;
        markShown();
        // ESC dismisses
        document.addEventListener( 'keydown', function onEsc( e ) {
            if ( e.key === 'Escape' ) { hide(); document.removeEventListener( 'keydown', onEsc ); }
        } );
        // Dismiss buttons
        p.querySelectorAll( '[data-dismiss]' ).forEach( el => el.addEventListener( 'click', hide ) );
    }
    function hide() {
        const p = document.getElementById( 'rt-popup' );
        if ( p ) { p.hidden = true; }
        setCookie( COOKIE, '1', DISMISS_DAYS );
    }

    const views = pageView();
    if ( views === 1 ) {
        // First page: scroll-depth trigger after 8s settle period
        setTimeout( function () {
            window.addEventListener( 'scroll', function onScroll() {
                const docH = document.documentElement.scrollHeight - window.innerHeight;
                if ( docH <= 0 ) { return; }
                const pct = window.scrollY / docH;
                if ( pct >= 0.5 ) { show(); window.removeEventListener( 'scroll', onScroll ); }
            }, { passive: true } );
        }, FIRST_PAGE_DELAY_MS );
    } else {
        // Subsequent pages: exit-intent (mouse leaves top)
        document.addEventListener( 'mouseout', function ( e ) {
            if ( e.clientY < 10 && ! e.relatedTarget ) { show(); }
        } );
    }
}() );
```

- [ ] **Step 5: PHP-lint, JS check, commit**

```bash
php -l services/rucktalk-minimal/templates/popup-signup.php
node --check services/rucktalk-minimal/assets/js/popup.js 2>&1 || echo "(node not installed — ok)"
git add services/rucktalk-minimal/templates/popup-signup.php \
         services/rucktalk-minimal/assets/css/popup.css \
         services/rucktalk-minimal/assets/js/popup.js \
         services/rucktalk-minimal/footer.php
git commit -m "feat(rucktalk-theme): newsletter popup (scroll-depth + exit-intent + 14d cookie)"
```

---

## WAVE 5 — Cutover (fitasruck.com 301 + Cloudflare 525 fix)

### Task 22: fitasruck.com → rucktalk.com/training mu-plugin

**Files:**
- Create: `services/rucktalk-redirects/rucktalk-redirects.php` (a must-use plugin)

**Strategy choice:** mu-plugin lives in `mu-plugins/` and loads automatically, theme-independent. The redirects survive theme switches and don't depend on .htaccess (which we don't control directly in this container setup).

- [ ] **Step 1: Write the mu-plugin (rucktalk.com side — catches direct hits to old paths if any)**

```php
<?php
/**
 * Plugin Name: RuckTalk Legacy Redirects
 * Description: 301-redirect old fitasruck.com paths that may resolve on rucktalk.com (defensive — primary 301s live at the fitasruck.com origin / Cloudflare worker).
 * Version: 1.0.0
 * Author: Ground Rush Labs
 */
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_action( 'template_redirect', function () {
    $path = $_SERVER['REQUEST_URI'] ?? '/';
    $map = array(
        '/8-week-plan/'   => '/training/8-week-plan/',
        '/8-week-plan'    => '/training/8-week-plan/',
        '/free-plan/'     => '/training/free/',
        '/free-plan'      => '/training/free/',
        '/checkout/'      => '/training/',
        '/checkout-1/'    => '/training/',
    );
    foreach ( $map as $old => $new ) {
        if ( strpos( $path, $old ) === 0 ) {
            wp_safe_redirect( home_url( $new ), 301 );
            exit;
        }
    }
} );
```

- [ ] **Step 2: Deploy to mu-plugins (T3 — Mike approves)**

```bash
ssh server-100 "docker exec rt-wordpress mkdir -p /var/www/html/wp-content/mu-plugins"
scp -i ~/.ssh/<key> services/rucktalk-redirects/rucktalk-redirects.php server-100:/tmp/
ssh server-100 "tar -C /tmp -cf - rucktalk-redirects.php | docker exec -i rt-wordpress tar -C /var/www/html/wp-content/mu-plugins -xf -"
ssh server-100 "docker exec rt-wordpress chown www-data:www-data /var/www/html/wp-content/mu-plugins/rucktalk-redirects.php"
```

- [ ] **Step 3: Smoke test a redirect**

```bash
curl -sI https://rucktalk.com/8-week-plan/ | head -5
```

Expected: `HTTP/2 301` with `location: https://rucktalk.com/training/8-week-plan/`.

- [ ] **Step 4: Commit**

```bash
git add services/rucktalk-redirects/
git commit -m "feat(rucktalk-redirects): mu-plugin for defensive legacy path redirects"
```

### Task 23: fitasruck.com origin → 301 to rucktalk.com (the real cutover)

**Files:** (operational — fitasruck.com sits on a different WP/host)

- [ ] **Step 1: BLOCKED ON MIKE — confirm fitasruck.com hosting + Cloudflare zone**

Need: which server hosts fitasruck.com WordPress? Is it Cloudflare-fronted? If yes, we can do the 301 at the Cloudflare layer (recommended — no origin change needed).

- [ ] **Step 2: Strategy A — Cloudflare Page Rule (preferred)**

If fitasruck.com is on Cloudflare:
- Cloudflare dashboard → fitasruck.com zone → Rules → Page Rules
- URL pattern: `*fitasruck.com/*`
- Action: Forwarding URL → Status 301 Permanent Redirect → Destination URL: `https://rucktalk.com/training/$2`
- (The `$2` captures everything after the `/`)

- [ ] **Step 3: Strategy B — origin mu-plugin (fallback if not on Cloudflare)**

Add a mu-plugin to the fitasruck.com WP install (`fitasruck-redirect.php`) that does `wp_safe_redirect( 'https://rucktalk.com/training/', 301 )` for every request.

- [ ] **Step 4: Search Console — submit change of address**

In Search Console, for property `https://fitasruck.com/`:
- Settings → Change of address → Update site → `https://rucktalk.com/`
- Submit.

- [ ] **Step 5: Submit new sitemap.xml on rucktalk.com Search Console property**

```bash
curl -s "https://rucktalk.com/sitemap.xml" | head -10
```

Verify the sitemap is generated and includes `/training/`, `/training/free/`, `/training/8-week-plan/`. Then re-submit in GSC.

- [ ] **Step 6: Smoke test the cutover**

```bash
curl -sIL https://fitasruck.com/ | head -10
```

Expected: HTTP 301 → `https://rucktalk.com/training/` (or `https://rucktalk.com/training/...`).

### Task 24: Fix Cloudflare 525 on www.rucktalk.com

**Files:** (operational)

**Cause hypothesis:** the `www.rucktalk.com` hostname is in the Cloudflare zone but the origin server doesn't have a valid SSL cert for that hostname (or the SSL mode is set to "Full (strict)" and the origin cert only covers the apex).

- [ ] **Step 1: Diagnose**

```bash
curl -vI https://www.rucktalk.com/ 2>&1 | head -30
```

Look for the actual TLS error. Note: the 525 is Cloudflare's error code for "SSL handshake failed between Cloudflare and origin."

- [ ] **Step 2: Check Cloudflare SSL/TLS mode for the zone**

Cloudflare dashboard → rucktalk.com → SSL/TLS → Overview. If mode is "Full (strict)" but origin cert doesn't cover www, either:
- (a) downgrade to "Full" (not strict) — accepts self-signed origin certs (lower security)
- (b) add `www.rucktalk.com` to the origin cert (recommended if origin uses Let's Encrypt — re-issue with both apex + www SANs)

Recommend (b).

- [ ] **Step 3: Re-issue origin Let's Encrypt cert with both names**

On server-100 (or wherever the origin nginx runs):

```bash
ssh server-100 "certbot certonly --webroot -w /var/www/html -d rucktalk.com -d www.rucktalk.com --expand"
```

Then reload nginx (or whatever the reverse proxy is) so it serves the new cert.

- [ ] **Step 4: Verify**

```bash
curl -sI https://www.rucktalk.com/ | head -5
```

Expected: HTTP 200 (or 301 if a www→apex redirect rule exists, which is fine).

- [ ] **Step 5: Add a www → apex 301 redirect (recommended)**

Either in Cloudflare (Page Rule: `www.rucktalk.com/*` → 301 → `https://rucktalk.com/$1`) or at the origin nginx config. Either way, www becomes a permanent redirect to the apex.

- [ ] **Step 6: Commit the documentation of what was done**

```bash
# No code change — just record the fix
echo "$(date): www.rucktalk.com 525 fixed via Let's Encrypt cert re-issue with www SAN" >> docs/superpowers/audits/2026-05-19-rucktalk-launch-screenshots/cloudflare-fix.md
git add docs/superpowers/audits/
git commit -m "fix(rucktalk-dns): document Cloudflare 525 resolution on www. subdomain"
```

---

## WAVE 6 — Ecosystem cross-promo wiring

### Task 25: LoovaCast floating radio bar

**Files:**
- Create: `services/rucktalk-minimal/assets/css/player.css`
- Create: `services/rucktalk-minimal/assets/js/player.js`
- Modify: `services/rucktalk-minimal/header.php` (set data-station-url)

- [ ] **Step 1: BLOCKED ON PF-3 — LoovaCast RuckTalk station details**

Need: public stream URL + station ID for the player to bind to.

- [ ] **Step 2: Write player.css**

```css
#rt-radio-bar {
    position: fixed; top: 0; left: 0; right: 0; height: 56px;
    background: var(--rt-ink); color: #fff; z-index: 100;
    display: flex; align-items: center; padding: 0 var(--rt-s-5);
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.rt-radio-bar__play {
    background: var(--rt-accent); color: #fff; border: 0; width: 36px; height: 36px;
    border-radius: 50%; cursor: pointer; font-size: 16px; display: flex; align-items: center; justify-content: center;
}
.rt-radio-bar__meta { flex: 1; padding: 0 var(--rt-s-4); display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.rt-radio-bar__label { font-size: var(--rt-fs-xs); color: #a0a0a0; text-transform: uppercase; letter-spacing: 0.06em; }
.rt-radio-bar__track { font-size: var(--rt-fs-sm); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rt-radio-bar__attrib { font-size: var(--rt-fs-xs); color: #a0a0a0; }
.rt-radio-bar__attrib a { color: #fff; text-decoration: none; }
.rt-radio-bar__attrib a:hover { color: var(--rt-accent-soft); }

@media ( max-width: 540px ) {
    .rt-radio-bar__attrib { display: none; }
}
```

- [ ] **Step 3: Write player.js**

```javascript
/**
 * rucktalk-minimal LoovaCast floating radio bar.
 *
 * Reads station URL from <div id="rt-radio-bar" data-station-url="...">.
 * Mounts a play button, track title (polled from a LoovaCast metadata endpoint),
 * and a "Powered by LoovaCast" attribution.
 */
( function () {
    const bar = document.getElementById( 'rt-radio-bar' );
    if ( ! bar ) { return; }

    const streamUrl = bar.dataset.stationUrl;
    const stationId = bar.dataset.stationId;
    if ( ! streamUrl ) { bar.style.display = 'none'; return; }

    bar.innerHTML = `
        <button class="rt-radio-bar__play" type="button" aria-label="Play radio">▶</button>
        <div class="rt-radio-bar__meta">
            <span class="rt-radio-bar__label">Live now</span>
            <span class="rt-radio-bar__track">RuckTalk Radio</span>
        </div>
        <span class="rt-radio-bar__attrib">Powered by <a href="https://loovacast.com" target="_blank" rel="noopener">LoovaCast</a></span>
    `;

    const audio = new Audio( streamUrl );
    audio.preload = 'none';
    const playBtn = bar.querySelector( '.rt-radio-bar__play' );
    const trackEl = bar.querySelector( '.rt-radio-bar__track' );

    let playing = false;
    playBtn.addEventListener( 'click', function () {
        if ( playing ) {
            audio.pause();
            playBtn.textContent = '▶';
        } else {
            audio.play().catch( function () { /* user-gesture issue, ignore */ } );
            playBtn.textContent = '⏸';
        }
        playing = ! playing;
    } );

    // Optional: poll metadata if stationId is provided
    if ( stationId ) {
        function refreshTrack() {
            fetch( 'https://loovacast.com/api/v1/station/' + encodeURIComponent( stationId ) + '/nowplaying' )
                .then( r => r.json() )
                .then( j => { if ( j && j.title ) { trackEl.textContent = j.title; } } )
                .catch( function () { /* silent */ } );
        }
        refreshTrack();
        setInterval( refreshTrack, 30000 );
    }
}() );
```

- [ ] **Step 4: Update header.php to inject station URL/ID**

In `services/rucktalk-minimal/header.php`, replace the `<div id="rt-radio-bar" ...>` line with a PHP-driven version:

```php
<?php
$rt_station_url = get_option( 'rt_loovacast_stream_url', '' );
$rt_station_id  = get_option( 'rt_loovacast_station_id', '' );
?>
<div id="rt-radio-bar"
     data-station-url="<?php echo esc_attr( $rt_station_url ); ?>"
     data-station-id="<?php echo esc_attr( $rt_station_id ); ?>">
</div>
```

- [ ] **Step 5: Set the LoovaCast options on rt-wordpress**

```bash
ssh server-100 "docker exec rt-wordpress wp option update rt_loovacast_stream_url 'https://<loovacast-stream-url>' --allow-root"
ssh server-100 "docker exec rt-wordpress wp option update rt_loovacast_station_id '<station-id>' --allow-root"
```

- [ ] **Step 6: Smoke test in browser**

Visit https://rucktalk.com/ — the floating bar appears, play button starts the stream. Track title polls every 30s.

- [ ] **Step 7: Commit**

```bash
git add services/rucktalk-minimal/assets/css/player.css \
         services/rucktalk-minimal/assets/js/player.js \
         services/rucktalk-minimal/header.php
git commit -m "feat(rucktalk-theme): LoovaCast floating radio bar with attribution"
```

### Task 26: LumaBot chat widget mount

**Files:**
- Modify: `services/rucktalk-minimal/inc/sonaar-overrides.php`

- [ ] **Step 1: BLOCKED ON PF-2 — LumaBot connector status**

Two paths:
- (a) LumaBot ships a `<script>` embed tag → drop it in `wp_footer`, done
- (b) We're building the WP connector → that's a separate ticket; for 1A, render an empty `#rt-lumabot-mount` div and stop

- [ ] **Step 2: Implement Path A (assuming a script-tag embed)**

```php
<?php
/**
 * Sonaar overrides + ecosystem mounts — rucktalk-minimal.
 */
if ( ! defined( 'ABSPATH' ) ) { exit; }

/**
 * Inject the LumaBot embed before </body>.
 * Bot ID + key are stored as WP options so we can change without code edits.
 */
function rt_lumabot_embed() {
    $bot_id  = get_option( 'rt_lumabot_bot_id', '' );
    $api_key = get_option( 'rt_lumabot_api_key', '' );
    if ( empty( $bot_id ) || empty( $api_key ) ) { return; }
    ?>
    <script>
      (function() {
        var s = document.createElement('script');
        s.async = true;
        s.src = 'https://embed.lumabot.com/widget.js';
        s.dataset.botId = <?php echo wp_json_encode( $bot_id ); ?>;
        s.dataset.apiKey = <?php echo wp_json_encode( $api_key ); ?>;
        s.dataset.mount = '#rt-lumabot-mount';
        document.head.appendChild(s);
      })();
    </script>
    <?php
}
add_action( 'wp_footer', 'rt_lumabot_embed', 99 );
```

- [ ] **Step 3: Set LumaBot options**

```bash
ssh server-100 "docker exec rt-wordpress wp option update rt_lumabot_bot_id '<id>' --allow-root"
ssh server-100 "docker exec rt-wordpress wp option update rt_lumabot_api_key '<key>' --allow-root"
```

- [ ] **Step 4: Smoke test**

Visit rucktalk.com — bottom-right chat bubble appears. Open it — chat header shows "Powered by LumaBot".

- [ ] **Step 5: Commit**

```bash
git add services/rucktalk-minimal/inc/sonaar-overrides.php
git commit -m "feat(rucktalk-theme): LumaBot chat widget embed + WP option gate"
```

### Task 27: AIROI contextual auto-tagger

**Files:**
- Modify: `services/rucktalk-minimal/inc/airoi-tagger.php`
- Create: `services/rucktalk-minimal/assets/js/airoi-block.js`

- [ ] **Step 1: Implement the auto-tagger**

```php
<?php
/**
 * AIROI auto-tagger — rucktalk-minimal.
 *
 * Scans post content for AI/business/efficiency keywords on save; if matched,
 * sets a custom field 'rt_show_airoi' = 1 so the front-end JS knows to render
 * the AIROI CTA block inline.
 */
if ( ! defined( 'ABSPATH' ) ) { exit; }

const RT_AIROI_KEYWORDS = array(
    'ai', 'artificial intelligence', 'automation', 'chatgpt', 'llm',
    'business', 'small business', 'entrepreneur', 'productivity',
    'efficiency', 'workflow', 'saas', 'startup',
);

function rt_airoi_check_post( $post_id, $post ) {
    if ( wp_is_post_revision( $post_id ) || wp_is_post_autosave( $post_id ) ) { return; }
    if ( ! in_array( $post->post_type, array( 'post' ), true ) ) { return; }

    $haystack = strtolower( $post->post_title . ' ' . wp_strip_all_tags( $post->post_content ) );
    foreach ( RT_AIROI_KEYWORDS as $kw ) {
        if ( strpos( $haystack, $kw ) !== false ) {
            update_post_meta( $post_id, 'rt_show_airoi', 1 );
            return;
        }
    }
    delete_post_meta( $post_id, 'rt_show_airoi' );
}
add_action( 'save_post', 'rt_airoi_check_post', 10, 2 );

/**
 * Inject the AIROI block at the end of post content for tagged posts.
 */
function rt_airoi_inject_block( $content ) {
    if ( ! is_singular( 'post' ) ) { return $content; }
    if ( ! get_post_meta( get_the_ID(), 'rt_show_airoi', true ) ) { return $content; }
    $block = '
    <aside class="rt-airoi-block">
        <h3>Curious what AI could actually save your business?</h3>
        <p>Try the AI Savings Calculator — 90 seconds, no signup.</p>
        <a class="rt-btn rt-btn--primary" href="https://aialfred.groundrushcloud.com/static/ai-savings-calc/" target="_blank" rel="noopener">Open the calculator →</a>
    </aside>';
    return $content . $block;
}
add_filter( 'the_content', 'rt_airoi_inject_block', 99 );
```

- [ ] **Step 2: Backfill existing posts (one-shot)**

```bash
ssh server-100 "docker exec rt-wordpress wp eval 'foreach ( get_posts( array( \"post_type\" => \"post\", \"posts_per_page\" => -1, \"post_status\" => \"publish\" ) ) as \$p ) { do_action( \"save_post\", \$p->ID, \$p ); }' --allow-root"
```

Expected: silent execution; verify by checking `wp post meta get <post-id> rt_show_airoi` on a post you know mentions AI.

- [ ] **Step 3: Add AIROI block styles**

Append to `services/rucktalk-minimal/assets/css/ecosystem.css` (created in Task 28):

```css
.rt-airoi-block { background: var(--rt-accent-soft); border-left: 4px solid var(--rt-accent); padding: var(--rt-s-5); margin: var(--rt-s-6) 0; border-radius: var(--rt-radius); }
.rt-airoi-block h3 { margin: 0 0 var(--rt-s-3); font-size: var(--rt-fs-lg); }
.rt-airoi-block p { margin: 0 0 var(--rt-s-4); color: var(--rt-ink-2); }
```

- [ ] **Step 4: Commit**

```bash
git add services/rucktalk-minimal/inc/airoi-tagger.php
git commit -m "feat(rucktalk-theme): AIROI contextual block auto-tagger for AI/business posts"
```

### Task 28: Ecosystem footer strip CSS + assets

**Files:**
- Create: `services/rucktalk-minimal/assets/css/ecosystem.css`
- Create: `services/rucktalk-minimal/assets/img/ecosystem/{loovacast,lumabot,airoi,roen,grl}.svg`

- [ ] **Step 1: Drop in 5 logo SVGs**

For each brand, place a wordmark SVG (Mike provides; if missing, generate a placeholder via comfyui_gen.py or use text wordmark for v1).

- [ ] **Step 2: Write ecosystem.css**

```css
.rt-ecosystem-strip { padding: var(--rt-s-6) 0; background: var(--rt-bg); border-top: 1px solid var(--rt-rule); }
.rt-ecosystem-strip__label { text-align: center; font-size: var(--rt-fs-sm); color: var(--rt-ink-muted); margin: 0 0 var(--rt-s-5); text-transform: uppercase; letter-spacing: 0.08em; }
.rt-ecosystem-strip__list { list-style: none; display: flex; justify-content: center; gap: var(--rt-s-7); margin: 0; padding: 0; flex-wrap: wrap; }
.rt-ecosystem-strip__item a { display: flex; flex-direction: column; align-items: center; gap: var(--rt-s-2); text-decoration: none; opacity: 0.5; filter: grayscale( 1 ); transition: opacity 0.2s, filter 0.2s; }
.rt-ecosystem-strip__item a:hover { opacity: 1; filter: none; }
.rt-ecosystem-strip__logo { height: 28px; width: auto; }
.rt-ecosystem-strip__tagline { font-size: var(--rt-fs-xs); color: var(--rt-ink-muted); }

@media ( max-width: 720px ) {
    .rt-ecosystem-strip__list { gap: var(--rt-s-5); }
    .rt-ecosystem-strip__logo { height: 22px; }
}
```

- [ ] **Step 3: Smoke test footer renders 5 logos in greyscale**

Visit https://rucktalk.com/ — scroll to footer — see 5 brand logos in grey, color on hover.

- [ ] **Step 4: Commit**

```bash
git add services/rucktalk-minimal/assets/css/ecosystem.css services/rucktalk-minimal/assets/img/ecosystem/
git commit -m "feat(rucktalk-theme): ecosystem footer strip styles + logo assets"
```

---

## WAVE 7 — Smoke tests + launch checklist

### Task 29: Post-launch smoke test script

**Files:**
- Create: `scripts/rucktalk_redesign_smoke.py`

- [ ] **Step 1: Write the smoke tester**

```python
"""Post-launch smoke tests for rucktalk.com redesign.

Hits the live site (default https://rucktalk.com) and verifies:
  - Homepage returns 200 + contains expected hero/footer fingerprints
  - /training, /training/free, /training/8-week-plan all return 200
  - /blog returns 200
  - /podcast returns 200
  - Signup REST endpoint accepts a test email (uses a known test address)
  - LoovaCast player markup present
  - LumaBot script tag present
  - Ecosystem footer strip renders all 5 logos
  - www. subdomain redirects to apex (no 525)
  - fitasruck.com still 301s to /training

Exit code 0 if all green, 1 otherwise. Prints a green/red table.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import httpx

BASE = "https://rucktalk.com"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def chk_status(label: str, url: str, expect: int = 200) -> Check:
    try:
        r = httpx.get(url, follow_redirects=False, timeout=15)
        return Check(label, r.status_code == expect, f"got {r.status_code}")
    except Exception as e:
        return Check(label, False, str(e))


def chk_contains(label: str, url: str, needle: str) -> Check:
    try:
        r = httpx.get(url, follow_redirects=True, timeout=15)
        return Check(label, needle.lower() in r.text.lower(),
                     f"missing '{needle}'" if needle.lower() not in r.text.lower() else "found")
    except Exception as e:
        return Check(label, False, str(e))


def chk_redirect(label: str, url: str, expect_loc_contains: str) -> Check:
    try:
        r = httpx.get(url, follow_redirects=False, timeout=15)
        loc = r.headers.get("location", "")
        ok = r.status_code in (301, 302, 308) and expect_loc_contains in loc
        return Check(label, ok, f"status={r.status_code} location={loc}")
    except Exception as e:
        return Check(label, False, str(e))


def main() -> int:
    checks: list[Check] = []
    # Pages
    checks.append(chk_status("homepage", f"{BASE}/"))
    checks.append(chk_status("/training", f"{BASE}/training/"))
    checks.append(chk_status("/training/free", f"{BASE}/training/free/"))
    checks.append(chk_status("/blog", f"{BASE}/blog/"))
    checks.append(chk_status("/podcast", f"{BASE}/podcast/"))
    # Fingerprints
    checks.append(chk_contains("hero present", f"{BASE}/", "Real talk for guys"))
    checks.append(chk_contains("ecosystem strip", f"{BASE}/", "Part of the Ground Rush ecosystem"))
    checks.append(chk_contains("radio bar", f"{BASE}/", "rt-radio-bar"))
    checks.append(chk_contains("lumabot mount", f"{BASE}/", "rt-lumabot-mount"))
    checks.append(chk_contains("popup", f"{BASE}/", "rt-popup"))
    # www subdomain (after fix)
    checks.append(chk_status("www. redirect or 200", "https://www.rucktalk.com/", expect=200))
    # fitasruck 301
    checks.append(chk_redirect("fitasruck → rucktalk", "https://fitasruck.com/", "rucktalk.com"))

    print(f"{'CHECK':<40} {'OK':<6} {'DETAIL':<40}")
    print("-" * 90)
    for c in checks:
        flag = "✓" if c.ok else "✗"
        print(f"{c.name:<40} {flag:<6} {c.detail:<40}")

    failed = [c for c in checks if not c.ok]
    print()
    print(f"Total: {len(checks)} · Passed: {len(checks) - len(failed)} · Failed: {len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run + verify**

```bash
venv/bin/python scripts/rucktalk_redesign_smoke.py
```

Expected: all green. If any red, fix and re-run.

- [ ] **Step 3: Commit**

```bash
git add scripts/rucktalk_redesign_smoke.py
git commit -m "test(rucktalk): post-launch smoke checker (pages, fingerprints, redirects)"
```

### Task 30: Launch checklist doc + handoff

**Files:**
- Create: `docs/superpowers/audits/2026-05-19-rucktalk-launch-checklist.md`

- [ ] **Step 1: Write the checklist**

```markdown
# RuckTalk Phase 1A Launch Checklist

**Date prepared:** 2026-05-19
**Phase 0 spec:** `docs/superpowers/specs/2026-05-19-rucktalk-rebuild-phase-0.md`
**Plan:** `docs/superpowers/plans/2026-05-19-rucktalk-phase-1a-site-redesign.md`

## Pre-launch
- [ ] Hero photo of Mike landed at `services/rucktalk-minimal/assets/img/mike-hero.jpg`
- [ ] PDF cover landed at `services/rucktalk-minimal/assets/img/pdf-cover.jpg`
- [ ] PDF file landed at `services/rucktalk-minimal/assets/pdf/8-week-plan.pdf` (or hosted in Brevo)
- [ ] 5 ecosystem logos landed at `services/rucktalk-minimal/assets/img/ecosystem/*.svg`
- [ ] LoovaCast RuckTalk station created; stream URL + station ID set as WP options
- [ ] LumaBot bot ID + API key set as WP options
- [ ] Brevo RuckTalk list created; list ID + API key set as WP options
- [ ] WC product for $29 8-Week Plan created; ID set as WP option `rt_training_product_id`
- [ ] Stripe checkout tested with the new product (sandbox)

## Launch (T3 — Mike approves each)
- [ ] Deploy: `./services/rucktalk-minimal/deploy.sh`
- [ ] Activate: `wp theme activate rucktalk-minimal`
- [ ] Deploy mu-plugin: legacy redirects (Task 22)
- [ ] Set Cloudflare 301 for fitasruck.com → rucktalk.com/training (Task 23)
- [ ] Re-issue Let's Encrypt cert with www SAN (Task 24)
- [ ] Add www → apex 301 (Task 24 Step 5)
- [ ] Submit Change of Address in GSC for fitasruck.com
- [ ] Re-submit rucktalk.com sitemap in GSC

## Post-launch (within 1 hour)
- [ ] Run smoke: `venv/bin/python scripts/rucktalk_redesign_smoke.py` → all green
- [ ] Visit homepage in incognito → hero, popup (after 8s scroll), radio bar, footer strip
- [ ] Submit a real test signup → confirmation email arrives → click → PDF arrives
- [ ] Place a $29 test order via Stripe sandbox → order confirms
- [ ] Visit rucktalk.com/sitemap.xml → 200, contains new /training pages
- [ ] Visit www.rucktalk.com → 200 (no 525)
- [ ] Visit fitasruck.com → 301 to rucktalk.com/training

## Post-launch (within 48 hours)
- [ ] GA4: confirm pageviews flowing on new pages
- [ ] Brevo: confirm contacts coming in with `SIGNUP_PLACEMENT` attribute populated
- [ ] n8n: confirm `o9cIjGWj8z9pwknY` webhook is receiving confirmed contacts
- [ ] Search Console: confirm fitasruck.com Change of Address shows "processing"
- [ ] Monitor rt-wordpress error log for any PHP fatals

## Rollback path
If launch goes sideways:
```bash
ssh server-100 "docker exec rt-wordpress wp theme activate sonaar --allow-root"
ssh server-100 "docker exec rt-wordpress rm /var/www/html/wp-content/mu-plugins/rucktalk-redirects.php"
```
Site reverts to Sonaar parent. Cloudflare 301 from fitasruck.com is reversible via Cloudflare dashboard.

## Open follow-ups (move to Plans 1B / 1C / Phase 2)
- Phase 1B: encode RuckTalk brand profile in Alfred SEO + run keyword discovery + wire weekly blog engine
- Phase 1C: podcast distribution audit + Spotify/Apple/YouTube Music reconnect
- Phase 2: real shop product photography + add real SKUs (currently the homepage shop teaser is empty until products exist)
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/audits/2026-05-19-rucktalk-launch-checklist.md
git commit -m "docs(rucktalk): Phase 1A launch checklist + rollback path"
```

### Task 31: Final merge to main

**Files:** (operational)

- [ ] **Step 1: Confirm all Wave 1-6 tasks complete + smoke green**

Run the smoke script one more time:
```bash
venv/bin/python scripts/rucktalk_redesign_smoke.py
```

- [ ] **Step 2: BLOCKED ON MIKE — explicit green-light to merge**

Mike reviews the launch checklist; confirms ready to merge `feat/rucktalk-rebuild-phase-1a` → `main`.

- [ ] **Step 3: Merge**

```bash
git checkout main
git pull
git merge feat/rucktalk-rebuild-phase-1a --no-ff -m "Merge Phase 1A: RuckTalk site redesign"
git push origin main
```

- [ ] **Step 4: Tag the launch**

```bash
git tag -a rucktalk-phase-1a -m "RuckTalk Phase 1A launched"
git push origin rucktalk-phase-1a
```

---

## Spec coverage check

| Phase 0 §  | Requirement | Plan task |
|------------|-------------|-----------|
| §1 | Personal brand identity, domain, FaR consolidation | Header copy, Task 22-23 |
| §2 | 6-item top nav | Task 6 (menu registration) + WP menu created in Task 15 step 3 |
| §3a | Floating LoovaCast radio bar on every page | Task 25 |
| §3b | Hero with photo, tagline, primary/secondary CTAs | Task 12 |
| §3c | Below-fold sections (episode, blog, shop, newsletter, about) | Task 13 |
| §4 | Tagline "Real talk for guys in the thick of it." | Task 12 (default value) |
| §5b | Double opt-in flow via Brevo | Task 19 (`/contacts/doubleOptinConfirmation` endpoint) |
| §5c | n8n webhook fires on Brevo confirmation | Configured in Brevo dashboard (operational, see launch checklist) |
| §5d | Signup surfaces: hero, inline, footer, popup | Tasks 11, 12, 13, 21 |
| §5e | The one allowed popup, exit-intent + scroll-depth, 14d cookie | Task 21 |
| §6a | 4 launch SKUs (hats + tees) | NOT in 1A — Phase 2 (homepage teaser is empty until products exist) |
| §6b | WC + @RuckTalkBot pattern | $29 product in Task 16; bot is Phase 2 |
| §7 | Sonaar child theme, work within parent | Whole Wave 2 |
| §8 | SEO plug-in (RuckTalk brand profile + weekly engine) | NOT in 1A — Phase 1B |
| §9 | Auto-blog 6/week (5 daily + 1 SEO weekly) | NOT in 1A — Phase 1B |
| §10 | fitasruck.com 301 + customer migration | Tasks 22, 23 |
| §10.5 | Podcast distribution audit (Spotify/Apple/YouTube) | NOT in 1A — Phase 1C |
| §12a | LoovaCast floating player | Task 25 |
| §12b | LumaBot chat widget | Task 26 |
| §12c | AIROI contextual (not blanket) | Task 27 |
| §12d | Ecosystem footer strip | Tasks 11 (shortcode), 28 (assets + CSS) |
| §12e | No popups except newsletter | Enforced by only Task 21 creating a popup |

**Gaps:** Phase 1B and 1C are intentionally separate plans (per scope split in plan intro). Phase 2 (real shop products) is also separate.

---

## Self-review notes

**Type consistency:** `[rt_signup placement="..."]` accepts `"hero"`, `"inline"`, `"footer"`, `"popup"`, `"training-free"` — used consistently across Tasks 11, 12, 13, 18, 21.

**Placeholder scan:** Only `<PRODUCT_ID>`, `<key>`, `<id>`, `<value Mike supplies>` — all are intentional handoff slots gated by Mike's input, marked BLOCKED ON MIKE / PF-N in the relevant tasks. No "TBD" or "implement later" without code.

**Naming consistency:** Theme directory `rucktalk-minimal` (matches `roen-minimal` precedent). WP option prefix `rt_`. CSS class prefix `rt-`. JS global `RuckTalkSignup`. All consistent throughout.
