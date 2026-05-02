# Roen Handmade — Website Brand & Layout Spec

**Date**: 2026-05-02
**Owner**: Mike Johnson
**Status**: Approved brand direction, ready for implementation plan

## Context

Roen Handmade (roenhandmade.com) is the handmade jewelry brand of Mike's wife. The site exists today on a default WordPress + WooCommerce install with the Twenty Twenty-Four theme — no real branding, 7 manually-uploaded products. This spec captures the brand and layout decisions that will drive the next phase of work: a custom theme/layout that turns the default install into a credible direct-to-consumer jewelry brand.

This work happens **before** the photo-intake bot, the Meta Shop submission, and the App Review. Reason: the website is the credibility floor for everything downstream. Meta Shop reviewers visit the site during approval; an unbranded WordPress default would risk rejection. The bot publishes to this site, so the product page template needs to exist before the bot starts pushing entries to it.

This spec covers **brand identity and layout structure only**. It does not cover the photo-intake pipeline, Meta Catalog sync, or the bundles system — those are separate downstream specs. See `.claude/plans/ok-so-here-s-the-wobbly-tarjan.md` for the broader project plan.

## Locked Brand Decisions

Decisions 1–4 were chosen by Mike from visual mockups (saved in `.superpowers/brainstorm/`). Decisions 5–7 follow from those choices via short clarifying conversations in chat.

### 1. Direction — Minimal Modern
- White-space-forward, thin sans-serif, restrained palette
- Reference points: Mejuri, Aritzia, Cuyana
- Lets the jewelry carry the page; the brand frame is quiet

### 2. Wordmark — `roen` (lowercase, ultra-thin)
- Single lowercase word, no tagline embedded in the lockup
- Font: Inter (200 weight) or system equivalent
- Letter-spacing: -2px at hero size, -0.5px at body size
- The brand mark is `roen`. The legal/business name is "Roen Handmade" (used in footer, legal copy, returns policy)

### 3. Photography — Marble Tabletop
- Cool grey-veined marble or polished stone background
- Subtle drop shadow under each piece
- Soft top-left light source, neutral white balance
- All AI-generated backdrops produced by the (future) intake bot must conform to this look. No props, no flat-lay clutter, no warm linen. One aesthetic, end to end.

### 4. Homepage Layout — Product-First Direct (L2)
- No big marble hero. The first thing a visitor sees on the homepage is products.
- Order, top to bottom:
  1. Thin nav: `roen` wordmark · shop · about · cart
  2. Compact tagline block (~80px tall): `handmade jewelry, made in atlanta.` / `new pieces every week.`
  3. Category pills: all · bracelets · earrings · necklaces · rings (clickable filters that reload the grid below)
  4. Product grid (4 columns desktop, 2 columns mobile, ~16 pieces)
  5. Footer: roen handmade · instagram · contact · privacy · returns

### 5. Color Palette
- **Primary background**: `#FFFFFF`
- **Secondary background** (sections, cards): `#FAF9F6` (soft bone)
- **Primary text**: `#1A1A1A`
- **Secondary text**: `#666666`
- **Hairlines / borders**: `#EEEEEE`
- **Accent — Warm Terracotta**: `#B85C3D`
  - Used only on: primary CTAs (`Add to Cart`, `Buy Now`), link hovers, sale price strikethrough, active category pill underline
  - Never used as background fill or for body text. The accent should feel like punctuation, not pigment.

### 6. Typography
- **Display / wordmark**: Inter, weight 200, tight tracking
- **Body**: Inter, weight 400
- **Product titles**: Inter, weight 400, mixed case, letter-spacing -0.2px
- **Prices**: Inter, weight 300
- **No serifs anywhere.** No display fonts beyond Inter. One typeface, multiple weights.

### 7. Brand Voice
- Roen is presented as a **brand name**, not a person
- About page copy uses third-person: "Roen is a small Atlanta jewelry studio..." — never "I" or "Roen Johnson"
- This is a deliberate choice; if Mike's wife wants a more personal voice later, it's a one-page rewrite, not a redesign

## Architecture & Implementation Approach

### WordPress Theme Strategy

**Chosen approach**: Build a child theme of an established WooCommerce-compatible base, NOT a custom-from-scratch theme.

**Why**: 
- A from-scratch theme is 2-3 weeks of work and creates a long-term maintenance burden. WooCommerce has hundreds of template hooks; replicating them all is wasted effort for a 1-employee jewelry brand.
- A child theme inherits all the WooCommerce plumbing (cart, checkout, account pages, hooks, payment integrations) and only overrides what we need: header, footer, homepage, product card, product detail page.
- Mike's existing AG Event Template System (`eveny-child` theme on 104) is precedent for this pattern in the fleet — proven to work.

**Base theme decision**: **Storefront** is already installed on this site (per `wp theme list`). It's the official WooCommerce theme, deeply tested, and explicitly designed to be child-themed.

**Child theme name**: `roen-minimal`

### File Structure (in container)
```
/var/www/html/wp-content/themes/roen-minimal/
├── style.css                    # Theme metadata + CSS custom properties (colors, type scale)
├── functions.php                # Enqueue parent + child styles, register theme supports
├── header.php                   # Custom thin nav (overrides Storefront's default)
├── footer.php                   # Custom minimal footer
├── front-page.php               # L2 homepage layout (tagline + pills + grid)
├── single-product.php           # Custom product detail page (marble bg, large image, terracotta CTA)
├── archive-product.php          # Shop / category archive page
└── assets/
    ├── css/roen.css             # Brand-specific overrides (typography, accent color, spacing)
    └── img/                     # Logo SVG, social icons (no large imagery — keep theme light)
```

### Page-Level Specifications

**Homepage (front-page.php)**:
- Set as static homepage in WP Settings → Reading
- Pulls 16 most recent products via WC_Product_Query
- Category pills filter via JS query string (?cat=bracelets), no page reload
- Mobile breakpoint at 768px: pills wrap, grid drops to 2 columns

**Product detail page (single-product.php)**:
- Override Storefront's default
- Layout: large product image left (60%), info column right (40%) on desktop; stacked on mobile
- Image gallery: vertical thumbnails on desktop, horizontal swipe on mobile
- Price in Inter 300 below title
- Description in Inter 400, single column, max 60ch line length
- "Add to Cart" button: terracotta fill, white text, no border radius, full-width on mobile
- Bundle/combo callout block (placeholder for downstream Chapter 5 work — render hidden until combos exist)

**About page (page-about.php template)**:
- Single-column, max 60ch
- One product hero image at top (marble flat-lay)
- Brand-voice copy as specified above
- No staff bio, no founder photo, no surname

**Shop / category archive**:
- Same product grid as homepage
- Pills shown as breadcrumb at top
- Sort dropdown (newest, price low-high, price high-low) in top-right

### Existing Infrastructure to Reuse

- `integrations/wordpress/client.py` (1359 lines) — already covers theme management, REST API calls, media upload. Use it for any programmatic theme installs or content seeding.
- `integrations/servers/manager.py` — SSH + Docker exec for WP-CLI commands inside `roenhandmade-wp` container.
- Existing 7 products, the Privacy Policy and Refund/Returns Policy pages (published in the previous session) all stay as-is.

### Out of Scope (Explicitly)

- **No new plugins**. Resist the urge to add page builders (Elementor, Beaver Builder), advanced custom post types, slider plugins, or AJAX product filters beyond simple JS. The site is a small product catalog — extra plugins are bloat and security surface.
- **No on-model product imagery**. Per the broader project plan, on-model jewelry compositing is parked. Marble flat-lay only.
- **No journal/blog**. Roen is presented as a brand, not a writer.
- **No fancy animations**. Minor hover state lifts (translate-y 2px) and color transitions (200ms) — that's it. No scroll-triggered reveals, no parallax, no carousels.
- **No newsletter popup**. If we add email capture later, it goes in the footer, not as a modal.

## Verification Criteria

When the implementation is complete, the spec is met if all of these are true:

- [ ] `wp theme list` shows `roen-minimal` as the active theme on roenhandmade-wp container
- [ ] roenhandmade.com renders the L2 homepage (tagline + pills + grid + footer) with no Storefront-default content visible
- [ ] All 7 existing products render correctly in the grid and on the new product detail page
- [ ] Add to Cart works end-to-end (product → cart → checkout pages exist and are reachable)
- [ ] Mobile rendering is clean at 375px, 768px, 1024px, 1440px viewports
- [ ] Color accent appears only on CTAs and hover states (not as background fill)
- [ ] Page weight: <500KB total transferred for homepage cold load (no heavy JS, no webfont sprawl)
- [ ] Lighthouse Performance score ≥85 on mobile
- [ ] Footer links to Privacy Policy and Refund/Returns Policy resolve to the existing published pages
- [ ] No console errors in Chrome DevTools on any page

## Open Questions / Future Decisions

These are intentionally deferred to later specs:

- **Custom font hosting** (self-host Inter vs Google Fonts CDN) — defer to implementation; default to Google Fonts unless privacy concerns surface
- **Newsletter signup** — out of scope for v1; revisit after Brevo email infra is wired up
- **Search** — WooCommerce has built-in search; revisit if catalog grows past ~50 products
- **Localization** — English-only, USD-only for v1
