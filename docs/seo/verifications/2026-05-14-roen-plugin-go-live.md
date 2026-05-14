# Roen alfred-seo plugin go-live verification

**Date:** 2026-05-14
**Site:** https://www.roenhandmade.com
**Plugin version:** 0.1.0 (deployed to `roenhandmade-wp` container on `server-104`)
**Deploy commit:** 61cc66b

## Plugin status

```
$ ssh server-104 'docker exec roenhandmade-wp wp plugin list ... | grep alfred-seo'
alfred-seo,active,none,0.1.0,,off
```

Initial settings written via `wp option update alfred_seo_settings`:
`site_slug=roen`, `business_name=Roen`, `business_type=LocalBusiness`,
IG/FB handles=`roenhandmade`, address=Atlanta GA US,
`alfred_endpoint=https://aialfred.groundrushcloud.com`,
`alt_text_enabled=true`, `sitemap_enabled=true`.

## Step 1 — Sitemap reachable

```
$ curl -s -o /dev/null -w "HTTP %{http_code}\n" https://www.roenhandmade.com/alfred-sitemap.xml
HTTP 200
```

First 6 lines of XML:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://www.roenhandmade.com/alfred-sitemap-pages.xml</loc><lastmod>2026-05-14T21:45:24+00:00</lastmod></sitemap>
  <sitemap><loc>https://www.roenhandmade.com/alfred-sitemap-posts.xml</loc><lastmod>2026-05-14T21:45:24+00:00</lastmod></sitemap>
  <sitemap><loc>https://www.roenhandmade.com/alfred-sitemap-products.xml</loc><lastmod>2026-05-14T21:45:24+00:00</lastmod></sitemap>
  <sitemap><loc>https://www.roenhandmade.com/alfred-sitemap-categories.xml</loc><lastmod>2026-05-14T21:45:24+00:00</lastmod></sitemap>
</sitemapindex>
```

PASS — 4 sub-sitemaps as expected.

## Step 2 — Products sitemap URL count

```
$ curl -s https://www.roenhandmade.com/alfred-sitemap-products.xml | grep -c '<url>'
93
```

PASS — 93 product URLs (≥90 expected).

## Step 3 — Product schema injection

URL: `https://www.roenhandmade.com/product/red-bead-toggle-necklace/`

Two `application/ld+json` blocks present:
1. **alfred-seo @graph block** (Product + BreadcrumbList + WebSite from our dispatcher).
   First 600 chars:
   ```json
   {
     "@context": "https://schema.org",
     "@graph": [
       {
         "@type": "Product",
         "name": "Red Bead Toggle Necklace",
         "description": "Large faceted red beads meet small matte blue beads on a knotted cord. A polished gold toggle clasp and textured pendant complete the piece.",
         "sku": "REDBTOG-7803",
         "url": "https://www.roenhandmade.com/product/red-bead-toggle-necklace/",
         "image": "https://www.roenhandmade.com/wp-content/uploads/2026/05/roen_upload_photo_1778107724644.jpg",
         "brand": { "@type": "Brand", "name": "Roen" },
         "offers": ...
   ```
2. **WooCommerce's pre-existing Product schema** (untouched — co-exists fine).

PASS — alfred-seo product schema valid with name, description, SKU, image, brand, offers.

## Step 4 — OG tags on a product page

Note: homepage tagline is unset in WP General Settings → its og:description is empty
and meta description falls through to bloginfo description (empty). Verified on a
**real product page** instead (homepage is a known config gap, not a plugin bug):

```
$ curl -s https://www.roenhandmade.com/product/red-bead-toggle-necklace/ \
    | grep -oE 'property="og:[a-z_]+" content="[^"]*"' | head -10
property="og:type" content="product"
property="og:title" content="Red Bead Toggle Necklace"
property="og:description" content="Large faceted red beads meet small matte blue beads on a knotted cord. A polished gold toggle clasp and textured pendant complete the piece."
property="og:url" content="https://www.roenhandmade.com/product/red-bead-toggle-necklace/"
property="og:site_name" content="Roen"
property="og:image" content="https://www.roenhandmade.com/wp-content/uploads/2026/05/roen_upload_photo_1778107724644.jpg"
```

Homepage output (for comparison — partial because tagline empty):

```
property="og:type" content="article"
property="og:title" content="Home"
property="og:description" content=""
property="og:url" content="https://www.roenhandmade.com/"
property="og:site_name" content="Roen"
```

PASS on product pages. Homepage flagged below.

## Step 5 — Meta description

Product page:

```
$ curl -s https://www.roenhandmade.com/product/red-bead-toggle-necklace/ \
    | grep -oE '<meta name="description" content="[^"]*"'
<meta name="description" content="Large faceted red beads meet small matte blue beads on a knotted cord. A polished gold toggle clasp and textured pendant complete the piece."
```

Homepage: empty (tagline unset — see Mike-actions below).

PASS on product pages.

## Step 6 — robots.txt sitemap line

```
$ curl -s https://www.roenhandmade.com/robots.txt | grep -i sitemap
(no output)
```

**Root cause:** `wp option get blog_public` returns `0` — Roen has "Discourage search
engines from indexing this site" enabled. Our `robots_txt` filter (per the plan spec)
honors this by early-returning when `$public === false`, so the `Sitemap:` line is
intentionally suppressed. WP-CLI confirms the filter works correctly when
`blog_public=1`:

```
$ wp eval 'echo apply_filters("robots_txt", "BASE\n", true);'
BASE
Sitemap: https://www.roenhandmade.com/alfred-sitemap.xml
```

NOT A PLUGIN BUG — flip `blog_public` to `1` to advertise the sitemap.

## Step 7 — Google Rich Results Test (Mike-action pending)

URL to test: https://search.google.com/test/rich-results?url=https%3A%2F%2Fwww.roenhandmade.com%2Fproduct%2Fred-bead-toggle-necklace%2F

Two Product blocks present (ours + WooCommerce's); the Rich Results Test may flag
the duplicate. If it does, the next iteration should suppress WooCommerce's schema
when alfred-seo is active.

**Status:** Mike-action pending — paste URL into the test, capture pass/fail screenshot
and any warnings.

## Step 8 — GSC + Bing sitemap submission (Mike-action pending)

- GSC: https://search.google.com/search-console → roenhandmade.com → Sitemaps → add `alfred-sitemap.xml`
- Bing: https://www.bing.com/webmasters → roenhandmade.com → Sitemaps → submit

**Status:** Mike-action pending. Note: with `blog_public=0`, GSC/Bing may refuse the
sitemap until the site is made public. Recommended sequence: (1) flip `blog_public=1`,
(2) verify `Sitemap:` line appears in robots.txt, (3) submit to GSC + Bing.

## Mike-actions summary

1. **Set `blog_public=1`** in WP Settings → Reading (uncheck "Discourage search
   engines"). Required for both robots.txt sitemap line AND for GSC/Bing to accept
   the submission. Or run:
   ```bash
   ssh server-104 'docker exec roenhandmade-wp wp option update blog_public 1 --allow-root --path=/var/www/html'
   ```
2. **Set Site Tagline** in WP Settings → General → Tagline. Currently empty, which
   makes homepage `<meta name="description">` and `og:description` empty.
3. **Google Rich Results Test** (Step 7) — confirm the duplicate Product schema
   doesn't blow up validation. If it does, we'll add a `remove_action` for WC's
   structured data in a follow-up patch.
4. **GSC + Bing sitemap submission** (Step 8) — after blog_public is flipped.

## What works out of the gate

- Sitemap index + 4 sub-sitemaps (HTTP 200, 93 product URLs)
- Product schema injection (with offers, image, brand, sku)
- OG tags on product pages (type, title, description, url, site_name, image)
- Meta description on product pages
- Plugin idempotent re-deploy via `services/alfred-seo/deploy.sh`
- Settings stored as JSON in `alfred_seo_settings` option

## What needs follow-up

- `blog_public=0` is suppressing robots.txt sitemap line (config, not plugin)
- Homepage tagline empty (config, not plugin)
- Possible duplicate Product schema with WooCommerce (Rich Results Test will tell)
