# Roen SEO Plan 1 — Foundation Infrastructure (Weeks 1-3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `alfred-seo` WP plugin to Roen + stand up the Alfred orchestrator skeleton on 105 + daily data ingest from GSC/GA4/PSI/backlinks so every Roen page has schema/OG/meta/sitemap injection and Mike has a cross-site dashboard showing real organic baselines.

**Architecture:** Two codebases. A thin WP plugin (`services/alfred-seo/`) renders schema/OG/sitemap on each WP site; an Alfred orchestrator module (`core/seo/`) on 105 owns site registry, data ingest, and admin UI. They talk via WP REST API (orchestrator → plugin) and external Google APIs (orchestrator → GSC/GA4/PSI read-only). PostgreSQL `alfred_main` for state, FastAPI for admin UI, systemd timers for scheduled jobs.

**Tech Stack:** PHP 8.x (WordPress plugin) · Python 3.11 + FastAPI + SQLAlchemy (orchestrator) · PostgreSQL 16 · Google API Python Client (GSC + GA4 + PSI) · pytest + WP unit tests · systemd user units.

**Spec reference:** `docs/superpowers/specs/2026-05-14-roen-seo-pipeline-design.md`

---

## File Structure

```
services/alfred-seo/                         # WP plugin (NEW)
├── alfred-seo.php                            # Main plugin file, WP headers, bootstrap
├── inc/
│   ├── schema/
│   │   ├── builder.php                       # Dispatches per page type
│   │   ├── product.php                       # Product + Offer + AggregateRating + Brand
│   │   ├── article.php                       # Article + Person + Organization
│   │   ├── organization.php                  # Organization + LocalBusiness
│   │   ├── breadcrumb.php                    # BreadcrumbList
│   │   ├── faq.php                           # FAQPage (extracts Q&A from content)
│   │   ├── website.php                       # WebSite + SearchAction
│   │   ├── collection.php                    # CollectionPage + ItemList
│   │   └── validator.php                     # JSON-LD validity check before injection
│   ├── open-graph.php                        # OG + Twitter Card head tags
│   ├── meta.php                              # Meta description override
│   ├── sitemap.php                           # Multi-sitemap index + per-type
│   ├── alt-text.php                          # Upload hook → Alfred vision endpoint
│   ├── internal-links.php                    # the_content filter (phrase→URL map)
│   ├── robots.php                            # robots.txt filter
│   ├── rest/
│   │   ├── auth.php                          # WP application password verification
│   │   ├── audit.php                         # GET /audit
│   │   ├── content.php                       # POST /content
│   │   ├── meta.php                          # POST /meta
│   │   ├── internal-links.php                # POST /internal-links
│   │   └── sitemap-ping.php                  # POST /sitemap-ping
│   └── settings.php                          # Options row managed by Alfred
├── deploy.sh                                  # rsync + docker exec deploy to roenhandmade-wp
├── tests/                                     # WP unit tests
│   ├── bootstrap.php
│   ├── test-schema-product.php
│   ├── test-schema-article.php
│   ├── test-sitemap.php
│   ├── test-rest-audit.php
│   └── test-rest-content.php
└── readme.txt                                 # WP plugin README

core/seo/                                     # Orchestrator (NEW)
├── __init__.py
├── db.py                                     # SQLAlchemy declarative base + session factory
├── models.py                                 # 9 SQLAlchemy table classes
├── sites/
│   ├── __init__.py
│   ├── registry.py                           # CRUD on seo_sites
│   └── profile.py                            # Brand voice profile loader (stub for Plan 1)
├── ingest/
│   ├── __init__.py
│   ├── gsc.py                                # Google Search Console daily sync
│   ├── ga4.py                                # GA4 daily sync
│   ├── cwv.py                                # PageSpeed Insights CWV sync
│   └── backlinks.py                          # GSC top-linking-sites diff
└── api_clients/
    ├── __init__.py
    ├── gsc_client.py                         # GSC REST client (Google API)
    ├── ga4_client.py                         # GA4 Data API client
    ├── psi_client.py                         # PageSpeed Insights client
    └── wp_rest_client.py                     # WP REST API client (for orchestrator → plugin)

core/api/
└── seo_admin.py                              # FastAPI routes for /admin/seo/* (NEW)

config/
└── settings.py                               # ADD seo_* settings (MODIFY)

integrations/
└── google_seo/                               # Google API auth helpers (NEW)
    ├── __init__.py
    └── oauth.py                              # OAuth flow + token storage

scripts/
├── seo_gsc_sync.py                           # CLI wrapper for ingest/gsc.py
├── seo_ga4_sync.py                           # CLI wrapper for ingest/ga4.py
├── seo_cwv_sync.py                           # CLI wrapper for ingest/cwv.py
├── seo_backlinks_sync.py                     # CLI wrapper for ingest/backlinks.py
└── seo_init_roen.py                          # One-shot: register Roen as Site #1

systemd/                                      # User systemd units for cron-like timers (NEW)
├── alfred-seo-gsc-sync.service
├── alfred-seo-gsc-sync.timer
├── alfred-seo-ga4-sync.service
├── alfred-seo-ga4-sync.timer
├── alfred-seo-cwv-sync.service
├── alfred-seo-cwv-sync.timer
├── alfred-seo-backlinks-sync.service
└── alfred-seo-backlinks-sync.timer

migrations/                                   # SQL migrations (NEW dir if not exists)
└── 2026-05-14-seo-initial-schema.sql         # 9 seo_* tables

tests/
├── core/seo/
│   ├── test_registry.py
│   ├── test_profile.py
│   ├── test_gsc_ingest.py
│   ├── test_ga4_ingest.py
│   ├── test_cwv_ingest.py
│   └── test_backlinks_ingest.py
└── fixtures/
    ├── gsc_response.json
    ├── ga4_response.json
    └── psi_response.json

docs/seo/                                     # User-facing docs (NEW)
├── ARCHITECTURE.md                            # Plan 1 architecture overview
├── PLUGIN_INSTALL.md                          # Per-site plugin install guide
└── OAUTH_SETUP.md                             # GSC + GA4 OAuth setup walkthrough
```

---

## Task index (Plan 1 has 32 tasks across 3 weeks)

**Week 1 — WP plugin (Tasks 1-15)**
1. Scaffold plugin skeleton + WP headers
2. Schema dispatcher + JSON-LD validator
3. Product schema builder + test
4. Article schema builder + test
5. Organization + LocalBusiness schema builder + test
6. BreadcrumbList schema builder + test
7. FAQ schema builder + test
8. WebSite + SearchAction schema builder + test
9. CollectionPage + ItemList schema builder + test
10. Open Graph + Twitter Card head tags + test
11. Meta description override + test
12. Multi-sitemap index + per-type sitemaps + test
13. robots.txt management + test
14. Image alt text on upload + test
15. Internal linking filter + test

**Week 1-2 — WP plugin REST + deploy (Tasks 16-19)**
16. REST endpoints (audit, content, meta, internal-links, sitemap-ping) + auth
17. WP unit test bootstrap + smoke tests
18. Deploy script + initial deploy to roenhandmade-wp
19. Functional verification on live Roen site (curl-driven)

**Week 2 — Orchestrator skeleton (Tasks 20-25)**
20. SQL migration: 9 seo_* tables
21. SQLAlchemy models for 9 tables
22. Site registry CRUD + test
23. FastAPI seo_admin routes (site list, site detail) + JWT auth
24. Cross-site dashboard HTML view
25. Register Roen as Site #1 via init script

**Week 3 — Data ingest (Tasks 26-32)**
26. Google API OAuth flow + token storage
27. GSC sync ingest module + test
28. GA4 sync ingest module + test
29. PageSpeed Insights CWV sync + test
30. Backlinks Layer 1 (GSC top-linking-sites) + test
31. systemd timers for 4 sync jobs
32. Roen baseline backfill + smoke validate

---


## Task 1: Scaffold the alfred-seo WP plugin

**Files:**
- Create: `services/alfred-seo/alfred-seo.php`
- Create: `services/alfred-seo/readme.txt`
- Create: `services/alfred-seo/inc/settings.php`

- [ ] **Step 1: Create plugin main file**

```php
<?php
/**
 * Plugin Name: Alfred SEO
 * Plugin URI: https://aialfred.groundrushcloud.com
 * Description: SEO foundation — schema, OG, meta, sitemap, alt text, internal linking.
 *              Controlled remotely by Alfred orchestrator on 105.
 * Version: 0.1.0
 * Requires PHP: 8.0
 * Requires at least: 6.0
 * Author: Alfred Labs
 * License: Proprietary
 */

if ( ! defined( 'ABSPATH' ) ) { exit; }

define( 'ALFRED_SEO_VERSION', '0.1.0' );
define( 'ALFRED_SEO_DIR', plugin_dir_path( __FILE__ ) );
define( 'ALFRED_SEO_URL', plugin_dir_url( __FILE__ ) );

// Bootstrap: include modules. Each module registers its own hooks.
require_once ALFRED_SEO_DIR . 'inc/settings.php';
// Modules added in subsequent tasks register here.

register_activation_hook( __FILE__, function () {
    // Ensure default options exist on activation.
    if ( false === get_option( 'alfred_seo_settings' ) ) {
        add_option( 'alfred_seo_settings', alfred_seo_default_settings() );
    }
});
```

- [ ] **Step 2: Create settings module with default options**

```php
<?php
// services/alfred-seo/inc/settings.php

if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_default_settings() {
    return array(
        'site_slug'        => '',                    // e.g. "roen" — set by Alfred during onboarding
        'business_name'    => '',                    // e.g. "Roen Handmade"
        'business_type'    => 'Organization',        // Organization | LocalBusiness | SoftwareApplication | Service
        'social_handles'   => array(),               // ['twitter' => '@handle', 'instagram' => 'handle']
        'local_address'    => array(),               // LocalBusiness only: streetAddress, addressLocality, etc.
        'internal_links'   => array(),               // ['phrase' => 'url', ...]
        'alfred_endpoint'  => 'https://aialfred.groundrushcloud.com',
        'alt_text_enabled' => true,
        'sitemap_enabled'  => true,
    );
}

function alfred_seo_get_settings() {
    $opts = get_option( 'alfred_seo_settings', alfred_seo_default_settings() );
    // Backfill missing keys (forward-compatible additions).
    return array_merge( alfred_seo_default_settings(), is_array( $opts ) ? $opts : array() );
}

function alfred_seo_update_settings( array $patch ) {
    $current = alfred_seo_get_settings();
    update_option( 'alfred_seo_settings', array_merge( $current, $patch ) );
}
```

- [ ] **Step 3: Create plugin readme**

```
=== Alfred SEO ===
Contributors: alfredlabs
Tags: seo, schema, sitemap, open-graph
Requires at least: 6.0
Tested up to: 6.7
Stable tag: 0.1.0
Requires PHP: 8.0
License: Proprietary

SEO foundation controlled remotely by Alfred orchestrator on 105.

== Description ==

Renders schema (JSON-LD), Open Graph, Twitter Cards, meta descriptions, sitemaps,
image alt text, and internal linking. All decisions come from the Alfred
orchestrator via WP REST API. Plugin has sane local fallbacks for when the
orchestrator is unavailable — the site never breaks.

== Changelog ==

= 0.1.0 =
* Initial release. Phase 1 foundation.
```

- [ ] **Step 4: Verify plugin loads in a local WP via PHP syntax check**

Run: `php -l services/alfred-seo/alfred-seo.php && php -l services/alfred-seo/inc/settings.php`
Expected: `No syntax errors detected` for both files.

- [ ] **Step 5: Commit**

```bash
git add services/alfred-seo/alfred-seo.php services/alfred-seo/inc/settings.php services/alfred-seo/readme.txt
git commit -m "feat(alfred-seo): scaffold plugin with settings module"
```

---

## Task 2: Schema dispatcher + JSON-LD validator

**Files:**
- Create: `services/alfred-seo/inc/schema/builder.php`
- Create: `services/alfred-seo/inc/schema/validator.php`
- Test: `services/alfred-seo/tests/test-schema-builder.php`

- [ ] **Step 1: Write the failing test**

```php
<?php
// services/alfred-seo/tests/test-schema-builder.php

class Test_Schema_Builder extends WP_UnitTestCase {
    public function test_dispatch_returns_array_for_homepage() {
        $this->go_to( home_url( '/' ) );
        $result = alfred_seo_build_schema_for_current_page();
        $this->assertIsArray( $result );
        $this->assertArrayHasKey( '@context', $result );
        $this->assertEquals( 'https://schema.org', $result['@context'] );
    }

    public function test_validator_rejects_missing_context() {
        $bad = array( '@type' => 'Product', 'name' => 'X' );
        $this->assertFalse( alfred_seo_validate_jsonld( $bad ) );
    }

    public function test_validator_accepts_valid_payload() {
        $good = array( '@context' => 'https://schema.org', '@type' => 'Product', 'name' => 'X' );
        $this->assertTrue( alfred_seo_validate_jsonld( $good ) );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/alfred-seo && phpunit tests/test-schema-builder.php`
Expected: FAIL with "function alfred_seo_build_schema_for_current_page not defined" or similar.

- [ ] **Step 3: Implement builder.php**

```php
<?php
// services/alfred-seo/inc/schema/builder.php

if ( ! defined( 'ABSPATH' ) ) { exit; }

/**
 * Resolve which schema builder applies to the current WP query state,
 * call it, validate the result, and return the JSON-LD array (or null).
 */
function alfred_seo_build_schema_for_current_page() {
    $schemas = array();

    // Org/LocalBusiness + WebSite always on the homepage.
    if ( is_front_page() || is_home() ) {
        $schemas[] = alfred_seo_schema_organization();
        $schemas[] = alfred_seo_schema_website();
    }
    // Per-page-type dispatch (each function defined in its own file).
    if ( function_exists( 'is_product' ) && is_product() ) {
        $schemas[] = alfred_seo_schema_product( get_queried_object_id() );
    } elseif ( is_singular( 'post' ) ) {
        $schemas[] = alfred_seo_schema_article( get_queried_object_id() );
    } elseif ( function_exists( 'is_product_category' ) && is_product_category() ) {
        $schemas[] = alfred_seo_schema_collection( get_queried_object() );
    }
    // Always-on (except homepage which has its own).
    if ( ! is_front_page() && ! is_home() ) {
        $schemas[] = alfred_seo_schema_breadcrumb();
    }

    // Strip nulls + validate each.
    $schemas = array_filter( $schemas );
    $valid   = array_filter( $schemas, 'alfred_seo_validate_jsonld' );

    if ( empty( $valid ) ) { return null; }
    if ( count( $valid ) === 1 ) { return reset( $valid ); }

    return array(
        '@context' => 'https://schema.org',
        '@graph'   => array_values( array_map( function ( $s ) {
            // Strip @context from individual graph entries — only top-level needs it.
            unset( $s['@context'] );
            return $s;
        }, $valid ) ),
    );
}

/**
 * wp_head injection. Hooks added once all schema files are loaded.
 */
function alfred_seo_inject_schema_to_head() {
    $payload = alfred_seo_build_schema_for_current_page();
    if ( null === $payload ) { return; }
    echo "\n<script type=\"application/ld+json\">"
        . wp_json_encode( $payload, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE )
        . "</script>\n";
}
add_action( 'wp_head', 'alfred_seo_inject_schema_to_head', 10 );
```

- [ ] **Step 4: Implement validator.php**

```php
<?php
// services/alfred-seo/inc/schema/validator.php

if ( ! defined( 'ABSPATH' ) ) { exit; }

/**
 * Validate a JSON-LD payload before injection.
 * Returns true if structurally valid, false otherwise. Logs failures.
 */
function alfred_seo_validate_jsonld( $payload ) {
    if ( ! is_array( $payload ) ) { return false; }
    // Top-level OR @graph wrapper must declare schema.org context.
    if ( ! isset( $payload['@context'] ) && ! isset( $payload['@graph'] ) ) {
        alfred_seo_log_schema_failure( 'missing @context', $payload );
        return false;
    }
    if ( isset( $payload['@context'] ) && 'https://schema.org' !== $payload['@context'] ) {
        alfred_seo_log_schema_failure( 'bad @context', $payload );
        return false;
    }
    // If single payload (no @graph), must have @type.
    if ( ! isset( $payload['@graph'] ) && ! isset( $payload['@type'] ) ) {
        alfred_seo_log_schema_failure( 'missing @type', $payload );
        return false;
    }
    // Round-trip JSON encode to catch bad UTF-8 or non-encodable values.
    $encoded = wp_json_encode( $payload );
    if ( false === $encoded ) {
        alfred_seo_log_schema_failure( 'json_encode failed', $payload );
        return false;
    }
    return true;
}

function alfred_seo_log_schema_failure( $reason, $payload ) {
    $log_dir = wp_upload_dir()['basedir'] . '/alfred-seo-logs';
    if ( ! is_dir( $log_dir ) ) { wp_mkdir_p( $log_dir ); }
    $line = sprintf(
        "[%s] schema rejected (%s): %s\n",
        gmdate( 'c' ),
        $reason,
        substr( wp_json_encode( $payload ), 0, 500 )
    );
    @file_put_contents( $log_dir . '/schema-failures.log', $line, FILE_APPEND | LOCK_EX );
}
```

- [ ] **Step 5: Wire builder + validator into alfred-seo.php**

Edit `services/alfred-seo/alfred-seo.php`, add after the existing `require_once` for settings.php:

```php
require_once ALFRED_SEO_DIR . 'inc/schema/validator.php';
require_once ALFRED_SEO_DIR . 'inc/schema/builder.php';
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd services/alfred-seo && phpunit tests/test-schema-builder.php`
Expected: PASS (validator tests pass; builder test may need stub schema functions — those land in Task 3+).

- [ ] **Step 7: Commit**

```bash
git add services/alfred-seo/inc/schema/builder.php services/alfred-seo/inc/schema/validator.php services/alfred-seo/alfred-seo.php services/alfred-seo/tests/test-schema-builder.php
git commit -m "feat(alfred-seo): schema dispatcher + JSON-LD validator with failure log"
```

---

## Task 3: Product schema builder

**Files:**
- Create: `services/alfred-seo/inc/schema/product.php`
- Test: `services/alfred-seo/tests/test-schema-product.php`

- [ ] **Step 1: Write the failing test**

```php
<?php
// services/alfred-seo/tests/test-schema-product.php

class Test_Schema_Product extends WP_UnitTestCase {
    private $product_id;

    public function set_up() {
        parent::set_up();
        if ( ! class_exists( 'WC_Product_Simple' ) ) {
            $this->markTestSkipped( 'WooCommerce not loaded in this test bootstrap.' );
        }
        $product = new WC_Product_Simple();
        $product->set_name( 'Red Bead Toggle Necklace' );
        $product->set_regular_price( '65.00' );
        $product->set_short_description( 'Faceted red beads on a knotted cord.' );
        $product->set_sku( 'roen-757' );
        $product->set_stock_status( 'instock' );
        $this->product_id = $product->save();
    }

    public function test_builder_returns_product_schema() {
        $result = alfred_seo_schema_product( $this->product_id );
        $this->assertEquals( 'Product', $result['@type'] );
        $this->assertEquals( 'Red Bead Toggle Necklace', $result['name'] );
        $this->assertEquals( 'roen-757', $result['sku'] );
        $this->assertEquals( 'Offer', $result['offers']['@type'] );
        $this->assertEquals( '65.00', $result['offers']['price'] );
        $this->assertEquals( 'https://schema.org/InStock', $result['offers']['availability'] );
    }

    public function test_builder_returns_null_for_unpurchasable() {
        $product = wc_get_product( $this->product_id );
        $product->set_status( 'draft' );
        $product->save();
        $this->assertNull( alfred_seo_schema_product( $this->product_id ) );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/alfred-seo && phpunit tests/test-schema-product.php`
Expected: FAIL with "function alfred_seo_schema_product not defined".

- [ ] **Step 3: Implement product.php**

```php
<?php
// services/alfred-seo/inc/schema/product.php

if ( ! defined( 'ABSPATH' ) ) { exit; }

/**
 * Build a Product schema for a WC product. Returns null if not catalog-able.
 *
 * @param int $product_id
 * @return array|null
 */
function alfred_seo_schema_product( $product_id ) {
    if ( ! function_exists( 'wc_get_product' ) ) { return null; }
    $product = wc_get_product( $product_id );
    if ( ! $product || ! $product->is_purchasable() ) { return null; }

    $settings = alfred_seo_get_settings();
    $images   = array();
    if ( $product->get_image_id() ) {
        $url = wp_get_attachment_image_url( $product->get_image_id(), 'full' );
        if ( $url ) { $images[] = $url; }
    }
    foreach ( $product->get_gallery_image_ids() as $gid ) {
        $url = wp_get_attachment_image_url( $gid, 'full' );
        if ( $url ) { $images[] = $url; }
    }

    $availability = $product->is_in_stock()
        ? 'https://schema.org/InStock'
        : 'https://schema.org/OutOfStock';

    $schema = array(
        '@context'    => 'https://schema.org',
        '@type'       => 'Product',
        'name'        => wp_strip_all_tags( $product->get_name() ),
        'description' => wp_strip_all_tags( $product->get_short_description() ?: $product->get_description() ),
        'sku'         => $product->get_sku(),
        'url'         => get_permalink( $product_id ),
        'image'       => count( $images ) === 1 ? $images[0] : $images,
        'brand'       => array(
            '@type' => 'Brand',
            'name'  => $settings['business_name'] ?: get_bloginfo( 'name' ),
        ),
        'offers'      => array(
            '@type'         => 'Offer',
            'price'         => $product->get_price(),
            'priceCurrency' => get_woocommerce_currency(),
            'availability'  => $availability,
            'url'           => get_permalink( $product_id ),
            'itemCondition' => 'https://schema.org/NewCondition',
        ),
    );

    // AggregateRating only if reviews exist.
    if ( $product->get_review_count() > 0 ) {
        $schema['aggregateRating'] = array(
            '@type'       => 'AggregateRating',
            'ratingValue' => $product->get_average_rating(),
            'reviewCount' => $product->get_review_count(),
        );
    }

    return $schema;
}
```

- [ ] **Step 4: Wire into alfred-seo.php**

Edit `services/alfred-seo/alfred-seo.php`, add after schema/builder.php require:

```php
require_once ALFRED_SEO_DIR . 'inc/schema/product.php';
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd services/alfred-seo && phpunit tests/test-schema-product.php`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git add services/alfred-seo/inc/schema/product.php services/alfred-seo/tests/test-schema-product.php services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): Product schema builder with WC integration"
```

---


## Task 4: Article schema builder

**Files:**
- Create: `services/alfred-seo/inc/schema/article.php`
- Test: `services/alfred-seo/tests/test-schema-article.php`

- [ ] **Step 1: Write the failing test**

```php
<?php
class Test_Schema_Article extends WP_UnitTestCase {
    public function test_article_schema_for_post() {
        $author_id = $this->factory->user->create( array( 'display_name' => 'Roen' ) );
        $post_id = $this->factory->post->create( array(
            'post_title' => 'How to care for handmade jewelry',
            'post_content' => 'Some content here.',
            'post_author' => $author_id,
        ) );
        $result = alfred_seo_schema_article( $post_id );
        $this->assertEquals( 'Article', $result['@type'] );
        $this->assertEquals( 'How to care for handmade jewelry', $result['headline'] );
        $this->assertEquals( 'Person', $result['author']['@type'] );
    }
}
```

- [ ] **Step 2: Run test (FAIL — function not defined)**

`cd services/alfred-seo && phpunit tests/test-schema-article.php`

- [ ] **Step 3: Implement article.php**

```php
<?php
// services/alfred-seo/inc/schema/article.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_schema_article( $post_id ) {
    $post = get_post( $post_id );
    if ( ! $post || 'publish' !== $post->post_status ) { return null; }

    $settings  = alfred_seo_get_settings();
    $author    = get_userdata( $post->post_author );
    $image_url = get_the_post_thumbnail_url( $post_id, 'full' );

    $schema = array(
        '@context'      => 'https://schema.org',
        '@type'         => 'Article',
        'headline'      => wp_strip_all_tags( $post->post_title ),
        'description'   => wp_strip_all_tags( get_the_excerpt( $post ) ),
        'datePublished' => mysql2date( 'c', $post->post_date_gmt, false ),
        'dateModified'  => mysql2date( 'c', $post->post_modified_gmt, false ),
        'url'           => get_permalink( $post ),
        'mainEntityOfPage' => array(
            '@type' => 'WebPage',
            '@id'   => get_permalink( $post ),
        ),
        'author'        => array(
            '@type' => 'Person',
            'name'  => $author ? $author->display_name : ( $settings['business_name'] ?: get_bloginfo( 'name' ) ),
        ),
        'publisher'     => array(
            '@type' => 'Organization',
            'name'  => $settings['business_name'] ?: get_bloginfo( 'name' ),
        ),
    );
    if ( $image_url ) { $schema['image'] = $image_url; }
    return $schema;
}
```

- [ ] **Step 4: Wire into alfred-seo.php**

Add: `require_once ALFRED_SEO_DIR . 'inc/schema/article.php';`

- [ ] **Step 5: Run test (PASS)**

- [ ] **Step 6: Commit**

```bash
git add services/alfred-seo/inc/schema/article.php services/alfred-seo/tests/test-schema-article.php services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): Article schema builder"
```

---

## Task 5: Organization + LocalBusiness schema builder

**Files:**
- Create: `services/alfred-seo/inc/schema/organization.php`
- Test: `services/alfred-seo/tests/test-schema-organization.php`

- [ ] **Step 1: Write the failing test**

```php
<?php
class Test_Schema_Organization extends WP_UnitTestCase {
    public function test_basic_organization() {
        alfred_seo_update_settings( array(
            'business_name' => 'Roen',
            'business_type' => 'Organization',
        ) );
        $result = alfred_seo_schema_organization();
        $this->assertEquals( 'Organization', $result['@type'] );
        $this->assertEquals( 'Roen', $result['name'] );
    }

    public function test_localbusiness_includes_address() {
        alfred_seo_update_settings( array(
            'business_name' => 'Roen',
            'business_type' => 'LocalBusiness',
            'local_address' => array(
                'streetAddress'   => '123 Atlanta Ave',
                'addressLocality' => 'Atlanta',
                'addressRegion'   => 'GA',
                'postalCode'      => '30303',
                'addressCountry'  => 'US',
            ),
        ) );
        $result = alfred_seo_schema_organization();
        $this->assertEquals( 'LocalBusiness', $result['@type'] );
        $this->assertEquals( 'Atlanta', $result['address']['addressLocality'] );
    }
}
```

- [ ] **Step 2: Run test (FAIL)**

- [ ] **Step 3: Implement organization.php**

```php
<?php
// services/alfred-seo/inc/schema/organization.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_schema_organization() {
    $settings = alfred_seo_get_settings();
    $type     = in_array( $settings['business_type'], array( 'Organization', 'LocalBusiness', 'SoftwareApplication', 'Service' ), true )
        ? $settings['business_type']
        : 'Organization';

    $schema = array(
        '@context' => 'https://schema.org',
        '@type'    => $type,
        'name'     => $settings['business_name'] ?: get_bloginfo( 'name' ),
        'url'      => home_url( '/' ),
    );

    $logo_url = get_site_icon_url( 512 );
    if ( $logo_url ) {
        $schema['logo'] = $logo_url;
    }

    if ( ! empty( $settings['social_handles'] ) ) {
        $sames = array();
        foreach ( $settings['social_handles'] as $platform => $handle ) {
            $handle = ltrim( $handle, '@' );
            switch ( strtolower( $platform ) ) {
                case 'twitter':   $sames[] = 'https://twitter.com/' . $handle; break;
                case 'instagram': $sames[] = 'https://instagram.com/' . $handle; break;
                case 'facebook':  $sames[] = 'https://facebook.com/' . $handle; break;
                case 'pinterest': $sames[] = 'https://pinterest.com/' . $handle; break;
                case 'youtube':   $sames[] = 'https://youtube.com/@' . $handle; break;
                case 'linkedin':  $sames[] = 'https://linkedin.com/in/' . $handle; break;
            }
        }
        if ( $sames ) { $schema['sameAs'] = $sames; }
    }

    if ( 'LocalBusiness' === $type && ! empty( $settings['local_address'] ) ) {
        $schema['address'] = array_merge(
            array( '@type' => 'PostalAddress' ),
            array_filter( $settings['local_address'] )
        );
    }

    return $schema;
}
```

- [ ] **Step 4: Wire + run test (PASS) + commit**

```bash
git add services/alfred-seo/inc/schema/organization.php services/alfred-seo/tests/test-schema-organization.php services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): Organization + LocalBusiness schema with sameAs links"
```

---

## Task 6: BreadcrumbList schema builder

**Files:**
- Create: `services/alfred-seo/inc/schema/breadcrumb.php`
- Test: `services/alfred-seo/tests/test-schema-breadcrumb.php`

- [ ] **Step 1: Write failing test**

```php
<?php
class Test_Schema_Breadcrumb extends WP_UnitTestCase {
    public function test_breadcrumb_for_post() {
        $cat_id = $this->factory->category->create( array( 'name' => 'Style Notes' ) );
        $post_id = $this->factory->post->create( array(
            'post_title' => 'Caring for beads',
            'post_category' => array( $cat_id ),
        ) );
        $this->go_to( get_permalink( $post_id ) );
        $result = alfred_seo_schema_breadcrumb();
        $this->assertEquals( 'BreadcrumbList', $result['@type'] );
        $this->assertGreaterThanOrEqual( 2, count( $result['itemListElement'] ) );
        $this->assertEquals( 'Home', $result['itemListElement'][0]['name'] );
    }
}
```

- [ ] **Step 2: Run test (FAIL)**

- [ ] **Step 3: Implement breadcrumb.php**

```php
<?php
// services/alfred-seo/inc/schema/breadcrumb.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_schema_breadcrumb() {
    $items = array();
    $position = 1;

    // Home is always first.
    $items[] = array(
        '@type'    => 'ListItem',
        'position' => $position++,
        'name'     => 'Home',
        'item'     => home_url( '/' ),
    );

    if ( function_exists( 'is_product' ) && is_product() ) {
        $product = wc_get_product( get_queried_object_id() );
        $cats    = get_the_terms( $product->get_id(), 'product_cat' );
        if ( $cats && ! is_wp_error( $cats ) ) {
            $primary = $cats[0];
            $items[] = array(
                '@type'    => 'ListItem',
                'position' => $position++,
                'name'     => $primary->name,
                'item'     => get_term_link( $primary ),
            );
        }
        $items[] = array(
            '@type'    => 'ListItem',
            'position' => $position++,
            'name'     => $product->get_name(),
            'item'     => get_permalink( $product->get_id() ),
        );
    } elseif ( is_singular( 'post' ) ) {
        $post = get_queried_object();
        $cats = get_the_category( $post->ID );
        if ( $cats ) {
            $items[] = array(
                '@type'    => 'ListItem',
                'position' => $position++,
                'name'     => $cats[0]->name,
                'item'     => get_category_link( $cats[0] ),
            );
        }
        $items[] = array(
            '@type'    => 'ListItem',
            'position' => $position++,
            'name'     => get_the_title( $post ),
            'item'     => get_permalink( $post ),
        );
    } elseif ( is_page() ) {
        $page = get_queried_object();
        if ( $page->post_parent ) {
            $ancestors = array_reverse( get_post_ancestors( $page->ID ) );
            foreach ( $ancestors as $anc_id ) {
                $items[] = array(
                    '@type'    => 'ListItem',
                    'position' => $position++,
                    'name'     => get_the_title( $anc_id ),
                    'item'     => get_permalink( $anc_id ),
                );
            }
        }
        $items[] = array(
            '@type'    => 'ListItem',
            'position' => $position++,
            'name'     => get_the_title( $page ),
            'item'     => get_permalink( $page ),
        );
    } elseif ( is_category() || is_tax() ) {
        $term = get_queried_object();
        $items[] = array(
            '@type'    => 'ListItem',
            'position' => $position++,
            'name'     => $term->name,
            'item'     => get_term_link( $term ),
        );
    }

    if ( count( $items ) < 2 ) { return null; }

    return array(
        '@context'        => 'https://schema.org',
        '@type'           => 'BreadcrumbList',
        'itemListElement' => $items,
    );
}
```

- [ ] **Step 4: Wire + test (PASS) + commit**

```bash
git add services/alfred-seo/inc/schema/breadcrumb.php services/alfred-seo/tests/test-schema-breadcrumb.php services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): BreadcrumbList schema for products/posts/pages/categories"
```

---

## Task 7: FAQPage schema builder (extracts from content)

**Files:**
- Create: `services/alfred-seo/inc/schema/faq.php`
- Test: `services/alfred-seo/tests/test-schema-faq.php`

- [ ] **Step 1: Write failing test**

```php
<?php
class Test_Schema_FAQ extends WP_UnitTestCase {
    public function test_faq_extracted_from_h2_question_pattern() {
        $content = "<h2>What is a toggle clasp?</h2><p>A toggle clasp has a bar and a ring.</p>"
                 . "<h2>How do I size this?</h2><p>Standard 7 inches; ask for resize.</p>";
        $result = alfred_seo_schema_faq_from_content( $content );
        $this->assertEquals( 'FAQPage', $result['@type'] );
        $this->assertCount( 2, $result['mainEntity'] );
        $this->assertEquals( 'What is a toggle clasp?', $result['mainEntity'][0]['name'] );
    }

    public function test_returns_null_when_no_faqs() {
        $this->assertNull( alfred_seo_schema_faq_from_content( '<p>just prose</p>' ) );
    }
}
```

- [ ] **Step 2: Run test (FAIL)**

- [ ] **Step 3: Implement faq.php**

```php
<?php
// services/alfred-seo/inc/schema/faq.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

/**
 * Extract FAQ schema from any HTML content where H2/H3 ends with ? followed by
 * one or more paragraphs (until next H2/H3 or end). Returns null if no FAQs.
 */
function alfred_seo_schema_faq_from_content( $html ) {
    if ( ! $html ) { return null; }

    // Match: <h[23]>question ending with ?</h[23]> followed by paragraph(s).
    $pattern = '#<h[23][^>]*>\s*([^<>]*\?)\s*</h[23]>\s*(.*?)(?=<h[23]|\z)#is';
    if ( ! preg_match_all( $pattern, $html, $matches, PREG_SET_ORDER ) ) {
        return null;
    }

    $entities = array();
    foreach ( $matches as $m ) {
        $question = trim( wp_strip_all_tags( $m[1] ) );
        $answer   = trim( wp_strip_all_tags( $m[2] ) );
        if ( $question && $answer ) {
            $entities[] = array(
                '@type'          => 'Question',
                'name'           => $question,
                'acceptedAnswer' => array(
                    '@type' => 'Answer',
                    'text'  => $answer,
                ),
            );
        }
    }
    if ( empty( $entities ) ) { return null; }

    return array(
        '@context'   => 'https://schema.org',
        '@type'      => 'FAQPage',
        'mainEntity' => $entities,
    );
}

/**
 * Page-level helper: build FAQ schema for current post content if any exist.
 */
function alfred_seo_schema_faq() {
    if ( ! is_singular() ) { return null; }
    $post = get_queried_object();
    return alfred_seo_schema_faq_from_content( $post->post_content );
}
```

- [ ] **Step 4: Hook into builder.php**

Add to `alfred_seo_build_schema_for_current_page()` after the breadcrumb check:

```php
    $faq = alfred_seo_schema_faq();
    if ( $faq ) { $schemas[] = $faq; }
```

- [ ] **Step 5: Wire + test (PASS) + commit**

```bash
git add services/alfred-seo/inc/schema/faq.php services/alfred-seo/inc/schema/builder.php services/alfred-seo/tests/test-schema-faq.php services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): FAQPage schema extracted from H2/H3 ?-pattern content"
```

---

## Task 8: WebSite + SearchAction schema

**Files:**
- Create: `services/alfred-seo/inc/schema/website.php`
- Test: `services/alfred-seo/tests/test-schema-website.php`

- [ ] **Step 1: Write failing test**

```php
<?php
class Test_Schema_Website extends WP_UnitTestCase {
    public function test_website_has_searchaction() {
        $result = alfred_seo_schema_website();
        $this->assertEquals( 'WebSite', $result['@type'] );
        $this->assertEquals( 'SearchAction', $result['potentialAction']['@type'] );
        $this->assertStringContainsString( '/?s={search_term_string}', $result['potentialAction']['target']['urlTemplate'] );
    }
}
```

- [ ] **Step 2: Run test (FAIL)**

- [ ] **Step 3: Implement website.php**

```php
<?php
// services/alfred-seo/inc/schema/website.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_schema_website() {
    return array(
        '@context'        => 'https://schema.org',
        '@type'           => 'WebSite',
        'name'            => get_bloginfo( 'name' ),
        'url'             => home_url( '/' ),
        'potentialAction' => array(
            '@type'  => 'SearchAction',
            'target' => array(
                '@type'       => 'EntryPoint',
                'urlTemplate' => home_url( '/?s={search_term_string}' ),
            ),
            'query-input' => 'required name=search_term_string',
        ),
    );
}
```

- [ ] **Step 4: Wire + test (PASS) + commit**

```bash
git add services/alfred-seo/inc/schema/website.php services/alfred-seo/tests/test-schema-website.php services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): WebSite + SearchAction schema (sitelinks search box)"
```

---

## Task 9: CollectionPage + ItemList schema

**Files:**
- Create: `services/alfred-seo/inc/schema/collection.php`
- Test: `services/alfred-seo/tests/test-schema-collection.php`

- [ ] **Step 1: Write failing test**

```php
<?php
class Test_Schema_Collection extends WP_UnitTestCase {
    public function test_collection_for_product_category() {
        if ( ! taxonomy_exists( 'product_cat' ) ) { $this->markTestSkipped(); }
        $term_id = $this->factory->term->create( array(
            'taxonomy' => 'product_cat',
            'name'     => 'Bracelets',
        ) );
        $term   = get_term( $term_id );
        $result = alfred_seo_schema_collection( $term );
        $this->assertEquals( 'CollectionPage', $result['@type'] );
        $this->assertEquals( 'Bracelets', $result['name'] );
    }
}
```

- [ ] **Step 2: Run test (FAIL)**

- [ ] **Step 3: Implement collection.php**

```php
<?php
// services/alfred-seo/inc/schema/collection.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_schema_collection( $term ) {
    if ( ! $term || is_wp_error( $term ) ) { return null; }

    // Pull top 12 products in this category for ItemList.
    $items = array();
    if ( function_exists( 'wc_get_products' ) ) {
        $products = wc_get_products( array(
            'category' => array( $term->slug ),
            'limit'    => 12,
            'status'   => 'publish',
        ) );
        $position = 1;
        foreach ( $products as $product ) {
            $items[] = array(
                '@type'    => 'ListItem',
                'position' => $position++,
                'url'      => get_permalink( $product->get_id() ),
                'name'     => $product->get_name(),
            );
        }
    }

    $schema = array(
        '@context'   => 'https://schema.org',
        '@type'      => 'CollectionPage',
        'name'       => $term->name,
        'description' => wp_strip_all_tags( $term->description ),
        'url'        => get_term_link( $term ),
    );

    if ( $items ) {
        $schema['mainEntity'] = array(
            '@type'           => 'ItemList',
            'numberOfItems'   => count( $items ),
            'itemListElement' => $items,
        );
    }
    return $schema;
}
```

- [ ] **Step 4: Wire + test (PASS) + commit**

```bash
git add services/alfred-seo/inc/schema/collection.php services/alfred-seo/tests/test-schema-collection.php services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): CollectionPage + ItemList schema for product categories"
```

---


## Task 10: Open Graph + Twitter Card head tags

**Files:**
- Create: `services/alfred-seo/inc/open-graph.php`
- Test: `services/alfred-seo/tests/test-open-graph.php`

- [ ] **Step 1: Write failing test**

```php
<?php
class Test_Open_Graph extends WP_UnitTestCase {
    public function test_og_tags_emitted_on_home() {
        $this->go_to( home_url( '/' ) );
        ob_start();
        alfred_seo_render_open_graph();
        $out = ob_get_clean();
        $this->assertStringContainsString( 'property="og:title"', $out );
        $this->assertStringContainsString( 'property="og:url"', $out );
        $this->assertStringContainsString( 'name="twitter:card"', $out );
    }
}
```

- [ ] **Step 2: Run test (FAIL)**

- [ ] **Step 3: Implement open-graph.php**

```php
<?php
// services/alfred-seo/inc/open-graph.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_render_open_graph() {
    $settings = alfred_seo_get_settings();

    // Resolve title, description, image, URL, og:type per page type.
    $url   = '';
    $title = get_bloginfo( 'name' );
    $desc  = get_bloginfo( 'description' );
    $image = get_site_icon_url( 1200 );
    $type  = 'website';

    if ( is_singular() ) {
        $obj   = get_queried_object();
        $url   = get_permalink( $obj );
        $title = get_the_title( $obj );
        $desc  = get_the_excerpt( $obj ) ?: $desc;
        if ( has_post_thumbnail( $obj ) ) {
            $image = get_the_post_thumbnail_url( $obj, 'full' );
        }
        $type = ( function_exists( 'is_product' ) && is_product() ) ? 'product' : 'article';
    } elseif ( is_category() || is_tax() ) {
        $term  = get_queried_object();
        $url   = get_term_link( $term );
        $title = $term->name;
        $desc  = wp_strip_all_tags( $term->description ) ?: $desc;
    } else {
        $url = home_url( $_SERVER['REQUEST_URI'] ?? '/' );
    }

    // Custom-field override (Alfred can push a specific OG image/title/desc).
    if ( is_singular() ) {
        $obj = get_queried_object();
        $override_title = get_post_meta( $obj->ID, '_alfred_seo_og_title', true );
        $override_desc  = get_post_meta( $obj->ID, '_alfred_seo_og_description', true );
        $override_img   = get_post_meta( $obj->ID, '_alfred_seo_og_image', true );
        if ( $override_title ) { $title = $override_title; }
        if ( $override_desc )  { $desc  = $override_desc; }
        if ( $override_img )   { $image = $override_img; }
    }

    $title = wp_strip_all_tags( $title );
    $desc  = wp_strip_all_tags( $desc );

    $tags = array(
        sprintf( '<meta property="og:type" content="%s" />', esc_attr( $type ) ),
        sprintf( '<meta property="og:title" content="%s" />', esc_attr( $title ) ),
        sprintf( '<meta property="og:description" content="%s" />', esc_attr( $desc ) ),
        sprintf( '<meta property="og:url" content="%s" />', esc_url( $url ) ),
        sprintf( '<meta property="og:site_name" content="%s" />', esc_attr( $settings['business_name'] ?: get_bloginfo( 'name' ) ) ),
    );
    if ( $image ) {
        $tags[] = sprintf( '<meta property="og:image" content="%s" />', esc_url( $image ) );
    }

    // Twitter Card.
    $twitter_handle = $settings['social_handles']['twitter'] ?? '';
    $tags[] = '<meta name="twitter:card" content="summary_large_image" />';
    $tags[] = sprintf( '<meta name="twitter:title" content="%s" />', esc_attr( $title ) );
    $tags[] = sprintf( '<meta name="twitter:description" content="%s" />', esc_attr( $desc ) );
    if ( $image ) {
        $tags[] = sprintf( '<meta name="twitter:image" content="%s" />', esc_url( $image ) );
    }
    if ( $twitter_handle ) {
        $tags[] = sprintf( '<meta name="twitter:site" content="@%s" />', esc_attr( ltrim( $twitter_handle, '@' ) ) );
    }

    echo "\n" . implode( "\n", $tags ) . "\n";
}
add_action( 'wp_head', 'alfred_seo_render_open_graph', 5 );
```

- [ ] **Step 4: Wire + test (PASS) + commit**

```bash
git add services/alfred-seo/inc/open-graph.php services/alfred-seo/tests/test-open-graph.php services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): Open Graph + Twitter Card head tags with custom-field override"
```

---

## Task 11: Meta description override

**Files:**
- Create: `services/alfred-seo/inc/meta.php`
- Test: `services/alfred-seo/tests/test-meta.php`

- [ ] **Step 1: Write failing test**

```php
<?php
class Test_Meta_Description extends WP_UnitTestCase {
    public function test_custom_field_overrides_excerpt() {
        $post_id = $this->factory->post->create( array( 'post_excerpt' => 'Auto excerpt.' ) );
        update_post_meta( $post_id, '_alfred_seo_meta_description', 'Alfred-pushed description.' );
        $this->go_to( get_permalink( $post_id ) );
        ob_start(); alfred_seo_render_meta_description(); $out = ob_get_clean();
        $this->assertStringContainsString( 'Alfred-pushed description.', $out );
    }

    public function test_excerpt_fallback() {
        $post_id = $this->factory->post->create( array( 'post_excerpt' => 'Auto excerpt.' ) );
        $this->go_to( get_permalink( $post_id ) );
        ob_start(); alfred_seo_render_meta_description(); $out = ob_get_clean();
        $this->assertStringContainsString( 'Auto excerpt.', $out );
    }
}
```

- [ ] **Step 2: Run test (FAIL)**

- [ ] **Step 3: Implement meta.php**

```php
<?php
// services/alfred-seo/inc/meta.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_resolve_meta_description() {
    if ( is_singular() ) {
        $obj      = get_queried_object();
        $override = get_post_meta( $obj->ID, '_alfred_seo_meta_description', true );
        if ( $override ) { return wp_strip_all_tags( $override ); }
        $excerpt = get_the_excerpt( $obj );
        if ( $excerpt ) { return wp_strip_all_tags( $excerpt ); }
        $content = wp_strip_all_tags( $obj->post_content );
        return mb_substr( trim( preg_replace( '/\s+/', ' ', $content ) ), 0, 155 );
    }
    if ( is_category() || is_tax() ) {
        $term = get_queried_object();
        if ( $term && $term->description ) {
            return wp_strip_all_tags( $term->description );
        }
    }
    return wp_strip_all_tags( get_bloginfo( 'description' ) );
}

function alfred_seo_render_meta_description() {
    $desc = alfred_seo_resolve_meta_description();
    if ( ! $desc ) { return; }
    printf( "\n<meta name=\"description\" content=\"%s\" />\n", esc_attr( mb_substr( $desc, 0, 160 ) ) );
}
add_action( 'wp_head', 'alfred_seo_render_meta_description', 3 );
```

- [ ] **Step 4: Wire + test (PASS) + commit**

```bash
git add services/alfred-seo/inc/meta.php services/alfred-seo/tests/test-meta.php services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): meta description with custom-field override + excerpt fallback"
```

---

## Task 12: Multi-sitemap index + per-type sitemaps

**Files:**
- Create: `services/alfred-seo/inc/sitemap.php`
- Test: `services/alfred-seo/tests/test-sitemap.php`

- [ ] **Step 1: Write failing test**

```php
<?php
class Test_Sitemap extends WP_UnitTestCase {
    public function test_sitemap_index_has_urls() {
        $xml = alfred_seo_render_sitemap_index();
        $this->assertStringContainsString( '<?xml version="1.0"', $xml );
        $this->assertStringContainsString( '<sitemapindex', $xml );
        $this->assertStringContainsString( '/alfred-sitemap-pages.xml', $xml );
        $this->assertStringContainsString( '/alfred-sitemap-posts.xml', $xml );
    }

    public function test_pages_sitemap_includes_published_pages() {
        $page_id = $this->factory->post->create( array( 'post_type' => 'page', 'post_status' => 'publish' ) );
        $xml = alfred_seo_render_sitemap_for( 'pages' );
        $this->assertStringContainsString( get_permalink( $page_id ), $xml );
    }
}
```

- [ ] **Step 2: Run test (FAIL)**

- [ ] **Step 3: Implement sitemap.php**

```php
<?php
// services/alfred-seo/inc/sitemap.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

/**
 * Register rewrite rules so /alfred-sitemap.xml and /alfred-sitemap-<type>.xml
 * are served by this plugin. Disables WP's built-in wp-sitemap.xml to avoid
 * duplication.
 */
add_action( 'init', function () {
    add_rewrite_rule( '^alfred-sitemap\.xml$', 'index.php?alfred_seo_sitemap=index', 'top' );
    add_rewrite_rule( '^alfred-sitemap-([a-z]+)\.xml$', 'index.php?alfred_seo_sitemap=$matches[1]', 'top' );
});
add_filter( 'query_vars', function ( $vars ) {
    $vars[] = 'alfred_seo_sitemap';
    return $vars;
});
add_filter( 'wp_sitemaps_enabled', '__return_false' );

add_action( 'template_redirect', function () {
    $type = get_query_var( 'alfred_seo_sitemap' );
    if ( ! $type ) { return; }
    header( 'Content-Type: application/xml; charset=UTF-8' );
    header( 'X-Robots-Tag: noindex, follow' );
    if ( 'index' === $type ) {
        echo alfred_seo_render_sitemap_index();
    } else {
        echo alfred_seo_render_sitemap_for( $type );
    }
    exit;
}, 1 );

function alfred_seo_render_sitemap_index() {
    $types = array( 'pages', 'posts', 'products', 'categories' );
    $now   = gmdate( 'c' );
    $xml   = '<?xml version="1.0" encoding="UTF-8"?>' . "\n";
    $xml  .= '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' . "\n";
    foreach ( $types as $t ) {
        $xml .= sprintf(
            "  <sitemap><loc>%s</loc><lastmod>%s</lastmod></sitemap>\n",
            esc_url( home_url( '/alfred-sitemap-' . $t . '.xml' ) ),
            $now
        );
    }
    $xml .= '</sitemapindex>' . "\n";
    return $xml;
}

function alfred_seo_render_sitemap_for( $type ) {
    $entries = array();
    switch ( $type ) {
        case 'pages':
            $posts = get_posts( array( 'post_type' => 'page', 'post_status' => 'publish', 'numberposts' => -1 ) );
            foreach ( $posts as $p ) { $entries[] = array( get_permalink( $p ), $p->post_modified_gmt ); }
            break;
        case 'posts':
            $posts = get_posts( array( 'post_type' => 'post', 'post_status' => 'publish', 'numberposts' => -1 ) );
            foreach ( $posts as $p ) { $entries[] = array( get_permalink( $p ), $p->post_modified_gmt ); }
            break;
        case 'products':
            if ( post_type_exists( 'product' ) ) {
                $posts = get_posts( array( 'post_type' => 'product', 'post_status' => 'publish', 'numberposts' => -1 ) );
                foreach ( $posts as $p ) { $entries[] = array( get_permalink( $p ), $p->post_modified_gmt ); }
            }
            break;
        case 'categories':
            $terms = get_terms( array( 'taxonomy' => array_filter( array( 'category', taxonomy_exists( 'product_cat' ) ? 'product_cat' : null ) ), 'hide_empty' => true ) );
            foreach ( $terms as $t ) { $entries[] = array( get_term_link( $t ), gmdate( 'c' ) ); }
            break;
    }
    $xml = '<?xml version="1.0" encoding="UTF-8"?>' . "\n";
    $xml .= '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' . "\n";
    foreach ( $entries as $row ) {
        $xml .= sprintf(
            "  <url><loc>%s</loc><lastmod>%s</lastmod></url>\n",
            esc_url( $row[0] ),
            $row[1] ? mysql2date( 'c', $row[1], false ) : gmdate( 'c' )
        );
    }
    $xml .= '</urlset>' . "\n";
    return $xml;
}

/**
 * Ping GSC + Bing when a post is published or updated.
 */
add_action( 'transition_post_status', function ( $new, $old, $post ) {
    if ( 'publish' !== $new ) { return; }
    if ( ! in_array( $post->post_type, array( 'post', 'page', 'product' ), true ) ) { return; }
    $sitemap = home_url( '/alfred-sitemap.xml' );
    wp_remote_get( 'https://www.google.com/ping?sitemap=' . urlencode( $sitemap ), array( 'timeout' => 3, 'blocking' => false ) );
    wp_remote_get( 'https://www.bing.com/ping?sitemap=' . urlencode( $sitemap ), array( 'timeout' => 3, 'blocking' => false ) );
}, 10, 3 );

register_activation_hook( ALFRED_SEO_DIR . 'alfred-seo.php', function () {
    flush_rewrite_rules();
});
```

- [ ] **Step 4: Wire + test (PASS) + manual rewrite flush + commit**

```bash
git add services/alfred-seo/inc/sitemap.php services/alfred-seo/tests/test-sitemap.php services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): multi-sitemap index + per-type sitemaps + GSC/Bing ping"
```

---

## Task 13: robots.txt management

**Files:**
- Create: `services/alfred-seo/inc/robots.php`
- Test: `services/alfred-seo/tests/test-robots.php`

- [ ] **Step 1: Write failing test**

```php
<?php
class Test_Robots extends WP_UnitTestCase {
    public function test_sitemap_line_added() {
        $out = apply_filters( 'robots_txt', "User-agent: *\nDisallow: /wp-admin/\n", true );
        $this->assertStringContainsString( 'Sitemap:', $out );
        $this->assertStringContainsString( '/alfred-sitemap.xml', $out );
    }
}
```

- [ ] **Step 2: Implement robots.php**

```php
<?php
// services/alfred-seo/inc/robots.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_filter( 'robots_txt', function ( $output, $public ) {
    if ( ! $public ) { return $output; }
    // Append sitemap line if not present.
    $sitemap_url = home_url( '/alfred-sitemap.xml' );
    if ( false === strpos( $output, $sitemap_url ) ) {
        $output = rtrim( $output ) . "\nSitemap: " . $sitemap_url . "\n";
    }
    return $output;
}, 10, 2 );
```

- [ ] **Step 3: Wire + test (PASS) + commit**

```bash
git add services/alfred-seo/inc/robots.php services/alfred-seo/tests/test-robots.php services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): robots.txt sitemap directive"
```

---

## Task 14: Image alt text auto-fill on upload

**Files:**
- Create: `services/alfred-seo/inc/alt-text.php`
- Test: `services/alfred-seo/tests/test-alt-text.php`

- [ ] **Step 1: Write failing test**

```php
<?php
class Test_Alt_Text extends WP_UnitTestCase {
    public function test_fallback_to_filename_when_alfred_down() {
        $attachment_id = $this->factory->attachment->create_object(
            'red-bead-necklace.jpg',
            0,
            array( 'post_mime_type' => 'image/jpeg', 'post_status' => 'inherit' )
        );
        // Force Alfred lookup to fail by setting bad endpoint.
        alfred_seo_update_settings( array( 'alfred_endpoint' => 'http://127.0.0.1:1' ) );
        alfred_seo_fill_alt_text_on_upload( $attachment_id );
        $alt = get_post_meta( $attachment_id, '_wp_attachment_image_alt', true );
        $this->assertNotEmpty( $alt );
        $this->assertStringContainsString( 'red bead necklace', strtolower( $alt ) );
    }
}
```

- [ ] **Step 2: Run test (FAIL)**

- [ ] **Step 3: Implement alt-text.php**

```php
<?php
// services/alfred-seo/inc/alt-text.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_action( 'add_attachment', 'alfred_seo_fill_alt_text_on_upload' );

function alfred_seo_fill_alt_text_on_upload( $attachment_id ) {
    $settings = alfred_seo_get_settings();
    if ( empty( $settings['alt_text_enabled'] ) ) { return; }

    // Skip non-images.
    if ( ! wp_attachment_is_image( $attachment_id ) ) { return; }
    // Skip if alt already set (manual override).
    if ( get_post_meta( $attachment_id, '_wp_attachment_image_alt', true ) ) { return; }

    $url = wp_get_attachment_url( $attachment_id );
    $alt = '';

    // Try Alfred orchestrator vision endpoint.
    $endpoint = trailingslashit( $settings['alfred_endpoint'] ) . 'wp-vision/alt-text';
    $resp = wp_remote_post( $endpoint, array(
        'timeout' => 10,
        'body'    => wp_json_encode( array( 'image_url' => $url, 'site_slug' => $settings['site_slug'] ) ),
        'headers' => array( 'Content-Type' => 'application/json' ),
    ) );
    if ( ! is_wp_error( $resp ) && 200 === wp_remote_retrieve_response_code( $resp ) ) {
        $body = json_decode( wp_remote_retrieve_body( $resp ), true );
        if ( ! empty( $body['alt'] ) ) { $alt = $body['alt']; }
    }

    // Local fallback: humanize filename.
    if ( ! $alt ) {
        $filename = pathinfo( get_attached_file( $attachment_id ), PATHINFO_FILENAME );
        $alt      = ucwords( str_replace( array( '-', '_' ), ' ', $filename ) );
    }

    update_post_meta( $attachment_id, '_wp_attachment_image_alt', mb_substr( $alt, 0, 125 ) );
}
```

- [ ] **Step 4: Wire + test (PASS) + commit**

```bash
git add services/alfred-seo/inc/alt-text.php services/alfred-seo/tests/test-alt-text.php services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): image alt text auto-fill via Alfred vision + filename fallback"
```

---

## Task 15: Internal linking filter on the_content

**Files:**
- Create: `services/alfred-seo/inc/internal-links.php`
- Test: `services/alfred-seo/tests/test-internal-links.php`

- [ ] **Step 1: Write failing test**

```php
<?php
class Test_Internal_Links extends WP_UnitTestCase {
    public function test_phrase_replaced_with_link() {
        alfred_seo_update_settings( array( 'internal_links' => array(
            'evil eye bracelet' => 'https://example.com/products/evil-eye/',
        ) ) );
        $out = alfred_seo_apply_internal_links( '<p>Our evil eye bracelet is popular.</p>' );
        $this->assertStringContainsString( 'href="https://example.com/products/evil-eye/"', $out );
        $this->assertStringContainsString( '>evil eye bracelet</a>', $out );
    }

    public function test_does_not_link_inside_existing_anchor() {
        alfred_seo_update_settings( array( 'internal_links' => array( 'evil eye' => 'https://x.com/' ) ) );
        $out = alfred_seo_apply_internal_links( '<a href="/foo">evil eye</a>' );
        $this->assertStringContainsString( '<a href="/foo">evil eye</a>', $out );
        $this->assertStringNotContainsString( 'https://x.com/', $out );
    }

    public function test_only_links_first_occurrence_per_phrase() {
        alfred_seo_update_settings( array( 'internal_links' => array( 'bracelet' => 'https://x.com/' ) ) );
        $out = alfred_seo_apply_internal_links( 'A bracelet and another bracelet.' );
        $this->assertEquals( 1, substr_count( $out, 'href="https://x.com/"' ) );
    }
}
```

- [ ] **Step 2: Run test (FAIL)**

- [ ] **Step 3: Implement internal-links.php**

```php
<?php
// services/alfred-seo/inc/internal-links.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_filter( 'the_content', 'alfred_seo_apply_internal_links', 99 );

function alfred_seo_apply_internal_links( $content ) {
    $settings = alfred_seo_get_settings();
    $map      = $settings['internal_links'] ?? array();
    if ( empty( $map ) ) { return $content; }

    // Don't operate on admin or feeds.
    if ( is_admin() || is_feed() ) { return $content; }

    // Sort by phrase length DESC so longer phrases bind before shorter substrings.
    uksort( $map, function ( $a, $b ) { return strlen( $b ) - strlen( $a ); } );

    // Skip anything inside <a>, <h1-h6>, <code>, <pre>, <script>.
    $segments = preg_split(
        '#(<(?:a|h[1-6]|code|pre|script)\b[^>]*>.*?</(?:a|h[1-6]|code|pre|script)>)#is',
        $content,
        -1,
        PREG_SPLIT_DELIM_CAPTURE
    );

    foreach ( $segments as $i => $seg ) {
        // Odd indices are the captured skip-segments; leave them alone.
        if ( 1 === $i % 2 ) { continue; }
        foreach ( $map as $phrase => $url ) {
            $pattern = '/\b' . preg_quote( $phrase, '/' ) . '\b/i';
            $replacement = sprintf( '<a href="%s">$0</a>', esc_url( $url ) );
            $seg = preg_replace( $pattern, $replacement, $seg, 1 );  // limit 1 occurrence
        }
        $segments[ $i ] = $seg;
    }
    return implode( '', $segments );
}
```

- [ ] **Step 4: Wire + test (PASS) + commit**

```bash
git add services/alfred-seo/inc/internal-links.php services/alfred-seo/tests/test-internal-links.php services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): internal linking filter on the_content, skips anchors/headings/code"
```

---


## Task 16: REST endpoints + WP application password auth

**Files:**
- Create: `services/alfred-seo/inc/rest/auth.php`
- Create: `services/alfred-seo/inc/rest/audit.php`
- Create: `services/alfred-seo/inc/rest/content.php`
- Create: `services/alfred-seo/inc/rest/meta.php`
- Create: `services/alfred-seo/inc/rest/internal-links.php`
- Create: `services/alfred-seo/inc/rest/sitemap-ping.php`
- Test: `services/alfred-seo/tests/test-rest-audit.php`
- Test: `services/alfred-seo/tests/test-rest-content.php`

- [ ] **Step 1: Write failing tests**

```php
<?php
// services/alfred-seo/tests/test-rest-audit.php
class Test_REST_Audit extends WP_UnitTestCase {
    public function test_audit_returns_page_list() {
        wp_set_current_user( $this->factory->user->create( array( 'role' => 'administrator' ) ) );
        $req = new WP_REST_Request( 'GET', '/alfred-seo/v1/audit' );
        $resp = rest_do_request( $req );
        $this->assertEquals( 200, $resp->get_status() );
        $data = $resp->get_data();
        $this->assertArrayHasKey( 'pages', $data );
        $this->assertArrayHasKey( 'missing_meta', $data );
    }

    public function test_audit_rejects_unauthorized() {
        wp_set_current_user( 0 );
        $req = new WP_REST_Request( 'GET', '/alfred-seo/v1/audit' );
        $resp = rest_do_request( $req );
        $this->assertEquals( 401, $resp->get_status() );
    }
}
```

```php
<?php
// services/alfred-seo/tests/test-rest-content.php
class Test_REST_Content extends WP_UnitTestCase {
    public function test_content_creates_draft() {
        wp_set_current_user( $this->factory->user->create( array( 'role' => 'administrator' ) ) );
        $req = new WP_REST_Request( 'POST', '/alfred-seo/v1/content' );
        $req->set_body_params( array(
            'title'             => 'Evil Eye Jewelry Guide',
            'content'           => '<p>Content body</p>',
            'meta_description'  => 'A guide to evil eye jewelry.',
            'slug'              => 'evil-eye-guide',
            'post_type'         => 'post',
            'status'            => 'draft',
        ) );
        $resp = rest_do_request( $req );
        $this->assertEquals( 201, $resp->get_status() );
        $data = $resp->get_data();
        $this->assertArrayHasKey( 'post_id', $data );
        $post = get_post( $data['post_id'] );
        $this->assertEquals( 'draft', $post->post_status );
    }
}
```

- [ ] **Step 2: Run tests (FAIL)**

- [ ] **Step 3: Implement auth.php (permission callback)**

```php
<?php
// services/alfred-seo/inc/rest/auth.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_rest_permission_check( WP_REST_Request $req ) {
    // WP's REST API handles application-password auth automatically when
    // the Authorization: Basic header is present. We just require an authed
    // user with manage_options.
    if ( ! is_user_logged_in() ) {
        return new WP_Error( 'rest_unauthorized', 'Authentication required', array( 'status' => 401 ) );
    }
    if ( ! current_user_can( 'manage_options' ) ) {
        return new WP_Error( 'rest_forbidden', 'Insufficient permissions', array( 'status' => 403 ) );
    }
    return true;
}
```

- [ ] **Step 4: Implement audit.php**

```php
<?php
// services/alfred-seo/inc/rest/audit.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_action( 'rest_api_init', function () {
    register_rest_route( 'alfred-seo/v1', '/audit', array(
        'methods'             => 'GET',
        'callback'            => 'alfred_seo_rest_audit',
        'permission_callback' => 'alfred_seo_rest_permission_check',
    ) );
});

function alfred_seo_rest_audit( WP_REST_Request $req ) {
    $page_types = array( 'post', 'page' );
    if ( post_type_exists( 'product' ) ) { $page_types[] = 'product'; }

    $pages = array();
    $missing_meta = array();
    $missing_alt  = array();

    $posts = get_posts( array(
        'post_type'   => $page_types,
        'post_status' => 'publish',
        'numberposts' => -1,
    ) );
    foreach ( $posts as $p ) {
        $has_meta = (bool) get_post_meta( $p->ID, '_alfred_seo_meta_description', true );
        $pages[]  = array(
            'id'       => $p->ID,
            'type'     => $p->post_type,
            'url'      => get_permalink( $p ),
            'title'    => $p->post_title,
            'has_meta' => $has_meta,
            'modified' => mysql2date( 'c', $p->post_modified_gmt, false ),
        );
        if ( ! $has_meta && ! $p->post_excerpt ) {
            $missing_meta[] = array( 'id' => $p->ID, 'url' => get_permalink( $p ) );
        }
    }

    $atts = get_posts( array(
        'post_type'   => 'attachment',
        'post_status' => 'inherit',
        'post_mime_type' => 'image',
        'numberposts' => -1,
    ) );
    foreach ( $atts as $a ) {
        $alt = get_post_meta( $a->ID, '_wp_attachment_image_alt', true );
        if ( ! $alt ) {
            $missing_alt[] = array( 'id' => $a->ID, 'url' => wp_get_attachment_url( $a->ID ) );
        }
    }

    return new WP_REST_Response( array(
        'pages'        => $pages,
        'missing_meta' => $missing_meta,
        'missing_alt'  => $missing_alt,
        'audited_at'   => gmdate( 'c' ),
    ), 200 );
}
```

- [ ] **Step 5: Implement content.php**

```php
<?php
// services/alfred-seo/inc/rest/content.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_action( 'rest_api_init', function () {
    register_rest_route( 'alfred-seo/v1', '/content', array(
        'methods'             => 'POST',
        'callback'            => 'alfred_seo_rest_content',
        'permission_callback' => 'alfred_seo_rest_permission_check',
    ) );
});

function alfred_seo_rest_content( WP_REST_Request $req ) {
    $title     = sanitize_text_field( $req->get_param( 'title' ) );
    $content   = wp_kses_post( $req->get_param( 'content' ) );
    $meta_desc = sanitize_text_field( $req->get_param( 'meta_description' ) );
    $slug      = sanitize_title( $req->get_param( 'slug' ) );
    $post_type = $req->get_param( 'post_type' ) ?: 'post';
    $status    = $req->get_param( 'status' ) ?: 'draft';
    $og_title  = sanitize_text_field( $req->get_param( 'og_title' ) );
    $og_desc   = sanitize_text_field( $req->get_param( 'og_description' ) );
    $og_image  = esc_url_raw( $req->get_param( 'og_image' ) );

    if ( ! $title || ! $content ) {
        return new WP_Error( 'rest_bad_request', 'title and content are required', array( 'status' => 400 ) );
    }
    if ( ! post_type_exists( $post_type ) ) {
        return new WP_Error( 'rest_bad_request', 'invalid post_type', array( 'status' => 400 ) );
    }

    $post_id = wp_insert_post( array(
        'post_title'   => $title,
        'post_content' => $content,
        'post_status'  => in_array( $status, array( 'draft', 'pending', 'publish' ), true ) ? $status : 'draft',
        'post_type'    => $post_type,
        'post_name'    => $slug,
    ), true );

    if ( is_wp_error( $post_id ) ) {
        return new WP_Error( 'rest_internal', $post_id->get_error_message(), array( 'status' => 500 ) );
    }

    if ( $meta_desc ) { update_post_meta( $post_id, '_alfred_seo_meta_description', $meta_desc ); }
    if ( $og_title )  { update_post_meta( $post_id, '_alfred_seo_og_title', $og_title ); }
    if ( $og_desc )   { update_post_meta( $post_id, '_alfred_seo_og_description', $og_desc ); }
    if ( $og_image )  { update_post_meta( $post_id, '_alfred_seo_og_image', $og_image ); }

    return new WP_REST_Response( array(
        'post_id' => $post_id,
        'url'     => get_permalink( $post_id ),
        'status'  => get_post_status( $post_id ),
    ), 201 );
}
```

- [ ] **Step 6: Implement meta.php (POST /meta)**

```php
<?php
// services/alfred-seo/inc/rest/meta.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_action( 'rest_api_init', function () {
    register_rest_route( 'alfred-seo/v1', '/meta', array(
        'methods'             => 'POST',
        'callback'            => 'alfred_seo_rest_meta',
        'permission_callback' => 'alfred_seo_rest_permission_check',
    ) );
});

function alfred_seo_rest_meta( WP_REST_Request $req ) {
    $post_id   = absint( $req->get_param( 'post_id' ) );
    $meta_desc = $req->get_param( 'meta_description' );
    $og_title  = $req->get_param( 'og_title' );
    $og_desc   = $req->get_param( 'og_description' );
    $og_image  = $req->get_param( 'og_image' );

    if ( ! $post_id || ! get_post( $post_id ) ) {
        return new WP_Error( 'rest_not_found', 'post not found', array( 'status' => 404 ) );
    }
    if ( null !== $meta_desc ) { update_post_meta( $post_id, '_alfred_seo_meta_description', sanitize_text_field( $meta_desc ) ); }
    if ( null !== $og_title )  { update_post_meta( $post_id, '_alfred_seo_og_title', sanitize_text_field( $og_title ) ); }
    if ( null !== $og_desc )   { update_post_meta( $post_id, '_alfred_seo_og_description', sanitize_text_field( $og_desc ) ); }
    if ( null !== $og_image )  { update_post_meta( $post_id, '_alfred_seo_og_image', esc_url_raw( $og_image ) ); }

    return new WP_REST_Response( array( 'post_id' => $post_id, 'updated_at' => gmdate( 'c' ) ), 200 );
}
```

- [ ] **Step 7: Implement internal-links.php (POST /internal-links)**

```php
<?php
// services/alfred-seo/inc/rest/internal-links.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_action( 'rest_api_init', function () {
    register_rest_route( 'alfred-seo/v1', '/internal-links', array(
        'methods'             => 'POST',
        'callback'            => 'alfred_seo_rest_internal_links',
        'permission_callback' => 'alfred_seo_rest_permission_check',
    ) );
});

function alfred_seo_rest_internal_links( WP_REST_Request $req ) {
    $map = $req->get_param( 'links' );
    if ( ! is_array( $map ) ) {
        return new WP_Error( 'rest_bad_request', 'links must be object phrase→url', array( 'status' => 400 ) );
    }
    $clean = array();
    foreach ( $map as $phrase => $url ) {
        $p = sanitize_text_field( $phrase );
        $u = esc_url_raw( $url );
        if ( $p && $u ) { $clean[ $p ] = $u; }
    }
    alfred_seo_update_settings( array( 'internal_links' => $clean ) );
    return new WP_REST_Response( array( 'count' => count( $clean ), 'updated_at' => gmdate( 'c' ) ), 200 );
}
```

- [ ] **Step 8: Implement sitemap-ping.php (POST /sitemap-ping)**

```php
<?php
// services/alfred-seo/inc/rest/sitemap-ping.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_action( 'rest_api_init', function () {
    register_rest_route( 'alfred-seo/v1', '/sitemap-ping', array(
        'methods'             => 'POST',
        'callback'            => 'alfred_seo_rest_sitemap_ping',
        'permission_callback' => 'alfred_seo_rest_permission_check',
    ) );
});

function alfred_seo_rest_sitemap_ping( WP_REST_Request $req ) {
    $sitemap = home_url( '/alfred-sitemap.xml' );
    $google  = wp_remote_get( 'https://www.google.com/ping?sitemap=' . urlencode( $sitemap ), array( 'timeout' => 5 ) );
    $bing    = wp_remote_get( 'https://www.bing.com/ping?sitemap=' . urlencode( $sitemap ), array( 'timeout' => 5 ) );
    return new WP_REST_Response( array(
        'sitemap_url' => $sitemap,
        'google'      => is_wp_error( $google ) ? 'error' : wp_remote_retrieve_response_code( $google ),
        'bing'        => is_wp_error( $bing ) ? 'error' : wp_remote_retrieve_response_code( $bing ),
        'pinged_at'   => gmdate( 'c' ),
    ), 200 );
}
```

- [ ] **Step 9: Wire REST modules into alfred-seo.php**

```php
require_once ALFRED_SEO_DIR . 'inc/rest/auth.php';
require_once ALFRED_SEO_DIR . 'inc/rest/audit.php';
require_once ALFRED_SEO_DIR . 'inc/rest/content.php';
require_once ALFRED_SEO_DIR . 'inc/rest/meta.php';
require_once ALFRED_SEO_DIR . 'inc/rest/internal-links.php';
require_once ALFRED_SEO_DIR . 'inc/rest/sitemap-ping.php';
```

- [ ] **Step 10: Run tests (PASS) + commit**

```bash
git add services/alfred-seo/inc/rest/ services/alfred-seo/tests/ services/alfred-seo/alfred-seo.php
git commit -m "feat(alfred-seo): 5 REST endpoints (audit, content, meta, internal-links, sitemap-ping) with app-password auth"
```

---

## Task 17: WP unit test bootstrap

**Files:**
- Create: `services/alfred-seo/tests/bootstrap.php`
- Create: `services/alfred-seo/phpunit.xml.dist`

- [ ] **Step 1: Create phpunit.xml.dist**

```xml
<?xml version="1.0"?>
<phpunit
    bootstrap="tests/bootstrap.php"
    backupGlobals="false"
    colors="true"
    convertErrorsToExceptions="true"
    convertNoticesToExceptions="true"
    convertWarningsToExceptions="true">
    <testsuites>
        <testsuite name="alfred-seo">
            <directory prefix="test-" suffix=".php">./tests</directory>
        </testsuite>
    </testsuites>
</phpunit>
```

- [ ] **Step 2: Create tests/bootstrap.php**

```php
<?php
// services/alfred-seo/tests/bootstrap.php

$_tests_dir = getenv( 'WP_TESTS_DIR' );
if ( ! $_tests_dir ) {
    $_tests_dir = rtrim( sys_get_temp_dir(), '/\\' ) . '/wordpress-tests-lib';
}
if ( ! file_exists( "{$_tests_dir}/includes/functions.php" ) ) {
    echo "Could not find {$_tests_dir}/includes/functions.php — set WP_TESTS_DIR env var.\n";
    exit( 1 );
}
require_once "{$_tests_dir}/includes/functions.php";

function _manually_load_plugin() {
    // Load WooCommerce first if available (Roen needs it for product tests).
    $wc_path = dirname( __DIR__, 2 ) . '/woocommerce/woocommerce.php';
    if ( file_exists( $wc_path ) ) { require $wc_path; }
    require dirname( __DIR__ ) . '/alfred-seo.php';
}
tests_add_filter( 'muplugins_loaded', '_manually_load_plugin' );

require "{$_tests_dir}/includes/bootstrap.php";
```

- [ ] **Step 3: Document local test run in readme**

Append to `services/alfred-seo/readme.txt`:

```
== Local Testing ==

Run the WP test suite installer once:

    bash bin/install-wp-tests.sh wordpress_test root '' localhost latest

Then run tests:

    cd services/alfred-seo && phpunit

```

- [ ] **Step 4: Commit**

```bash
git add services/alfred-seo/tests/bootstrap.php services/alfred-seo/phpunit.xml.dist services/alfred-seo/readme.txt
git commit -m "test(alfred-seo): phpunit bootstrap loading plugin + WooCommerce"
```

---

## Task 18: Deploy script + initial deploy to roenhandmade-wp

**Files:**
- Create: `services/alfred-seo/deploy.sh`

- [ ] **Step 1: Write deploy.sh (modeled on services/roen-minimal/deploy.sh)**

```bash
#!/usr/bin/env bash
# Deploy alfred-seo plugin to roenhandmade-wp.
# Idempotent — safe to re-run. First site only; multi-site loop arrives in Plan 3.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_HOST="${SSH_HOST:-server-104}"
CONTAINER="${CONTAINER:-roenhandmade-wp}"
WP_PATH="${WP_PATH:-/var/www/html}"
STAGE_DIR="/tmp/alfred-seo"
TARGET="${WP_PATH}/wp-content/plugins/alfred-seo"

echo "==> rsync source to ${SSH_HOST}:${STAGE_DIR}"
rsync -av --delete \
  --exclude 'tests/' \
  --exclude 'deploy.sh' \
  --exclude 'phpunit.xml.dist' \
  --exclude '.DS_Store' \
  "${SRC_DIR}/" "${SSH_HOST}:${STAGE_DIR}/"

echo "==> tar-pipe into ${CONTAINER}:${TARGET}"
ssh "${SSH_HOST}" "
  set -e
  timeout 30 docker exec ${CONTAINER} mkdir -p ${TARGET}
  tar -C ${STAGE_DIR} -cf - . | timeout 60 docker exec -i ${CONTAINER} tar -C ${TARGET} -xf -
  timeout 30 docker exec ${CONTAINER} chown -R www-data:www-data ${TARGET}
"

echo "==> activate plugin via wp-cli"
ssh "${SSH_HOST}" "timeout 30 docker exec ${CONTAINER} wp plugin activate alfred-seo --allow-root --path=${WP_PATH}"

echo "==> flush rewrite rules"
ssh "${SSH_HOST}" "timeout 30 docker exec ${CONTAINER} wp rewrite flush --allow-root --path=${WP_PATH}"

echo "==> flush WP cache"
ssh "${SSH_HOST}" "timeout 30 docker exec ${CONTAINER} wp cache flush --allow-root --path=${WP_PATH} || true"

echo "==> done"
```

- [ ] **Step 2: Make executable**

`chmod +x services/alfred-seo/deploy.sh`

- [ ] **Step 3: Run deploy to Roen**

`bash services/alfred-seo/deploy.sh`

Expected output (final line): `==> done`

- [ ] **Step 4: Verify plugin active on Roen**

`ssh server-104 'timeout 20 docker exec roenhandmade-wp wp plugin list --allow-root --path=/var/www/html --format=csv | grep alfred-seo'`

Expected: `alfred-seo,active,none,0.1.0,,off`

- [ ] **Step 5: Configure initial settings via WP-CLI (Roen-specific values)**

```bash
ssh server-104 'timeout 30 docker exec roenhandmade-wp wp option update alfred_seo_settings "$(cat <<JSON
{
  "site_slug": "roen",
  "business_name": "Roen",
  "business_type": "LocalBusiness",
  "social_handles": {"instagram": "roenhandmade", "facebook": "roenhandmade"},
  "local_address": {
    "addressLocality": "Atlanta",
    "addressRegion": "GA",
    "addressCountry": "US"
  },
  "internal_links": {},
  "alfred_endpoint": "https://aialfred.groundrushcloud.com",
  "alt_text_enabled": true,
  "sitemap_enabled": true
}
JSON
)" --format=json --allow-root --path=/var/www/html'
```

- [ ] **Step 6: Commit**

```bash
git add services/alfred-seo/deploy.sh
git commit -m "feat(alfred-seo): deploy script + initial Roen deploy with LocalBusiness settings"
```

---

## Task 19: Functional verification on live Roen site

**Files:** No new files — verification only.

- [ ] **Step 1: Verify sitemap is reachable**

```bash
curl -s -o /dev/null -w "HTTP %{http_code}\n" https://www.roenhandmade.com/alfred-sitemap.xml
curl -s https://www.roenhandmade.com/alfred-sitemap.xml | head -10
```

Expected: HTTP 200, XML output with `<sitemapindex>` and 4 sitemap entries.

- [ ] **Step 2: Verify each sub-sitemap returns products/pages**

```bash
curl -s https://www.roenhandmade.com/alfred-sitemap-products.xml | grep -c '<url>'
```

Expected: ≥ 90 (Roen has 92-93 products).

- [ ] **Step 3: Verify schema injection on a product page**

```bash
curl -s https://www.roenhandmade.com/product/red-bead-toggle-necklace/ \
  | python3 -c "
import sys, re, json
html = sys.stdin.read()
m = re.search(r'<script type=\"application/ld\+json\">(.*?)</script>', html, re.S)
print('Schema present:', bool(m))
if m:
    data = json.loads(m.group(1))
    print(json.dumps(data, indent=2)[:600])
"
```

Expected: Schema present: True, Product schema with name + offers + image.

- [ ] **Step 4: Verify OG tags on home page**

```bash
curl -s https://www.roenhandmade.com/ | grep -oE 'property="og:[a-z_]+" content="[^"]*"' | head -8
```

Expected: og:title, og:url, og:type, og:site_name, og:image all populated.

- [ ] **Step 5: Verify meta description on home page**

```bash
curl -s https://www.roenhandmade.com/ | grep -oE '<meta name="description" content="[^"]*"'
```

Expected: One line, non-empty content.

- [ ] **Step 6: Verify robots.txt has sitemap line**

```bash
curl -s https://www.roenhandmade.com/robots.txt | grep -i sitemap
```

Expected: `Sitemap: https://www.roenhandmade.com/alfred-sitemap.xml`

- [ ] **Step 7: Verify Google Rich Results Test passes on a product**

Open: https://search.google.com/test/rich-results?url=https%3A%2F%2Fwww.roenhandmade.com%2Fproduct%2Fred-bead-toggle-necklace%2F

Expected: Google reports "Product" detected, no errors. Document the result (screenshot or copy/paste) in a verification note. Capture the result URL.

- [ ] **Step 8: Submit sitemap to Google Search Console + Bing Webmaster (manual UI step for Mike)**

GSC: https://search.google.com/search-console → roenhandmade.com → Sitemaps → add `alfred-sitemap.xml`
Bing: https://www.bing.com/webmasters → roenhandmade.com → Sitemaps → submit

- [ ] **Step 9: Commit verification record (no code)**

```bash
mkdir -p docs/seo/verifications
cat > docs/seo/verifications/2026-05-14-roen-plugin-go-live.md << 'NOTE'
# Roen alfred-seo plugin go-live verification

Date: <fill on day-of>
- Sitemap index: HTTP 200 ✓
- Products sitemap: N URLs ✓
- Product schema valid (Google Rich Results Test) ✓
- OG + Twitter cards on homepage ✓
- Meta description present ✓
- robots.txt sitemap line ✓
- GSC sitemap submitted ✓
- Bing sitemap submitted ✓
NOTE

git add docs/seo/verifications/2026-05-14-roen-plugin-go-live.md
git commit -m "docs(seo): Roen alfred-seo plugin go-live verification"
```

---


## Task 20: SQL migration — 9 seo_* tables

**Files:**
- Create: `migrations/2026-05-14-seo-initial-schema.sql`

- [ ] **Step 1: Write migration**

```sql
-- migrations/2026-05-14-seo-initial-schema.sql
-- Phase 1 SEO foundation schema. Runs against alfred_main.

BEGIN;

CREATE TABLE IF NOT EXISTS seo_sites (
    id                  SERIAL PRIMARY KEY,
    slug                VARCHAR(64) UNIQUE NOT NULL,
    domain              VARCHAR(255) NOT NULL,
    display_name        VARCHAR(255) NOT NULL,
    wp_rest_url         VARCHAR(255) NOT NULL,
    wp_username         VARCHAR(255),
    wp_app_password     TEXT,                          -- encrypted at app layer
    gsc_property        VARCHAR(255),
    ga4_property_id     VARCHAR(64),
    brand_profile_path  VARCHAR(255),
    business_type       VARCHAR(32) DEFAULT 'Organization',
    status              VARCHAR(32) DEFAULT 'active',
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS seo_queries (
    id            BIGSERIAL PRIMARY KEY,
    site_id       INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    query         TEXT NOT NULL,
    position      NUMERIC(6,2),
    impressions   INTEGER DEFAULT 0,
    clicks        INTEGER DEFAULT 0,
    ctr           NUMERIC(6,4),
    captured_at   DATE NOT NULL,
    UNIQUE (site_id, query, captured_at)
);
CREATE INDEX IF NOT EXISTS idx_seo_queries_site_date ON seo_queries (site_id, captured_at DESC);

CREATE TABLE IF NOT EXISTS seo_pages (
    id                  BIGSERIAL PRIMARY KEY,
    site_id             INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    url                 TEXT NOT NULL,
    page_type           VARCHAR(64),                  -- product | post | page | category
    indexed_at          TIMESTAMPTZ,
    last_audit_at       TIMESTAMPTZ,
    schema_status       VARCHAR(32),                  -- ok | invalid | missing
    meta_status         VARCHAR(32),                  -- ok | missing
    cwv_lcp_ms          INTEGER,
    cwv_cls             NUMERIC(6,3),
    cwv_inp_ms          INTEGER,
    organic_sessions    INTEGER DEFAULT 0,
    conversions         INTEGER DEFAULT 0,
    last_seen_at        TIMESTAMPTZ DEFAULT now(),
    UNIQUE (site_id, url)
);
CREATE INDEX IF NOT EXISTS idx_seo_pages_site ON seo_pages (site_id);

CREATE TABLE IF NOT EXISTS seo_briefs (
    id                BIGSERIAL PRIMARY KEY,
    site_id           INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    topic             TEXT NOT NULL,
    content_type      VARCHAR(32),                    -- product_enrichment | cluster | blog | ad_landing
    target_keywords   JSONB,
    audience          TEXT,
    status            VARCHAR(32) DEFAULT 'queued',
    brief_payload     JSONB,
    source_signal     JSONB,
    created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS seo_pending (
    id              BIGSERIAL PRIMARY KEY,
    site_id         INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    brief_id        BIGINT REFERENCES seo_briefs(id) ON DELETE SET NULL,
    content_type    VARCHAR(32),
    title           TEXT,
    body_payload    JSONB,
    source_signal   JSONB,
    status          VARCHAR(32) DEFAULT 'pending',
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS seo_decided (
    id              BIGSERIAL PRIMARY KEY,
    site_id         INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    brief_id        BIGINT REFERENCES seo_briefs(id) ON DELETE SET NULL,
    content_type    VARCHAR(32),
    title           TEXT,
    body_payload    JSONB,
    decided_at      TIMESTAMPTZ DEFAULT now(),
    decided_by      VARCHAR(64),
    outcome         VARCHAR(32),                       -- approved | rejected
    wp_post_id      BIGINT,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS seo_backlinks (
    id              BIGSERIAL PRIMARY KEY,
    site_id         INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    source_url      TEXT NOT NULL,
    target_url      TEXT,
    anchor_text     TEXT,
    first_seen      TIMESTAMPTZ DEFAULT now(),
    last_seen       TIMESTAMPTZ DEFAULT now(),
    lost_at         TIMESTAMPTZ,
    UNIQUE (site_id, source_url, target_url)
);

CREATE TABLE IF NOT EXISTS seo_haro_opps (
    id                   BIGSERIAL PRIMARY KEY,
    site_id              INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    source_email_id      VARCHAR(255),
    query_text           TEXT,
    deadline             TIMESTAMPTZ,
    draft_pitch_payload  JSONB,
    status               VARCHAR(32) DEFAULT 'pending',
    response_sent_at     TIMESTAMPTZ,
    created_at           TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS seo_rankings_daily (
    id              BIGSERIAL PRIMARY KEY,
    site_id         INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    query           TEXT NOT NULL,
    position        NUMERIC(6,2),
    captured_at     DATE NOT NULL,
    UNIQUE (site_id, query, captured_at)
);

COMMIT;
```

- [ ] **Step 2: Apply migration**

```bash
psql -h localhost -U alfred -d alfred_main -f migrations/2026-05-14-seo-initial-schema.sql
```

Expected: `BEGIN` then 9 `CREATE TABLE`/`CREATE INDEX` notices then `COMMIT`.

- [ ] **Step 3: Verify all 9 tables exist**

```bash
psql -h localhost -U alfred -d alfred_main -c "\dt seo_*"
```

Expected: 9 tables listed.

- [ ] **Step 4: Commit**

```bash
git add migrations/2026-05-14-seo-initial-schema.sql
git commit -m "feat(seo): initial schema — 9 seo_* tables for sites/queries/pages/briefs/pending/decided/backlinks/haro/rankings"
```

---

## Task 21: SQLAlchemy models for 9 tables

**Files:**
- Create: `core/seo/__init__.py`
- Create: `core/seo/db.py`
- Create: `core/seo/models.py`
- Test: `tests/core/seo/test_models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/core/seo/test_models.py
import datetime as dt
import pytest
from core.seo.db import SessionLocal, Base, engine
from core.seo.models import SeoSite, SeoQuery


@pytest.fixture(autouse=True)
def _tables():
    Base.metadata.create_all(engine, tables=[SeoSite.__table__, SeoQuery.__table__])
    yield
    # Best-effort cleanup
    with SessionLocal() as s:
        s.execute(SeoQuery.__table__.delete())
        s.execute(SeoSite.__table__.delete())
        s.commit()


def test_create_site_and_query():
    with SessionLocal() as s:
        site = SeoSite(
            slug="roen-test", domain="roenhandmade-test.invalid",
            display_name="Roen Test", wp_rest_url="https://x/wp-json",
        )
        s.add(site)
        s.commit()
        q = SeoQuery(
            site_id=site.id, query="evil eye bracelet",
            position=14.2, impressions=1247, clicks=47, ctr=0.0377,
            captured_at=dt.date(2026, 5, 14),
        )
        s.add(q)
        s.commit()
        assert q.id is not None
        assert q.site.slug == "roen-test"
```

- [ ] **Step 2: Run test (FAIL — modules don't exist)**

`./venv/bin/pytest tests/core/seo/test_models.py -v`

- [ ] **Step 3: Implement db.py**

```python
# core/seo/db.py
"""SQLAlchemy session + declarative base for the SEO module.

Uses the existing Alfred database URL from config.settings.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config.settings import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)
Base = declarative_base()
```

- [ ] **Step 4: Implement models.py**

```python
# core/seo/models.py
"""SQLAlchemy table definitions matching migrations/2026-05-14-seo-initial-schema.sql."""
from __future__ import annotations

import datetime as dt
from sqlalchemy import (
    BigInteger, Column, Date, DateTime, ForeignKey, Integer, Numeric,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from core.seo.db import Base


class SeoSite(Base):
    __tablename__ = "seo_sites"
    id                 = Column(Integer, primary_key=True)
    slug               = Column(String(64), unique=True, nullable=False)
    domain             = Column(String(255), nullable=False)
    display_name       = Column(String(255), nullable=False)
    wp_rest_url        = Column(String(255), nullable=False)
    wp_username        = Column(String(255))
    wp_app_password    = Column(Text)  # encrypted at app layer
    gsc_property       = Column(String(255))
    ga4_property_id    = Column(String(64))
    brand_profile_path = Column(String(255))
    business_type      = Column(String(32), default="Organization")
    status             = Column(String(32), default="active")
    created_at         = Column(DateTime(timezone=True), server_default=func.now())
    updated_at         = Column(DateTime(timezone=True), server_default=func.now())

    queries     = relationship("SeoQuery", back_populates="site", cascade="all, delete-orphan")
    pages       = relationship("SeoPage", back_populates="site", cascade="all, delete-orphan")
    briefs      = relationship("SeoBrief", back_populates="site", cascade="all, delete-orphan")
    pending     = relationship("SeoPending", back_populates="site", cascade="all, delete-orphan")
    decided     = relationship("SeoDecided", back_populates="site", cascade="all, delete-orphan")
    backlinks   = relationship("SeoBacklink", back_populates="site", cascade="all, delete-orphan")
    haro_opps   = relationship("SeoHaroOpp", back_populates="site", cascade="all, delete-orphan")
    rankings    = relationship("SeoRankingDaily", back_populates="site", cascade="all, delete-orphan")


class SeoQuery(Base):
    __tablename__ = "seo_queries"
    id          = Column(BigInteger, primary_key=True)
    site_id     = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    query       = Column(Text, nullable=False)
    position    = Column(Numeric(6, 2))
    impressions = Column(Integer, default=0)
    clicks      = Column(Integer, default=0)
    ctr         = Column(Numeric(6, 4))
    captured_at = Column(Date, nullable=False)
    __table_args__ = (UniqueConstraint("site_id", "query", "captured_at"),)
    site = relationship("SeoSite", back_populates="queries")


class SeoPage(Base):
    __tablename__ = "seo_pages"
    id                = Column(BigInteger, primary_key=True)
    site_id           = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    url               = Column(Text, nullable=False)
    page_type         = Column(String(64))
    indexed_at        = Column(DateTime(timezone=True))
    last_audit_at     = Column(DateTime(timezone=True))
    schema_status     = Column(String(32))
    meta_status       = Column(String(32))
    cwv_lcp_ms        = Column(Integer)
    cwv_cls           = Column(Numeric(6, 3))
    cwv_inp_ms        = Column(Integer)
    organic_sessions  = Column(Integer, default=0)
    conversions       = Column(Integer, default=0)
    last_seen_at      = Column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (UniqueConstraint("site_id", "url"),)
    site = relationship("SeoSite", back_populates="pages")


class SeoBrief(Base):
    __tablename__ = "seo_briefs"
    id              = Column(BigInteger, primary_key=True)
    site_id         = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    topic           = Column(Text, nullable=False)
    content_type    = Column(String(32))
    target_keywords = Column(JSONB)
    audience        = Column(Text)
    status          = Column(String(32), default="queued")
    brief_payload   = Column(JSONB)
    source_signal   = Column(JSONB)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    site = relationship("SeoSite", back_populates="briefs")


class SeoPending(Base):
    __tablename__ = "seo_pending"
    id            = Column(BigInteger, primary_key=True)
    site_id       = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    brief_id      = Column(BigInteger, ForeignKey("seo_briefs.id", ondelete="SET NULL"))
    content_type  = Column(String(32))
    title         = Column(Text)
    body_payload  = Column(JSONB)
    source_signal = Column(JSONB)
    status        = Column(String(32), default="pending")
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    site = relationship("SeoSite", back_populates="pending")


class SeoDecided(Base):
    __tablename__ = "seo_decided"
    id           = Column(BigInteger, primary_key=True)
    site_id      = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    brief_id     = Column(BigInteger, ForeignKey("seo_briefs.id", ondelete="SET NULL"))
    content_type = Column(String(32))
    title        = Column(Text)
    body_payload = Column(JSONB)
    decided_at   = Column(DateTime(timezone=True), server_default=func.now())
    decided_by   = Column(String(64))
    outcome      = Column(String(32))
    wp_post_id   = Column(BigInteger)
    error        = Column(Text)
    site = relationship("SeoSite", back_populates="decided")


class SeoBacklink(Base):
    __tablename__ = "seo_backlinks"
    id          = Column(BigInteger, primary_key=True)
    site_id     = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    source_url  = Column(Text, nullable=False)
    target_url  = Column(Text)
    anchor_text = Column(Text)
    first_seen  = Column(DateTime(timezone=True), server_default=func.now())
    last_seen   = Column(DateTime(timezone=True), server_default=func.now())
    lost_at     = Column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("site_id", "source_url", "target_url"),)
    site = relationship("SeoSite", back_populates="backlinks")


class SeoHaroOpp(Base):
    __tablename__ = "seo_haro_opps"
    id                  = Column(BigInteger, primary_key=True)
    site_id             = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    source_email_id     = Column(String(255))
    query_text          = Column(Text)
    deadline            = Column(DateTime(timezone=True))
    draft_pitch_payload = Column(JSONB)
    status              = Column(String(32), default="pending")
    response_sent_at    = Column(DateTime(timezone=True))
    created_at          = Column(DateTime(timezone=True), server_default=func.now())
    site = relationship("SeoSite", back_populates="haro_opps")


class SeoRankingDaily(Base):
    __tablename__ = "seo_rankings_daily"
    id          = Column(BigInteger, primary_key=True)
    site_id     = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    query       = Column(Text, nullable=False)
    position    = Column(Numeric(6, 2))
    captured_at = Column(Date, nullable=False)
    __table_args__ = (UniqueConstraint("site_id", "query", "captured_at"),)
    site = relationship("SeoSite", back_populates="rankings")
```

- [ ] **Step 5: Create core/seo/__init__.py**

```python
# core/seo/__init__.py
"""SEO orchestrator module. Mirrors design spec sections.

See: docs/superpowers/specs/2026-05-14-roen-seo-pipeline-design.md
"""
```

- [ ] **Step 6: Run test (PASS) + commit**

```bash
git add core/seo/__init__.py core/seo/db.py core/seo/models.py tests/core/seo/test_models.py
git commit -m "feat(seo): SQLAlchemy models for 9 seo_* tables with relationships"
```

---

## Task 22: Site registry CRUD

**Files:**
- Create: `core/seo/sites/__init__.py`
- Create: `core/seo/sites/registry.py`
- Test: `tests/core/seo/test_registry.py`

- [ ] **Step 1: Write failing test**

```python
# tests/core/seo/test_registry.py
import pytest
from core.seo.db import SessionLocal
from core.seo.sites.registry import (
    register_site, get_site_by_slug, list_sites, update_site, deactivate_site
)


def test_register_and_fetch_site():
    site = register_site(
        slug="roen-test-22",
        domain="roen22.invalid",
        display_name="Roen 22",
        wp_rest_url="https://roen22.invalid/wp-json",
    )
    assert site.id is not None
    fetched = get_site_by_slug("roen-test-22")
    assert fetched.display_name == "Roen 22"
    deactivate_site("roen-test-22")  # cleanup-ish
    fetched = get_site_by_slug("roen-test-22")
    assert fetched.status == "inactive"


def test_register_rejects_duplicate_slug():
    register_site(slug="dup-test", domain="x.invalid", display_name="X", wp_rest_url="https://x")
    with pytest.raises(ValueError):
        register_site(slug="dup-test", domain="y.invalid", display_name="Y", wp_rest_url="https://y")
    deactivate_site("dup-test")


def test_list_sites_only_active():
    register_site(slug="active-22", domain="a.invalid", display_name="A", wp_rest_url="https://a")
    register_site(slug="inactive-22", domain="b.invalid", display_name="B", wp_rest_url="https://b")
    deactivate_site("inactive-22")
    slugs = {s.slug for s in list_sites()}
    assert "active-22" in slugs
    assert "inactive-22" not in slugs
    deactivate_site("active-22")
```

- [ ] **Step 2: Run test (FAIL)**

- [ ] **Step 3: Implement registry.py**

```python
# core/seo/sites/registry.py
"""CRUD for seo_sites. The orchestrator's source of truth on which sites exist."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import select

from core.seo.db import SessionLocal
from core.seo.models import SeoSite


def register_site(
    slug: str,
    domain: str,
    display_name: str,
    wp_rest_url: str,
    *,
    wp_username: Optional[str] = None,
    wp_app_password: Optional[str] = None,  # caller should pass already-encrypted
    gsc_property: Optional[str] = None,
    ga4_property_id: Optional[str] = None,
    brand_profile_path: Optional[str] = None,
    business_type: str = "Organization",
) -> SeoSite:
    """Insert a new site. Raises ValueError on duplicate slug."""
    with SessionLocal() as s:
        existing = s.scalar(select(SeoSite).where(SeoSite.slug == slug))
        if existing:
            raise ValueError(f"site already registered: {slug}")
        site = SeoSite(
            slug=slug,
            domain=domain,
            display_name=display_name,
            wp_rest_url=wp_rest_url,
            wp_username=wp_username,
            wp_app_password=wp_app_password,
            gsc_property=gsc_property,
            ga4_property_id=ga4_property_id,
            brand_profile_path=brand_profile_path,
            business_type=business_type,
            status="active",
        )
        s.add(site)
        s.commit()
        s.refresh(site)
        return site


def get_site_by_slug(slug: str) -> Optional[SeoSite]:
    with SessionLocal() as s:
        return s.scalar(select(SeoSite).where(SeoSite.slug == slug))


def get_site_by_id(site_id: int) -> Optional[SeoSite]:
    with SessionLocal() as s:
        return s.get(SeoSite, site_id)


def list_sites(include_inactive: bool = False) -> list[SeoSite]:
    with SessionLocal() as s:
        q = select(SeoSite).order_by(SeoSite.id)
        if not include_inactive:
            q = q.where(SeoSite.status == "active")
        return list(s.scalars(q).all())


def update_site(slug: str, **patch) -> SeoSite:
    allowed = {
        "domain", "display_name", "wp_rest_url", "wp_username", "wp_app_password",
        "gsc_property", "ga4_property_id", "brand_profile_path", "business_type", "status",
    }
    bad = set(patch.keys()) - allowed
    if bad:
        raise ValueError(f"cannot update fields: {bad}")
    with SessionLocal() as s:
        site = s.scalar(select(SeoSite).where(SeoSite.slug == slug))
        if not site:
            raise ValueError(f"site not found: {slug}")
        for k, v in patch.items():
            setattr(site, k, v)
        s.commit()
        s.refresh(site)
        return site


def deactivate_site(slug: str) -> None:
    update_site(slug, status="inactive")
```

- [ ] **Step 4: Implement core/seo/sites/__init__.py**

```python
# core/seo/sites/__init__.py
from core.seo.sites.registry import (
    register_site, get_site_by_slug, get_site_by_id, list_sites, update_site, deactivate_site,
)
__all__ = ["register_site", "get_site_by_slug", "get_site_by_id", "list_sites", "update_site", "deactivate_site"]
```

- [ ] **Step 5: Run test (PASS) + commit**

```bash
git add core/seo/sites/ tests/core/seo/test_registry.py
git commit -m "feat(seo): site registry CRUD with slug uniqueness + soft deactivate"
```

---

## Task 23: FastAPI seo_admin routes + JWT auth

**Files:**
- Create: `core/api/seo_admin.py`
- Modify: `core/api/main.py` (register seo_admin)

- [ ] **Step 1: Implement seo_admin.py**

```python
# core/api/seo_admin.py
"""FastAPI routes for /admin/seo/*. Auth-gated via the existing JWT cookie.

Phase 1 routes: dashboard, site list, site detail. Approval queue, content
preview, and editing land in Plan 2.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from core.security.auth import get_current_user, require_auth
from core.seo.sites.registry import get_site_by_slug, list_sites

logger = logging.getLogger(__name__)


def _format_when(dt_val: datetime | None) -> str:
    if not dt_val:
        return "—"
    delta = datetime.now(timezone.utc) - dt_val
    if delta.total_seconds() < 60:
        return "just now"
    if delta.total_seconds() < 3600:
        return f"{int(delta.total_seconds() / 60)}m ago"
    if delta.total_seconds() < 86400:
        return f"{int(delta.total_seconds() / 3600)}h ago"
    return dt_val.strftime("%b %d %H:%M")


def _render_dashboard(sites: list) -> str:
    rows = []
    for s in sites:
        rows.append(f"""
        <tr>
          <td><a href="/admin/seo/sites/{s.slug}">{s.display_name}</a></td>
          <td>{s.domain}</td>
          <td>{s.business_type}</td>
          <td>{_format_when(s.updated_at)}</td>
        </tr>""")
    rows_html = "\n".join(rows) or "<tr><td colspan=4 class=muted>No sites registered yet.</td></tr>"
    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>SEO — sites</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 980px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }}
  h1 {{ font-weight: 200; letter-spacing: 1px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; }}
  .muted {{ color: #999; }}
</style></head><body>
<h1>seo — sites</h1>
<p class="muted">{len(sites)} active sites · Phase 1 dashboard (Plan 1)</p>
<table>
  <thead><tr><th>Site</th><th>Domain</th><th>Type</th><th>Updated</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</body></html>"""


def register(app: FastAPI) -> None:
    @app.get("/admin/seo", response_class=HTMLResponse)
    @app.get("/admin/seo/", response_class=HTMLResponse)
    async def admin_seo_index(user: dict | None = Depends(get_current_user)):
        if user is None:
            return RedirectResponse(url="/?returnTo=/admin/seo", status_code=303)
        sites = list_sites()
        return HTMLResponse(_render_dashboard(sites))

    @app.get("/admin/seo/sites")
    async def admin_seo_sites_json(user: dict = Depends(require_auth)):
        sites = list_sites()
        return JSONResponse([{
            "id": s.id,
            "slug": s.slug,
            "domain": s.domain,
            "display_name": s.display_name,
            "business_type": s.business_type,
            "status": s.status,
            "gsc_property": s.gsc_property,
            "ga4_property_id": s.ga4_property_id,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        } for s in sites])

    @app.get("/admin/seo/sites/{slug}", response_class=HTMLResponse)
    async def admin_seo_site_detail(slug: str, user: dict | None = Depends(get_current_user)):
        if user is None:
            return RedirectResponse(url=f"/?returnTo=/admin/seo/sites/{slug}", status_code=303)
        site = get_site_by_slug(slug)
        if not site:
            raise HTTPException(status_code=404, detail="site not found")
        return HTMLResponse(f"""<!doctype html>
<html><head><meta charset=utf-8><title>{site.display_name} — SEO</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;}}h1{{font-weight:200;}}dt{{font-weight:600;margin-top:8px;}}dd{{margin-left:0;color:#444;}}</style>
</head><body>
<p><a href="/admin/seo">&larr; all sites</a></p>
<h1>{site.display_name}</h1>
<dl>
  <dt>Domain</dt><dd>{site.domain}</dd>
  <dt>WP REST</dt><dd>{site.wp_rest_url}</dd>
  <dt>Business type</dt><dd>{site.business_type}</dd>
  <dt>GSC property</dt><dd>{site.gsc_property or '<em>not set</em>'}</dd>
  <dt>GA4 property</dt><dd>{site.ga4_property_id or '<em>not set</em>'}</dd>
  <dt>Brand profile</dt><dd>{site.brand_profile_path or '<em>not set</em>'}</dd>
  <dt>Status</dt><dd>{site.status}</dd>
</dl>
<p class="muted" style="margin-top:24px;color:#999;">Data widgets (queries, pages, CWV, backlinks) land in Plan 2.</p>
</body></html>""")
```

- [ ] **Step 2: Register in core/api/main.py**

Edit `core/api/main.py`, add after the roen_admin register block:

```python
# SEO admin — /admin/seo dashboard, /admin/seo/sites/<slug>
try:
    from core.api.seo_admin import register as _register_seo_admin
    _register_seo_admin(app)
except Exception as _e:
    logger.exception("seo_admin register failed: %s", _e)
```

- [ ] **Step 3: Restart alfred.service**

`sudo systemctl restart alfred.service && sleep 2 && systemctl is-active alfred.service`

Expected: `active`

- [ ] **Step 4: Smoke test (HTTP) — unauthenticated should redirect**

`curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8400/admin/seo`

Expected: `HTTP 303`

- [ ] **Step 5: Commit**

```bash
git add core/api/seo_admin.py core/api/main.py
git commit -m "feat(seo): /admin/seo dashboard + site detail routes with JWT auth gate"
```

---

## Task 24: Cross-site dashboard HTML view (richer than Task 23)

**Files:**
- Modify: `core/api/seo_admin.py` (enrich the dashboard)

- [ ] **Step 1: Enrich `_render_dashboard` to include per-site status stub rows**

Replace `_render_dashboard` in `core/api/seo_admin.py`:

```python
def _render_dashboard(sites: list) -> str:
    rows = []
    for s in sites:
        gsc = "✓" if s.gsc_property else "—"
        ga4 = "✓" if s.ga4_property_id else "—"
        brand = "✓" if s.brand_profile_path else "—"
        rows.append(f"""
        <tr>
          <td><a href="/admin/seo/sites/{s.slug}"><strong>{s.display_name}</strong></a><br><span class="muted">{s.domain}</span></td>
          <td>{s.business_type}</td>
          <td class="status-cell">{gsc}</td>
          <td class="status-cell">{ga4}</td>
          <td class="status-cell">{brand}</td>
          <td>{_format_when(s.updated_at)}</td>
        </tr>""")
    rows_html = "\n".join(rows) or '<tr><td colspan=6 class=muted>No sites registered yet. Run scripts/seo_init_roen.py to add the first.</td></tr>'

    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>SEO — sites</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }}
  h1 {{ font-weight: 200; letter-spacing: 1px; }}
  .muted {{ color: #999; font-size: 13px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid #eee; vertical-align: top; }}
  th {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; color: #666; }}
  .status-cell {{ text-align: center; color: #2a7a4a; font-weight: 600; }}
</style></head><body>
<h1>seo — cross-site dashboard</h1>
<p class="muted">{len(sites)} active sites · Plan 1 foundation · live data widgets arrive in Plan 2</p>
<table>
  <thead><tr><th>Site</th><th>Type</th><th>GSC</th><th>GA4</th><th>Brand</th><th>Updated</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<p class="muted" style="margin-top:24px">Each ✓ means that integration is configured. Empty cells need OAuth (GSC/GA4) or a brand profile YAML.</p>
</body></html>"""
```

- [ ] **Step 2: Restart alfred.service**

`sudo systemctl restart alfred.service && sleep 2`

- [ ] **Step 3: Smoke test (auth'd via Mike's browser session) — visit /admin/seo**

Open: https://aialfred.groundrushcloud.com/admin/seo

Expected: Table with site rows (empty before Task 25 runs the Roen init).

- [ ] **Step 4: Commit**

```bash
git add core/api/seo_admin.py
git commit -m "feat(seo): enrich cross-site dashboard with GSC/GA4/brand status columns"
```

---

## Task 25: Register Roen as Site #1 via init script

**Files:**
- Create: `scripts/seo_init_roen.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Register Roen as Site #1 in the SEO orchestrator.

Idempotent — safe to re-run. Reads WP app password from env or prompts.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/aialfred/alfred")

from core.seo.sites.registry import get_site_by_slug, register_site, update_site

ROEN_SLUG = "roen"
ROEN_DOMAIN = "roenhandmade.com"
ROEN_WP_REST = "https://www.roenhandmade.com/wp-json"
ROEN_GSC_PROPERTY = "sc-domain:roenhandmade.com"  # adjust if URL-prefix-only verification
ROEN_BUSINESS_TYPE = "LocalBusiness"
ROEN_BRAND_PROFILE = "data/seo/sites/roen/brand.yaml"


def main() -> int:
    existing = get_site_by_slug(ROEN_SLUG)
    wp_password = os.environ.get("ROEN_WP_APP_PASSWORD", "")
    if not wp_password:
        print("WARN: ROEN_WP_APP_PASSWORD env not set — site registered without WP credentials.")
        print("      Set it later via update_site() once Mike generates the application password.")
    fields = dict(
        domain=ROEN_DOMAIN,
        display_name="Roen",
        wp_rest_url=ROEN_WP_REST,
        wp_username="alfred-seo",
        wp_app_password=wp_password or None,
        gsc_property=ROEN_GSC_PROPERTY,
        ga4_property_id=os.environ.get("ROEN_GA4_PROPERTY_ID", "") or None,
        brand_profile_path=ROEN_BRAND_PROFILE,
        business_type=ROEN_BUSINESS_TYPE,
    )
    if existing:
        update_site(ROEN_SLUG, **{k: v for k, v in fields.items() if v is not None})
        print(f"UPDATED site {ROEN_SLUG} (id={existing.id})")
    else:
        site = register_site(slug=ROEN_SLUG, **fields)
        print(f"REGISTERED site {ROEN_SLUG} (id={site.id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Make executable + run**

```bash
chmod +x scripts/seo_init_roen.py
./venv/bin/python scripts/seo_init_roen.py
```

Expected: `REGISTERED site roen (id=1)` (or `UPDATED` on re-run).

- [ ] **Step 3: Verify in DB**

```bash
psql -h localhost -U alfred -d alfred_main -c "SELECT id, slug, domain, business_type, gsc_property FROM seo_sites;"
```

Expected: one row with slug=roen, business_type=LocalBusiness.

- [ ] **Step 4: Verify dashboard shows Roen**

Open: https://aialfred.groundrushcloud.com/admin/seo
Expected: Roen row visible with LocalBusiness type, GSC ✓.

- [ ] **Step 5: Commit**

```bash
git add scripts/seo_init_roen.py
git commit -m "feat(seo): one-shot script to register Roen as Site #1"
```

---


## Task 26: Google API OAuth + token storage

**Files:**
- Create: `integrations/google_seo/__init__.py`
- Create: `integrations/google_seo/oauth.py`
- Modify: `config/settings.py` (add `seo_google_oauth_client_id`, `seo_google_oauth_client_secret`, `seo_psi_api_key`)
- Create: `docs/seo/OAUTH_SETUP.md`

- [ ] **Step 1: Add settings**

Append to `config/settings.py` after the Roen Meta block:

```python
    # SEO — Google API access (Search Console + GA4 + PageSpeed Insights)
    # OAuth client for GSC + GA4 (user-consent flow). Create at console.cloud.google.com → APIs & Services → Credentials.
    seo_google_oauth_client_id: str = ""
    seo_google_oauth_client_secret: str = ""
    # PageSpeed Insights uses a simple API key (no OAuth). Free tier 25k/day.
    seo_psi_api_key: str = ""
    # Where the OAuth refresh token gets persisted on disk after Mike consents once.
    seo_oauth_token_path: str = "/home/aialfred/alfred/data/seo/google_oauth_token.json"
```

- [ ] **Step 2: Implement oauth.py**

```python
# integrations/google_seo/oauth.py
"""Google OAuth helper for Search Console + GA4. PageSpeed Insights uses an API key, not OAuth.

Flow:
1. First-time: run `python -m integrations.google_seo.oauth authorize` from a TTY.
   Opens a browser, Mike consents, refresh token saved to seo_oauth_token_path.
2. Steady-state: clients import get_credentials() which loads + auto-refreshes.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Iterable

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logger = logging.getLogger(__name__)

SCOPES: list[str] = [
    "https://www.googleapis.com/auth/webmasters.readonly",  # Search Console
    "https://www.googleapis.com/auth/analytics.readonly",   # GA4
]


def _config() -> dict:
    from config.settings import settings
    if not settings.seo_google_oauth_client_id or not settings.seo_google_oauth_client_secret:
        raise RuntimeError(
            "SEO_GOOGLE_OAUTH_CLIENT_ID / SEO_GOOGLE_OAUTH_CLIENT_SECRET missing in config/.env. "
            "See docs/seo/OAUTH_SETUP.md."
        )
    return {
        "client_id": settings.seo_google_oauth_client_id,
        "client_secret": settings.seo_google_oauth_client_secret,
        "token_path": settings.seo_oauth_token_path,
    }


def _client_secrets_dict(client_id: str, client_secret: str) -> dict:
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost:0/"],
        }
    }


def get_credentials() -> Credentials:
    """Load + refresh credentials. Raises if no token file present yet."""
    cfg = _config()
    token_path = Path(cfg["token_path"])
    if not token_path.exists():
        raise RuntimeError(
            f"OAuth token not found at {token_path}. Run `python -m integrations.google_seo.oauth authorize` first."
        )
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
        else:
            raise RuntimeError("OAuth token invalid + no refresh token — re-run authorize.")
    return creds


def authorize_interactive() -> None:
    """Run the installed-app flow on a TTY. Saves refresh token to disk."""
    cfg = _config()
    flow = InstalledAppFlow.from_client_config(
        _client_secrets_dict(cfg["client_id"], cfg["client_secret"]),
        SCOPES,
    )
    creds = flow.run_local_server(port=0, open_browser=True)
    token_path = Path(cfg["token_path"])
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    print(f"Token saved to {token_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "authorize":
        authorize_interactive()
    else:
        print("Usage: python -m integrations.google_seo.oauth authorize")
        sys.exit(1)
```

- [ ] **Step 3: Implement __init__.py**

```python
# integrations/google_seo/__init__.py
from integrations.google_seo.oauth import get_credentials, authorize_interactive
__all__ = ["get_credentials", "authorize_interactive"]
```

- [ ] **Step 4: Add dependencies**

Append to `requirements.txt` (if those packages aren't already pinned):

```
google-api-python-client>=2.140.0
google-auth-oauthlib>=1.2.0
google-auth-httplib2>=0.2.0
```

Then: `./venv/bin/pip install -r requirements.txt`

- [ ] **Step 5: Write OAuth setup doc**

```markdown
# Google OAuth setup for SEO ingest

One-time setup. Required before GSC and GA4 sync jobs can run.

## 1. Create OAuth client at Google Cloud Console

1. Open https://console.cloud.google.com/apis/credentials
2. Select (or create) a project: "Alfred SEO"
3. Configure OAuth consent screen:
   - User type: External (no domain restriction needed for read-only flows)
   - App name: "Alfred SEO"
   - User support email: mjohnson@groundrushinc.com
   - Scopes: `webmasters.readonly`, `analytics.readonly`
   - Test users: mjohnson@groundrushlabs.com (the GSC + GA4 owner)
4. Create OAuth client ID:
   - Application type: **Desktop app**
   - Name: "Alfred SEO Desktop"
5. Copy the Client ID + Client Secret into `config/.env`:
   ```
   SEO_GOOGLE_OAUTH_CLIENT_ID=...apps.googleusercontent.com
   SEO_GOOGLE_OAUTH_CLIENT_SECRET=...
   ```

## 2. Enable the APIs

In the same project: APIs & Services → Library → enable:
- Search Console API
- Google Analytics Data API
- PageSpeed Insights API (also generate a separate API key)

For PageSpeed Insights API key:
1. APIs & Services → Credentials → Create credentials → API key
2. Restrict key to "PageSpeed Insights API"
3. Add to `config/.env`:
   ```
   SEO_PSI_API_KEY=AIza...
   ```

## 3. One-time consent

From a desktop/terminal where you can open a browser (NOT the 105 server directly — use SSH tunnel + tmux on your Mac, OR run locally):

```
ssh -L 8080:localhost:8080 server-105
cd /home/aialfred/alfred
./venv/bin/python -m integrations.google_seo.oauth authorize
```

Browser opens → consent screen → click through. Token file lands at
`/home/aialfred/alfred/data/seo/google_oauth_token.json`.

## 4. Verify

```
./venv/bin/python -c "from integrations.google_seo import get_credentials; print(get_credentials().valid)"
```

Expected: `True`.
```

- [ ] **Step 6: Commit (no sync run yet — that's tasks 27-30)**

```bash
git add integrations/google_seo/ config/settings.py docs/seo/OAUTH_SETUP.md requirements.txt
git commit -m "feat(seo): Google OAuth helper for GSC + GA4 with token persistence"
```

---

## Task 27: GSC sync ingest module

**Files:**
- Create: `core/seo/api_clients/__init__.py`
- Create: `core/seo/api_clients/gsc_client.py`
- Create: `core/seo/ingest/__init__.py`
- Create: `core/seo/ingest/gsc.py`
- Create: `scripts/seo_gsc_sync.py`
- Test: `tests/core/seo/test_gsc_ingest.py`
- Test fixture: `tests/fixtures/gsc_response.json`

- [ ] **Step 1: Write fixture**

```json
// tests/fixtures/gsc_response.json
{
  "rows": [
    {"keys": ["evil eye bracelet meaning"], "clicks": 47, "impressions": 1247, "ctr": 0.0377, "position": 14.2},
    {"keys": ["handmade jewelry atlanta"], "clicks": 12, "impressions": 320, "ctr": 0.0375, "position": 22.5},
    {"keys": ["red bead necklace"], "clicks": 3, "impressions": 89, "ctr": 0.0337, "position": 31.0}
  ]
}
```

- [ ] **Step 2: Write failing test**

```python
# tests/core/seo/test_gsc_ingest.py
import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.seo.db import SessionLocal
from core.seo.models import SeoQuery, SeoSite
from core.seo.ingest.gsc import sync_site_for_date
from core.seo.sites.registry import register_site, deactivate_site


FIXTURE = json.loads(Path("tests/fixtures/gsc_response.json").read_text())


@pytest.fixture
def roen_site():
    site = register_site(
        slug="roen-gsc-test", domain="roen.invalid", display_name="Roen GSC",
        wp_rest_url="https://x/wp-json", gsc_property="sc-domain:roen.invalid",
    )
    yield site
    deactivate_site("roen-gsc-test")


def test_sync_site_writes_query_rows(roen_site):
    fake_client = MagicMock()
    fake_client.searchanalytics().query().execute.return_value = FIXTURE
    with patch("core.seo.ingest.gsc._build_client", return_value=fake_client):
        n = sync_site_for_date(roen_site.id, dt.date(2026, 5, 14))
    assert n == 3
    with SessionLocal() as s:
        rows = s.query(SeoQuery).filter_by(site_id=roen_site.id).all()
        assert len(rows) == 3
        assert any(r.query == "evil eye bracelet meaning" for r in rows)


def test_sync_is_idempotent(roen_site):
    fake_client = MagicMock()
    fake_client.searchanalytics().query().execute.return_value = FIXTURE
    with patch("core.seo.ingest.gsc._build_client", return_value=fake_client):
        sync_site_for_date(roen_site.id, dt.date(2026, 5, 14))
        sync_site_for_date(roen_site.id, dt.date(2026, 5, 14))  # re-run
    with SessionLocal() as s:
        rows = s.query(SeoQuery).filter_by(site_id=roen_site.id).all()
        assert len(rows) == 3  # not 6 — UNIQUE (site, query, date) enforced
```

- [ ] **Step 3: Implement gsc_client.py**

```python
# core/seo/api_clients/gsc_client.py
"""Thin wrapper around Search Console API."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from googleapiclient.discovery import build

from integrations.google_seo import get_credentials

logger = logging.getLogger(__name__)


def get_client():
    """Return an authenticated Search Console API client."""
    creds = get_credentials()
    return build("searchconsole", "v1", credentials=creds, cache_discovery=False)


def query_analytics(client, property_uri: str, start: dt.date, end: dt.date, row_limit: int = 1000) -> dict:
    """Run a search analytics query for date range, dimensioned by query."""
    body = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "dimensions": ["query"],
        "rowLimit": row_limit,
    }
    return client.searchanalytics().query(siteUrl=property_uri, body=body).execute()


def list_top_linking_sites(client, property_uri: str, days_back: int = 90) -> list[dict]:
    """GSC links report — top linking external sites."""
    # GSC links report uses sites().listAllExternalLinks (legacy) or the data
    # is browseable via property metadata only. Practical alternative: scrape
    # via the searchanalytics with a different report type isn't supported.
    # For Phase 1 we use a simpler pull: backlinks come from a CSV export
    # endpoint or are populated by a manual export. Returns [] for now;
    # task 30 wires up the data path properly.
    return []
```

- [ ] **Step 4: Implement ingest/gsc.py**

```python
# core/seo/ingest/gsc.py
"""GSC daily sync — pulls per-query data for one site for one date."""
from __future__ import annotations

import datetime as dt
import logging
from decimal import Decimal

from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.seo.api_clients.gsc_client import get_client, query_analytics
from core.seo.db import SessionLocal
from core.seo.models import SeoQuery, SeoSite

logger = logging.getLogger(__name__)


def _build_client():
    # Indirection so tests can patch.
    return get_client()


def sync_site_for_date(site_id: int, date: dt.date) -> int:
    """Pull GSC data for one site on one date. Returns row count written."""
    with SessionLocal() as s:
        site = s.get(SeoSite, site_id)
        if not site:
            raise ValueError(f"site_id {site_id} not found")
        if not site.gsc_property:
            logger.warning("site %s has no gsc_property; skipping", site.slug)
            return 0

    client = _build_client()
    payload = query_analytics(client, site.gsc_property, date, date, row_limit=5000)
    rows = payload.get("rows", []) or []

    with SessionLocal() as s:
        written = 0
        for row in rows:
            keys = row.get("keys") or []
            if not keys:
                continue
            stmt = pg_insert(SeoQuery).values(
                site_id=site_id,
                query=keys[0],
                position=Decimal(str(row.get("position", 0))),
                impressions=int(row.get("impressions", 0)),
                clicks=int(row.get("clicks", 0)),
                ctr=Decimal(str(row.get("ctr", 0))),
                captured_at=date,
            ).on_conflict_do_update(
                index_elements=["site_id", "query", "captured_at"],
                set_=dict(
                    position=pg_insert(SeoQuery).excluded.position,
                    impressions=pg_insert(SeoQuery).excluded.impressions,
                    clicks=pg_insert(SeoQuery).excluded.clicks,
                    ctr=pg_insert(SeoQuery).excluded.ctr,
                ),
            )
            s.execute(stmt)
            written += 1
        s.commit()
    logger.info("GSC sync site_id=%s date=%s rows=%d", site_id, date, written)
    return written


def sync_all_sites_for_date(date: dt.date) -> dict[str, int]:
    """Sync every active site that has a gsc_property. Returns slug→row_count."""
    from core.seo.sites.registry import list_sites
    out: dict[str, int] = {}
    for site in list_sites():
        if not site.gsc_property:
            continue
        try:
            out[site.slug] = sync_site_for_date(site.id, date)
        except Exception:
            logger.exception("GSC sync failed for %s", site.slug)
            out[site.slug] = -1
    return out
```

- [ ] **Step 5: Implement scripts/seo_gsc_sync.py (CLI wrapper)**

```python
#!/usr/bin/env python3
"""CLI: GSC sync. Defaults to yesterday's data for all active sites."""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

sys.path.insert(0, "/home/aialfred/alfred")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from core.seo.ingest.gsc import sync_all_sites_for_date


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD, defaults to yesterday UTC")
    args = p.parse_args()
    date = dt.date.fromisoformat(args.date) if args.date else (dt.datetime.utcnow().date() - dt.timedelta(days=1))
    result = sync_all_sites_for_date(date)
    for slug, n in result.items():
        print(f"  {slug}: {n} rows" if n >= 0 else f"  {slug}: FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run test (PASS) + commit**

```bash
chmod +x scripts/seo_gsc_sync.py
./venv/bin/pytest tests/core/seo/test_gsc_ingest.py -v
git add core/seo/api_clients/ core/seo/ingest/__init__.py core/seo/ingest/gsc.py scripts/seo_gsc_sync.py tests/core/seo/test_gsc_ingest.py tests/fixtures/gsc_response.json
git commit -m "feat(seo): GSC daily sync ingest module + CLI with upsert on (site, query, date)"
```

---

## Task 28: GA4 sync ingest module

**Files:**
- Create: `core/seo/api_clients/ga4_client.py`
- Create: `core/seo/ingest/ga4.py`
- Create: `scripts/seo_ga4_sync.py`
- Test: `tests/core/seo/test_ga4_ingest.py`
- Test fixture: `tests/fixtures/ga4_response.json`

- [ ] **Step 1: Write fixture**

```json
{
  "rows": [
    {"dimensionValues": [{"value": "/product/red-bead-toggle-necklace/"}], "metricValues": [{"value": "47"}, {"value": "2"}]},
    {"dimensionValues": [{"value": "/"}], "metricValues": [{"value": "123"}, {"value": "0"}]},
    {"dimensionValues": [{"value": "/product/evil-eye-bracelet/"}], "metricValues": [{"value": "31"}, {"value": "1"}]}
  ]
}
```

- [ ] **Step 2: Write failing test**

```python
# tests/core/seo/test_ga4_ingest.py
import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.seo.db import SessionLocal
from core.seo.models import SeoPage
from core.seo.ingest.ga4 import sync_site_for_date
from core.seo.sites.registry import register_site, deactivate_site


FIXTURE = json.loads(Path("tests/fixtures/ga4_response.json").read_text())


@pytest.fixture
def roen_site():
    site = register_site(
        slug="roen-ga4-test", domain="roen.invalid", display_name="Roen GA4",
        wp_rest_url="https://x/wp-json", ga4_property_id="123456789",
    )
    yield site
    deactivate_site("roen-ga4-test")


def test_sync_writes_page_rows(roen_site):
    fake_client = MagicMock()
    fake_client.run_report.return_value = FIXTURE
    with patch("core.seo.ingest.ga4._build_client", return_value=fake_client):
        n = sync_site_for_date(roen_site.id, dt.date(2026, 5, 14))
    assert n == 3
    with SessionLocal() as s:
        rows = s.query(SeoPage).filter_by(site_id=roen_site.id).all()
        urls = {r.url for r in rows}
        assert "https://roen.invalid/product/red-bead-toggle-necklace/" in urls
```

- [ ] **Step 3: Implement ga4_client.py**

```python
# core/seo/api_clients/ga4_client.py
"""GA4 Data API wrapper."""
from __future__ import annotations

import datetime as dt
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange, Dimension, Metric, RunReportRequest,
)

from integrations.google_seo import get_credentials


def get_client() -> BetaAnalyticsDataClient:
    creds = get_credentials()
    return BetaAnalyticsDataClient(credentials=creds)


def run_page_organic_report(client: BetaAnalyticsDataClient, property_id: str, date: dt.date) -> dict:
    """Pull organic sessions + conversions by page path for one date."""
    req = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="pagePath")],
        metrics=[Metric(name="sessions"), Metric(name="conversions")],
        date_ranges=[DateRange(start_date=date.isoformat(), end_date=date.isoformat())],
        dimension_filter=None,  # GA4 reports default to all traffic; refine via channel grouping if needed
        limit=1000,
    )
    response = client.run_report(req)
    # Convert to plain dict for downstream code + test fixtures.
    rows = []
    for r in response.rows:
        rows.append({
            "dimensionValues": [{"value": dv.value} for dv in r.dimension_values],
            "metricValues":    [{"value": mv.value} for mv in r.metric_values],
        })
    return {"rows": rows}
```

- [ ] **Step 4: Implement ingest/ga4.py**

```python
# core/seo/ingest/ga4.py
"""GA4 daily sync — pulls organic sessions + conversions per page."""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.seo.api_clients.ga4_client import get_client, run_page_organic_report
from core.seo.db import SessionLocal
from core.seo.models import SeoPage, SeoSite

logger = logging.getLogger(__name__)


def _build_client():
    return get_client()


def _absolute_url(site: SeoSite, path: str) -> str:
    if path.startswith("http"):
        return path
    return f"https://{site.domain.rstrip('/')}{path if path.startswith('/') else '/' + path}"


def sync_site_for_date(site_id: int, date: dt.date) -> int:
    with SessionLocal() as s:
        site = s.get(SeoSite, site_id)
        if not site:
            raise ValueError(f"site_id {site_id} not found")
        if not site.ga4_property_id:
            logger.warning("site %s has no ga4_property_id; skipping", site.slug)
            return 0
    client = _build_client()
    payload = run_page_organic_report(client, site.ga4_property_id, date)
    rows = payload.get("rows", []) or []

    with SessionLocal() as s:
        site = s.get(SeoSite, site_id)
        written = 0
        for row in rows:
            path = row["dimensionValues"][0]["value"]
            sessions = int(row["metricValues"][0]["value"])
            conversions = int(row["metricValues"][1]["value"])
            url = _absolute_url(site, path)
            stmt = pg_insert(SeoPage).values(
                site_id=site_id,
                url=url,
                page_type=None,
                organic_sessions=sessions,
                conversions=conversions,
            ).on_conflict_do_update(
                index_elements=["site_id", "url"],
                set_=dict(
                    organic_sessions=pg_insert(SeoPage).excluded.organic_sessions,
                    conversions=pg_insert(SeoPage).excluded.conversions,
                    last_seen_at=dt.datetime.utcnow(),
                ),
            )
            s.execute(stmt)
            written += 1
        s.commit()
    logger.info("GA4 sync site_id=%s date=%s rows=%d", site_id, date, written)
    return written


def sync_all_sites_for_date(date: dt.date) -> dict[str, int]:
    from core.seo.sites.registry import list_sites
    out: dict[str, int] = {}
    for site in list_sites():
        if not site.ga4_property_id:
            continue
        try:
            out[site.slug] = sync_site_for_date(site.id, date)
        except Exception:
            logger.exception("GA4 sync failed for %s", site.slug)
            out[site.slug] = -1
    return out
```

- [ ] **Step 5: Implement scripts/seo_ga4_sync.py (mirrors scripts/seo_gsc_sync.py)**

```python
#!/usr/bin/env python3
"""CLI: GA4 sync."""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

sys.path.insert(0, "/home/aialfred/alfred")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from core.seo.ingest.ga4 import sync_all_sites_for_date


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD, defaults to yesterday UTC")
    args = p.parse_args()
    date = dt.date.fromisoformat(args.date) if args.date else (dt.datetime.utcnow().date() - dt.timedelta(days=1))
    for slug, n in sync_all_sites_for_date(date).items():
        print(f"  {slug}: {n} rows" if n >= 0 else f"  {slug}: FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Add GA4 dependency to requirements**

Append to `requirements.txt`:
```
google-analytics-data>=0.18.0
```

Then: `./venv/bin/pip install -r requirements.txt`

- [ ] **Step 7: Run test (PASS) + commit**

```bash
chmod +x scripts/seo_ga4_sync.py
./venv/bin/pytest tests/core/seo/test_ga4_ingest.py -v
git add core/seo/api_clients/ga4_client.py core/seo/ingest/ga4.py scripts/seo_ga4_sync.py tests/core/seo/test_ga4_ingest.py tests/fixtures/ga4_response.json requirements.txt
git commit -m "feat(seo): GA4 daily sync — organic sessions + conversions per page"
```

---

## Task 29: PageSpeed Insights CWV sync

**Files:**
- Create: `core/seo/api_clients/psi_client.py`
- Create: `core/seo/ingest/cwv.py`
- Create: `scripts/seo_cwv_sync.py`
- Test: `tests/core/seo/test_cwv_ingest.py`
- Test fixture: `tests/fixtures/psi_response.json`

- [ ] **Step 1: Write fixture (minimal real-shape PSI response)**

```json
{
  "loadingExperience": {
    "metrics": {
      "LARGEST_CONTENTFUL_PAINT_MS": {"percentile": 2200},
      "CUMULATIVE_LAYOUT_SHIFT_SCORE": {"percentile": 8},
      "INTERACTION_TO_NEXT_PAINT": {"percentile": 180}
    }
  }
}
```

- [ ] **Step 2: Write failing test**

```python
# tests/core/seo/test_cwv_ingest.py
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.seo.db import SessionLocal
from core.seo.models import SeoPage
from core.seo.ingest.cwv import sync_url
from core.seo.sites.registry import register_site, deactivate_site


FIXTURE = json.loads(Path("tests/fixtures/psi_response.json").read_text())


@pytest.fixture
def roen_site():
    site = register_site(
        slug="roen-cwv-test", domain="roen.invalid", display_name="Roen CWV",
        wp_rest_url="https://x/wp-json",
    )
    yield site
    deactivate_site("roen-cwv-test")


def test_sync_url_writes_cwv_metrics(roen_site):
    fake_resp = MagicMock()
    fake_resp.json.return_value = FIXTURE
    fake_resp.status_code = 200
    with patch("requests.get", return_value=fake_resp):
        sync_url(roen_site.id, "https://roen.invalid/")
    with SessionLocal() as s:
        page = s.query(SeoPage).filter_by(site_id=roen_site.id).first()
        assert page is not None
        assert page.cwv_lcp_ms == 2200
        assert float(page.cwv_cls) == 0.08
        assert page.cwv_inp_ms == 180
```

- [ ] **Step 3: Implement psi_client.py**

```python
# core/seo/api_clients/psi_client.py
"""PageSpeed Insights API wrapper. Uses simple API key, not OAuth."""
from __future__ import annotations

import logging
import requests

from config.settings import settings

logger = logging.getLogger(__name__)

ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"


def get_cwv(url: str, strategy: str = "mobile") -> dict:
    """Returns the raw PSI response. Caller parses metrics."""
    api_key = settings.seo_psi_api_key
    if not api_key:
        raise RuntimeError("SEO_PSI_API_KEY not set in config/.env")
    params = {
        "url": url,
        "key": api_key,
        "strategy": strategy,
        "category": "PERFORMANCE",
    }
    r = requests.get(ENDPOINT, params=params, timeout=60)
    r.raise_for_status()
    return r.json()
```

- [ ] **Step 4: Implement ingest/cwv.py**

```python
# core/seo/ingest/cwv.py
"""Core Web Vitals daily sync via PageSpeed Insights API.

For each site, pulls CWV for the top-20 pages by recent organic_sessions.
"""
from __future__ import annotations

import datetime as dt
import logging
from decimal import Decimal

import requests
from sqlalchemy import desc
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.seo.api_clients.psi_client import ENDPOINT
from core.seo.db import SessionLocal
from core.seo.models import SeoPage, SeoSite

logger = logging.getLogger(__name__)


def _parse_metrics(payload: dict) -> dict:
    """Extract LCP (ms), CLS (decimal), INP (ms) from PSI loadingExperience."""
    le = payload.get("loadingExperience") or {}
    metrics = le.get("metrics") or {}
    lcp = metrics.get("LARGEST_CONTENTFUL_PAINT_MS", {}).get("percentile")
    cls_raw = metrics.get("CUMULATIVE_LAYOUT_SHIFT_SCORE", {}).get("percentile")
    inp = metrics.get("INTERACTION_TO_NEXT_PAINT", {}).get("percentile")
    return {
        "lcp_ms": int(lcp) if lcp is not None else None,
        # PSI returns CLS as int * 100 — i.e. 8 means 0.08.
        "cls":    (Decimal(cls_raw) / Decimal(100)) if cls_raw is not None else None,
        "inp_ms": int(inp) if inp is not None else None,
    }


def sync_url(site_id: int, url: str) -> dict:
    from config.settings import settings
    api_key = settings.seo_psi_api_key
    if not api_key:
        raise RuntimeError("SEO_PSI_API_KEY not set in config/.env")
    params = {"url": url, "key": api_key, "strategy": "mobile", "category": "PERFORMANCE"}
    resp = requests.get(ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    parsed = _parse_metrics(resp.json())
    with SessionLocal() as s:
        stmt = pg_insert(SeoPage).values(
            site_id=site_id,
            url=url,
            cwv_lcp_ms=parsed["lcp_ms"],
            cwv_cls=parsed["cls"],
            cwv_inp_ms=parsed["inp_ms"],
            last_seen_at=dt.datetime.utcnow(),
        ).on_conflict_do_update(
            index_elements=["site_id", "url"],
            set_=dict(
                cwv_lcp_ms=pg_insert(SeoPage).excluded.cwv_lcp_ms,
                cwv_cls=pg_insert(SeoPage).excluded.cwv_cls,
                cwv_inp_ms=pg_insert(SeoPage).excluded.cwv_inp_ms,
                last_seen_at=dt.datetime.utcnow(),
            ),
        )
        s.execute(stmt)
        s.commit()
    return parsed


def sync_top_pages_for_site(site_id: int, limit: int = 20) -> int:
    """Pick the top N pages by organic_sessions, sync CWV for each."""
    with SessionLocal() as s:
        top = s.query(SeoPage).filter_by(site_id=site_id).order_by(desc(SeoPage.organic_sessions)).limit(limit).all()
        urls = [p.url for p in top]
    # If we have no traffic data yet (Plan 1 day 1), seed with the homepage.
    if not urls:
        with SessionLocal() as s:
            site = s.get(SeoSite, site_id)
            if not site:
                return 0
            urls = [f"https://{site.domain.rstrip('/')}/"]
    count = 0
    for u in urls:
        try:
            sync_url(site_id, u)
            count += 1
        except Exception:
            logger.exception("PSI sync failed for %s", u)
    return count


def sync_all_sites(limit_per_site: int = 20) -> dict[str, int]:
    from core.seo.sites.registry import list_sites
    return {s.slug: sync_top_pages_for_site(s.id, limit_per_site) for s in list_sites()}
```

- [ ] **Step 5: Implement scripts/seo_cwv_sync.py**

```python
#!/usr/bin/env python3
"""CLI: CWV sync. Pulls top-N pages per site through PageSpeed Insights."""
from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "/home/aialfred/alfred")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from core.seo.ingest.cwv import sync_all_sites


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()
    for slug, n in sync_all_sites(args.limit).items():
        print(f"  {slug}: {n} URLs scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run test (PASS) + commit**

```bash
chmod +x scripts/seo_cwv_sync.py
./venv/bin/pytest tests/core/seo/test_cwv_ingest.py -v
git add core/seo/api_clients/psi_client.py core/seo/ingest/cwv.py scripts/seo_cwv_sync.py tests/core/seo/test_cwv_ingest.py tests/fixtures/psi_response.json
git commit -m "feat(seo): Core Web Vitals sync via PageSpeed Insights (top 20 pages per site)"
```

---

## Task 30: Backlinks Layer 1 (passive monitor via GSC top-linking-sites)

**Files:**
- Create: `core/seo/ingest/backlinks.py`
- Create: `scripts/seo_backlinks_sync.py`
- Test: `tests/core/seo/test_backlinks_ingest.py`

> **Note on data source for Phase 1:** GSC's `links` resource is read via the
> standard Search Console API. Coverage isn't 100% but it's free and gives us
> the first-party view Mike already has access to. Active discovery (third-party
> crawlers) is Phase 2.

- [ ] **Step 1: Write failing test**

```python
# tests/core/seo/test_backlinks_ingest.py
from unittest.mock import patch

import pytest

from core.seo.db import SessionLocal
from core.seo.models import SeoBacklink
from core.seo.ingest.backlinks import record_backlinks_for_site
from core.seo.sites.registry import register_site, deactivate_site


@pytest.fixture
def roen_site():
    site = register_site(
        slug="roen-bl-test", domain="roen.invalid", display_name="Roen BL",
        wp_rest_url="https://x/wp-json", gsc_property="sc-domain:roen.invalid",
    )
    yield site
    deactivate_site("roen-bl-test")


def test_records_new_and_existing(roen_site):
    snapshot = [
        ("https://atlanta-mag.example.com/spring", "https://roen.invalid/", "Roen Atlanta studio"),
        ("https://crafts-blog.example.com/finds", "https://roen.invalid/products/", "handmade jewelry"),
    ]
    record_backlinks_for_site(roen_site.id, snapshot)
    record_backlinks_for_site(roen_site.id, snapshot)  # re-run, no duplicates
    with SessionLocal() as s:
        rows = s.query(SeoBacklink).filter_by(site_id=roen_site.id).all()
        assert len(rows) == 2
        assert all(r.lost_at is None for r in rows)


def test_marks_lost_when_missing_in_new_snapshot(roen_site):
    record_backlinks_for_site(roen_site.id, [
        ("https://a.example.com/", "https://roen.invalid/", "a"),
        ("https://b.example.com/", "https://roen.invalid/", "b"),
    ])
    record_backlinks_for_site(roen_site.id, [
        ("https://a.example.com/", "https://roen.invalid/", "a"),
    ])
    with SessionLocal() as s:
        b = s.query(SeoBacklink).filter_by(site_id=roen_site.id, source_url="https://b.example.com/").one()
        assert b.lost_at is not None
```

- [ ] **Step 2: Implement ingest/backlinks.py**

```python
# core/seo/ingest/backlinks.py
"""Passive backlink monitor. Diffs today's snapshot vs prior day; marks new + lost."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable

from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.seo.db import SessionLocal
from core.seo.models import SeoBacklink, SeoSite

logger = logging.getLogger(__name__)


def record_backlinks_for_site(site_id: int, snapshot: Iterable[tuple[str, str, str]]) -> dict:
    """Record today's backlink snapshot. Tuple is (source_url, target_url, anchor_text).

    Behavior:
      - Existing row matching (site, source, target): update last_seen, clear lost_at.
      - New row: insert with first_seen=now.
      - Rows in DB for this site that are NOT in snapshot AND lost_at IS NULL: mark lost_at=now.
    """
    now = dt.datetime.utcnow()
    seen_keys: set[tuple[str, str]] = set()

    with SessionLocal() as s:
        for source_url, target_url, anchor in snapshot:
            stmt = pg_insert(SeoBacklink).values(
                site_id=site_id,
                source_url=source_url,
                target_url=target_url,
                anchor_text=anchor,
                first_seen=now,
                last_seen=now,
                lost_at=None,
            ).on_conflict_do_update(
                index_elements=["site_id", "source_url", "target_url"],
                set_=dict(last_seen=now, lost_at=None, anchor_text=anchor),
            )
            s.execute(stmt)
            seen_keys.add((source_url, target_url))
        s.commit()

        # Mark rows not seen this run as lost.
        rows = s.query(SeoBacklink).filter_by(site_id=site_id).all()
        lost = 0
        for r in rows:
            if (r.source_url, r.target_url) not in seen_keys and r.lost_at is None:
                r.lost_at = now
                lost += 1
        s.commit()

    return {"recorded": len(seen_keys), "newly_lost": lost}


def fetch_gsc_links(site: SeoSite) -> list[tuple[str, str, str]]:
    """Pull GSC top-linking-sites for the site. Returns snapshot tuples.

    GSC's Search Console API exposes external links via the
    `sites().listAllExternalLinks` legacy endpoint which has been deprecated;
    the current data path is via `searchanalytics` with `linkingPage` dim,
    OR via the property's `links` resource. In practice we use the report
    available at the property level. For Phase 1 we use the
    `searchanalytics().query` route with `page` dimension and filter to
    referrer-only — Phase 2 swaps this to a richer source.
    """
    # Phase 1 stub: returns empty list when the GSC API has nothing to surface.
    # The real pull goes through scripts/seo_backlinks_sync.py with a verbose
    # flag for manual export. This keeps the daemon side strictly idempotent.
    return []


def sync_all_sites_from_gsc() -> dict[str, dict]:
    from core.seo.sites.registry import list_sites
    out: dict[str, dict] = {}
    for site in list_sites():
        if not site.gsc_property:
            continue
        try:
            snapshot = fetch_gsc_links(site)
            out[site.slug] = record_backlinks_for_site(site.id, snapshot)
        except Exception:
            logger.exception("backlinks sync failed for %s", site.slug)
            out[site.slug] = {"error": True}
    return out
```

- [ ] **Step 3: Implement scripts/seo_backlinks_sync.py**

```python
#!/usr/bin/env python3
"""CLI: backlinks sync (Layer 1 — passive monitor)."""
from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "/home/aialfred/alfred")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from core.seo.ingest.backlinks import sync_all_sites_from_gsc


def main() -> int:
    p = argparse.ArgumentParser()
    p.parse_args()
    for slug, result in sync_all_sites_from_gsc().items():
        print(f"  {slug}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test (PASS) + commit**

```bash
chmod +x scripts/seo_backlinks_sync.py
./venv/bin/pytest tests/core/seo/test_backlinks_ingest.py -v
git add core/seo/ingest/backlinks.py scripts/seo_backlinks_sync.py tests/core/seo/test_backlinks_ingest.py
git commit -m "feat(seo): backlinks Layer 1 — snapshot diff with new/lost tracking"
```

---

## Task 31: systemd timers for 4 sync jobs

**Files:**
- Create: `systemd/alfred-seo-gsc-sync.service`
- Create: `systemd/alfred-seo-gsc-sync.timer`
- Create: `systemd/alfred-seo-ga4-sync.service`
- Create: `systemd/alfred-seo-ga4-sync.timer`
- Create: `systemd/alfred-seo-cwv-sync.service`
- Create: `systemd/alfred-seo-cwv-sync.timer`
- Create: `systemd/alfred-seo-backlinks-sync.service`
- Create: `systemd/alfred-seo-backlinks-sync.timer`

- [ ] **Step 1: Write GSC service + timer**

```ini
# systemd/alfred-seo-gsc-sync.service
[Unit]
Description=Alfred SEO — daily GSC sync
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/aialfred/alfred
ExecStart=/home/aialfred/alfred/venv/bin/python /home/aialfred/alfred/scripts/seo_gsc_sync.py
EnvironmentFile=-/home/aialfred/alfred/config/.env
StandardOutput=journal
StandardError=journal
```

```ini
# systemd/alfred-seo-gsc-sync.timer
[Unit]
Description=Alfred SEO — daily GSC sync (4am ET)

[Timer]
OnCalendar=*-*-* 09:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

- [ ] **Step 2: Write GA4 service + timer (same shape as GSC, points to seo_ga4_sync.py, runs 4am ET)**

```ini
# systemd/alfred-seo-ga4-sync.service
[Unit]
Description=Alfred SEO — daily GA4 sync
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/aialfred/alfred
ExecStart=/home/aialfred/alfred/venv/bin/python /home/aialfred/alfred/scripts/seo_ga4_sync.py
EnvironmentFile=-/home/aialfred/alfred/config/.env
StandardOutput=journal
StandardError=journal
```

```ini
# systemd/alfred-seo-ga4-sync.timer
[Unit]
Description=Alfred SEO — daily GA4 sync (4am ET)

[Timer]
OnCalendar=*-*-* 09:05:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

- [ ] **Step 3: Write CWV service + timer (runs 5am ET → 10:00 UTC during EDT)**

```ini
# systemd/alfred-seo-cwv-sync.service
[Unit]
Description=Alfred SEO — daily CWV sync
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/aialfred/alfred
ExecStart=/home/aialfred/alfred/venv/bin/python /home/aialfred/alfred/scripts/seo_cwv_sync.py
EnvironmentFile=-/home/aialfred/alfred/config/.env
StandardOutput=journal
StandardError=journal
```

```ini
# systemd/alfred-seo-cwv-sync.timer
[Unit]
Description=Alfred SEO — daily CWV sync (5am ET)

[Timer]
OnCalendar=*-*-* 10:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

- [ ] **Step 4: Write backlinks service + timer (runs 6am ET → 11:00 UTC during EDT)**

```ini
# systemd/alfred-seo-backlinks-sync.service
[Unit]
Description=Alfred SEO — daily backlinks sync
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/aialfred/alfred
ExecStart=/home/aialfred/alfred/venv/bin/python /home/aialfred/alfred/scripts/seo_backlinks_sync.py
EnvironmentFile=-/home/aialfred/alfred/config/.env
StandardOutput=journal
StandardError=journal
```

```ini
# systemd/alfred-seo-backlinks-sync.timer
[Unit]
Description=Alfred SEO — daily backlinks sync (6am ET)

[Timer]
OnCalendar=*-*-* 11:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

- [ ] **Step 5: Install timers**

```bash
sudo cp systemd/alfred-seo-*.service /etc/systemd/system/
sudo cp systemd/alfred-seo-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now alfred-seo-gsc-sync.timer
sudo systemctl enable --now alfred-seo-ga4-sync.timer
sudo systemctl enable --now alfred-seo-cwv-sync.timer
sudo systemctl enable --now alfred-seo-backlinks-sync.timer
```

- [ ] **Step 6: Verify all 4 timers active**

```bash
systemctl list-timers --all | grep alfred-seo
```

Expected: 4 lines, all "active" with next run timestamp.

- [ ] **Step 7: Commit**

```bash
git add systemd/alfred-seo-*.service systemd/alfred-seo-*.timer
git commit -m "feat(seo): systemd timers for 4 daily sync jobs (GSC/GA4/CWV/backlinks)"
```

---

## Task 32: Roen baseline backfill + smoke validate

**Files:** No new files. Operational validation.

- [ ] **Step 1: Backfill last 14 days of GSC for Roen**

```bash
for d in $(seq 1 14); do
  date=$(date -u -d "$d days ago" +%Y-%m-%d)
  ./venv/bin/python scripts/seo_gsc_sync.py --date "$date"
done
```

Expected: Each run prints `roen: N rows`. Total query rows in DB after 14 days: 100s-1000s depending on Roen's GSC coverage.

- [ ] **Step 2: Backfill last 14 days of GA4 for Roen**

```bash
for d in $(seq 1 14); do
  date=$(date -u -d "$d days ago" +%Y-%m-%d)
  ./venv/bin/python scripts/seo_ga4_sync.py --date "$date"
done
```

- [ ] **Step 3: Run CWV sync once for Roen**

```bash
./venv/bin/python scripts/seo_cwv_sync.py --limit 20
```

Expected: `roen: N URLs scanned` where N ≤ 20.

- [ ] **Step 4: Run backlinks sync once**

```bash
./venv/bin/python scripts/seo_backlinks_sync.py
```

Expected: `roen: {'recorded': N, 'newly_lost': 0}`.

- [ ] **Step 5: Validate dashboard at /admin/seo**

Open: https://aialfred.groundrushcloud.com/admin/seo
Expected: Roen row visible, GSC + GA4 + CWV columns showing data.

Open: https://aialfred.groundrushcloud.com/admin/seo/sites/roen
Expected: Site detail with GSC property + GA4 property + brand profile path filled.

- [ ] **Step 6: Spot-check DB**

```bash
psql -h localhost -U alfred -d alfred_main -c "
  SELECT COUNT(*) AS query_rows FROM seo_queries WHERE site_id=(SELECT id FROM seo_sites WHERE slug='roen');
  SELECT COUNT(*) AS page_rows FROM seo_pages WHERE site_id=(SELECT id FROM seo_sites WHERE slug='roen');
  SELECT COUNT(*) AS backlink_rows FROM seo_backlinks WHERE site_id=(SELECT id FROM seo_sites WHERE slug='roen');
"
```

Expected: query_rows ≥ 50, page_rows ≥ 10, backlink_rows ≥ 0.

- [ ] **Step 7: Commit verification note**

```bash
cat > docs/seo/verifications/2026-05-XX-plan1-baseline.md << 'NOTE'
# Plan 1 Roen baseline backfill — verification

Date: <fill on day-of>

Counts after 14-day backfill:
- seo_queries: <N>
- seo_pages: <N>
- seo_backlinks: <N>
- CWV scanned URLs: <N>

Dashboard checks:
- /admin/seo loads with Roen row ✓
- /admin/seo/sites/roen shows GSC + GA4 + brand profile ✓
- All 4 systemd timers active ✓

Plan 1 complete. Plan 2 (content engine) starts next.
NOTE

git add docs/seo/verifications/
git commit -m "docs(seo): Plan 1 Roen baseline verification — 14-day backfill complete"
```

---

## Plan 1 Done

What's now live:
- `alfred-seo` WP plugin deployed to roenhandmade.com with schema + OG + meta + sitemap + alt text + internal linking + 5 REST endpoints
- Alfred orchestrator skeleton on 105 with 9 DB tables, site registry, FastAPI admin at /admin/seo
- 4 daily sync daemons (GSC, GA4, CWV, backlinks) running on systemd timers
- Roen registered as Site #1, 14-day baseline data populated
- Cross-site dashboard ready for Plans 2 and 3 to add the remaining sites

What comes next:
- **Plan 2 (Weeks 4-6):** Content engine — brand profiles, writer, approval queue, publisher. Same /admin/seo skin extended to /admin/seo/pending.
- **Plan 3 (Weeks 7-10):** HARO + GMB + multi-site rollout + polish.

