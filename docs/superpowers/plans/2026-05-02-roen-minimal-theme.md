# roen-minimal Child Theme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a Storefront child theme (`roen-minimal`) that turns roenhandmade.com from a default WordPress install into the minimal-modern brand defined in `docs/superpowers/specs/2026-05-02-roen-handmade-website-design.md` — lowercase `roen` wordmark, marble-tabletop product photography aesthetic, product-first homepage, terracotta accent color.

**Architecture:** Child theme of the official WooCommerce Storefront theme (already installed on the site). All theme files live in a Git-tracked directory inside the Alfred Labs repo and are deployed to `/var/www/html/wp-content/themes/roen-minimal/` inside the `roenhandmade-wp` Docker container on server-104 via a deploy script. Storefront's parent theme stays activated alongside the child via WordPress's standard `Template:` mechanism. WooCommerce template overrides (single product page, archive, product card) are placed in `themes/roen-minimal/woocommerce/` per WooCommerce's documented override convention.

**Tech Stack:**
- WordPress 6.x + WooCommerce 10.7.0 (already installed)
- Storefront 4.6.2 parent theme (already installed)
- PHP 8.2 (container default)
- Inter font (Google Fonts CDN)
- Vanilla CSS — no preprocessor, no build step
- Vanilla JS for category-pill filtering — no framework
- WP-CLI 2.12.0 for activation, settings, smoke tests
- Bash deploy script via SSH + `docker cp`

---

## File Structure

All theme source lives at `/home/aialfred/alfred/services/roen-minimal/` (Git-tracked). Deployment copies it into the container.

```
services/roen-minimal/
├── style.css                       # Theme metadata header + CSS custom properties (colors, type scale, spacing)
├── functions.php                   # Enqueue parent + child styles, theme supports, deregister Storefront cruft, register custom image sizes
├── header.php                      # Custom thin nav with `roen` wordmark, no Storefront top bar
├── footer.php                      # Minimal footer with brand mark, social, legal links
├── front-page.php                  # L2 homepage: tagline + category pills + product grid
├── page-about.php                  # About page template (single column, brand voice copy)
├── assets/
│   ├── css/roen.css                # Brand-specific CSS overrides (typography scale, accent, spacing)
│   ├── css/roen-product.css        # Product detail page specific styles
│   ├── js/category-pills.js        # Client-side category filter (no page reload)
│   └── img/roen-wordmark.svg       # Lowercase `roen` SVG logo (vector for crispness)
├── woocommerce/
│   ├── content-product.php         # Product card override (used in shop / archive / homepage grid)
│   ├── single-product.php          # Wrapper for single product page
│   ├── single-product/
│   │   ├── price.php               # Price markup (Inter 300, no decimals if whole dollar)
│   │   └── add-to-cart/simple.php  # Terracotta CTA button override
│   └── archive-product.php         # Shop archive page override
├── inc/
│   └── theme-cleanup.php           # Removes Storefront homepage components, header credits, footer credit
└── README.md                       # Deploy instructions, override map, color/type tokens reference

services/roen-minimal/deploy.sh     # Bash deploy script (rsync to server-104, docker cp into container, wp cache flush)
```

**File responsibility boundaries:**
- `style.css` is the design tokens file. Anything color/spacing/type lives here as CSS custom properties. No layout rules.
- `assets/css/roen.css` is global structural CSS (header, footer, homepage). One concern per selector group.
- `assets/css/roen-product.css` is product-page only — split because product page styles are the largest concern.
- `functions.php` is wiring only. No HTML output, no business logic. Each function ≤20 lines.
- `inc/theme-cleanup.php` is the "remove Storefront defaults" file. Isolated so it's easy to delete if we ever change parent themes.
- `woocommerce/` overrides follow WooCommerce's exact path convention. Each override file is the smallest possible diff from the WC default.

**Why one CSS file per concern**: keeps cognitive load low and makes hot-reload faster during development. The total CSS payload after gzip will be under 12KB combined — no real performance penalty.

---

## Pre-Flight (Run Once Before Task 1)

- [ ] **Step 1: Verify branch is clean and we're on main**

Run: `git status -sb`
Expected: `## main` and no uncommitted theme files. If there are uncommitted Roen-spec edits from earlier sessions, leave them — the spec was already committed.

- [ ] **Step 2: Create the source directory tree**

Run:
```bash
mkdir -p /home/aialfred/alfred/services/roen-minimal/{assets/css,assets/js,assets/img,woocommerce/single-product/add-to-cart,inc}
ls -la /home/aialfred/alfred/services/roen-minimal/
```
Expected: directory tree shown.

- [ ] **Step 3: Verify SSH + container access**

Run:
```bash
ssh server-104 'timeout 10 docker exec roenhandmade-wp wp option get blogname --allow-root --path=/var/www/html'
```
Expected: `Roen Handmade`

- [ ] **Step 4: Take a snapshot backup of the current active theme setting**

Run:
```bash
ssh server-104 'docker exec roenhandmade-wp wp option get template --allow-root --path=/var/www/html; docker exec roenhandmade-wp wp option get stylesheet --allow-root --path=/var/www/html' | tee /tmp/roen-theme-pre-deploy.txt
```
Expected: `twentytwentyfour` printed twice. The file becomes our rollback reference if we need to revert.

- [ ] **Step 5: Commit the empty directory placeholder**

Run:
```bash
cd /home/aialfred/alfred
echo "# roen-minimal child theme — see plan: docs/superpowers/plans/2026-05-02-roen-minimal-theme.md" > services/roen-minimal/README.md
git add services/roen-minimal/README.md
git commit -m "chore(roen): scaffold roen-minimal theme directory"
```
Expected: 1 file changed.

---

## Task 1: Theme Metadata + Design Tokens (`style.css`)

**Files:**
- Create: `services/roen-minimal/style.css`

WordPress identifies a theme by the header in `style.css`. The `Template:` line is what makes this a child theme of Storefront. We also use this file as our design-tokens manifest — all colors, type sizes, spacing, and breakpoints are declared as CSS custom properties on `:root`. No layout rules in this file.

- [ ] **Step 1: Write `style.css` with theme header + design tokens**

Create `/home/aialfred/alfred/services/roen-minimal/style.css`:

```css
/*
Theme Name: Roen Minimal
Theme URI: https://www.roenhandmade.com/
Author: Roen Handmade
Author URI: https://www.roenhandmade.com/
Description: Minimal-modern child theme for Roen Handmade. Lowercase wordmark, marble photography, product-first homepage, terracotta accent. Built on Storefront.
Template: storefront
Version: 1.0.0
License: GNU General Public License v2 or later
Text Domain: roen-minimal
*/

:root {
  /* Brand color tokens */
  --roen-bg-primary: #FFFFFF;
  --roen-bg-secondary: #FAF9F6;
  --roen-text-primary: #1A1A1A;
  --roen-text-secondary: #666666;
  --roen-hairline: #EEEEEE;
  --roen-accent: #B85C3D;
  --roen-accent-hover: #9C4A30;

  /* Typography tokens */
  --roen-font-stack: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --roen-fw-thin: 200;
  --roen-fw-light: 300;
  --roen-fw-regular: 400;
  --roen-fw-medium: 500;

  --roen-fs-wordmark: 28px;
  --roen-fs-tagline: 22px;
  --roen-fs-tagline-sub: 13px;
  --roen-fs-product-title: 14px;
  --roen-fs-product-price: 13px;
  --roen-fs-body: 15px;
  --roen-fs-nav: 13px;
  --roen-fs-footer: 12px;

  /* Spacing scale */
  --roen-space-1: 4px;
  --roen-space-2: 8px;
  --roen-space-3: 12px;
  --roen-space-4: 16px;
  --roen-space-5: 24px;
  --roen-space-6: 32px;
  --roen-space-7: 48px;
  --roen-space-8: 64px;

  /* Layout */
  --roen-content-max: 1280px;
  --roen-content-pad-mobile: 20px;
  --roen-content-pad-desktop: 48px;
  --roen-breakpoint-mobile: 768px;
  --roen-breakpoint-tablet: 1024px;

  /* Transitions */
  --roen-transition: 200ms ease;
}
```

- [ ] **Step 2: Validate the header parses (deploy + verify)**

We can't run a unit test on theme metadata, so the test is "WP-CLI sees the theme." Defer that to Task 16 (deploy + activate). For now just commit.

- [ ] **Step 3: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/style.css
git commit -m "feat(roen): theme metadata + design tokens (style.css)"
```

---

## Task 2: functions.php — Enqueue Parent + Child Styles

**Files:**
- Create: `services/roen-minimal/functions.php`

WordPress requires `functions.php` for theme bootstrap. The standard child-theme pattern is to enqueue the parent's `style.css` first, then the child's. We also register theme supports here (title-tag, post-thumbnails, etc. — most are already inherited from Storefront, but we declare ours explicitly).

- [ ] **Step 1: Write the failing test**

Test approach: a simple smoke test via WP-CLI that asserts the child theme loads without PHP errors. We'll write a Bash test now and run it after Task 16 (deploy).

Create `/home/aialfred/alfred/services/roen-minimal/tests/test_no_php_errors.sh`:

```bash
#!/usr/bin/env bash
# Test: child theme activates without PHP fatals
set -euo pipefail

ssh server-104 'docker exec roenhandmade-wp wp theme activate roen-minimal --allow-root --path=/var/www/html' \
  | grep -q "Success" || { echo "FAIL: theme did not activate cleanly"; exit 1; }

# Hit the homepage and confirm 200 + no PHP error string
HTTP=$(timeout 15 curl -ksS -o /tmp/roen-home.html -w "%{http_code}" https://www.roenhandmade.com/)
[ "$HTTP" = "200" ] || { echo "FAIL: homepage returned $HTTP"; exit 1; }

if grep -qiE "fatal error|warning:|notice:|deprecated" /tmp/roen-home.html; then
  echo "FAIL: PHP error markers found in homepage HTML"
  grep -iE "fatal error|warning:|notice:|deprecated" /tmp/roen-home.html | head -5
  exit 1
fi

echo "PASS: theme activated and homepage rendered without PHP errors"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x /home/aialfred/alfred/services/roen-minimal/tests/test_no_php_errors.sh`

- [ ] **Step 3: Write `functions.php`**

Create `/home/aialfred/alfred/services/roen-minimal/functions.php`:

```php
<?php
/**
 * roen-minimal child theme bootstrap.
 *
 * Enqueues parent + child styles, declares theme supports, includes cleanup module.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

/**
 * Enqueue parent (Storefront) and child styles + Inter font + child JS.
 */
function roen_enqueue_assets() {
    $version = wp_get_theme()->get( 'Version' );

    // Inter font — single request, all weights we use.
    wp_enqueue_style(
        'roen-inter',
        'https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500&display=swap',
        array(),
        null
    );

    // Parent stylesheet first.
    wp_enqueue_style(
        'storefront-style',
        get_template_directory_uri() . '/style.css',
        array(),
        wp_get_theme( 'storefront' )->get( 'Version' )
    );

    // Child design tokens.
    wp_enqueue_style(
        'roen-tokens',
        get_stylesheet_directory_uri() . '/style.css',
        array( 'storefront-style' ),
        $version
    );

    // Child structural CSS.
    wp_enqueue_style(
        'roen-structure',
        get_stylesheet_directory_uri() . '/assets/css/roen.css',
        array( 'roen-tokens' ),
        $version
    );

    // Product page CSS (loaded site-wide; <4KB after gzip).
    wp_enqueue_style(
        'roen-product',
        get_stylesheet_directory_uri() . '/assets/css/roen-product.css',
        array( 'roen-structure' ),
        $version
    );

    // Category pills filter on homepage / archive.
    wp_enqueue_script(
        'roen-category-pills',
        get_stylesheet_directory_uri() . '/assets/js/category-pills.js',
        array(),
        $version,
        true
    );
}
add_action( 'wp_enqueue_scripts', 'roen_enqueue_assets', 20 );

/**
 * Theme supports the child explicitly opts into.
 * Most are inherited from Storefront, but declaring is documentation.
 */
function roen_theme_supports() {
    add_theme_support( 'title-tag' );
    add_theme_support( 'post-thumbnails' );
    add_theme_support( 'woocommerce', array(
        'thumbnail_image_width' => 600,
        'single_image_width'    => 1200,
    ) );
    add_theme_support( 'wc-product-gallery-zoom' );
    add_theme_support( 'wc-product-gallery-lightbox' );
    add_theme_support( 'wc-product-gallery-slider' );
}
add_action( 'after_setup_theme', 'roen_theme_supports' );

/**
 * Storefront cleanup (header credits, default homepage components, etc.)
 */
require_once get_stylesheet_directory() . '/inc/theme-cleanup.php';
```

- [ ] **Step 4: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/functions.php services/roen-minimal/tests/test_no_php_errors.sh
git commit -m "feat(roen): functions.php — enqueue parent+child styles, Inter font, theme supports"
```

---

## Task 3: Storefront Cleanup Module

**Files:**
- Create: `services/roen-minimal/inc/theme-cleanup.php`

Storefront ships with a homepage builder, "Storefront powered" footer credit, and a top-bar header. We disable all of that.

- [ ] **Step 1: Write `inc/theme-cleanup.php`**

Create `/home/aialfred/alfred/services/roen-minimal/inc/theme-cleanup.php`:

```php
<?php
/**
 * Strip Storefront's homepage components, header bar, and footer credit.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

/**
 * Remove the Storefront homepage builder action.
 * Our front-page.php takes over completely.
 */
function roen_remove_storefront_homepage_actions() {
    remove_action( 'homepage', 'storefront_homepage_content', 10 );
    remove_action( 'homepage', 'storefront_product_categories', 20 );
    remove_action( 'homepage', 'storefront_recent_products', 30 );
    remove_action( 'homepage', 'storefront_featured_products', 40 );
    remove_action( 'homepage', 'storefront_popular_products', 50 );
    remove_action( 'homepage', 'storefront_on_sale_products', 60 );
    remove_action( 'homepage', 'storefront_best_selling_products', 70 );
}
add_action( 'init', 'roen_remove_storefront_homepage_actions' );

/**
 * Replace Storefront's footer credit with Roen's own.
 */
function roen_remove_storefront_footer_credit() {
    remove_action( 'storefront_footer', 'storefront_credit', 20 );
}
add_action( 'init', 'roen_remove_storefront_footer_credit' );

/**
 * Remove Storefront's site-branding-with-tagline header element.
 * Our header.php replaces it entirely; this prevents double-rendering.
 */
function roen_remove_storefront_header_elements() {
    remove_action( 'storefront_header', 'storefront_header_container', 0 );
    remove_action( 'storefront_header', 'storefront_skip_links', 5 );
    remove_action( 'storefront_header', 'storefront_site_branding', 20 );
    remove_action( 'storefront_header', 'storefront_secondary_navigation', 30 );
    remove_action( 'storefront_header', 'storefront_product_search', 40 );
    remove_action( 'storefront_header', 'storefront_header_container_close', 41 );
    remove_action( 'storefront_header', 'storefront_primary_navigation_wrapper', 42 );
    remove_action( 'storefront_header', 'storefront_primary_navigation', 50 );
    remove_action( 'storefront_header', 'storefront_header_cart', 60 );
    remove_action( 'storefront_header', 'storefront_primary_navigation_wrapper_close', 68 );
}
add_action( 'init', 'roen_remove_storefront_header_elements' );
```

- [ ] **Step 2: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/inc/theme-cleanup.php
git commit -m "feat(roen): cleanup module — strip Storefront homepage, header, footer defaults"
```

---

## Task 4: Wordmark SVG

**Files:**
- Create: `services/roen-minimal/assets/img/roen-wordmark.svg`

Lowercase `roen` as vector. Used in header + footer + favicon source.

- [ ] **Step 1: Write the SVG**

Create `/home/aialfred/alfred/services/roen-minimal/assets/img/roen-wordmark.svg`:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 32" role="img" aria-label="roen">
  <text x="0" y="24"
        font-family="Inter, system-ui, sans-serif"
        font-size="28"
        font-weight="200"
        letter-spacing="-1.2"
        fill="currentColor">roen</text>
</svg>
```

The `currentColor` fill means the SVG inherits the text color from its parent — same wordmark works on light or dark surfaces.

- [ ] **Step 2: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/assets/img/roen-wordmark.svg
git commit -m "feat(roen): roen wordmark SVG (currentColor, Inter 200)"
```

---

## Task 5: Custom Header Template

**Files:**
- Create: `services/roen-minimal/header.php`

Replaces Storefront's default header. Thin, minimal, lowercase wordmark on the left, three nav items + cart on the right.

- [ ] **Step 1: Write `header.php`**

Create `/home/aialfred/alfred/services/roen-minimal/header.php`:

```php
<?php
/**
 * roen-minimal header
 *
 * Replaces Storefront's default header entirely.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}
?><!doctype html>
<html <?php language_attributes(); ?>>
<head>
    <meta charset="<?php bloginfo( 'charset' ); ?>" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="profile" href="https://gmpg.org/xfn/11" />
    <?php wp_head(); ?>
</head>

<body <?php body_class(); ?>>
<?php wp_body_open(); ?>

<header class="roen-header" role="banner">
    <div class="roen-container roen-header__inner">
        <a class="roen-header__brand" href="<?php echo esc_url( home_url( '/' ) ); ?>" aria-label="<?php esc_attr_e( 'Roen home', 'roen-minimal' ); ?>">
            <?php echo file_get_contents( get_stylesheet_directory() . '/assets/img/roen-wordmark.svg' ); // phpcs:ignore -- SVG inline ?>
        </a>

        <nav class="roen-header__nav" role="navigation" aria-label="<?php esc_attr_e( 'Primary', 'roen-minimal' ); ?>">
            <a href="<?php echo esc_url( get_permalink( wc_get_page_id( 'shop' ) ) ); ?>">shop</a>
            <a href="<?php echo esc_url( home_url( '/about/' ) ); ?>">about</a>
            <a class="roen-header__cart" href="<?php echo esc_url( wc_get_cart_url() ); ?>">
                cart (<span class="roen-cart-count"><?php echo (int) WC()->cart->get_cart_contents_count(); ?></span>)
            </a>
        </nav>
    </div>
</header>

<main id="content" class="roen-main">
```

Note: `</main>` and `</body>` close in `footer.php` (Task 6). This is standard WordPress convention.

- [ ] **Step 2: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/header.php
git commit -m "feat(roen): custom header — wordmark + 3-link nav + cart count"
```

---

## Task 6: Custom Footer Template

**Files:**
- Create: `services/roen-minimal/footer.php`

Closes the main and body tags. Minimal three-column footer: brand mark, social links, legal links.

- [ ] **Step 1: Write `footer.php`**

Create `/home/aialfred/alfred/services/roen-minimal/footer.php`:

```php
<?php
/**
 * roen-minimal footer
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}
?>
</main><?php // close .roen-main from header.php ?>

<footer class="roen-footer" role="contentinfo">
    <div class="roen-container roen-footer__inner">
        <div class="roen-footer__brand">
            <?php echo file_get_contents( get_stylesheet_directory() . '/assets/img/roen-wordmark.svg' ); // phpcs:ignore ?>
            <p class="roen-footer__legal">© <?php echo (int) date( 'Y' ); ?> Roen Handmade.</p>
        </div>

        <nav class="roen-footer__col" aria-label="<?php esc_attr_e( 'Social', 'roen-minimal' ); ?>">
            <h4 class="roen-footer__heading">follow</h4>
            <a href="https://www.instagram.com/roenhandmade/" rel="noopener" target="_blank">instagram</a>
            <a href="https://www.facebook.com/roenhandmade/" rel="noopener" target="_blank">facebook</a>
        </nav>

        <nav class="roen-footer__col" aria-label="<?php esc_attr_e( 'Help', 'roen-minimal' ); ?>">
            <h4 class="roen-footer__heading">help</h4>
            <a href="<?php echo esc_url( home_url( '/about/' ) ); ?>">about</a>
            <a href="mailto:mjohnson@groundrushinc.com">contact</a>
            <a href="<?php echo esc_url( home_url( '/privacy-policy/' ) ); ?>">privacy</a>
            <a href="<?php echo esc_url( home_url( '/refund_returns/' ) ); ?>">returns</a>
        </nav>
    </div>
</footer>

<?php wp_footer(); ?>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/footer.php
git commit -m "feat(roen): custom footer — brand + social + help links"
```

---

## Task 7: Front Page Template (Homepage L2)

**Files:**
- Create: `services/roen-minimal/front-page.php`

The L2 layout: tagline block, category pills, product grid. WordPress automatically uses `front-page.php` when `Settings → Reading → Front page displays` is set to a static page (we'll set this in Task 16).

- [ ] **Step 1: Write `front-page.php`**

Create `/home/aialfred/alfred/services/roen-minimal/front-page.php`:

```php
<?php
/**
 * Front page — L2 product-first layout.
 *
 * Tagline → category pills → product grid (16 most recent).
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

get_header();

// Pull product categories that have at least 1 in-stock product.
$product_cats = get_terms( array(
    'taxonomy'   => 'product_cat',
    'hide_empty' => true,
    'orderby'    => 'name',
    'order'      => 'ASC',
) );
?>

<section class="roen-tagline">
    <div class="roen-container">
        <h1 class="roen-tagline__head">handmade jewelry, made in atlanta.</h1>
        <p class="roen-tagline__sub">new pieces every week.</p>
    </div>
</section>

<?php if ( ! empty( $product_cats ) && ! is_wp_error( $product_cats ) ) : ?>
<nav class="roen-pills" aria-label="<?php esc_attr_e( 'Filter by category', 'roen-minimal' ); ?>">
    <div class="roen-container roen-pills__row">
        <button class="roen-pill is-active" data-cat="all" type="button">all</button>
        <?php foreach ( $product_cats as $cat ) : ?>
            <button class="roen-pill" data-cat="<?php echo esc_attr( $cat->slug ); ?>" type="button"><?php echo esc_html( strtolower( $cat->name ) ); ?></button>
        <?php endforeach; ?>
    </div>
</nav>
<?php endif; ?>

<section class="roen-grid-section">
    <div class="roen-container">
        <?php
        $products = wc_get_products( array(
            'status'  => 'publish',
            'limit'   => 16,
            'orderby' => 'date',
            'order'   => 'DESC',
        ) );

        if ( empty( $products ) ) :
            ?>
            <p class="roen-empty">no pieces in the shop yet — new drops are on the way.</p>
            <?php
        else :
            ?>
            <ul class="roen-grid" role="list">
                <?php
                foreach ( $products as $product ) {
                    $post_object = get_post( $product->get_id() );
                    setup_postdata( $GLOBALS['post'] =& $post_object ); // phpcs:ignore
                    wc_get_template_part( 'content', 'product' );
                }
                wp_reset_postdata();
                ?>
            </ul>
            <?php
        endif;
        ?>
    </div>
</section>

<?php get_footer();
```

- [ ] **Step 2: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/front-page.php
git commit -m "feat(roen): front-page.php — L2 layout (tagline, pills, 16 products)"
```

---

## Task 8: WooCommerce Product Card Override

**Files:**
- Create: `services/roen-minimal/woocommerce/content-product.php`

WooCommerce calls `content-product.php` for each product card in any grid. Our override emits a clean structure: image, title, price. No "Add to Cart" button on cards (per spec — minimal).

- [ ] **Step 1: Write `woocommerce/content-product.php`**

Create `/home/aialfred/alfred/services/roen-minimal/woocommerce/content-product.php`:

```php
<?php
/**
 * roen-minimal product card (loops on home, shop, archive).
 *
 * Override of woocommerce/templates/content-product.php
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

global $product;

if ( empty( $product ) || ! $product->is_visible() ) {
    return;
}

$cat_slugs = array();
foreach ( wp_get_post_terms( $product->get_id(), 'product_cat' ) as $cat ) {
    $cat_slugs[] = $cat->slug;
}
?>

<li class="roen-card" data-cats="<?php echo esc_attr( implode( ' ', $cat_slugs ) ); ?>">
    <a class="roen-card__link" href="<?php the_permalink(); ?>">
        <div class="roen-card__media">
            <?php
            echo $product->get_image( 'woocommerce_thumbnail', array( // phpcs:ignore
                'class' => 'roen-card__img',
                'loading' => 'lazy',
            ) );
            ?>
        </div>
        <div class="roen-card__body">
            <h3 class="roen-card__title"><?php echo esc_html( $product->get_name() ); ?></h3>
            <div class="roen-card__price"><?php echo $product->get_price_html(); // phpcs:ignore ?></div>
        </div>
    </a>
</li>
```

- [ ] **Step 2: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/woocommerce/content-product.php
git commit -m "feat(roen): WC product card override — image + title + price, no CTA on cards"
```

---

## Task 9: Single Product Page Override

**Files:**
- Create: `services/roen-minimal/woocommerce/single-product.php`
- Create: `services/roen-minimal/woocommerce/single-product/price.php`
- Create: `services/roen-minimal/woocommerce/single-product/add-to-cart/simple.php`

Single-product is the highest-stakes page — this is where the buy decision happens. 60/40 split desktop, stacked mobile, terracotta CTA, gallery on the left, info on the right.

- [ ] **Step 1: Write the wrapper `single-product.php`**

Create `/home/aialfred/alfred/services/roen-minimal/woocommerce/single-product.php`:

```php
<?php
/**
 * roen-minimal single product wrapper.
 * Loads WC's content-single-product.php inside our layout.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

get_header( 'shop' );
?>

<section class="roen-single roen-container">
    <?php while ( have_posts() ) : the_post(); ?>
        <?php wc_get_template_part( 'content', 'single-product' ); ?>
    <?php endwhile; ?>
</section>

<?php get_footer( 'shop' );
```

- [ ] **Step 2: Write the price override**

Create `/home/aialfred/alfred/services/roen-minimal/woocommerce/single-product/price.php`:

```php
<?php
/**
 * roen-minimal price markup on single product.
 * Inter 300, no extra chrome.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

global $product;
?>
<p class="roen-single__price price"><?php echo $product->get_price_html(); // phpcs:ignore ?></p>
```

- [ ] **Step 3: Write the simple add-to-cart override**

Create `/home/aialfred/alfred/services/roen-minimal/woocommerce/single-product/add-to-cart/simple.php`:

```php
<?php
/**
 * roen-minimal simple-product add-to-cart form.
 *
 * Override of woocommerce/templates/single-product/add-to-cart/simple.php
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

global $product;

if ( ! $product->is_purchasable() ) {
    return;
}

echo wc_get_stock_html( $product ); // phpcs:ignore

if ( $product->is_in_stock() ) :
    do_action( 'woocommerce_before_add_to_cart_form' );
    ?>
    <form class="cart roen-atc-form" action="<?php echo esc_url( apply_filters( 'woocommerce_add_to_cart_form_action', $product->get_permalink() ) ); ?>" method="post" enctype="multipart/form-data">
        <?php do_action( 'woocommerce_before_add_to_cart_button' ); ?>
        <?php do_action( 'woocommerce_before_add_to_cart_quantity' ); ?>

        <?php
        woocommerce_quantity_input( array(
            'min_value'   => apply_filters( 'woocommerce_quantity_input_min', $product->get_min_purchase_quantity(), $product ),
            'max_value'   => apply_filters( 'woocommerce_quantity_input_max', $product->get_max_purchase_quantity(), $product ),
            'input_value' => isset( $_POST['quantity'] ) ? wc_stock_amount( wp_unslash( $_POST['quantity'] ) ) : $product->get_min_purchase_quantity(),
        ) );
        ?>

        <?php do_action( 'woocommerce_after_add_to_cart_quantity' ); ?>

        <button type="submit"
                name="add-to-cart"
                value="<?php echo esc_attr( $product->get_id() ); ?>"
                class="single_add_to_cart_button button alt roen-atc-btn">
            add to cart
        </button>

        <?php do_action( 'woocommerce_after_add_to_cart_button' ); ?>
    </form>
    <?php
    do_action( 'woocommerce_after_add_to_cart_form' );
endif;
```

- [ ] **Step 4: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/woocommerce/single-product.php services/roen-minimal/woocommerce/single-product/price.php services/roen-minimal/woocommerce/single-product/add-to-cart/simple.php
git commit -m "feat(roen): single product page + price + atc button overrides"
```

---

## Task 10: WooCommerce Archive Page Override

**Files:**
- Create: `services/roen-minimal/woocommerce/archive-product.php`

The shop page (`/shop/`) and category archives (`/product-category/<slug>/`) all use this template. Same look as the homepage grid — same `content-product.php` cards.

- [ ] **Step 1: Write `archive-product.php`**

Create `/home/aialfred/alfred/services/roen-minimal/woocommerce/archive-product.php`:

```php
<?php
/**
 * roen-minimal product archive (shop page, category archives).
 *
 * Override of woocommerce/templates/archive-product.php
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

get_header( 'shop' );
?>

<section class="roen-tagline roen-tagline--small">
    <div class="roen-container">
        <h1 class="roen-tagline__head"><?php woocommerce_page_title(); ?></h1>
    </div>
</section>

<section class="roen-grid-section">
    <div class="roen-container">
        <?php if ( woocommerce_product_loop() ) : ?>
            <?php woocommerce_product_loop_start(); ?>

            <?php
            if ( wc_get_loop_prop( 'total' ) ) {
                while ( have_posts() ) {
                    the_post();
                    do_action( 'woocommerce_shop_loop' );
                    wc_get_template_part( 'content', 'product' );
                }
            }
            ?>

            <?php woocommerce_product_loop_end(); ?>

            <?php do_action( 'woocommerce_after_shop_loop' ); ?>

        <?php else : ?>
            <p class="roen-empty">no pieces in this category yet.</p>
        <?php endif; ?>
    </div>
</section>

<?php
get_footer( 'shop' );
```

- [ ] **Step 2: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/woocommerce/archive-product.php
git commit -m "feat(roen): WC archive override — shared grid layout for shop + categories"
```

---

## Task 11: About Page Template

**Files:**
- Create: `services/roen-minimal/page-about.php`

A WordPress page template Roen can pick from the page editor. Single column, max 60ch, brand-voice copy. Mike's wife can edit the body content via the WP admin; the template provides the layout.

- [ ] **Step 1: Write `page-about.php`**

Create `/home/aialfred/alfred/services/roen-minimal/page-about.php`:

```php
<?php
/**
 * Template Name: About — Roen Minimal
 *
 * Single column, narrow line length, brand-voice tone.
 * Body content is editable via the WP page editor.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

get_header();
?>

<article class="roen-about roen-container">
    <?php while ( have_posts() ) : the_post(); ?>
        <h1 class="roen-about__title"><?php the_title(); ?></h1>
        <div class="roen-about__body">
            <?php the_content(); ?>
        </div>
    <?php endwhile; ?>
</article>

<?php get_footer();
```

- [ ] **Step 2: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/page-about.php
git commit -m "feat(roen): About page template — single column, narrow line length"
```

---

## Task 12: Structural CSS (`assets/css/roen.css`)

**Files:**
- Create: `services/roen-minimal/assets/css/roen.css`

Global structural CSS. Resets a few Storefront defaults, sets typography baseline, lays out the header / footer / homepage / product grid.

- [ ] **Step 1: Write `roen.css`**

Create `/home/aialfred/alfred/services/roen-minimal/assets/css/roen.css`:

```css
/* ============================================================
   roen-minimal — structural CSS
   Tokens defined in style.css. This file owns layout only.
   ============================================================ */

/* Reset Storefront cruft we never want */
.site-header,
.storefront-handheld-footer-bar,
.storefront-credit { display: none !important; }

body, button, input, select, textarea {
    font-family: var(--roen-font-stack);
    font-weight: var(--roen-fw-regular);
    font-size: var(--roen-fs-body);
    color: var(--roen-text-primary);
    background: var(--roen-bg-primary);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

a { color: var(--roen-text-primary); text-decoration: none; transition: color var(--roen-transition); }
a:hover, a:focus { color: var(--roen-accent); }

img { max-width: 100%; display: block; height: auto; }

button { font-family: var(--roen-font-stack); cursor: pointer; }

.roen-container {
    max-width: var(--roen-content-max);
    margin: 0 auto;
    padding: 0 var(--roen-content-pad-mobile);
}
@media (min-width: 768px) { .roen-container { padding: 0 var(--roen-content-pad-desktop); } }

/* ---------- Header ---------- */

.roen-header {
    border-bottom: 1px solid var(--roen-hairline);
    background: var(--roen-bg-primary);
}

.roen-header__inner {
    display: flex; align-items: center; justify-content: space-between;
    padding: var(--roen-space-4) var(--roen-content-pad-mobile);
    gap: var(--roen-space-4);
}
@media (min-width: 768px) {
    .roen-header__inner { padding: var(--roen-space-5) var(--roen-content-pad-desktop); }
}

.roen-header__brand { display: flex; align-items: center; color: var(--roen-text-primary); }
.roen-header__brand svg { height: 28px; width: auto; }

.roen-header__nav {
    display: flex; align-items: center; gap: var(--roen-space-5);
    font-size: var(--roen-fs-nav);
    font-weight: var(--roen-fw-regular);
    color: var(--roen-text-secondary);
}
.roen-header__nav a { color: inherit; }
.roen-header__nav a:hover { color: var(--roen-accent); }

/* ---------- Tagline ---------- */

.roen-tagline { padding: var(--roen-space-7) 0 var(--roen-space-5); text-align: center; }
.roen-tagline--small { padding: var(--roen-space-5) 0 var(--roen-space-3); }
.roen-tagline__head {
    font-size: var(--roen-fs-tagline);
    font-weight: var(--roen-fw-thin);
    letter-spacing: -1px;
    color: var(--roen-text-primary);
    margin: 0 0 var(--roen-space-2) 0;
    line-height: 1.2;
}
.roen-tagline__sub {
    font-size: var(--roen-fs-tagline-sub);
    color: var(--roen-text-secondary);
    margin: 0;
}

/* ---------- Category pills ---------- */

.roen-pills { padding: 0 0 var(--roen-space-5); }
.roen-pills__row { display: flex; gap: var(--roen-space-2); flex-wrap: wrap; justify-content: center; }
.roen-pill {
    background: transparent;
    border: 1px solid var(--roen-hairline);
    border-radius: 24px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: var(--roen-fw-regular);
    color: var(--roen-text-secondary);
    transition: color var(--roen-transition), border-color var(--roen-transition), background var(--roen-transition);
}
.roen-pill:hover { color: var(--roen-text-primary); border-color: var(--roen-text-primary); }
.roen-pill.is-active {
    color: var(--roen-bg-primary);
    background: var(--roen-text-primary);
    border-color: var(--roen-text-primary);
}

/* ---------- Product grid ---------- */

.roen-grid-section { padding: 0 0 var(--roen-space-7); }
.roen-grid {
    list-style: none; margin: 0; padding: 0;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: var(--roen-space-4);
}
@media (min-width: 768px) { .roen-grid { grid-template-columns: repeat(3, 1fr); gap: var(--roen-space-5); } }
@media (min-width: 1024px) { .roen-grid { grid-template-columns: repeat(4, 1fr); } }

.roen-card {
    background: var(--roen-bg-primary);
    transition: transform var(--roen-transition);
}
.roen-card.is-hidden { display: none; }
.roen-card:hover { transform: translateY(-2px); }
.roen-card__link { display: block; color: inherit; }
.roen-card__media {
    aspect-ratio: 4 / 5;
    background: var(--roen-bg-secondary);
    overflow: hidden;
    margin-bottom: var(--roen-space-3);
}
.roen-card__img { width: 100%; height: 100%; object-fit: cover; }
.roen-card__title {
    font-size: var(--roen-fs-product-title);
    font-weight: var(--roen-fw-regular);
    letter-spacing: -0.2px;
    color: var(--roen-text-primary);
    margin: 0 0 var(--roen-space-1) 0;
    line-height: 1.4;
}
.roen-card__price {
    font-size: var(--roen-fs-product-price);
    font-weight: var(--roen-fw-light);
    color: var(--roen-text-secondary);
}
.roen-card__price del { color: var(--roen-text-secondary); margin-right: 6px; }
.roen-card__price ins { color: var(--roen-accent); text-decoration: none; }

.roen-empty {
    text-align: center;
    color: var(--roen-text-secondary);
    padding: var(--roen-space-7) 0;
}

/* ---------- Footer ---------- */

.roen-footer {
    background: var(--roen-bg-secondary);
    border-top: 1px solid var(--roen-hairline);
    margin-top: var(--roen-space-8);
}
.roen-footer__inner {
    display: grid;
    grid-template-columns: 1fr;
    gap: var(--roen-space-6);
    padding: var(--roen-space-6) var(--roen-content-pad-mobile);
}
@media (min-width: 768px) {
    .roen-footer__inner {
        grid-template-columns: 1.5fr 1fr 1fr;
        padding: var(--roen-space-7) var(--roen-content-pad-desktop);
    }
}
.roen-footer__brand svg { height: 24px; width: auto; color: var(--roen-text-primary); margin-bottom: var(--roen-space-3); }
.roen-footer__legal { font-size: var(--roen-fs-footer); color: var(--roen-text-secondary); margin: 0; }
.roen-footer__heading {
    font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px;
    font-weight: var(--roen-fw-medium); color: var(--roen-text-primary);
    margin: 0 0 var(--roen-space-3) 0;
}
.roen-footer__col a {
    display: block;
    font-size: var(--roen-fs-footer);
    color: var(--roen-text-secondary);
    line-height: 1.9;
}
.roen-footer__col a:hover { color: var(--roen-accent); }

/* ---------- About ---------- */

.roen-about { max-width: 640px; padding-top: var(--roen-space-7); padding-bottom: var(--roen-space-8); }
.roen-about__title {
    font-size: 32px; font-weight: var(--roen-fw-thin); letter-spacing: -1px;
    margin: 0 0 var(--roen-space-5) 0;
}
.roen-about__body { font-size: 16px; line-height: 1.7; color: var(--roen-text-primary); }
.roen-about__body p { margin: 0 0 var(--roen-space-4) 0; }
```

- [ ] **Step 2: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/assets/css/roen.css
git commit -m "feat(roen): structural CSS (header, footer, homepage, grid, about)"
```

---

## Task 13: Product Page CSS (`assets/css/roen-product.css`)

**Files:**
- Create: `services/roen-minimal/assets/css/roen-product.css`

Single-product page styles — gallery left / info right, terracotta CTA. Kept separate so we can iterate on it without touching the rest.

- [ ] **Step 1: Write `roen-product.css`**

Create `/home/aialfred/alfred/services/roen-minimal/assets/css/roen-product.css`:

```css
/* ============================================================
   roen-minimal — single product page
   ============================================================ */

.roen-single {
    padding-top: var(--roen-space-6);
    padding-bottom: var(--roen-space-8);
}

.roen-single .product {
    display: grid;
    grid-template-columns: 1fr;
    gap: var(--roen-space-5);
}
@media (min-width: 1024px) {
    .roen-single .product {
        grid-template-columns: 60fr 40fr;
        gap: var(--roen-space-7);
    }
}

.roen-single .woocommerce-product-gallery {
    background: var(--roen-bg-secondary);
}
.roen-single .woocommerce-product-gallery__image img {
    width: 100%; height: auto; display: block;
}

.roen-single .summary {
    padding: var(--roen-space-3) 0;
}

.roen-single .product_title {
    font-size: 28px;
    font-weight: var(--roen-fw-thin);
    letter-spacing: -0.5px;
    margin: 0 0 var(--roen-space-3) 0;
    line-height: 1.2;
}

.roen-single__price {
    font-size: 20px;
    font-weight: var(--roen-fw-light);
    color: var(--roen-text-primary);
    margin: 0 0 var(--roen-space-5) 0;
}
.roen-single__price del { color: var(--roen-text-secondary); margin-right: 8px; }
.roen-single__price ins { color: var(--roen-accent); text-decoration: none; }

.roen-single .woocommerce-product-details__short-description {
    color: var(--roen-text-primary);
    line-height: 1.7;
    margin-bottom: var(--roen-space-5);
}

/* ATC form */
.roen-atc-form {
    display: flex;
    align-items: stretch;
    gap: var(--roen-space-3);
    flex-wrap: wrap;
    margin-bottom: var(--roen-space-5);
}
.roen-atc-form .quantity {
    display: flex; align-items: stretch;
}
.roen-atc-form .quantity input.qty {
    width: 64px;
    padding: 0 12px;
    border: 1px solid var(--roen-hairline);
    background: var(--roen-bg-primary);
    font-size: 14px;
    text-align: center;
    height: 44px;
}

.roen-atc-btn {
    background: var(--roen-accent);
    color: var(--roen-bg-primary);
    border: none;
    padding: 0 var(--roen-space-6);
    height: 44px;
    font-size: 13px;
    font-weight: var(--roen-fw-medium);
    letter-spacing: 1px;
    text-transform: lowercase;
    cursor: pointer;
    transition: background var(--roen-transition);
    flex: 1 1 auto;
    min-width: 200px;
}
.roen-atc-btn:hover, .roen-atc-btn:focus { background: var(--roen-accent-hover); }

@media (max-width: 767px) {
    .roen-atc-form { flex-direction: column; }
    .roen-atc-btn { width: 100%; min-width: 0; }
}

/* Stock + tabs */
.roen-single .stock { font-size: 13px; color: var(--roen-text-secondary); margin-bottom: var(--roen-space-3); }
.roen-single .out-of-stock { color: var(--roen-accent); }

.woocommerce-tabs {
    border-top: 1px solid var(--roen-hairline);
    padding-top: var(--roen-space-6);
    margin-top: var(--roen-space-7);
    grid-column: 1 / -1;
}

.woocommerce-tabs ul.tabs { list-style: none; padding: 0; display: flex; gap: var(--roen-space-5); margin: 0 0 var(--roen-space-5) 0; border-bottom: 1px solid var(--roen-hairline); }
.woocommerce-tabs ul.tabs li { padding: 0 0 var(--roen-space-2) 0; }
.woocommerce-tabs ul.tabs li a { font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--roen-text-secondary); }
.woocommerce-tabs ul.tabs li.active { border-bottom: 1px solid var(--roen-accent); }
.woocommerce-tabs ul.tabs li.active a { color: var(--roen-text-primary); }
```

- [ ] **Step 2: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/assets/css/roen-product.css
git commit -m "feat(roen): single product page CSS — 60/40 split, terracotta CTA"
```

---

## Task 14: Category Pills JS Filter

**Files:**
- Create: `services/roen-minimal/assets/js/category-pills.js`
- Test: `services/roen-minimal/tests/test_category_pills.html` (manual smoke test page)

Vanilla JS. No frameworks, no dependencies. The pills filter the existing rendered grid by toggling a `is-hidden` class on cards based on their `data-cats` attribute. No network requests.

- [ ] **Step 1: Write the smoke-test HTML**

Create `/home/aialfred/alfred/services/roen-minimal/tests/test_category_pills.html`:

```html
<!doctype html>
<html><head><title>category pills test</title>
<style>
  .roen-card.is-hidden { display: none; }
  .roen-pill.is-active { font-weight: bold; }
  .roen-card { display: block; padding: 8px; }
</style>
</head>
<body>
<nav class="roen-pills">
  <div class="roen-pills__row">
    <button class="roen-pill is-active" data-cat="all">all</button>
    <button class="roen-pill" data-cat="bracelets">bracelets</button>
    <button class="roen-pill" data-cat="earrings">earrings</button>
  </div>
</nav>
<ul class="roen-grid">
  <li class="roen-card" data-cats="bracelets">A — bracelet</li>
  <li class="roen-card" data-cats="earrings">B — earring</li>
  <li class="roen-card" data-cats="bracelets necklaces">C — both</li>
</ul>
<script src="../assets/js/category-pills.js"></script>
<script>
  // Manual checklist: click each pill, confirm visibility:
  //   all       -> A, B, C all visible
  //   bracelets -> A, C visible; B hidden
  //   earrings  -> B visible; A, C hidden
</script>
</body></html>
```

- [ ] **Step 2: Write `category-pills.js`**

Create `/home/aialfred/alfred/services/roen-minimal/assets/js/category-pills.js`:

```javascript
(function () {
    'use strict';

    function init() {
        var pills = document.querySelectorAll('.roen-pill');
        var cards = document.querySelectorAll('.roen-card');
        if (pills.length === 0 || cards.length === 0) {
            return;
        }

        function applyFilter(slug) {
            cards.forEach(function (card) {
                var cats = (card.getAttribute('data-cats') || '').split(/\s+/);
                var match = (slug === 'all') || cats.indexOf(slug) !== -1;
                card.classList.toggle('is-hidden', !match);
            });
        }

        pills.forEach(function (pill) {
            pill.addEventListener('click', function () {
                pills.forEach(function (p) { p.classList.remove('is-active'); });
                pill.classList.add('is-active');
                applyFilter(pill.getAttribute('data-cat'));
            });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
}());
```

- [ ] **Step 3: Manually verify the smoke test**

Open `services/roen-minimal/tests/test_category_pills.html` in a browser.
- Click `bracelets` → A and C visible, B hidden.
- Click `earrings` → B visible, A and C hidden.
- Click `all` → all three visible.

If any case fails, fix `category-pills.js` and repeat.

- [ ] **Step 4: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/assets/js/category-pills.js services/roen-minimal/tests/test_category_pills.html
git commit -m "feat(roen): category pills JS filter (vanilla, no deps)"
```

---

## Task 15: Deploy Script

**Files:**
- Create: `services/roen-minimal/deploy.sh`

A bash script that rsyncs the source to a staging directory on server-104, then `docker cp`s into the container, then flushes the WP cache. Idempotent. Should be the only way the theme reaches production.

- [ ] **Step 1: Write `deploy.sh`**

Create `/home/aialfred/alfred/services/roen-minimal/deploy.sh`:

```bash
#!/usr/bin/env bash
# Deploy the roen-minimal child theme to roenhandmade.com.
# Idempotent — safe to re-run.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_HOST="server-104"
STAGE_DIR="/tmp/roen-minimal"
CONTAINER="roenhandmade-wp"
WP_PATH="/var/www/html"
TARGET="${WP_PATH}/wp-content/themes/roen-minimal"

echo "==> rsync source to ${SSH_HOST}:${STAGE_DIR}"
rsync -av --delete \
  --exclude 'tests/' \
  --exclude 'deploy.sh' \
  --exclude 'README.md' \
  --exclude '.DS_Store' \
  "${SRC_DIR}/" "${SSH_HOST}:${STAGE_DIR}/"

echo "==> docker cp into ${CONTAINER}:${TARGET}"
ssh "${SSH_HOST}" "
  set -e
  timeout 30 docker exec ${CONTAINER} mkdir -p ${TARGET}
  timeout 60 docker cp ${STAGE_DIR}/. ${CONTAINER}:${TARGET}/
  timeout 30 docker exec ${CONTAINER} chown -R www-data:www-data ${TARGET}
"

echo "==> verify theme is recognized"
ssh "${SSH_HOST}" "timeout 20 docker exec ${CONTAINER} wp theme list --allow-root --path=${WP_PATH} --format=csv --fields=name,status,version" \
  | grep -q '^roen-minimal' || { echo "ERROR: roen-minimal not seen by WP-CLI"; exit 1; }

echo "==> flush WP cache"
ssh "${SSH_HOST}" "timeout 20 docker exec ${CONTAINER} wp cache flush --allow-root --path=${WP_PATH}" || true

echo "==> done. To activate: ssh ${SSH_HOST} 'docker exec ${CONTAINER} wp theme activate roen-minimal --allow-root --path=${WP_PATH}'"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x /home/aialfred/alfred/services/roen-minimal/deploy.sh`

- [ ] **Step 3: Commit**

```bash
cd /home/aialfred/alfred
git add services/roen-minimal/deploy.sh
git commit -m "feat(roen): deploy script — rsync + docker cp + cache flush"
```

---

## Task 16: First Deploy + Activation

**Files:** none — this is an operational step.

Now we wire up the actual site. Deploy, activate, set as homepage, smoke test.

- [ ] **Step 1: Deploy**

Run from `/home/aialfred/alfred/`:
```bash
./services/roen-minimal/deploy.sh
```
Expected: ends with `==> done.` line. If anything fails, do not proceed — fix the deploy issue first.

- [ ] **Step 2: Activate**

Run:
```bash
ssh server-104 'timeout 20 docker exec roenhandmade-wp wp theme activate roen-minimal --allow-root --path=/var/www/html'
```
Expected: `Success: Switched to 'Roen Minimal' theme.`

- [ ] **Step 3: Run the no-PHP-errors smoke test from Task 2**

Run: `bash /home/aialfred/alfred/services/roen-minimal/tests/test_no_php_errors.sh`
Expected: `PASS: theme activated and homepage rendered without PHP errors`

If FAIL: deactivate immediately and investigate.
```bash
ssh server-104 'timeout 20 docker exec roenhandmade-wp wp theme activate twentytwentyfour --allow-root --path=/var/www/html'
```

- [ ] **Step 4: Configure WP to use a static homepage**

Currently the homepage is the blog index. Switch it to use `front-page.php` by setting Reading → static page. We don't actually need a Page entry — `front-page.php` takes precedence over any Reading setting. But to be safe and idiomatic:

Run:
```bash
ssh server-104 '
  timeout 20 docker exec roenhandmade-wp wp option update show_on_front page --allow-root --path=/var/www/html
  timeout 20 docker exec roenhandmade-wp wp post create --post_type=page --post_title="Home" --post_status=publish --porcelain --allow-root --path=/var/www/html
'
```
Capture the page ID returned by the `wp post create` command. Then:

```bash
ssh server-104 'timeout 20 docker exec roenhandmade-wp wp option update page_on_front <PAGE_ID_HERE> --allow-root --path=/var/www/html'
```
(Substitute the captured ID.)

- [ ] **Step 5: Verify the homepage renders the new layout**

Run: `timeout 12 curl -ksS https://www.roenhandmade.com/ | grep -E 'roen-tagline|roen-grid|roen-pill' | head -3`
Expected: at least 3 lines matching our class names. If empty, the front-page template isn't being used — check Reading settings.

- [ ] **Step 6: Verify all 7 products appear in the grid**

Run: `timeout 12 curl -ksS https://www.roenhandmade.com/ | grep -c 'roen-card'`
Expected: 7

- [ ] **Step 7: Check a product detail page**

Run: `timeout 12 curl -ksS -o /tmp/roen-product.html -w "%{http_code}" https://www.roenhandmade.com/product/charming-pink-heart-bracelet/`
Expected: `200`. Then:
```bash
grep -E 'roen-single|roen-atc-btn' /tmp/roen-product.html | head -2
```
Expected: at least 2 matches.

- [ ] **Step 8: Create the About page using our template**

Run:
```bash
ssh server-104 'timeout 20 docker exec roenhandmade-wp wp post create --post_type=page --post_title="About" --post_name="about" --post_status=publish --post_content="Roen is a small Atlanta jewelry studio. Each piece is sketched, soldered, set, and polished by hand — no assembly line, no outsourced casting. New pieces drop weekly." --page_template="page-about.php" --porcelain --allow-root --path=/var/www/html'
```
Capture the ID. Verify:
```bash
timeout 12 curl -ksS -o /tmp/about.html -w "%{http_code}" https://www.roenhandmade.com/about/
grep -c 'roen-about' /tmp/about.html
```
Expected: 200 status, at least 1 match.

- [ ] **Step 9: Commit a deployment record**

```bash
cd /home/aialfred/alfred
git add -A
git commit --allow-empty -m "deploy(roen): roen-minimal v1.0.0 activated on roenhandmade.com"
```

---

## Task 17: Spec Verification Checklist

Run through every checkbox in the spec's "Verification Criteria" section. If any fails, file a follow-up commit.

- [ ] **Step 1: Theme list shows roen-minimal active**

Run: `ssh server-104 'docker exec roenhandmade-wp wp theme list --status=active --allow-root --path=/var/www/html --format=csv --fields=name'`
Expected: `name\nroen-minimal`

- [ ] **Step 2: Homepage renders L2 (no Storefront default content)**

Run: `timeout 12 curl -ksS https://www.roenhandmade.com/ | grep -cE 'storefront-credit|storefront_homepage|storefront-handheld'`
Expected: `0`

- [ ] **Step 3: All 7 products in the grid**

Run: `timeout 12 curl -ksS https://www.roenhandmade.com/ | grep -c 'roen-card'`
Expected: `7`

- [ ] **Step 4: Add to Cart works end-to-end**

Manual smoke test:
1. Visit https://www.roenhandmade.com/product/charming-pink-heart-bracelet/ in a browser.
2. Click `add to cart`. Confirm a redirect to /cart/ or a "Added to your cart" notice appears.
3. Visit /cart/ — product is listed.
4. Click "Proceed to checkout" — checkout page loads with product line item.

- [ ] **Step 5: Mobile rendering at 375 / 768 / 1024 / 1440**

Manual test in Chrome DevTools (Toggle Device Toolbar):
- 375px: nav stacks gracefully, grid is 2 cols, ATC button full width on product page.
- 768px: nav inline, grid is 3 cols.
- 1024px: grid is 4 cols, product page is 60/40 split.
- 1440px: container is centered, doesn't stretch full bleed.

- [ ] **Step 6: Color accent only on CTAs and hover**

Visual scan: load homepage and a product page. Terracotta should appear ONLY on:
- The "add to cart" button on product page
- Link hovers in nav and footer
- Active "ins" price (sale price markup)

If terracotta appears on any background fill, body text, or borders — fix `roen.css`.

- [ ] **Step 7: Page weight under 500KB on cold load**

Run: `timeout 30 curl -ksS -o /dev/null -w 'TOTAL: %{size_download} bytes\n' https://www.roenhandmade.com/`
Add the asset weights:
```bash
for url in https://www.roenhandmade.com/wp-content/themes/roen-minimal/style.css https://www.roenhandmade.com/wp-content/themes/roen-minimal/assets/css/roen.css https://www.roenhandmade.com/wp-content/themes/roen-minimal/assets/css/roen-product.css https://www.roenhandmade.com/wp-content/themes/roen-minimal/assets/js/category-pills.js; do
  timeout 10 curl -ksS -o /dev/null -w "${url}: %{size_download} bytes\n" "$url"
done
```
Expected: HTML+CSS+JS sum (excluding product images and the Storefront parent CSS) under 30KB. Total page weight including images and parent CSS should be well under 500KB; if not, dial back image sizes.

- [ ] **Step 8: Lighthouse Performance ≥85 mobile**

Manual test: open Chrome DevTools → Lighthouse → Mobile → Run.
Expected: Performance score ≥85.
If fails, the most likely culprits are: too-large product images (regenerate via `wp media regenerate`), unminified Inter font (switch to `display=swap` parameters), or Storefront's parent CSS bloat. Investigate in that order.

- [ ] **Step 9: Footer policy links resolve**

Run:
```bash
for path in /privacy-policy/ /refund_returns/; do
  timeout 10 curl -ksS -o /dev/null -w "$path -> %{http_code}\n" "https://www.roenhandmade.com${path}"
done
```
Expected: both return `200`.

- [ ] **Step 10: No console errors**

Manual test: open homepage and a product page in Chrome with DevTools → Console. Refresh. Expected: no red error messages. Yellow warnings from Storefront or WooCommerce are acceptable as long as we didn't introduce them.

- [ ] **Step 11: Tag the release**

```bash
cd /home/aialfred/alfred
git tag -a roen-minimal-v1.0.0 -m "roen-minimal child theme v1.0.0 — initial deploy"
```

---

## Verification Summary

When this plan is complete:
- `services/roen-minimal/` exists in the repo with full theme source
- `roenhandmade-wp` container has `roen-minimal` as the active theme
- Homepage at https://www.roenhandmade.com/ renders the L2 product-first layout with 7 existing products
- Product detail pages render the 60/40 layout with terracotta `add to cart`
- About page exists at https://www.roenhandmade.com/about/ with brand-voice copy
- All spec verification criteria pass
- Deploy is idempotent — re-running `deploy.sh` after edits ships changes safely

After this plan:
- The site is ready for Meta Shop reviewers (real branding, real product detail pages, working checkout)
- The next downstream plan is the Telegram intake bot + AI product analyzer + WooCommerce publisher (Chapter 1-4 of `.claude/plans/ok-so-here-s-the-wobbly-tarjan.md`)
