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

    // Product gallery JS — only on single product pages.
    if ( function_exists( 'is_product' ) && is_product() ) {
        wp_enqueue_script(
            'roen-gallery',
            get_stylesheet_directory_uri() . '/assets/js/gallery.js',
            array(),
            $version,
            true
        );
    }

    // /pick landing page CSS — only loaded on that page template.
    if ( is_page_template( 'page-pick.php' ) ) {
        wp_enqueue_style(
            'roen-pick',
            get_stylesheet_directory_uri() . '/assets/css/roen-pick.css',
            array( 'roen-structure' ),
            $version
        );
    }
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
        'single_image_width'    => 1500,
    ) );
    // Note: deliberately NOT registering wc-product-gallery-* — we ship our
    // own gallery template (woocommerce/single-product/product-image.php) so
    // Flexslider/zoom/lightbox aren't enqueued on the single-product page.
}

/**
 * Override WC catalog image sizes — Storefront's theme support otherwise wins.
 * Source photos from the Telegram intake are ~721×1280 portrait, so we cap
 * thumbnail width at 600 for retina sharpness on a 4-up grid (~300px CSS).
 */
function roen_image_size_thumbnail( $size ) {
    return array( 'width' => 600, 'height' => 750, 'crop' => 1 );
}
add_filter( 'woocommerce_get_image_size_thumbnail', 'roen_image_size_thumbnail', 99 );
add_action( 'after_setup_theme', 'roen_theme_supports' );

/**
 * Single-product page: print a small uppercase category eyebrow above the title.
 * Hooks at priority 4 — woocommerce_template_single_title runs at 5, so we land
 * just before it. Falls back silently if the product has no category assigned.
 */
function roen_product_eyebrow() {
    if ( ! function_exists( 'is_product' ) || ! is_product() ) {
        return;
    }
    global $product;
    if ( ! $product instanceof WC_Product ) {
        return;
    }
    $terms = get_the_terms( $product->get_id(), 'product_cat' );
    if ( ! $terms || is_wp_error( $terms ) ) {
        return;
    }
    // Pick the most-specific category (deepest term) so "bracelet > beaded" picks "beaded".
    usort( $terms, function ( $a, $b ) { return $b->term_id - $a->term_id; } );
    $name = $terms[0]->name;
    echo '<p class="roen-single__eyebrow">' . esc_html( strtoupper( $name ) ) . '</p>';
}
add_action( 'woocommerce_single_product_summary', 'roen_product_eyebrow', 4 );

/**
 * Mobile: inline the long product description directly inside the summary
 * panel, right after the short description and before add-to-cart. On mobile
 * the default tabs widget pushes the real description below meta + tab chrome,
 * so the section reads as photo → title → short-desc → meta → tabs → desc,
 * which is the wrong order. We emit the description inline here and hide the
 * tabs widget on mobile via CSS (desktop keeps the tabs panel for breathing room).
 *
 * Priority 25 sits between woocommerce_template_single_excerpt (20) and
 * woocommerce_template_single_add_to_cart (30).
 */
function roen_inject_mobile_description() {
    if ( ! function_exists( 'is_product' ) || ! is_product() ) {
        return;
    }
    global $product;
    if ( ! $product instanceof WC_Product ) {
        return;
    }
    $raw = $product->get_description();
    if ( ! trim( wp_strip_all_tags( $raw ) ) ) {
        return;
    }
    $content = apply_filters( 'the_content', $raw );
    echo '<div class="roen-mobile-desc">' . $content . '</div>';
}
add_action( 'woocommerce_single_product_summary', 'roen_inject_mobile_description', 25 );

/**
 * Rename the related-products heading so the section reads as a continuation
 * of the same page, not a navigation jump (especially on mobile, where the
 * 2-col grid otherwise looks identical to the shop archive).
 */
function roen_related_products_args( $args ) {
    $args['heading'] = __( 'You might also like', 'roen-minimal' );
    return $args;
}
add_filter( 'woocommerce_output_related_products_args', 'roen_related_products_args', 20 );

// Fallback: the related.php template still calls this filter directly in
// current WC versions, so set it here too in case the args route is bypassed.
add_filter( 'woocommerce_product_related_products_heading', function () {
    return __( 'You might also like', 'roen-minimal' );
}, 20 );

/**
 * Storefront cleanup (header credits, default homepage components, etc.)
 */
require_once get_stylesheet_directory() . '/inc/theme-cleanup.php';

/**
 * Shared template helpers (wordmark SVG, cart count, fragment filter).
 */
require_once get_stylesheet_directory() . '/inc/template-helpers.php';

/**
 * Roen Bracelet Box — server-side glue.
 */
require_once get_stylesheet_directory() . '/inc/bracelet-box.php';

/**
 * Meta Shop checkout deep-link handler — parses /meta-checkout?products=ID:QTY,..&coupon=CODE.
 */
require_once get_stylesheet_directory() . '/inc/meta-checkout.php';

/**
 * SEO meta description + noindex injector. Reads assets/seo/meta_descriptions.json.
 */
require_once get_stylesheet_directory() . '/inc/seo-meta.php';

/**
 * Self-contained XML sitemap (WP core's /wp-sitemap.xml is 404ing on this install).
 * Routes: /sitemap.xml (index), /sitemap-{products,content,categories}.xml.
 */
require_once get_stylesheet_directory() . '/inc/seo-sitemap.php';
