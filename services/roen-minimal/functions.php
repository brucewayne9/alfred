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
 * Storefront cleanup (header credits, default homepage components, etc.)
 */
require_once get_stylesheet_directory() . '/inc/theme-cleanup.php';

/**
 * Shared template helpers (wordmark SVG, cart count, fragment filter).
 */
require_once get_stylesheet_directory() . '/inc/template-helpers.php';
