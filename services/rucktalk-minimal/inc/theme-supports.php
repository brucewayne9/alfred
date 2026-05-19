<?php
/**
 * Theme supports — rucktalk-minimal.
 *
 * Sonaar parent already declares custom-logo, post-thumbnails, title-tag,
 * html5 (per the 2026-05-19 audit at docs/superpowers/audits/2026-05-19-
 * sonaar-theme-audit.md). We only add WooCommerce support — the parent
 * doesn't include WC, but our /shop is critical.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

function rucktalk_after_setup_theme() {
    // WooCommerce integration so the shop renders inside the child theme.
    add_theme_support( 'woocommerce' );
    add_theme_support( 'wc-product-gallery-zoom' );
    add_theme_support( 'wc-product-gallery-lightbox' );
    add_theme_support( 'wc-product-gallery-slider' );
}
add_action( 'after_setup_theme', 'rucktalk_after_setup_theme', 20 );
