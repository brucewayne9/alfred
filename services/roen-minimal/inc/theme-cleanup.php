<?php
/**
 * Strip Storefront's homepage components, header bar, and footer credit.
 *
 * Priority numbers below mirror Storefront's own registrations in:
 *   wp-content/themes/storefront/inc/storefront-template-hooks.php
 *   wp-content/themes/storefront/inc/woocommerce/storefront-woocommerce-template-hooks.php
 * Re-verify these on Storefront upgrades — a priority drift in the parent
 * silently no-ops the matching remove_action() call here.
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
add_action( 'after_setup_theme', 'roen_remove_storefront_homepage_actions', 11 );

/**
 * Replace Storefront's footer credit + remove the mobile handheld footer bar.
 */
function roen_remove_storefront_footer_credit() {
    remove_action( 'storefront_footer', 'storefront_credit', 20 );
    remove_action( 'wp_footer', 'storefront_handheld_footer_bar' );
}
add_action( 'after_setup_theme', 'roen_remove_storefront_footer_credit', 11 );

/**
 * Strip Storefront's prev/next product pagination on single-product pages.
 * It absolute-positions thumbnail arrows that float off the viewport edges
 * — visual noise our minimal layout doesn't want.
 */
function roen_remove_storefront_product_pagination() {
    remove_action( 'woocommerce_after_single_product_summary', 'storefront_product_pagination', 30 );
}
add_action( 'after_setup_theme', 'roen_remove_storefront_product_pagination', 11 );

/**
 * Force Storefront into a no-sidebar (full-width content) layout site-wide.
 * Sarah is not blogging — there's no Recent Posts, Recent Comments, Archives,
 * Categories rail to surface yet. Re-enable later by removing this filter
 * and registering widgets in sidebar-1.
 *
 * This also stops the sidebar template from being included by Storefront's
 * page templates, eliminating the empty <aside> from the DOM.
 */
add_filter( 'theme_mod_storefront_layout', function () { return 'content'; } );

/**
 * Unregister Storefront's `sidebar-1` and `sidebar-shop` so the widget area
 * cannot be populated through Appearance → Widgets while we're sidebar-less.
 */
function roen_unregister_default_sidebars() {
    unregister_sidebar( 'sidebar-1' );
    unregister_sidebar( 'sidebar-shop' );
}
add_action( 'widgets_init', 'roen_unregister_default_sidebars', 99 );

/**
 * Belt-and-braces: also strip the legacy storefront_sidebar() action call
 * for any template that bypasses the layout filter above.
 */
function roen_remove_sidebar_action() {
    remove_action( 'storefront_sidebar', 'storefront_get_sidebar', 10 );
}
add_action( 'after_setup_theme', 'roen_remove_sidebar_action', 11 );

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
add_action( 'after_setup_theme', 'roen_remove_storefront_header_elements', 11 );
