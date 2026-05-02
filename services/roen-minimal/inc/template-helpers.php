<?php
/**
 * Shared template helpers for roen-minimal.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

/**
 * Return the inlined wordmark SVG markup, memoized per request.
 *
 * The SVG never changes between deploys, so file_get_contents() runs at
 * most once per page render even if the wordmark appears in both header
 * and footer.
 */
function roen_wordmark_svg() {
    static $svg = null;
    if ( $svg === null ) {
        $svg = (string) file_get_contents( get_stylesheet_directory() . '/assets/img/roen-wordmark.svg' );
    }
    return $svg;
}

/**
 * Live cart-contents count, safe for any request context.
 *
 * WC()->cart is null on REST, login, and cron requests. This guard returns
 * 0 in those cases instead of fataling.
 */
function roen_cart_count() {
    if ( ! function_exists( 'WC' ) || ! WC()->cart ) {
        return 0;
    }
    return (int) WC()->cart->get_cart_contents_count();
}

/**
 * Refresh the header cart count via WooCommerce's AJAX fragments system.
 *
 * Without this, clicking add-to-cart updates Storefront's default count
 * but leaves our .roen-cart-count span stale until the next full reload.
 */
function roen_cart_count_fragment( $fragments ) {
    ob_start();
    ?><span class="roen-cart-count" aria-live="polite"><?php echo (int) roen_cart_count(); ?></span><?php
    $fragments['span.roen-cart-count'] = ob_get_clean();
    return $fragments;
}
add_filter( 'woocommerce_add_to_cart_fragments', 'roen_cart_count_fragment' );
