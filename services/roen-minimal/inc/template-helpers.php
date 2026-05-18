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
 * Return the product categories worth surfacing in the pill nav.
 *
 * Pulls all non-empty product_cat terms, then drops the WP-default
 * "Uncategorized" bucket — products without a real category land there
 * silently and we never want it as a browsable filter.
 *
 * Used by both front-page.php and woocommerce/archive-product.php so the
 * homepage and the shop archive can never drift out of sync on this rule.
 */
function roen_browseable_product_cats() {
    $cats = get_terms( array(
        'taxonomy'   => 'product_cat',
        'hide_empty' => true,
        'orderby'    => 'name',
        'order'      => 'ASC',
    ) );
    if ( is_wp_error( $cats ) || empty( $cats ) ) {
        return array();
    }
    return array_values( array_filter( $cats, function ( $cat ) {
        return $cat->slug !== 'uncategorized'
            && strtolower( $cat->name ) !== 'uncategorized';
    } ) );
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

/**
 * Meta (Facebook) domain verification tag.
 *
 * Required for the Roen Meta Shop to activate "Checkout on another website."
 * Business Verification + Catalog are green; this is the last gate.
 */
function roen_emit_meta_domain_verification() {
    echo '<meta name="facebook-domain-verification" content="3d7nx9vbpsjvju91wqzyn6dnl9lteg" />' . "\n";
}
add_action( 'wp_head', 'roen_emit_meta_domain_verification', 1 );
add_filter( 'woocommerce_add_to_cart_fragments', 'roen_cart_count_fragment' );
