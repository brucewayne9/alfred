<?php
/**
 * Meta Shop checkout deep-link handler.
 *
 * Meta's "Checkout on another website" Shop sends buyers to a URL pattern like
 *   https://roenhandmade.com/meta-checkout?products=ID:QTY,ID:QTY&coupon=CODE
 *
 * This handler parses that, builds the WC cart server-side, applies the coupon
 * (if any), then redirects to /checkout/. Tolerant of both raw WC IDs (757) and
 * the retailer_id prefix used in our Meta catalog (roen-757).
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

/**
 * Register the rewrite rule so /meta-checkout matches without a real page.
 */
add_action( 'init', function () {
    add_rewrite_rule( '^meta-checkout/?$', 'index.php?roen_meta_checkout=1', 'top' );
});

add_filter( 'query_vars', function ( $vars ) {
    $vars[] = 'roen_meta_checkout';
    return $vars;
});

/**
 * Strip the catalog retailer-id prefix if present, return numeric WC ID.
 * "roen-757" -> "757". "757" -> "757". "garbage" -> "".
 */
function roen_meta_parse_product_id( $raw ) {
    $raw = trim( (string) $raw );
    if ( strpos( $raw, 'roen-' ) === 0 ) {
        $raw = substr( $raw, 5 );
    }
    return ctype_digit( $raw ) ? $raw : '';
}

/**
 * Parse "ID:QTY,ID:QTY" into [[product_id, qty], ...].
 * Default qty = 1 when missing or invalid.
 */
function roen_meta_parse_products( $products_param ) {
    $out = array();
    if ( ! is_string( $products_param ) || $products_param === '' ) {
        return $out;
    }
    foreach ( explode( ',', $products_param ) as $entry ) {
        $entry = trim( $entry );
        if ( $entry === '' ) {
            continue;
        }
        $parts = explode( ':', $entry );
        $pid = roen_meta_parse_product_id( $parts[0] );
        if ( $pid === '' ) {
            continue;
        }
        $qty = isset( $parts[1] ) ? intval( $parts[1] ) : 1;
        if ( $qty < 1 ) {
            $qty = 1;
        }
        $out[] = array( (int) $pid, $qty );
    }
    return $out;
}

/**
 * Main handler — fires early so we can short-circuit before template rendering.
 */
add_action( 'template_redirect', function () {
    if ( ! intval( get_query_var( 'roen_meta_checkout' ) ) ) {
        return;
    }
    if ( ! function_exists( 'WC' ) ) {
        wp_die( 'WooCommerce is not active.', 'roen-meta-checkout', array( 'response' => 500 ) );
    }

    $products_raw = isset( $_GET['products'] ) ? wp_unslash( $_GET['products'] ) : '';
    $coupon_raw   = isset( $_GET['coupon'] )   ? wp_unslash( $_GET['coupon'] )   : '';

    $items = roen_meta_parse_products( $products_raw );
    if ( empty( $items ) ) {
        // Nothing valid — just send them to the catalog so they aren't dumped on a blank page.
        wp_safe_redirect( wc_get_page_permalink( 'shop' ) );
        exit;
    }

    // Start the customer session so the cart persists into checkout.
    if ( ! WC()->session ) {
        return;
    }
    if ( ! WC()->session->has_session() ) {
        WC()->session->set_customer_session_cookie( true );
    }
    if ( ! WC()->cart ) {
        return;
    }

    // Clear any prior cart so Meta deep-links are deterministic.
    WC()->cart->empty_cart();

    $added_any = false;
    foreach ( $items as $pair ) {
        list( $pid, $qty ) = $pair;
        $product = wc_get_product( $pid );
        if ( ! $product || ! $product->is_purchasable() || ! $product->is_in_stock() ) {
            continue;
        }
        $cart_item_key = WC()->cart->add_to_cart( $pid, $qty );
        if ( $cart_item_key ) {
            $added_any = true;
        }
    }

    if ( ! $added_any ) {
        wp_safe_redirect( wc_get_page_permalink( 'shop' ) );
        exit;
    }

    // Apply coupon if provided.
    if ( $coupon_raw !== '' ) {
        $coupon = sanitize_text_field( $coupon_raw );
        if ( $coupon !== '' && ! WC()->cart->has_discount( $coupon ) ) {
            WC()->cart->apply_coupon( $coupon );
        }
    }

    // Persist + redirect.
    WC()->cart->calculate_totals();
    wp_safe_redirect( wc_get_checkout_url() );
    exit;
}, 1 );

/**
 * Flush rewrite rules once after theme update so /meta-checkout starts matching.
 * Stored option prevents reflushing on every load.
 */
add_action( 'after_switch_theme', function () {
    flush_rewrite_rules();
});
add_action( 'init', function () {
    if ( get_option( 'roen_meta_checkout_rewrites_flushed' ) !== 'v1' ) {
        flush_rewrite_rules();
        update_option( 'roen_meta_checkout_rewrites_flushed', 'v1' );
    }
}, 99 );
