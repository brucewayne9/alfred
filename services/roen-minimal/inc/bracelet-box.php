<?php
/**
 * Roen Bracelet Box — server-side glue.
 *
 *  - Recomputes the hidden 'bracelet-box' product's stock_quantity as
 *    floor(eligible_in_stock_bracelets / 5). Runs on a 15-minute wp-cron
 *    AND on relevant product hooks (so the homepage/landing-page state
 *    flips instantly when Sarah publishes or sells a piece).
 *  - Bounds cart quantity for that SKU to current stock (so a customer
 *    can't add 5 boxes when only 2 are available).
 *
 * The eligible count is cached in a 60-second transient to avoid
 * hammering the DB on every page render.
 */

if ( ! defined( 'ABSPATH' ) ) { exit; }

const ROEN_BOX_SKU = 'bracelet-box';
const ROEN_BOX_TRANSIENT = 'roen_box_eligible_count';
const ROEN_BOX_TRANSIENT_TTL = 60;
const ROEN_BOX_BUNDLE_SIZE = 5;

/**
 * Count published bracelets that are in stock.
 *
 * Roen's catalog uses manage_stock=false for handmade 1-of-1 pieces, so
 * the only reliable signal of availability is _stock_status. We do NOT
 * also check _stock because it's unset for manage-stock-off products.
 * Cached in a transient.
 */
function roen_box_eligible_count(): int {
    $cached = get_transient( ROEN_BOX_TRANSIENT );
    if ( false !== $cached ) {
        return (int) $cached;
    }

    global $wpdb;
    $sql = "
        SELECT COUNT(DISTINCT p.ID)
        FROM {$wpdb->posts} p
        INNER JOIN {$wpdb->term_relationships} tr ON tr.object_id = p.ID
        INNER JOIN {$wpdb->term_taxonomy} tt ON tt.term_taxonomy_id = tr.term_taxonomy_id
        INNER JOIN {$wpdb->terms} t ON t.term_id = tt.term_id
        INNER JOIN {$wpdb->postmeta} sm ON sm.post_id = p.ID AND sm.meta_key = '_stock_status'
        WHERE p.post_type = 'product'
          AND p.post_status = 'publish'
          AND tt.taxonomy = 'product_cat'
          AND t.slug = 'bracelet'
          AND sm.meta_value = 'instock'
    ";
    $count = (int) $wpdb->get_var( $sql );

    set_transient( ROEN_BOX_TRANSIENT, $count, ROEN_BOX_TRANSIENT_TTL );
    return $count;
}

/**
 * Compute the box's stock quantity and write it back to the bracelet-box product.
 * Excludes the box product itself from the eligible count (it's not a bracelet,
 * but defensive — also prevents recursion if stock-set hooks fire on the box).
 */
function roen_box_recompute_stock(): int {
    $box_id = wc_get_product_id_by_sku( ROEN_BOX_SKU );
    if ( ! $box_id ) {
        return 0;
    }

    delete_transient( ROEN_BOX_TRANSIENT );  // force fresh count
    $eligible = roen_box_eligible_count();
    $stock = (int) floor( $eligible / ROEN_BOX_BUNDLE_SIZE );

    $box = wc_get_product( $box_id );
    if ( ! $box ) {
        return 0;
    }
    $box->set_manage_stock( true );
    $box->set_stock_quantity( $stock );
    $box->set_stock_status( $stock > 0 ? 'instock' : 'outofstock' );
    $box->save();

    return $stock;
}

/**
 * Schedule the recompute every 15 minutes.
 */
add_action( 'init', function () {
    if ( ! wp_next_scheduled( 'roen_box_stock_recompute_cron' ) ) {
        wp_schedule_event( time() + 60, 'roen_box_15min', 'roen_box_stock_recompute_cron' );
    }
} );

add_filter( 'cron_schedules', function ( $sched ) {
    $sched['roen_box_15min'] = array( 'interval' => 900, 'display' => 'Every 15 minutes' );
    return $sched;
} );

add_action( 'roen_box_stock_recompute_cron', 'roen_box_recompute_stock' );

/**
 * Instant recompute when individual product stock changes.
 * Skip if the product whose stock changed IS the box itself, to avoid recursion.
 */
add_action( 'woocommerce_product_set_stock', function ( $product ) {
    if ( $product && $product->get_sku() !== ROEN_BOX_SKU ) {
        roen_box_recompute_stock();
    }
} );

/**
 * Instant recompute when a bracelet product's status changes
 * (publish, trash, draft, etc.). Only fires on actual transitions.
 */
add_action( 'transition_post_status', function ( $new, $old, $post ) {
    if ( $post && $post->post_type === 'product' && $new !== $old ) {
        roen_box_recompute_stock();
    }
}, 10, 3 );

/**
 * Bound cart qty for the box to current stock. WooCommerce normally lets
 * users type any number; we cap at the current floor.
 */
add_filter( 'woocommerce_quantity_input_args', function ( $args, $product ) {
    if ( $product && $product->get_sku() === ROEN_BOX_SKU ) {
        $args['max_value'] = max( 1, (int) $product->get_stock_quantity() );
        $args['min_value'] = 1;
    }
    return $args;
}, 10, 2 );

/**
 * On cancel/refund of an order containing the box SKU: trigger a stock
 * recompute and add an admin note. The bot's pick records (in jewelry.db)
 * remain in their last state — Mike or Sarah can manually mark the
 * specific pick session as cancelled if needed. For v1 the customer-facing
 * fix is the stock recompute; pick-record cleanup is best-effort manual.
 */
add_action( 'woocommerce_order_status_cancelled', 'roen_box_handle_order_void' );
add_action( 'woocommerce_order_status_refunded',  'roen_box_handle_order_void' );

function roen_box_handle_order_void( $order_id ) {
    $order = wc_get_order( $order_id );
    if ( ! $order ) {
        return;
    }

    $has_box = false;
    foreach ( $order->get_items() as $item ) {
        $product = $item->get_product();
        if ( $product && $product->get_sku() === ROEN_BOX_SKU ) {
            $has_box = true;
            break;
        }
    }
    if ( ! $has_box ) {
        return;
    }

    roen_box_recompute_stock();

    $order->add_order_note( sprintf(
        'Box order #%d voided. Reserved bracelets may need manual restock — check the bot pick record.',
        $order_id
    ) );
}

/**
 * WC orders admin column — flags which orders contain the bracelet-box SKU
 * with a 📦 icon and the qty. Lets Mike scan the list for box-containing
 * orders without opening each one.
 */
add_filter( 'manage_edit-shop_order_columns', function ( $cols ) {
    $cols['roen_box_pick'] = '📦 Box';
    return $cols;
} );

add_action( 'manage_shop_order_posts_custom_column', function ( $col, $post_id ) {
    if ( $col !== 'roen_box_pick' ) {
        return;
    }
    $order = wc_get_order( $post_id );
    if ( ! $order ) {
        return;
    }
    foreach ( $order->get_items() as $item ) {
        $product = $item->get_product();
        if ( $product && $product->get_sku() === ROEN_BOX_SKU ) {
            $qty = (int) $item->get_quantity();
            echo '<span title="contains bracelet-box, qty ' . esc_attr( $qty ) . '">📦 ' . esc_html( $qty ) . '</span>';
            return;
        }
    }
    echo '—';
}, 10, 2 );
