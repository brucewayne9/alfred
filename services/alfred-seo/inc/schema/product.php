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
