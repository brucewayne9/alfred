<?php
// services/alfred-seo/inc/schema/collection.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_schema_collection( $term ) {
    if ( ! $term || is_wp_error( $term ) ) { return null; }

    // Pull top 12 products in this category for ItemList.
    $items = array();
    if ( function_exists( 'wc_get_products' ) ) {
        $products = wc_get_products( array(
            'category' => array( $term->slug ),
            'limit'    => 12,
            'status'   => 'publish',
        ) );
        $position = 1;
        foreach ( $products as $product ) {
            $items[] = array(
                '@type'    => 'ListItem',
                'position' => $position++,
                'url'      => get_permalink( $product->get_id() ),
                'name'     => $product->get_name(),
            );
        }
    }

    $schema = array(
        '@context'   => 'https://schema.org',
        '@type'      => 'CollectionPage',
        'name'       => $term->name,
        'description' => wp_strip_all_tags( $term->description ),
        'url'        => get_term_link( $term ),
    );

    if ( $items ) {
        $schema['mainEntity'] = array(
            '@type'           => 'ItemList',
            'numberOfItems'   => count( $items ),
            'itemListElement' => $items,
        );
    }
    return $schema;
}
