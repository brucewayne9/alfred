<?php
// services/alfred-seo/inc/schema/breadcrumb.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_schema_breadcrumb() {
    $items = array();
    $position = 1;

    // Home is always first.
    $items[] = array(
        '@type'    => 'ListItem',
        'position' => $position++,
        'name'     => 'Home',
        'item'     => home_url( '/' ),
    );

    if ( function_exists( 'is_product' ) && is_product() ) {
        $product = wc_get_product( get_queried_object_id() );
        $cats    = get_the_terms( $product->get_id(), 'product_cat' );
        if ( $cats && ! is_wp_error( $cats ) ) {
            $primary = $cats[0];
            $items[] = array(
                '@type'    => 'ListItem',
                'position' => $position++,
                'name'     => $primary->name,
                'item'     => get_term_link( $primary ),
            );
        }
        $items[] = array(
            '@type'    => 'ListItem',
            'position' => $position++,
            'name'     => $product->get_name(),
            'item'     => get_permalink( $product->get_id() ),
        );
    } elseif ( is_singular( 'post' ) ) {
        $post = get_queried_object();
        $cats = get_the_category( $post->ID );
        if ( $cats ) {
            $items[] = array(
                '@type'    => 'ListItem',
                'position' => $position++,
                'name'     => $cats[0]->name,
                'item'     => get_category_link( $cats[0] ),
            );
        }
        $items[] = array(
            '@type'    => 'ListItem',
            'position' => $position++,
            'name'     => get_the_title( $post ),
            'item'     => get_permalink( $post ),
        );
    } elseif ( is_page() ) {
        $page = get_queried_object();
        if ( $page->post_parent ) {
            $ancestors = array_reverse( get_post_ancestors( $page->ID ) );
            foreach ( $ancestors as $anc_id ) {
                $items[] = array(
                    '@type'    => 'ListItem',
                    'position' => $position++,
                    'name'     => get_the_title( $anc_id ),
                    'item'     => get_permalink( $anc_id ),
                );
            }
        }
        $items[] = array(
            '@type'    => 'ListItem',
            'position' => $position++,
            'name'     => get_the_title( $page ),
            'item'     => get_permalink( $page ),
        );
    } elseif ( is_category() || is_tax() ) {
        $term = get_queried_object();
        $items[] = array(
            '@type'    => 'ListItem',
            'position' => $position++,
            'name'     => $term->name,
            'item'     => get_term_link( $term ),
        );
    }

    if ( count( $items ) < 2 ) { return null; }

    return array(
        '@context'        => 'https://schema.org',
        '@type'           => 'BreadcrumbList',
        'itemListElement' => $items,
    );
}
