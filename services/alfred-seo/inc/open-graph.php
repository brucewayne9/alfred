<?php
// services/alfred-seo/inc/open-graph.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_render_open_graph() {
    $settings = alfred_seo_get_settings();

    // Resolve title, description, image, URL, og:type per page type.
    $url   = '';
    $title = get_bloginfo( 'name' );
    $desc  = get_bloginfo( 'description' );
    $image = get_site_icon_url( 1200 );
    $type  = 'website';

    if ( is_singular() ) {
        $obj   = get_queried_object();
        $url   = get_permalink( $obj );
        $title = get_the_title( $obj );
        $desc  = get_the_excerpt( $obj ) ?: $desc;
        if ( has_post_thumbnail( $obj ) ) {
            $image = get_the_post_thumbnail_url( $obj, 'full' );
        }
        $type = ( function_exists( 'is_product' ) && is_product() ) ? 'product' : 'article';
    } elseif ( is_category() || is_tax() ) {
        $term  = get_queried_object();
        $url   = get_term_link( $term );
        $title = $term->name;
        $desc  = wp_strip_all_tags( $term->description ) ?: $desc;
    } else {
        $url = home_url( $_SERVER['REQUEST_URI'] ?? '/' );
    }

    // Custom-field override (Alfred can push a specific OG image/title/desc).
    if ( is_singular() ) {
        $obj = get_queried_object();
        $override_title = get_post_meta( $obj->ID, '_alfred_seo_og_title', true );
        $override_desc  = get_post_meta( $obj->ID, '_alfred_seo_og_description', true );
        $override_img   = get_post_meta( $obj->ID, '_alfred_seo_og_image', true );
        if ( $override_title ) { $title = $override_title; }
        if ( $override_desc )  { $desc  = $override_desc; }
        if ( $override_img )   { $image = $override_img; }
    }

    $title = wp_strip_all_tags( $title );
    $desc  = wp_strip_all_tags( $desc );

    $tags = array(
        sprintf( '<meta property="og:type" content="%s" />', esc_attr( $type ) ),
        sprintf( '<meta property="og:title" content="%s" />', esc_attr( $title ) ),
        sprintf( '<meta property="og:description" content="%s" />', esc_attr( $desc ) ),
        sprintf( '<meta property="og:url" content="%s" />', esc_url( $url ) ),
        sprintf( '<meta property="og:site_name" content="%s" />', esc_attr( $settings['business_name'] ?: get_bloginfo( 'name' ) ) ),
    );
    if ( $image ) {
        $tags[] = sprintf( '<meta property="og:image" content="%s" />', esc_url( $image ) );
    }

    // Twitter Card.
    $twitter_handle = $settings['social_handles']['twitter'] ?? '';
    $tags[] = '<meta name="twitter:card" content="summary_large_image" />';
    $tags[] = sprintf( '<meta name="twitter:title" content="%s" />', esc_attr( $title ) );
    $tags[] = sprintf( '<meta name="twitter:description" content="%s" />', esc_attr( $desc ) );
    if ( $image ) {
        $tags[] = sprintf( '<meta name="twitter:image" content="%s" />', esc_url( $image ) );
    }
    if ( $twitter_handle ) {
        $tags[] = sprintf( '<meta name="twitter:site" content="@%s" />', esc_attr( ltrim( $twitter_handle, '@' ) ) );
    }

    echo "\n" . implode( "\n", $tags ) . "\n";
}
add_action( 'wp_head', 'alfred_seo_render_open_graph', 5 );
