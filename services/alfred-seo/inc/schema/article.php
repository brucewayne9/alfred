<?php
// services/alfred-seo/inc/schema/article.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_schema_article( $post_id ) {
    $post = get_post( $post_id );
    if ( ! $post || 'publish' !== $post->post_status ) { return null; }

    $settings  = alfred_seo_get_settings();
    $author    = get_userdata( $post->post_author );
    $image_url = get_the_post_thumbnail_url( $post_id, 'full' );

    $schema = array(
        '@context'      => 'https://schema.org',
        '@type'         => 'Article',
        'headline'      => wp_strip_all_tags( $post->post_title ),
        'description'   => wp_strip_all_tags( get_the_excerpt( $post ) ),
        'datePublished' => mysql2date( 'c', $post->post_date_gmt, false ),
        'dateModified'  => mysql2date( 'c', $post->post_modified_gmt, false ),
        'url'           => get_permalink( $post ),
        'mainEntityOfPage' => array(
            '@type' => 'WebPage',
            '@id'   => get_permalink( $post ),
        ),
        'author'        => array(
            '@type' => 'Person',
            'name'  => $author ? $author->display_name : ( $settings['business_name'] ?: get_bloginfo( 'name' ) ),
        ),
        'publisher'     => array(
            '@type' => 'Organization',
            'name'  => $settings['business_name'] ?: get_bloginfo( 'name' ),
        ),
    );
    if ( $image_url ) { $schema['image'] = $image_url; }
    return $schema;
}
