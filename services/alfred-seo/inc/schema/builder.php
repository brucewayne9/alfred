<?php
// services/alfred-seo/inc/schema/builder.php

if ( ! defined( 'ABSPATH' ) ) { exit; }

/**
 * Resolve which schema builder applies to the current WP query state,
 * call it, validate the result, and return the JSON-LD array (or null).
 */
function alfred_seo_build_schema_for_current_page() {
    $schemas        = array();
    $settings       = alfred_seo_get_settings();
    $audio_post_type = $settings['audio_post_type'] ?? 'podcast';

    // Org/LocalBusiness + WebSite always on the homepage.
    if ( is_front_page() || is_home() ) {
        $schemas[] = alfred_seo_schema_organization();
        $schemas[] = alfred_seo_schema_website();
        if ( ! empty( $settings['is_podcast_site'] ) ) {
            $schemas[] = alfred_seo_schema_podcast_series();
        }
    }
    // Per-page-type dispatch (each function defined in its own file).
    if ( function_exists( 'is_product' ) && is_product() ) {
        $schemas[] = alfred_seo_schema_product( get_queried_object_id() );
    } elseif ( is_singular( $audio_post_type ) ) {
        $schemas[] = alfred_seo_schema_podcast_episode( get_queried_object_id() );
    } elseif ( is_singular( 'post' ) ) {
        $schemas[] = alfred_seo_schema_article( get_queried_object_id() );
    } elseif ( function_exists( 'is_product_category' ) && is_product_category() ) {
        $schemas[] = alfred_seo_schema_collection( get_queried_object() );
    }
    // Always-on (except homepage which has its own).
    if ( ! is_front_page() && ! is_home() ) {
        $schemas[] = alfred_seo_schema_breadcrumb();
    }

    $faq = alfred_seo_schema_faq();
    if ( $faq ) { $schemas[] = $faq; }

    // Strip nulls + validate each.
    $schemas = array_filter( $schemas );
    $valid   = array_filter( $schemas, 'alfred_seo_validate_jsonld' );

    if ( empty( $valid ) ) { return null; }
    if ( count( $valid ) === 1 ) { return reset( $valid ); }

    return array(
        '@context' => 'https://schema.org',
        '@graph'   => array_values( array_map( function ( $s ) {
            unset( $s['@context'] );
            return $s;
        }, $valid ) ),
    );
}

/**
 * wp_head injection. Hooks added once all schema files are loaded.
 */
function alfred_seo_inject_schema_to_head() {
    $payload = alfred_seo_build_schema_for_current_page();
    if ( null === $payload ) { return; }
    echo "\n<script type=\"application/ld+json\">"
        . wp_json_encode( $payload, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE )
        . "</script>\n";
}
add_action( 'wp_head', 'alfred_seo_inject_schema_to_head', 10 );
