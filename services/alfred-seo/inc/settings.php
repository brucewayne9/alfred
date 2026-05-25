<?php
// services/alfred-seo/inc/settings.php

if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_default_settings() {
    return array(
        'site_slug'        => '',                    // e.g. "roen" — set by Alfred during onboarding
        'business_name'    => '',                    // e.g. "Roen Handmade"
        'business_type'    => 'Organization',        // Organization | LocalBusiness | SoftwareApplication | Service
        'social_handles'   => array(),               // ['twitter' => '@handle', 'instagram' => 'handle']
        'local_address'    => array(),               // LocalBusiness only: streetAddress, addressLocality, etc.
        'internal_links'   => array(),               // ['phrase' => 'url', ...]
        'alfred_endpoint'  => 'https://aialfred.groundrushcloud.com',
        'alt_text_enabled'  => true,
        'sitemap_enabled'   => true,
        'audio_post_type'   => 'podcast',
        'podcast_feed_url'  => '',
        'is_podcast_site'   => false,
    );
}

function alfred_seo_get_settings() {
    $opts = get_option( 'alfred_seo_settings', alfred_seo_default_settings() );
    // Backfill missing keys (forward-compatible additions).
    return array_merge( alfred_seo_default_settings(), is_array( $opts ) ? $opts : array() );
}

function alfred_seo_update_settings( array $patch ) {
    $current = alfred_seo_get_settings();
    update_option( 'alfred_seo_settings', array_merge( $current, $patch ) );
}
