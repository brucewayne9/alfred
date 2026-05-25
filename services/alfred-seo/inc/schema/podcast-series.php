<?php
// services/alfred-seo/inc/schema/podcast-series.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_schema_podcast_series() {
    $settings = alfred_seo_get_settings();
    $name     = $settings['business_name'] ?? '';
    if ( empty( $name ) ) { return null; }

    $tagline  = $settings['site_tagline'] ?? get_bloginfo( 'description' );
    $feed_url = $settings['podcast_feed_url'] ?? get_feed_link();

    return array(
        '@context'    => 'https://schema.org',
        '@type'       => 'PodcastSeries',
        '@id'         => home_url( '/#podcastseries' ),
        'name'        => $name,
        'url'         => home_url( '/' ),
        'description' => wp_strip_all_tags( $tagline ),
        'webFeed'     => $feed_url,
        'inLanguage'  => get_bloginfo( 'language' ),
        'image'       => $settings['default_image_url'] ?? '',
    );
}
