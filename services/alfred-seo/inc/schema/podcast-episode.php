<?php
// services/alfred-seo/inc/schema/podcast-episode.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_schema_podcast_episode( $post_id ) {
    $settings  = alfred_seo_get_settings();
    $post_type = get_post_type( $post_id );
    $expected  = $settings['audio_post_type'] ?? 'podcast';
    if ( $post_type !== $expected ) { return null; }

    $post = get_post( $post_id );
    if ( ! $post || 'publish' !== $post->post_status ) { return null; }

    $duration       = get_post_meta( $post_id, '_rt_audio_duration', true );
    $episode_number = get_post_meta( $post_id, '_rt_episode_number', true );
    $season_number  = get_post_meta( $post_id, '_rt_season_number', true );
    $image_url      = get_the_post_thumbnail_url( $post_id, 'full' ) ?: ( $settings['default_image_url'] ?? '' );
    $audio          = alfred_seo_schema_audio_object( $post_id );

    $schema = array(
        '@context'      => 'https://schema.org',
        '@type'         => 'PodcastEpisode',
        '@id'           => get_permalink( $post ) . '#episode',
        'name'          => wp_strip_all_tags( $post->post_title ),
        'description'   => wp_strip_all_tags( get_the_excerpt( $post ) ),
        'url'           => get_permalink( $post ),
        'datePublished' => mysql2date( 'c', $post->post_date_gmt, false ),
        'partOfSeries'  => array(
            '@type' => 'PodcastSeries',
            '@id'   => home_url( '/#podcastseries' ),
            'name'  => $settings['business_name'] ?? get_bloginfo( 'name' ),
        ),
    );
    if ( $episode_number )      { $schema['episodeNumber'] = $episode_number; }
    if ( $season_number )       { $schema['partOfSeason']  = array( '@type' => 'PodcastSeason', 'seasonNumber' => $season_number ); }
    if ( $duration )            { $schema['timeRequired']  = $duration; }
    if ( $image_url )           { $schema['image']         = $image_url; }
    if ( $audio )               { unset( $audio['@context'] ); $schema['associatedMedia'] = $audio; }
    return $schema;
}
