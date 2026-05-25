<?php
// services/alfred-seo/inc/schema/audio-object.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_schema_audio_object( $post_id ) {
    $audio_url = get_post_meta( $post_id, '_rt_audio_url', true );
    if ( empty( $audio_url ) ) { return null; }

    $duration = get_post_meta( $post_id, '_rt_audio_duration', true );  // ISO-8601, e.g. PT42M18S
    $bytes    = get_post_meta( $post_id, '_rt_audio_bytes', true );
    $mime     = get_post_meta( $post_id, '_rt_audio_mime', true ) ?: 'audio/mpeg';

    $schema = array(
        '@context'       => 'https://schema.org',
        '@type'          => 'AudioObject',
        '@id'            => get_permalink( $post_id ) . '#audio',
        'contentUrl'     => esc_url_raw( $audio_url ),
        'encodingFormat' => $mime,
        'name'           => wp_strip_all_tags( get_the_title( $post_id ) ),
    );
    if ( $duration ) { $schema['duration']    = $duration; }
    if ( $bytes )    { $schema['contentSize'] = $bytes; }
    return $schema;
}
