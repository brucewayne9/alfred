<?php
// services/alfred-seo/inc/schema/audio-object.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_schema_audio_object( $post_id ) {
    // 1. Explicit _rt_audio_url always wins (override hatch for non-Sonaar sites
    //    like Roen, and for any case where Mike wants to override Sonaar's value).
    $audio_url = get_post_meta( $post_id, '_rt_audio_url', true );
    $duration  = get_post_meta( $post_id, '_rt_audio_duration', true );
    $bytes     = get_post_meta( $post_id, '_rt_audio_bytes', true );
    $mime      = get_post_meta( $post_id, '_rt_audio_mime', true );

    // 2. Fall back to Sonaar's track_mp3_podcast attachment for fields not
    //    already set by _rt_* meta. wp_get_attachment_metadata handles the
    //    PHP-serialized _wp_attachment_metadata array transparently.
    if ( empty( $audio_url ) ) {
        $att_id = get_post_meta( $post_id, 'track_mp3_podcast', true );
        if ( $att_id ) {
            $audio_url = wp_get_attachment_url( (int) $att_id );
            $meta      = wp_get_attachment_metadata( (int) $att_id );
            if ( is_array( $meta ) ) {
                if ( empty( $bytes )    && isset( $meta['filesize'] ) )  { $bytes    = $meta['filesize']; }
                if ( empty( $mime )     && isset( $meta['mime_type'] ) ) { $mime     = $meta['mime_type']; }
                if ( empty( $duration ) && isset( $meta['length'] ) )    { $duration = alfred_seo_iso8601_duration( (int) $meta['length'] ); }
            }
        }
    }

    if ( empty( $audio_url ) ) { return null; }
    $mime = $mime ?: 'audio/mpeg';

    $schema = array(
        '@context'       => 'https://schema.org',
        '@type'          => 'AudioObject',
        '@id'            => get_permalink( $post_id ) . '#audio',
        'contentUrl'     => esc_url_raw( $audio_url ),
        'encodingFormat' => $mime,
        'name'           => wp_strip_all_tags( get_the_title( $post_id ) ),
    );
    if ( $duration ) { $schema['duration']    = $duration; }
    if ( $bytes )    { $schema['contentSize'] = is_numeric( $bytes ) ? (int) $bytes : $bytes; }
    return $schema;
}

/**
 * Convert raw seconds to ISO-8601 duration (PT#H#M#S). Schema.org expects
 * ISO-8601; Sonaar stores duration as raw seconds in attachment metadata's
 * "length" key.
 */
function alfred_seo_iso8601_duration( $seconds ) {
    $seconds = (int) $seconds;
    if ( $seconds <= 0 ) { return ''; }
    $h = intdiv( $seconds, 3600 );
    $m = intdiv( $seconds % 3600, 60 );
    $s = $seconds % 60;
    $out = 'PT';
    if ( $h ) { $out .= $h . 'H'; }
    if ( $m ) { $out .= $m . 'M'; }
    if ( $s || ( ! $h && ! $m ) ) { $out .= $s . 'S'; }
    return $out;
}
