<?php
// services/alfred-seo/inc/rest/transcript.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

/**
 * Dedicated transcript-write endpoint.
 *
 * POST /alfred-seo/v1/transcript
 *   { "post_id": <int>, "transcript": "<string>" }
 *
 * Writes to the post_meta key configured in alfred_seo_settings
 * (transcript_post_meta_key, default _rt_transcript). The /meta endpoint
 * whitelists only meta_description + og_* keys, so transcripts need their
 * own route. Auth is the same alfred-seo permission check (WP application
 * password as the alfred-seo user).
 */
add_action( 'rest_api_init', function () {
    register_rest_route( 'alfred-seo/v1', '/transcript', array(
        'methods'             => 'POST',
        'callback'            => 'alfred_seo_rest_transcript',
        'permission_callback' => 'alfred_seo_rest_permission_check',
        'args'                => array(
            'post_id'    => array( 'required' => true, 'type' => 'integer' ),
            'transcript' => array( 'required' => true, 'type' => 'string' ),
        ),
    ) );
});

function alfred_seo_rest_transcript( WP_REST_Request $req ) {
    $post_id    = absint( $req->get_param( 'post_id' ) );
    $transcript = (string) $req->get_param( 'transcript' );

    $post = get_post( $post_id );
    if ( ! $post_id || ! $post ) {
        return new WP_Error( 'rest_not_found', 'post not found', array( 'status' => 404 ) );
    }

    // Reject if the post isn't the configured audio post type — prevents accidental
    // writes to articles/products via this endpoint.
    $settings        = alfred_seo_get_settings();
    $audio_post_type = $settings['audio_post_type'] ?? 'podcast';
    if ( $post->post_type !== $audio_post_type ) {
        return new WP_Error(
            'rest_wrong_post_type',
            "transcript endpoint only accepts {$audio_post_type} posts, got {$post->post_type}",
            array( 'status' => 400 )
        );
    }

    // Plain-text transcripts. wp_kses_post allows safe HTML for the rare case
    // someone passes formatted text; the render filter strips tags anyway.
    $clean    = wp_kses_post( wp_unslash( $transcript ) );
    $meta_key = $settings['transcript_post_meta_key'] ?? '_rt_transcript';
    update_post_meta( $post_id, $meta_key, $clean );

    return new WP_REST_Response( array(
        'post_id'    => $post_id,
        'meta_key'   => $meta_key,
        'chars'      => strlen( $clean ),
        'updated_at' => gmdate( 'c' ),
    ), 200 );
}
