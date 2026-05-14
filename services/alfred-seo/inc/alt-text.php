<?php
// services/alfred-seo/inc/alt-text.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_action( 'add_attachment', 'alfred_seo_fill_alt_text_on_upload' );

function alfred_seo_fill_alt_text_on_upload( $attachment_id ) {
    $settings = alfred_seo_get_settings();
    if ( empty( $settings['alt_text_enabled'] ) ) { return; }

    // Skip non-images.
    if ( ! wp_attachment_is_image( $attachment_id ) ) { return; }
    // Skip if alt already set (manual override).
    if ( get_post_meta( $attachment_id, '_wp_attachment_image_alt', true ) ) { return; }

    $url = wp_get_attachment_url( $attachment_id );
    $alt = '';

    // Try Alfred orchestrator vision endpoint.
    $endpoint = trailingslashit( $settings['alfred_endpoint'] ) . 'wp-vision/alt-text';
    $resp = wp_remote_post( $endpoint, array(
        'timeout' => 10,
        'body'    => wp_json_encode( array( 'image_url' => $url, 'site_slug' => $settings['site_slug'] ) ),
        'headers' => array( 'Content-Type' => 'application/json' ),
    ) );
    if ( ! is_wp_error( $resp ) && 200 === wp_remote_retrieve_response_code( $resp ) ) {
        $body = json_decode( wp_remote_retrieve_body( $resp ), true );
        if ( ! empty( $body['alt'] ) ) { $alt = $body['alt']; }
    }

    // Local fallback: humanize filename.
    if ( ! $alt ) {
        $filename = pathinfo( get_attached_file( $attachment_id ), PATHINFO_FILENAME );
        $alt      = ucwords( str_replace( array( '-', '_' ), ' ', $filename ) );
    }

    update_post_meta( $attachment_id, '_wp_attachment_image_alt', mb_substr( $alt, 0, 125 ) );
}
