<?php
// services/alfred-seo/inc/rest/internal-links.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_action( 'rest_api_init', function () {
    register_rest_route( 'alfred-seo/v1', '/internal-links', array(
        'methods'             => 'POST',
        'callback'            => 'alfred_seo_rest_internal_links',
        'permission_callback' => 'alfred_seo_rest_permission_check',
    ) );
});

function alfred_seo_rest_internal_links( WP_REST_Request $req ) {
    $map = $req->get_param( 'links' );
    if ( ! is_array( $map ) ) {
        return new WP_Error( 'rest_bad_request', 'links must be object phrase->url', array( 'status' => 400 ) );
    }
    $clean = array();
    foreach ( $map as $phrase => $url ) {
        $p = sanitize_text_field( $phrase );
        $u = esc_url_raw( $url );
        if ( $p && $u ) { $clean[ $p ] = $u; }
    }
    alfred_seo_update_settings( array( 'internal_links' => $clean ) );
    return new WP_REST_Response( array( 'count' => count( $clean ), 'updated_at' => gmdate( 'c' ) ), 200 );
}
