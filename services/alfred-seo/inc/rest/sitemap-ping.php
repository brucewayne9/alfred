<?php
// services/alfred-seo/inc/rest/sitemap-ping.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_action( 'rest_api_init', function () {
    register_rest_route( 'alfred-seo/v1', '/sitemap-ping', array(
        'methods'             => 'POST',
        'callback'            => 'alfred_seo_rest_sitemap_ping',
        'permission_callback' => 'alfred_seo_rest_permission_check',
    ) );
});

function alfred_seo_rest_sitemap_ping( WP_REST_Request $req ) {
    $sitemap = home_url( '/alfred-sitemap.xml' );
    $google  = wp_remote_get( 'https://www.google.com/ping?sitemap=' . urlencode( $sitemap ), array( 'timeout' => 5 ) );
    $bing    = wp_remote_get( 'https://www.bing.com/ping?sitemap=' . urlencode( $sitemap ), array( 'timeout' => 5 ) );
    return new WP_REST_Response( array(
        'sitemap_url' => $sitemap,
        'google'      => is_wp_error( $google ) ? 'error' : wp_remote_retrieve_response_code( $google ),
        'bing'        => is_wp_error( $bing ) ? 'error' : wp_remote_retrieve_response_code( $bing ),
        'pinged_at'   => gmdate( 'c' ),
    ), 200 );
}
