<?php
// services/alfred-seo/inc/rest/meta.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_action( 'rest_api_init', function () {
    register_rest_route( 'alfred-seo/v1', '/meta', array(
        'methods'             => 'POST',
        'callback'            => 'alfred_seo_rest_meta',
        'permission_callback' => 'alfred_seo_rest_permission_check',
    ) );
});

function alfred_seo_rest_meta( WP_REST_Request $req ) {
    $post_id   = absint( $req->get_param( 'post_id' ) );
    $meta_desc = $req->get_param( 'meta_description' );
    $og_title  = $req->get_param( 'og_title' );
    $og_desc   = $req->get_param( 'og_description' );
    $og_image  = $req->get_param( 'og_image' );

    if ( ! $post_id || ! get_post( $post_id ) ) {
        return new WP_Error( 'rest_not_found', 'post not found', array( 'status' => 404 ) );
    }
    if ( null !== $meta_desc ) { update_post_meta( $post_id, '_alfred_seo_meta_description', sanitize_text_field( $meta_desc ) ); }
    if ( null !== $og_title )  { update_post_meta( $post_id, '_alfred_seo_og_title', sanitize_text_field( $og_title ) ); }
    if ( null !== $og_desc )   { update_post_meta( $post_id, '_alfred_seo_og_description', sanitize_text_field( $og_desc ) ); }
    if ( null !== $og_image )  { update_post_meta( $post_id, '_alfred_seo_og_image', esc_url_raw( $og_image ) ); }

    return new WP_REST_Response( array( 'post_id' => $post_id, 'updated_at' => gmdate( 'c' ) ), 200 );
}
