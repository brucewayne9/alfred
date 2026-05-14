<?php
// services/alfred-seo/inc/rest/content.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_action( 'rest_api_init', function () {
    register_rest_route( 'alfred-seo/v1', '/content', array(
        'methods'             => 'POST',
        'callback'            => 'alfred_seo_rest_content',
        'permission_callback' => 'alfred_seo_rest_permission_check',
    ) );
});

function alfred_seo_rest_content( WP_REST_Request $req ) {
    $title     = sanitize_text_field( $req->get_param( 'title' ) );
    $content   = wp_kses_post( $req->get_param( 'content' ) );
    $meta_desc = sanitize_text_field( $req->get_param( 'meta_description' ) );
    $slug      = sanitize_title( $req->get_param( 'slug' ) );
    $post_type = $req->get_param( 'post_type' ) ?: 'post';
    $status    = $req->get_param( 'status' ) ?: 'draft';
    $og_title  = sanitize_text_field( $req->get_param( 'og_title' ) );
    $og_desc   = sanitize_text_field( $req->get_param( 'og_description' ) );
    $og_image  = esc_url_raw( $req->get_param( 'og_image' ) );

    if ( ! $title || ! $content ) {
        return new WP_Error( 'rest_bad_request', 'title and content are required', array( 'status' => 400 ) );
    }
    if ( ! post_type_exists( $post_type ) ) {
        return new WP_Error( 'rest_bad_request', 'invalid post_type', array( 'status' => 400 ) );
    }

    $post_id = wp_insert_post( array(
        'post_title'   => $title,
        'post_content' => $content,
        'post_status'  => in_array( $status, array( 'draft', 'pending', 'publish' ), true ) ? $status : 'draft',
        'post_type'    => $post_type,
        'post_name'    => $slug,
    ), true );

    if ( is_wp_error( $post_id ) ) {
        return new WP_Error( 'rest_internal', $post_id->get_error_message(), array( 'status' => 500 ) );
    }

    if ( $meta_desc ) { update_post_meta( $post_id, '_alfred_seo_meta_description', $meta_desc ); }
    if ( $og_title )  { update_post_meta( $post_id, '_alfred_seo_og_title', $og_title ); }
    if ( $og_desc )   { update_post_meta( $post_id, '_alfred_seo_og_description', $og_desc ); }
    if ( $og_image )  { update_post_meta( $post_id, '_alfred_seo_og_image', $og_image ); }

    return new WP_REST_Response( array(
        'post_id' => $post_id,
        'url'     => get_permalink( $post_id ),
        'status'  => get_post_status( $post_id ),
    ), 201 );
}
