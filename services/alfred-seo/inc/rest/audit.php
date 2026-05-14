<?php
// services/alfred-seo/inc/rest/audit.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_action( 'rest_api_init', function () {
    register_rest_route( 'alfred-seo/v1', '/audit', array(
        'methods'             => 'GET',
        'callback'            => 'alfred_seo_rest_audit',
        'permission_callback' => 'alfred_seo_rest_permission_check',
    ) );
});

function alfred_seo_rest_audit( WP_REST_Request $req ) {
    $page_types = array( 'post', 'page' );
    if ( post_type_exists( 'product' ) ) { $page_types[] = 'product'; }

    $pages = array();
    $missing_meta = array();
    $missing_alt  = array();

    $posts = get_posts( array(
        'post_type'   => $page_types,
        'post_status' => 'publish',
        'numberposts' => -1,
    ) );
    foreach ( $posts as $p ) {
        $has_meta = (bool) get_post_meta( $p->ID, '_alfred_seo_meta_description', true );
        $pages[]  = array(
            'id'       => $p->ID,
            'type'     => $p->post_type,
            'url'      => get_permalink( $p ),
            'title'    => $p->post_title,
            'has_meta' => $has_meta,
            'modified' => mysql2date( 'c', $p->post_modified_gmt, false ),
        );
        if ( ! $has_meta && ! $p->post_excerpt ) {
            $missing_meta[] = array( 'id' => $p->ID, 'url' => get_permalink( $p ) );
        }
    }

    $atts = get_posts( array(
        'post_type'   => 'attachment',
        'post_status' => 'inherit',
        'post_mime_type' => 'image',
        'numberposts' => -1,
    ) );
    foreach ( $atts as $a ) {
        $alt = get_post_meta( $a->ID, '_wp_attachment_image_alt', true );
        if ( ! $alt ) {
            $missing_alt[] = array( 'id' => $a->ID, 'url' => wp_get_attachment_url( $a->ID ) );
        }
    }

    return new WP_REST_Response( array(
        'pages'        => $pages,
        'missing_meta' => $missing_meta,
        'missing_alt'  => $missing_alt,
        'audited_at'   => gmdate( 'c' ),
    ), 200 );
}
