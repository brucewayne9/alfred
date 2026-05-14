<?php
// services/alfred-seo/inc/meta.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_resolve_meta_description() {
    if ( is_singular() ) {
        $obj      = get_queried_object();
        $override = get_post_meta( $obj->ID, '_alfred_seo_meta_description', true );
        if ( $override ) { return wp_strip_all_tags( $override ); }
        $excerpt = get_the_excerpt( $obj );
        if ( $excerpt ) { return wp_strip_all_tags( $excerpt ); }
        $content = wp_strip_all_tags( $obj->post_content );
        return mb_substr( trim( preg_replace( '/\s+/', ' ', $content ) ), 0, 155 );
    }
    if ( is_category() || is_tax() ) {
        $term = get_queried_object();
        if ( $term && $term->description ) {
            return wp_strip_all_tags( $term->description );
        }
    }
    return wp_strip_all_tags( get_bloginfo( 'description' ) );
}

function alfred_seo_render_meta_description() {
    $desc = alfred_seo_resolve_meta_description();
    if ( ! $desc ) { return; }
    printf( "\n<meta name=\"description\" content=\"%s\" />\n", esc_attr( mb_substr( $desc, 0, 160 ) ) );
}
add_action( 'wp_head', 'alfred_seo_render_meta_description', 3 );
