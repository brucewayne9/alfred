<?php
// services/alfred-seo/inc/sitemap.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

/**
 * Register rewrite rules so /alfred-sitemap.xml and /alfred-sitemap-<type>.xml
 * are served by this plugin. Disables WP's built-in wp-sitemap.xml to avoid
 * duplication.
 */
add_action( 'init', function () {
    add_rewrite_rule( '^alfred-sitemap\.xml$', 'index.php?alfred_seo_sitemap=index', 'top' );
    add_rewrite_rule( '^alfred-sitemap-([a-z]+)\.xml$', 'index.php?alfred_seo_sitemap=$matches[1]', 'top' );
});
add_filter( 'query_vars', function ( $vars ) {
    $vars[] = 'alfred_seo_sitemap';
    return $vars;
});
add_filter( 'wp_sitemaps_enabled', '__return_false' );

add_action( 'template_redirect', function () {
    $type = get_query_var( 'alfred_seo_sitemap' );
    if ( ! $type ) { return; }
    header( 'Content-Type: application/xml; charset=UTF-8' );
    header( 'X-Robots-Tag: noindex, follow' );
    if ( 'index' === $type ) {
        echo alfred_seo_render_sitemap_index();
    } else {
        echo alfred_seo_render_sitemap_for( $type );
    }
    exit;
}, 1 );

function alfred_seo_render_sitemap_index() {
    $types = array( 'pages', 'posts', 'products', 'categories' );
    $now   = gmdate( 'c' );
    $xml   = '<?xml version="1.0" encoding="UTF-8"?>' . "\n";
    $xml  .= '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' . "\n";
    foreach ( $types as $t ) {
        $xml .= sprintf(
            "  <sitemap><loc>%s</loc><lastmod>%s</lastmod></sitemap>\n",
            esc_url( home_url( '/alfred-sitemap-' . $t . '.xml' ) ),
            $now
        );
    }
    $xml .= '</sitemapindex>' . "\n";
    return $xml;
}

function alfred_seo_render_sitemap_for( $type ) {
    $entries = array();
    switch ( $type ) {
        case 'pages':
            $posts = get_posts( array( 'post_type' => 'page', 'post_status' => 'publish', 'numberposts' => -1 ) );
            foreach ( $posts as $p ) { $entries[] = array( get_permalink( $p ), $p->post_modified_gmt ); }
            break;
        case 'posts':
            $posts = get_posts( array( 'post_type' => 'post', 'post_status' => 'publish', 'numberposts' => -1 ) );
            foreach ( $posts as $p ) { $entries[] = array( get_permalink( $p ), $p->post_modified_gmt ); }
            break;
        case 'products':
            if ( post_type_exists( 'product' ) ) {
                $posts = get_posts( array( 'post_type' => 'product', 'post_status' => 'publish', 'numberposts' => -1 ) );
                foreach ( $posts as $p ) { $entries[] = array( get_permalink( $p ), $p->post_modified_gmt ); }
            }
            break;
        case 'categories':
            $terms = get_terms( array( 'taxonomy' => array_filter( array( 'category', taxonomy_exists( 'product_cat' ) ? 'product_cat' : null ) ), 'hide_empty' => true ) );
            foreach ( $terms as $t ) { $entries[] = array( get_term_link( $t ), gmdate( 'c' ) ); }
            break;
    }
    $xml = '<?xml version="1.0" encoding="UTF-8"?>' . "\n";
    $xml .= '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' . "\n";
    foreach ( $entries as $row ) {
        $xml .= sprintf(
            "  <url><loc>%s</loc><lastmod>%s</lastmod></url>\n",
            esc_url( $row[0] ),
            $row[1] ? mysql2date( 'c', $row[1], false ) : gmdate( 'c' )
        );
    }
    $xml .= '</urlset>' . "\n";
    return $xml;
}

/**
 * Ping GSC + Bing when a post is published or updated.
 */
add_action( 'transition_post_status', function ( $new, $old, $post ) {
    if ( 'publish' !== $new ) { return; }
    if ( ! in_array( $post->post_type, array( 'post', 'page', 'product' ), true ) ) { return; }
    $sitemap = home_url( '/alfred-sitemap.xml' );
    wp_remote_get( 'https://www.google.com/ping?sitemap=' . urlencode( $sitemap ), array( 'timeout' => 3, 'blocking' => false ) );
    wp_remote_get( 'https://www.bing.com/ping?sitemap=' . urlencode( $sitemap ), array( 'timeout' => 3, 'blocking' => false ) );
}, 10, 3 );

register_activation_hook( ALFRED_SEO_DIR . 'alfred-seo.php', function () {
    flush_rewrite_rules();
});
