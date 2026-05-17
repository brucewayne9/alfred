<?php
/**
 * Self-contained XML sitemap for Roen.
 *
 * WP core's /wp-sitemap.xml is 404ing on this install (likely a disabled
 * filter from a long-removed plugin or a permalink rewrite that never got
 * flushed). Rather than chase that ghost, this file owns the sitemap.
 *
 * Routes:
 *   /sitemap.xml          — sitemap index (links to the three below)
 *   /sitemap-products.xml — all published WooCommerce products
 *   /sitemap-content.xml  — pages + blog posts
 *   /sitemap-categories.xml — product categories (excluding empty)
 *
 * Excluded:
 *   - Product tags (noindexed per seo-meta.php)
 *   - Paginated archives (noindexed per seo-meta.php)
 *   - Author archives, attachment pages
 *
 * Cached for 1 hour via transient. Purged on save_post / saved_term so
 * new products show up fast.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

const ROEN_SITEMAP_CACHE_TTL = HOUR_IN_SECONDS;

function roen_sitemap_register_rewrites() {
    // Tolerate optional trailing slash (WP's redirect_canonical otherwise
    // appends one to .xml URLs and creates an infinite redirect loop).
    add_rewrite_rule( '^sitemap\.xml/?$',            'index.php?roen_sitemap=index',      'top' );
    add_rewrite_rule( '^sitemap-products\.xml/?$',   'index.php?roen_sitemap=products',   'top' );
    add_rewrite_rule( '^sitemap-content\.xml/?$',    'index.php?roen_sitemap=content',    'top' );
    add_rewrite_rule( '^sitemap-categories\.xml/?$', 'index.php?roen_sitemap=categories', 'top' );
}
add_action( 'init', 'roen_sitemap_register_rewrites' );

// Disable canonical redirect for our sitemap routes — it tries to add
// a trailing slash and loops.
function roen_sitemap_skip_canonical( $redirect_url, $requested_url ) {
    if ( get_query_var( 'roen_sitemap' ) ) {
        return false;
    }
    return $redirect_url;
}
add_filter( 'redirect_canonical', 'roen_sitemap_skip_canonical', 10, 2 );

function roen_sitemap_query_var( $vars ) {
    $vars[] = 'roen_sitemap';
    return $vars;
}
add_filter( 'query_vars', 'roen_sitemap_query_var' );

function roen_sitemap_template_redirect() {
    $kind = get_query_var( 'roen_sitemap' );
    if ( ! $kind ) {
        return;
    }
    $cache_key = "roen_sitemap_{$kind}";
    $xml       = get_transient( $cache_key );
    if ( false === $xml ) {
        switch ( $kind ) {
            case 'index':      $xml = roen_sitemap_build_index(); break;
            case 'products':   $xml = roen_sitemap_build_products(); break;
            case 'content':    $xml = roen_sitemap_build_content(); break;
            case 'categories': $xml = roen_sitemap_build_categories(); break;
            default:
                status_header( 404 );
                exit;
        }
        set_transient( $cache_key, $xml, ROEN_SITEMAP_CACHE_TTL );
    }
    status_header( 200 );
    header( 'Content-Type: application/xml; charset=UTF-8' );
    header( 'X-Robots-Tag: noindex, follow' );
    echo $xml;
    exit;
}
add_action( 'template_redirect', 'roen_sitemap_template_redirect' );

function roen_sitemap_purge() {
    delete_transient( 'roen_sitemap_index' );
    delete_transient( 'roen_sitemap_products' );
    delete_transient( 'roen_sitemap_content' );
    delete_transient( 'roen_sitemap_categories' );
}
add_action( 'save_post',  'roen_sitemap_purge' );
add_action( 'saved_term', 'roen_sitemap_purge' );

// Tell crawlers where the sitemap lives via robots.txt.
function roen_sitemap_robots( $output, $public ) {
    if ( '1' === (string) $public ) {
        $output .= "\nSitemap: " . esc_url( home_url( '/sitemap.xml' ) ) . "\n";
    }
    return $output;
}
add_filter( 'robots_txt', 'roen_sitemap_robots', 10, 2 );

function roen_sitemap_xml_header() {
    return '<?xml version="1.0" encoding="UTF-8"?>' . "\n";
}

function roen_sitemap_build_index() {
    $base = home_url( '/' );
    $now  = gmdate( 'c' );
    $xml  = roen_sitemap_xml_header()
          . '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' . "\n";
    foreach ( array( 'products', 'content', 'categories' ) as $kind ) {
        $loc = esc_url( $base . "sitemap-{$kind}.xml" );
        $xml .= "  <sitemap><loc>{$loc}</loc><lastmod>{$now}</lastmod></sitemap>\n";
    }
    $xml .= '</sitemapindex>';
    return $xml;
}

function roen_sitemap_build_products() {
    $q = new WP_Query( array(
        'post_type'      => 'product',
        'post_status'    => 'publish',
        'posts_per_page' => 1000,
        'no_found_rows'  => true,
        'fields'         => 'ids',
    ) );
    return roen_sitemap_render_urlset( $q->posts, 0.8 );
}

function roen_sitemap_build_content() {
    $q = new WP_Query( array(
        'post_type'      => array( 'post', 'page' ),
        'post_status'    => 'publish',
        'posts_per_page' => 500,
        'no_found_rows'  => true,
        'fields'         => 'ids',
    ) );
    return roen_sitemap_render_urlset( $q->posts, 0.6 );
}

function roen_sitemap_build_categories() {
    $terms = get_terms( array(
        'taxonomy'   => 'product_cat',
        'hide_empty' => true,
    ) );
    $xml = roen_sitemap_xml_header()
         . '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' . "\n";
    foreach ( (array) $terms as $term ) {
        if ( is_wp_error( $term ) ) continue;
        $loc = esc_url( get_term_link( $term ) );
        $xml .= "  <url><loc>{$loc}</loc><changefreq>weekly</changefreq><priority>0.7</priority></url>\n";
    }
    $xml .= '</urlset>';
    return $xml;
}

function roen_sitemap_render_urlset( array $post_ids, float $priority ) {
    $xml = roen_sitemap_xml_header()
         . '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' . "\n";
    foreach ( $post_ids as $pid ) {
        $loc     = esc_url( get_permalink( $pid ) );
        $lastmod = get_the_modified_date( 'c', $pid );
        $xml .= "  <url><loc>{$loc}</loc><lastmod>{$lastmod}</lastmod><priority>" . number_format( $priority, 1 ) . "</priority></url>\n";
    }
    $xml .= '</urlset>';
    return $xml;
}
