<?php
/**
 * SEO meta description + noindex injector.
 *
 * Reads meta_descriptions.json (deployed alongside the theme) and emits
 * <meta name="description"> + matching og:/twitter: tags on pages we own.
 *
 * Also emits noindex,follow on product-tag archives and paginated archives
 * (page 2+), per audit recommendation 2026-05-17.
 *
 * Why a custom injector instead of an SEO plugin: Roen has no SEO plugin
 * installed. Adding RankMath/Yoast pulls in 2–4 MB of plugin code we don't
 * need. This 90-line file does the high-impact 10%.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

/**
 * Lazy-load and cache the meta config JSON.
 */
function roen_seo_meta_config() {
    static $cfg = null;
    if ( $cfg !== null ) {
        return $cfg;
    }
    $path = get_stylesheet_directory() . '/assets/seo/meta_descriptions.json';
    if ( ! file_exists( $path ) ) {
        $cfg = array();
        return $cfg;
    }
    $raw = file_get_contents( $path );
    $cfg = json_decode( $raw, true ) ?: array();
    return $cfg;
}

/**
 * Resolve the meta description for the current request.
 * Returns null if no override applies (let other sources handle it).
 */
function roen_seo_resolve_description() {
    $cfg = roen_seo_meta_config();
    if ( empty( $cfg ) ) {
        return null;
    }

    if ( is_front_page() ) {
        return $cfg['pages']['front_page']['description'] ?? null;
    }
    if ( is_page() ) {
        $slug = get_post_field( 'post_name', get_queried_object_id() );
        return $cfg['pages'][ $slug ]['description'] ?? null;
    }
    if ( function_exists( 'is_shop' ) && is_shop() ) {
        return $cfg['shop']['description'] ?? null;
    }
    if ( function_exists( 'is_product_category' ) && is_product_category() ) {
        $term = get_queried_object();
        return $cfg['product_categories'][ $term->slug ]['description'] ?? null;
    }
    return null;
}

/**
 * Decide whether the current request should be noindexed.
 */
function roen_seo_should_noindex() {
    $cfg   = roen_seo_meta_config();
    $rules = $cfg['noindex_rules'] ?? array();

    if ( ! empty( $rules['product_tag_archive'] ) && function_exists( 'is_product_tag' ) && is_product_tag() ) {
        return true;
    }
    if ( ! empty( $rules['paged_archive_after_page_1'] ) && is_paged() ) {
        return true;
    }
    return false;
}

/**
 * Emit meta description + og/twitter mirrors when we have one.
 */
function roen_seo_emit_description() {
    $desc = roen_seo_resolve_description();
    if ( ! $desc ) {
        return;
    }
    $escaped = esc_attr( $desc );
    echo "<meta name=\"description\" content=\"{$escaped}\" />\n";
    echo "<meta property=\"og:description\" content=\"{$escaped}\" />\n";
    echo "<meta name=\"twitter:description\" content=\"{$escaped}\" />\n";
}

/**
 * Emit noindex,follow on archives we don't want indexed.
 */
function roen_seo_emit_robots() {
    if ( ! roen_seo_should_noindex() ) {
        return;
    }
    echo "<meta name=\"robots\" content=\"noindex,follow\" />\n";
}

// Priority 1 so we beat any other plugin/theme that might emit blanks.
add_action( 'wp_head', 'roen_seo_emit_description', 1 );
add_action( 'wp_head', 'roen_seo_emit_robots', 1 );
