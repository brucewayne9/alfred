<?php
// services/alfred-seo/inc/robots.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_filter( 'robots_txt', function ( $output, $public ) {
    if ( ! $public ) { return $output; }
    // Append sitemap line if not present.
    $sitemap_url = home_url( '/alfred-sitemap.xml' );
    if ( false === strpos( $output, $sitemap_url ) ) {
        $output = rtrim( $output ) . "\nSitemap: " . $sitemap_url . "\n";
    }
    return $output;
}, 10, 2 );
