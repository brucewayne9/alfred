<?php
// services/alfred-seo/inc/internal-links.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

add_filter( 'the_content', 'alfred_seo_apply_internal_links', 99 );

function alfred_seo_apply_internal_links( $content ) {
    $settings = alfred_seo_get_settings();
    $map      = $settings['internal_links'] ?? array();
    if ( empty( $map ) ) { return $content; }

    // Don't operate on admin or feeds.
    if ( is_admin() || is_feed() ) { return $content; }

    // Sort by phrase length DESC so longer phrases bind before shorter substrings.
    uksort( $map, function ( $a, $b ) { return strlen( $b ) - strlen( $a ); } );

    // Skip anything inside <a>, <h1-h6>, <code>, <pre>, <script>.
    $segments = preg_split(
        '#(<(?:a|h[1-6]|code|pre|script)\b[^>]*>.*?</(?:a|h[1-6]|code|pre|script)>)#is',
        $content,
        -1,
        PREG_SPLIT_DELIM_CAPTURE
    );

    foreach ( $segments as $i => $seg ) {
        // Odd indices are the captured skip-segments; leave them alone.
        if ( 1 === $i % 2 ) { continue; }
        foreach ( $map as $phrase => $url ) {
            $pattern = '/\b' . preg_quote( $phrase, '/' ) . '\b/i';
            $replacement = sprintf( '<a href="%s">$0</a>', esc_url( $url ) );
            $seg = preg_replace( $pattern, $replacement, $seg, 1 );  // limit 1 occurrence
        }
        $segments[ $i ] = $seg;
    }
    return implode( '', $segments );
}
