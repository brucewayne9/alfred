<?php
// services/alfred-seo/inc/schema/faq.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

/**
 * Extract FAQ schema from any HTML content where H2/H3 ends with ? followed by
 * one or more paragraphs (until next H2/H3 or end). Returns null if no FAQs.
 */
function alfred_seo_schema_faq_from_content( $html ) {
    if ( ! $html ) { return null; }

    // Match: <h[23]>question ending with ?</h[23]> followed by paragraph(s).
    $pattern = '#<h[23][^>]*>\s*([^<>]*\?)\s*</h[23]>\s*(.*?)(?=<h[23]|\z)#is';
    if ( ! preg_match_all( $pattern, $html, $matches, PREG_SET_ORDER ) ) {
        return null;
    }

    $entities = array();
    foreach ( $matches as $m ) {
        $question = trim( wp_strip_all_tags( $m[1] ) );
        $answer   = trim( wp_strip_all_tags( $m[2] ) );
        if ( $question && $answer ) {
            $entities[] = array(
                '@type'          => 'Question',
                'name'           => $question,
                'acceptedAnswer' => array(
                    '@type' => 'Answer',
                    'text'  => $answer,
                ),
            );
        }
    }
    if ( empty( $entities ) ) { return null; }

    return array(
        '@context'   => 'https://schema.org',
        '@type'      => 'FAQPage',
        'mainEntity' => $entities,
    );
}

/**
 * Page-level helper: build FAQ schema for current post content if any exist.
 */
function alfred_seo_schema_faq() {
    if ( ! is_singular() ) { return null; }
    $post = get_queried_object();
    return alfred_seo_schema_faq_from_content( $post->post_content );
}
