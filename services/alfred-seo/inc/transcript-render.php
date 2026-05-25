<?php
// services/alfred-seo/inc/transcript-render.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

/**
 * On the_content of any singular audio-post-type page, append a collapsible
 * transcript block read from the post_meta key configured in alfred_seo_settings
 * (default `_rt_transcript`). Plain-text input; render-time paragraph wrapping.
 *
 * Why <details>: search engines index collapsed <details> content (verified
 * Google + Bing 2024+), and the closed-by-default state keeps the page short
 * for non-readers. Matches the Apple Podcasts Transcript UX.
 */
function alfred_seo_append_transcript( $content ) {
    if ( ! is_singular() || ! in_the_loop() || ! is_main_query() ) {
        return $content;
    }
    $settings        = alfred_seo_get_settings();
    $audio_post_type = $settings['audio_post_type'] ?? 'podcast';
    if ( get_post_type() !== $audio_post_type ) { return $content; }

    $meta_key   = $settings['transcript_post_meta_key'] ?? '_rt_transcript';
    $transcript = get_post_meta( get_the_ID(), $meta_key, true );
    if ( empty( $transcript ) || ! is_string( $transcript ) ) { return $content; }

    $word_count = str_word_count( wp_strip_all_tags( $transcript ) );
    $body       = alfred_seo_format_transcript_html( $transcript );

    $label = sprintf(
        /* translators: %s: word count */
        __( 'Transcript (%s words)', 'alfred-seo' ),
        number_format_i18n( $word_count )
    );

    $block  = '<details class="alfred-seo-transcript" data-word-count="' . esc_attr( $word_count ) . '">';
    $block .= '<summary>' . esc_html( $label ) . '</summary>';
    $block .= '<div class="alfred-seo-transcript__body">' . $body . '</div>';
    $block .= '</details>';

    return $content . "\n" . $block;
}
add_filter( 'the_content', 'alfred_seo_append_transcript', 20 );

/**
 * Wrap plain-text transcript in <p> tags per blank-line block. wpautop is too
 * heavy-handed (turns single newlines into <br>); we want clean paragraphs.
 */
function alfred_seo_format_transcript_html( $text ) {
    $text = wp_strip_all_tags( $text );
    $text = trim( $text );
    if ( '' === $text ) { return ''; }
    $blocks = preg_split( '/\n{2,}/', $text );
    $html   = '';
    foreach ( $blocks as $block ) {
        $block = trim( $block );
        if ( '' === $block ) { continue; }
        $html .= '<p>' . esc_html( $block ) . '</p>';
    }
    return $html;
}
