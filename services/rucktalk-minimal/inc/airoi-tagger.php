<?php
/**
 * AIROI auto-tagger — rucktalk-minimal.
 *
 * On post save, scans `post_title + post_content` (lowercased, stripped of
 * HTML) for any keyword in RT_AIROI_KEYWORDS. If matched, sets post meta
 * `rt_show_airoi = 1`; otherwise the meta is removed so re-edits that drop
 * the AI angle correctly stop showing the block.
 *
 * The `the_content` filter then injects a single AIROI CTA aside at the
 * end of tagged posts on singular `post` views only. The podcast CPT is
 * NEVER touched (Phase 0 §12c: contextual, not blanket).
 */

if ( ! defined( 'ABSPATH' ) ) {
	exit;
}

/**
 * Keyword set for AI / business / efficiency detection. Lowercase,
 * substring-matched against the lowercased title + stripped content.
 * Keep this list tight — false positives dilute the placement.
 */
const RT_AIROI_KEYWORDS = array(
	'ai',
	'artificial intelligence',
	'automation',
	'chatgpt',
	'llm',
	'business',
	'small business',
	'entrepreneur',
	'productivity',
	'efficiency',
	'workflow',
	'saas',
	'startup',
);

/**
 * save_post callback. Decides whether the post should surface the AIROI
 * block by writing/removing the `rt_show_airoi` meta flag.
 *
 * @param int     $post_id Post ID.
 * @param WP_Post $post    Post object.
 */
function rt_airoi_check_post( $post_id, $post ) {
	if ( wp_is_post_revision( $post_id ) || wp_is_post_autosave( $post_id ) ) {
		return;
	}
	if ( ! ( $post instanceof WP_Post ) ) {
		return;
	}
	// Only standard blog posts. Pages, podcast CPT, products etc. are skipped.
	if ( 'post' !== $post->post_type ) {
		return;
	}

	$haystack = strtolower( $post->post_title . ' ' . wp_strip_all_tags( (string) $post->post_content ) );

	foreach ( RT_AIROI_KEYWORDS as $kw ) {
		if ( '' !== $kw && false !== strpos( $haystack, $kw ) ) {
			update_post_meta( $post_id, 'rt_show_airoi', 1 );
			return;
		}
	}

	delete_post_meta( $post_id, 'rt_show_airoi' );
}
add_action( 'save_post', 'rt_airoi_check_post', 10, 2 );

/**
 * the_content filter. Appends the AIROI CTA aside on singular `post`
 * views whose meta `rt_show_airoi` is truthy. No-ops everywhere else.
 *
 * @param string $content Post content.
 * @return string
 */
function rt_airoi_inject_block( $content ) {
	if ( ! is_singular( 'post' ) ) {
		return $content;
	}
	if ( ! get_post_meta( get_the_ID(), 'rt_show_airoi', true ) ) {
		return $content;
	}

	$cta_url = 'https://aialfred.groundrushcloud.com/static/ai-savings-calc/';

	$aside  = '<aside class="rt-airoi-block">';
	$aside .= '<h3>' . esc_html__( 'Curious what AI could actually save your business?', 'rucktalk-minimal' ) . '</h3>';
	$aside .= '<p>' . esc_html__( 'Try the AI Savings Calculator — 90 seconds, no signup.', 'rucktalk-minimal' ) . '</p>';
	$aside .= '<a class="btn btn--primary" href="' . esc_url( $cta_url ) . '" target="_blank" rel="noopener">'
		. esc_html__( 'Open the calculator →', 'rucktalk-minimal' )
		. '</a>';
	$aside .= '</aside>';

	return $content . $aside;
}
add_filter( 'the_content', 'rt_airoi_inject_block', 99 );
