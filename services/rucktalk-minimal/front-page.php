<?php
/**
 * Homepage — rucktalk-minimal.
 *
 * Wires the homepage section partials in the order defined by the
 * approved mockup (rucktalk-homepage-mockup.html) and the Phase 0
 * spec §3:
 *
 *   1.  .radio  + .head  — rendered by header.php
 *   2.  .hero            — templates/parts/hero
 *   3.  .live            — templates/parts/live-player
 *   4.  .pillars         — templates/parts/pillars
 *   5.  .episode         — templates/parts/episode-card
 *   6.  .blog            — templates/parts/blog-card
 *   7.  .shop            — templates/parts/shop-teaser
 *   8.  .coupon          — templates/parts/newsletter-inline
 *   9.  .about           — templates/parts/about-blurb
 *   10. .footer + .popup — rendered by footer.php
 *
 * Each section is its own partial so we can iterate independently in
 * later waves without touching this orchestrator. Anyone editing this
 * file should keep it that way — composition only, no markup.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

get_header();

/**
 * Action hook: rucktalk_homepage_before_sections
 * Lets plugins (e.g., a future seasonal banner) inject content above
 * the hero without touching the theme.
 */
do_action( 'rucktalk_homepage_before_sections' );

get_template_part( 'templates/parts/hero' );
get_template_part( 'templates/parts/live-player' );
get_template_part( 'templates/parts/pillars' );
get_template_part( 'templates/parts/episode-card' );
get_template_part( 'templates/parts/blog-card' );
get_template_part( 'templates/parts/shop-teaser' );
get_template_part( 'templates/parts/newsletter-inline' );
get_template_part( 'templates/parts/about-blurb' );

/**
 * Action hook: rucktalk_homepage_after_sections
 * Mirror of the "before" hook for tail-end content.
 */
do_action( 'rucktalk_homepage_after_sections' );

get_footer();
