<?php
/**
 * Template Name: About — Roen Minimal
 *
 * Single column, narrow line length, brand-voice tone.
 * Body content is editable via the WP page editor.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

get_header();
?>

<article class="roen-about roen-container">
    <?php while ( have_posts() ) : the_post(); ?>
        <h1 class="roen-about__title"><?php the_title(); ?></h1>
        <div class="roen-about__body">
            <?php the_content(); ?>
        </div>
    <?php endwhile; ?>
</article>

<?php get_footer();
