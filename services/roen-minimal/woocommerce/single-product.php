<?php
/**
 * roen-minimal single product wrapper.
 * Loads WC's content-single-product.php inside our layout.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

get_header( 'shop' );
?>

<section class="roen-single roen-container roen-container--product">
    <?php while ( have_posts() ) : the_post(); ?>
        <?php wc_get_template_part( 'content', 'single-product' ); ?>
    <?php endwhile; ?>
</section>

<?php get_footer( 'shop' );
