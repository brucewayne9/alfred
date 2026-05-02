<?php
/**
 * roen-minimal product archive (shop page, category archives).
 *
 * Override of woocommerce/templates/archive-product.php
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

get_header( 'shop' );
?>

<section class="roen-tagline roen-tagline--small">
    <div class="roen-container">
        <h1 class="roen-tagline__head"><?php woocommerce_page_title(); ?></h1>
    </div>
</section>

<section class="roen-grid-section">
    <div class="roen-container">
        <?php if ( woocommerce_product_loop() ) : ?>
            <?php woocommerce_product_loop_start(); ?>

            <?php
            if ( wc_get_loop_prop( 'total' ) ) {
                while ( have_posts() ) {
                    the_post();
                    do_action( 'woocommerce_shop_loop' );
                    wc_get_template_part( 'content', 'product' );
                }
            }
            ?>

            <?php woocommerce_product_loop_end(); ?>

            <?php do_action( 'woocommerce_after_shop_loop' ); ?>

        <?php else : ?>
            <p class="roen-empty">no pieces in this category yet.</p>
        <?php endif; ?>
    </div>
</section>

<?php
get_footer( 'shop' );
