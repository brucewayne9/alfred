<?php
/**
 * roen-minimal product card (loops on home, shop, archive).
 *
 * Override of woocommerce/templates/content-product.php
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

global $product;

if ( empty( $product ) || ! $product->is_visible() ) {
    return;
}

$cat_slugs = array();
foreach ( wp_get_post_terms( $product->get_id(), 'product_cat' ) as $cat ) {
    $cat_slugs[] = $cat->slug;
}
?>

<li class="roen-card" data-cats="<?php echo esc_attr( implode( ' ', $cat_slugs ) ); ?>">
    <a class="roen-card__link" href="<?php the_permalink(); ?>">
        <div class="roen-card__media">
            <?php
            echo $product->get_image( 'woocommerce_thumbnail', array( // phpcs:ignore
                'class' => 'roen-card__img',
                'loading' => 'lazy',
            ) );
            ?>
        </div>
        <div class="roen-card__body">
            <h2 class="roen-card__title"><?php echo esc_html( $product->get_name() ); ?></h2>
            <div class="roen-card__price"><?php echo $product->get_price_html(); // phpcs:ignore ?></div>
        </div>
    </a>
</li>
