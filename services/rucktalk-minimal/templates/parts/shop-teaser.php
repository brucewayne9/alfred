<?php
/**
 * Shop teaser — rucktalk-minimal homepage partial.
 *
 * Mirrors mockup .shop block. Full-bleed --paper-deep background with
 * three product cards. If WooCommerce isn't active, returns silently —
 * no fatal, no empty section.
 *
 * Tries featured products first, falls back to most-recent. Each card
 * uses the mockup's .product / .product__visual / .product__title /
 * .product__price classes verbatim so the lifted CSS lights up as-is.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

if ( ! class_exists( 'WooCommerce' ) ) {
    return;
}

$q = new WP_Query(
    array(
        'post_type'           => 'product',
        'posts_per_page'      => 3,
        'post_status'         => 'publish',
        'orderby'             => 'date',
        'order'               => 'DESC',
        'ignore_sticky_posts' => true,
        'no_found_rows'       => true,
        'tax_query'           => array(
            array(
                'taxonomy' => 'product_visibility',
                'field'    => 'name',
                'terms'    => 'featured',
                'operator' => 'IN',
            ),
        ),
    )
);

if ( ! $q->have_posts() ) {
    $q = new WP_Query(
        array(
            'post_type'      => 'product',
            'posts_per_page' => 3,
            'post_status'    => 'publish',
            'orderby'        => 'date',
            'order'          => 'DESC',
            'no_found_rows'  => true,
        )
    );
    if ( ! $q->have_posts() ) {
        return;
    }
}

$shop_url = function_exists( 'wc_get_page_id' ) ? get_permalink( wc_get_page_id( 'shop' ) ) : home_url( '/shop/' );
if ( ! $shop_url ) {
    $shop_url = home_url( '/shop/' );
}
?>
<section class="shop">
    <div class="shop__head">
        <span class="section__eyebrow"><?php esc_html_e( 'Shop · Small batch', 'rucktalk-minimal' ); ?></span>
        <a class="btn btn--ghost" href="<?php echo esc_url( $shop_url ); ?>"><?php esc_html_e( 'All gear', 'rucktalk-minimal' ); ?> &rarr;</a>
    </div>

    <div class="shop__grid">
        <?php
        while ( $q->have_posts() ) :
            $q->the_post();
            $product = function_exists( 'wc_get_product' ) ? wc_get_product( get_the_ID() ) : null;
            $price_html = $product ? $product->get_price_html() : '';
            $img_html   = $product ? $product->get_image( 'medium_large' ) : get_the_post_thumbnail( get_the_ID(), 'medium_large' );
            ?>
            <a class="product" href="<?php echo esc_url( get_permalink() ); ?>">
                <div class="product__visual">
                    <?php
                    if ( $img_html ) {
                        echo $img_html; // phpcs:ignore WordPress.Security.EscapeOutput -- WC-built img markup, escaped at source.
                    } else {
                        the_title();
                    }
                    ?>
                </div>
                <h3 class="product__title"><?php the_title(); ?></h3>
                <?php if ( '' !== $price_html ) : ?>
                    <p class="product__price"><?php echo wp_kses_post( $price_html ); ?></p>
                <?php endif; ?>
            </a>
        <?php endwhile; ?>
    </div>
</section>
<?php
wp_reset_postdata();
