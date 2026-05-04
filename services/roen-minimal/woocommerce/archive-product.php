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

$current_term  = is_product_category() ? get_queried_object() : null;
$current_slug  = $current_term ? $current_term->slug : '';
$product_cats  = get_terms( array(
    'taxonomy'   => 'product_cat',
    'hide_empty' => true,
    'orderby'    => 'name',
    'order'      => 'ASC',
) );
?>

<section class="roen-tagline roen-tagline--small">
    <div class="roen-container">
        <h1 class="roen-tagline__head"><?php woocommerce_page_title(); ?></h1>
    </div>
</section>

<?php if ( ! empty( $product_cats ) && ! is_wp_error( $product_cats ) ) : ?>
<nav class="roen-pills" aria-label="<?php esc_attr_e( 'Browse by category', 'roen-minimal' ); ?>">
    <div class="roen-container roen-pills__row">
        <a class="roen-pill<?php echo $current_slug ? '' : ' is-active'; ?>" href="<?php echo esc_url( wc_get_page_permalink( 'shop' ) ); ?>">all</a>
        <?php foreach ( $product_cats as $cat ) : ?>
            <a class="roen-pill<?php echo $current_slug === $cat->slug ? ' is-active' : ''; ?>" href="<?php echo esc_url( get_term_link( $cat ) ); ?>"><?php echo esc_html( strtolower( $cat->name ) ); ?></a>
        <?php endforeach; ?>
    </div>
</nav>
<?php endif; ?>

<section class="roen-grid-section">
    <div class="roen-container">
        <?php if ( woocommerce_product_loop() && wc_get_loop_prop( 'total' ) ) : ?>
            <ul class="roen-grid" role="list">
                <?php
                while ( have_posts() ) {
                    the_post();
                    wc_get_template_part( 'content', 'product' );
                }
                ?>
            </ul>

            <?php do_action( 'woocommerce_after_shop_loop' ); ?>

        <?php else : ?>
            <p class="roen-empty">no pieces in this category yet.</p>
        <?php endif; ?>
    </div>
</section>

<?php
get_footer( 'shop' );
