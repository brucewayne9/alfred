<?php
/**
 * Front page — L2 product-first layout.
 *
 * Tagline → category pills → product grid (16 most recent).
 */

if ( ! defined( 'ABSPATH' ) ) {
	exit;
}

get_header();

// Pull product categories that have at least 1 published product.
// (hide_empty filters by attached post count, not stock status.)
$product_cats = get_terms( array(
	'taxonomy'   => 'product_cat',
	'hide_empty' => true,
	'orderby'    => 'name',
	'order'      => 'ASC',
) );
?>

<section class="roen-tagline">
	<div class="roen-container">
		<h1 class="roen-tagline__head">decorate yourself.</h1>
		<p class="roen-tagline__sub">new pieces every week.</p>
	</div>
</section>

<?php if ( ! empty( $product_cats ) && ! is_wp_error( $product_cats ) ) : ?>
<nav class="roen-pills" aria-label="<?php esc_attr_e( 'Filter by category', 'roen-minimal' ); ?>">
	<div class="roen-container roen-pills__row">
		<button class="roen-pill is-active" data-cat="all" type="button">all</button>
		<?php foreach ( $product_cats as $cat ) : ?>
			<button class="roen-pill" data-cat="<?php echo esc_attr( $cat->slug ); ?>" type="button"><?php echo esc_html( strtolower( $cat->name ) ); ?></button>
		<?php endforeach; ?>
	</div>
</nav>
<?php endif; ?>

<section class="roen-grid-section">
	<div class="roen-container">
		<?php
		$products = wc_get_products( array(
			'status'  => 'publish',
			'limit'   => 16,
			'orderby' => 'date',
			'order'   => 'DESC',
		) );

		if ( empty( $products ) ) :
			?>
			<p class="roen-empty">no pieces in the shop yet — new drops are on the way.</p>
			<?php
		else :
			?>
			<ul class="roen-grid" role="list">
				<?php
				foreach ( $products as $product ) {
					$post_object = get_post( $product->get_id() );
					setup_postdata( $GLOBALS['post'] =& $post_object ); // phpcs:ignore
					wc_get_template_part( 'content', 'product' );
				}
				wp_reset_postdata();
				?>
			</ul>

			<p class="roen-grid__more">
				<a href="<?php echo esc_url( wc_get_page_permalink( 'shop' ) ); ?>">shop all pieces &rarr;</a>
			</p>
			<?php
		endif;
		?>
	</div>
</section>

<?php get_footer();
