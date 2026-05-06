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
// Drop the WP-default "Uncategorized" bucket — products without a real
// category land there silently and we never want it as a browsable pill.
$product_cats = get_terms( array(
	'taxonomy'   => 'product_cat',
	'hide_empty' => true,
	'orderby'    => 'name',
	'order'      => 'ASC',
) );
if ( ! is_wp_error( $product_cats ) && is_array( $product_cats ) ) {
	$product_cats = array_values( array_filter( $product_cats, function ( $cat ) {
		return $cat->slug !== 'uncategorized' && strtolower( $cat->name ) !== 'uncategorized';
	} ) );
}

// Bracelet-box promo banner — points at /pick. Background image is loaded
// via stylesheet so we can swap it without editing this template.
$pick_url = home_url( '/pick/' );
?>

<section class="roen-promo-banner" aria-label="<?php esc_attr_e( 'Roen bracelet box', 'roen-minimal' ); ?>">
	<div class="roen-container">
		<a class="roen-promo-banner__link" href="<?php echo esc_url( $pick_url ); ?>">
			<div class="roen-promo-banner__image" role="img" aria-label="<?php esc_attr_e( 'Five handmade bracelets arranged on marble', 'roen-minimal' ); ?>"></div>
			<div class="roen-promo-banner__overlay">
				<p class="roen-promo-banner__eyebrow">curated</p>
				<h2 class="roen-promo-banner__title">the bracelet box</h2>
				<p class="roen-promo-banner__caption">five handmade pieces &middot; <span>$25</span></p>
				<span class="roen-promo-banner__cta">shop the box &rarr;</span>
			</div>
		</a>
	</div>
</section>

<section class="roen-tagline">
	<div class="roen-container">
		<h1 class="roen-tagline__head">decorate yourself.</h1>
		<p class="roen-tagline__sub">new pieces every week.</p>
	</div>
</section>

<?php if ( ! empty( $product_cats ) && ! is_wp_error( $product_cats ) ) : ?>
<nav class="roen-pills" aria-label="<?php esc_attr_e( 'Browse by category', 'roen-minimal' ); ?>">
	<div class="roen-container roen-pills__row">
		<a class="roen-pill is-active" href="<?php echo esc_url( wc_get_page_permalink( 'shop' ) ); ?>">all</a>
		<?php foreach ( $product_cats as $cat ) : ?>
			<a class="roen-pill" href="<?php echo esc_url( get_term_link( $cat ) ); ?>"><?php echo esc_html( strtolower( $cat->name ) ); ?></a>
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
