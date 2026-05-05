<?php
/**
 * Template Name: Roen's Bracelet Box (/pick)
 *
 * Custom landing page that wraps the hidden 'bracelet-box' WooCommerce
 * product. The "Reserve your box" button adds that product to the cart;
 * checkout continues through the existing PayPal flow.
 *
 * If the box is out of stock (eligible bracelets < 5), renders a
 * "back soon" state and disables the CTA.
 */

if ( ! defined( 'ABSPATH' ) ) { exit; }

get_header();

// Resolve the hidden box product. If it doesn't exist, render a soft fallback.
$box_product = wc_get_product( wc_get_product_id_by_sku( 'bracelet-box' ) );
$in_stock = $box_product && $box_product->is_in_stock();
$available = $box_product ? (int) $box_product->get_stock_quantity() : 0;

$mark_url = get_stylesheet_directory_uri() . '/assets/svg/rowan-mark.svg';
?>

<main id="primary" class="site-main pick-page">

  <section class="pick-hero">
    <img src="<?php echo esc_url( $mark_url ); ?>" alt="" class="pick-mark" />
    <h1 class="pick-h1">Can't decide? Roen will.</h1>
    <p class="pick-sub">Five hand-picked bracelets. One curated note. $25, shipped within five business days.</p>

    <?php if ( $in_stock && $available > 0 ) : ?>
      <form class="pick-cta" method="post" action="<?php echo esc_url( wc_get_cart_url() ); ?>">
        <input type="hidden" name="add-to-cart" value="<?php echo esc_attr( $box_product->get_id() ); ?>" />
        <input type="hidden" name="quantity" value="1" />
        <button type="submit" class="button alt pick-button">Reserve your box — $25</button>
      </form>
      <p class="pick-availability">
        <?php printf( esc_html__( '%d %s available right now.', 'roen-minimal' ),
                      $available, _n( 'box', 'boxes', $available, 'roen-minimal' ) ); ?>
      </p>
    <?php else : ?>
      <button class="button pick-button pick-button--disabled" disabled>Roen is restocking</button>
      <p class="pick-availability">Back soon — usually within a few days.</p>
    <?php endif; ?>
  </section>

  <section class="pick-howitworks">
    <ol>
      <li><strong>You reserve a box.</strong> Pay $25, that's it.</li>
      <li><strong>Roen hand-picks five pieces.</strong> Curated to work as a set.</li>
      <li><strong>It ships in five business days</strong> with a personal card explaining the choices.</li>
    </ol>
  </section>

  <section class="pick-faq">
    <h2>A few quick answers</h2>
    <details>
      <summary>What sizes are the bracelets?</summary>
      <p>Most pieces fit wrists 6–8 inches. If you have a smaller or larger wrist, leave a note at checkout.</p>
    </details>
    <details>
      <summary>Returns?</summary>
      <p>Because each box is hand-curated, it's final sale. See <a href="/refund_returns/">refund policy</a> for details.</p>
    </details>
    <details>
      <summary>Is it gift-friendly?</summary>
      <p>Yes — leave a note at checkout if you'd like Roen to ship directly to the recipient.</p>
    </details>
    <details>
      <summary>Allergies?</summary>
      <p>Materials vary across the catalog. If you have specific metal or material sensitivities, leave a note at checkout and Roen will exclude those.</p>
    </details>
  </section>

</main>

<?php
get_footer();
