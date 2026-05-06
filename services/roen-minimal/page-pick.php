<?php
/**
 * Template Name: Roen's Bracelet Box (/pick)
 *
 * Curated landing page that wraps the hidden 'bracelet-box' WooCommerce
 * product. The "reserve your box" button adds that product to the cart;
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

// Resolve the refund-policy URL — robust to slug changes / subdirectory installs.
$refund_page = get_page_by_path( 'refund_returns' );
$refund_url  = $refund_page ? get_permalink( $refund_page ) : '/refund_returns/';

// Asset URLs.
$hero_img    = get_stylesheet_directory_uri() . '/assets/img/pick-hero.jpg';
$unbox_img   = get_stylesheet_directory_uri() . '/assets/img/pick-unboxing.jpg';
$wordmark    = function_exists( 'roen_wordmark_svg' ) ? roen_wordmark_svg() : '';
?>

<main id="primary" class="site-main pick-page">

  <header class="pick-eyebrow-bar">
    <?php if ( $wordmark ) : ?>
      <span class="pick-wordmark" aria-hidden="true"><?php echo $wordmark; // safe: hardcoded theme asset ?></span>
    <?php endif; ?>
    <span class="pick-eyebrow-tag"><?php esc_html_e( 'the bracelet box', 'roen-minimal' ); ?></span>
  </header>

  <section class="pick-hero">
    <div class="pick-hero__media">
      <img src="<?php echo esc_url( $hero_img ); ?>"
           alt="<?php esc_attr_e( 'A cluster of handmade beaded bracelets on cream linen', 'roen-minimal' ); ?>"
           class="pick-hero__img"
           loading="eager"
           width="1024" height="1024" />
    </div>

    <div class="pick-hero__panel">
      <p class="pick-hero__eyebrow"><?php esc_html_e( 'curated for you', 'roen-minimal' ); ?></p>
      <h1 class="pick-hero__title"><?php esc_html_e( "Can't decide? Roen will.", 'roen-minimal' ); ?></h1>
      <p class="pick-hero__sub">
        <?php esc_html_e( 'Five hand-picked bracelets. One curated note. $25, shipped within five business days.', 'roen-minimal' ); ?>
      </p>

      <?php if ( $in_stock ) : ?>
        <form class="pick-cta" method="post" action="<?php echo esc_url( wc_get_cart_url() ); ?>">
          <input type="hidden" name="add-to-cart" value="<?php echo esc_attr( $box_product->get_id() ); ?>" />
          <input type="hidden" name="quantity" value="1" />
          <button type="submit" class="pick-button"><?php esc_html_e( 'reserve your box · $25', 'roen-minimal' ); ?></button>
        </form>
        <?php if ( $available > 0 ) : ?>
          <p class="pick-availability">
            <span class="pick-availability__dot" aria-hidden="true"></span>
            <?php
            printf(
                /* translators: 1: count, 2: "box" or "boxes" */
                esc_html__( '%1$d %2$s available right now', 'roen-minimal' ),
                $available,
                esc_html( _n( 'box', 'boxes', $available, 'roen-minimal' ) )
            );
            ?>
          </p>
        <?php endif; ?>
      <?php else : ?>
        <button class="pick-button pick-button--disabled" disabled><?php esc_html_e( 'roen is restocking', 'roen-minimal' ); ?></button>
        <p class="pick-availability"><?php esc_html_e( 'Back soon — usually within a few days.', 'roen-minimal' ); ?></p>
      <?php endif; ?>
    </div>
  </section>

  <section class="pick-howitworks">
    <p class="pick-section-eyebrow"><?php esc_html_e( 'how it works', 'roen-minimal' ); ?></p>
    <ol class="pick-steps">
      <li class="pick-step">
        <span class="pick-step__num">01</span>
        <h3 class="pick-step__title"><?php esc_html_e( 'You reserve a box', 'roen-minimal' ); ?></h3>
        <p class="pick-step__body"><?php esc_html_e( 'Pay $25, that\'s it. Leave a note about colour, vibe, or wrist size if you like — or leave it blank and trust the curation.', 'roen-minimal' ); ?></p>
      </li>
      <li class="pick-step">
        <span class="pick-step__num">02</span>
        <h3 class="pick-step__title"><?php esc_html_e( 'Roen hand-picks five pieces', 'roen-minimal' ); ?></h3>
        <p class="pick-step__body"><?php esc_html_e( 'Five bracelets chosen from the current catalogue, curated to wear together as a stack — never random, never repeated.', 'roen-minimal' ); ?></p>
      </li>
      <li class="pick-step">
        <span class="pick-step__num">03</span>
        <h3 class="pick-step__title"><?php esc_html_e( 'It ships in five business days', 'roen-minimal' ); ?></h3>
        <p class="pick-step__body"><?php esc_html_e( 'Tucked into a paper-wrapped box with a hand-signed card explaining each piece and how the stack was built.', 'roen-minimal' ); ?></p>
      </li>
    </ol>
  </section>

  <?php if ( file_exists( get_stylesheet_directory() . '/assets/img/pick-unboxing.jpg' ) ) : ?>
  <section class="pick-unboxing">
    <img src="<?php echo esc_url( $unbox_img ); ?>"
         alt="<?php esc_attr_e( 'An opened paper jewelry box with five bracelets and a card inside', 'roen-minimal' ); ?>"
         class="pick-unboxing__img"
         loading="lazy"
         width="1280" height="768" />
    <p class="pick-unboxing__caption"><?php esc_html_e( 'what arrives at your door', 'roen-minimal' ); ?></p>
  </section>
  <?php endif; ?>

  <section class="pick-faq">
    <p class="pick-section-eyebrow"><?php esc_html_e( 'questions', 'roen-minimal' ); ?></p>
    <details>
      <summary><?php esc_html_e( 'What sizes are the bracelets?', 'roen-minimal' ); ?></summary>
      <p><?php esc_html_e( 'Most pieces fit wrists 6–8 inches. If you have a smaller or larger wrist, leave a note at checkout.', 'roen-minimal' ); ?></p>
    </details>
    <details>
      <summary><?php esc_html_e( 'Returns?', 'roen-minimal' ); ?></summary>
      <p>
        <?php
        printf(
            /* translators: %s: refund policy URL */
            wp_kses_post( __( 'Because each box is hand-curated, it\'s final sale. See <a href="%s">refund policy</a> for details.', 'roen-minimal' ) ),
            esc_url( $refund_url )
        );
        ?>
      </p>
    </details>
    <details>
      <summary><?php esc_html_e( 'Is it gift-friendly?', 'roen-minimal' ); ?></summary>
      <p><?php esc_html_e( 'Yes — leave a note at checkout if you\'d like Roen to ship directly to the recipient.', 'roen-minimal' ); ?></p>
    </details>
    <details>
      <summary><?php esc_html_e( 'Allergies?', 'roen-minimal' ); ?></summary>
      <p><?php esc_html_e( 'Materials vary across the catalogue. If you have specific metal or material sensitivities, leave a note at checkout and Roen will exclude those.', 'roen-minimal' ); ?></p>
    </details>
  </section>

</main>

<?php
get_footer();
