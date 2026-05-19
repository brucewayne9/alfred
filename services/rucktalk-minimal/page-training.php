<?php
/**
 * Template Name: RuckTalk Training
 *
 * Landing page for /training/ — the FaR migration target (Phase 0 §10a).
 *
 * Two side-by-side cards:
 *   1. Free 8-Week Plan  → /training/free/ (email-gated PDF)
 *   2. Full 8-Week Plan ($29) → WooCommerce product set via the
 *      `rt_training_product_id` WP option. If the option is unset or
 *      WooCommerce isn't active, the paid button renders disabled with
 *      "Coming soon" copy rather than producing a broken link.
 *
 * Visual language follows the locked design tokens in style.css; the
 * /training-only structural styles live in assets/css/training.css
 * (conditionally enqueued via functions.php on this template only).
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

get_header();

/**
 * Resolve the paid SKU CTA up front so the markup stays clean.
 *
 * Defensive: wc_get_product() doesn't exist if WC isn't loaded, and the
 * product ID can be 0 (unset) or point at a trashed product. Each branch
 * is explicit — silent failures here mean dead "Buy now" buttons.
 */
$rt_paid_url      = '';
$rt_paid_label    = __( 'Buy now &mdash; $29', 'rucktalk-minimal' );
$rt_paid_disabled = false;
$rt_product_id    = (int) get_option( 'rt_training_product_id', 0 );

if ( $rt_product_id > 0 && function_exists( 'wc_get_product' ) ) {
    $rt_product = wc_get_product( $rt_product_id );
    if ( $rt_product && $rt_product->is_visible() ) {
        $rt_paid_url = get_permalink( $rt_product_id );
    }
}

if ( '' === $rt_paid_url ) {
    // Fall back to /shop/ if it exists and WC is loaded; otherwise disable
    // the button entirely. Either way, no broken link.
    $rt_shop_url = function_exists( 'wc_get_page_permalink' ) ? wc_get_page_permalink( 'shop' ) : '';
    if ( $rt_shop_url ) {
        $rt_paid_url = $rt_shop_url;
    } else {
        $rt_paid_disabled = true;
        $rt_paid_label    = __( 'Coming soon', 'rucktalk-minimal' );
    }
}
?>
<section class="training-hero">
    <div class="wrap training-hero__inner">
        <h1 class="training-hero__title"><?php esc_html_e( 'RuckTalk Training', 'rucktalk-minimal' ); ?></h1>
        <p class="training-hero__sub">
            <?php esc_html_e( 'Two ways in: free plan in your inbox, or the full PDF.', 'rucktalk-minimal' ); ?>
        </p>
    </div>
</section>

<section class="training-cards">
    <div class="wrap training-cards__grid">

        <article class="training-card training-card--free">
            <span class="training-card__eyebrow"><?php esc_html_e( 'Free', 'rucktalk-minimal' ); ?></span>
            <h2 class="training-card__title"><?php esc_html_e( 'Free 8-Week Plan', 'rucktalk-minimal' ); ?></h2>
            <p class="training-card__sub">
                <?php esc_html_e( 'What I would do in your first 8 weeks if I were starting over. Sent by email after you verify.', 'rucktalk-minimal' ); ?>
            </p>
            <a class="btn btn--primary training-card__cta"
               href="<?php echo esc_url( home_url( '/training/free/' ) ); ?>">
                <?php esc_html_e( 'Get it free', 'rucktalk-minimal' ); ?>
            </a>
        </article>

        <article class="training-card training-card--paid">
            <span class="training-card__eyebrow"><?php esc_html_e( '$29 · printable', 'rucktalk-minimal' ); ?></span>
            <h2 class="training-card__title">
                <?php
                echo wp_kses_post(
                    sprintf(
                        /* translators: %s = price phrase styled by .training-card__title em */
                        __( 'The Full 8-Week Plan %s', 'rucktalk-minimal' ),
                        '<em>&mdash; $29</em>'
                    )
                );
                ?>
            </h2>
            <p class="training-card__sub">
                <?php esc_html_e( 'Printable PDF with the full progression, daily prompts, and recovery protocols.', 'rucktalk-minimal' ); ?>
            </p>
            <?php if ( $rt_paid_disabled ) : ?>
                <button class="btn btn--secondary training-card__cta is-disabled"
                        type="button"
                        disabled
                        aria-disabled="true">
                    <?php echo esc_html( $rt_paid_label ); ?>
                </button>
            <?php else : ?>
                <a class="btn btn--secondary training-card__cta"
                   href="<?php echo esc_url( $rt_paid_url ); ?>">
                    <?php echo wp_kses_post( $rt_paid_label ); ?>
                </a>
            <?php endif; ?>
        </article>

    </div>
</section>

<?php get_footer();
