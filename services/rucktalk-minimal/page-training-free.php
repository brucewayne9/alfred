<?php
/**
 * Template Name: RuckTalk Training Free
 *
 * Email-gate landing for the free 8-week PDF (/training/free/).
 *
 * Two-column layout:
 *   - Left: headline + sub + [rt_signup placement="training-free"] form
 *           + footer micro-link to the paid PDF
 *   - Right: PDF cover image (placeholder block when the JPG is missing,
 *            same pattern as templates/parts/hero.php)
 *
 * Confirmation handoff:
 *   Brevo's doubleOptinConfirmation flow uses
 *   redirectionUrl = home_url('/training/free/?confirmed=1')
 *   (see inc/rest-signup.php). When that query param is present, the
 *   form is swapped for a "✓ Confirmed" message — no form re-submit.
 *
 * Structural styles live in assets/css/training.css (conditionally
 * enqueued by functions.php on this template only).
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

get_header();

$rt_cover_path  = get_stylesheet_directory() . '/assets/img/pdf-cover.jpg';
$rt_cover_uri   = get_stylesheet_directory_uri() . '/assets/img/pdf-cover.jpg';
$rt_has_cover   = file_exists( $rt_cover_path );
$rt_confirmed   = isset( $_GET['confirmed'] ) && '1' === sanitize_text_field( wp_unslash( $_GET['confirmed'] ) );
?>
<section class="training-free">
    <div class="wrap training-free__grid">

        <div class="training-free__copy">
            <h1 class="training-free__title">
                <?php esc_html_e( 'The Free 8-Week RuckTalk Plan', 'rucktalk-minimal' ); ?>
            </h1>
            <p class="training-free__sub">
                <?php esc_html_e( "What I would do in your first 8 weeks if I were starting over. Drop your email — we'll send you a verification link, then the plan lands in your inbox.", 'rucktalk-minimal' ); ?>
            </p>

            <?php if ( $rt_confirmed ) : ?>
                <div class="training-free__confirmed" role="status" aria-live="polite">
                    <p class="training-free__confirmed-title">
                        <?php esc_html_e( '✓ Confirmed — check your inbox.', 'rucktalk-minimal' ); ?>
                    </p>
                    <p class="training-free__confirmed-sub">
                        <?php esc_html_e( 'The 8-week plan is on its way.', 'rucktalk-minimal' ); ?>
                    </p>
                </div>
            <?php else : ?>
                <?php echo do_shortcode( '[rt_signup placement="training-free" button="Send me the plan"]' ); ?>
            <?php endif; ?>

            <p class="training-free__paid">
                <?php
                printf(
                    /* translators: %s = link to /training/ */
                    esc_html__( 'Already done the free one? %s', 'rucktalk-minimal' ),
                    '<a href="' . esc_url( home_url( '/training/' ) ) . '"><strong>' . esc_html__( 'Get the full PDF for $29', 'rucktalk-minimal' ) . '</strong> &rarr;</a>'
                );
                ?>
            </p>
        </div>

        <div class="training-free__visual">
            <?php if ( $rt_has_cover ) : ?>
                <img class="training-free__cover"
                     src="<?php echo esc_url( $rt_cover_uri ); ?>"
                     alt="<?php esc_attr_e( 'RuckTalk 8-week plan PDF cover', 'rucktalk-minimal' ); ?>"
                     loading="lazy"
                     decoding="async">
            <?php else : ?>
                <div class="training-free__cover-placeholder" aria-hidden="true">
                    <span class="training-free__cover-label"><?php esc_html_e( 'PDF Cover', 'rucktalk-minimal' ); ?></span>
                    <span class="training-free__cover-meta"><?php esc_html_e( 'placeholder · ships at launch', 'rucktalk-minimal' ); ?></span>
                </div>
            <?php endif; ?>
        </div>

    </div>
</section>
<?php get_footer();
