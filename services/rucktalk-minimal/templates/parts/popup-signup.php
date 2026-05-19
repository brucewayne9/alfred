<?php
/**
 * Newsletter popup — rucktalk-minimal partial.
 *
 * Mirrors mockup .popup block. Hidden by default (data-open="0"); the
 * trigger logic lives in assets/js/popup.js (scroll-depth 50% on first
 * page OR exit-intent on second page, with 14-day dismiss cookie).
 *
 * Included from footer.php so the popup is available site-wide, not
 * just on the homepage.
 *
 * PDF cover placeholder ships with the theme. When the real cover PDF
 * lands at /assets/pdf/rucktalk-plan.pdf (Phase 3), the asset will be
 * regenerated as a JPG cover at /assets/img/pdf-cover.jpg.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

$cover_path = get_stylesheet_directory() . '/assets/img/pdf-cover.jpg';
$cover_uri  = get_stylesheet_directory_uri() . '/assets/img/pdf-cover.jpg';
$has_cover  = file_exists( $cover_path );
?>
<div class="popup"
     id="rt-popup"
     data-open="0"
     role="dialog"
     aria-modal="true"
     aria-labelledby="rt-popup-title"
     aria-hidden="true">
    <div class="popup__backdrop" data-rt-popup-close="1"></div>
    <div class="popup__panel">
        <button class="popup__close" type="button" data-rt-popup-close="1" aria-label="<?php esc_attr_e( 'Close', 'rucktalk-minimal' ); ?>">&times;</button>

        <div class="popup__cover">
            <?php if ( $has_cover ) : ?>
                <img src="<?php echo esc_url( $cover_uri ); ?>"
                     alt="<?php esc_attr_e( '8-week RuckTalk plan PDF cover', 'rucktalk-minimal' ); ?>"
                     loading="lazy">
            <?php else : ?>
                <?php esc_html_e( 'Actual PDF', 'rucktalk-minimal' ); ?><br>
                <?php esc_html_e( 'cover here', 'rucktalk-minimal' ); ?><br>
                <small style="opacity:0.5;font-size:10px;text-transform:uppercase;letter-spacing:0.1em">
                    <?php esc_html_e( '(at launch)', 'rucktalk-minimal' ); ?>
                </small>
            <?php endif; ?>
        </div>

        <h3 class="popup__title" id="rt-popup-title">
            <?php
            echo wp_kses_post(
                sprintf(
                    /* translators: %s = italic phrase styled by .popup__title em */
                    __( 'Get the %s RuckTalk plan', 'rucktalk-minimal' ),
                    '<em>' . esc_html__( 'free 8-week', 'rucktalk-minimal' ) . '</em>'
                )
            );
            ?>
        </h3>
        <p class="popup__sub">
            <?php esc_html_e( "What I'd do in your first eight weeks if I were starting over.", 'rucktalk-minimal' ); ?>
        </p>
        <?php
        echo do_shortcode(
            '[rt_signup placement="popup" button="' . esc_attr__( 'Send it', 'rucktalk-minimal' ) . '" micro=""]'
        );
        ?>
    </div>
</div>
