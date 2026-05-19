<?php
/**
 * Inline newsletter (coupon) — rucktalk-minimal homepage partial.
 *
 * Mirrors mockup .coupon block — 2px dashed --ink border, scissors
 * emoji top-left (rotated -20deg via CSS), italic forest stamp top-
 * right (rotated 8deg). All styling lives in assets/css/rucktalk.css.
 *
 * The signup form itself is rendered by [rt_signup] so the back-end
 * wiring (Brevo via REST, Task 19) only needs to be written once.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}
?>
<section class="coupon">
    <div class="wrap">
        <div class="coupon__card" id="rt-hero-signup">
            <span class="coupon__stamp"><?php esc_html_e( 'Free · Email-gated', 'rucktalk-minimal' ); ?></span>
            <p class="coupon__eyebrow"><?php esc_html_e( 'Lead magnet', 'rucktalk-minimal' ); ?></p>
            <h2 class="coupon__title">
                <?php
                echo wp_kses_post(
                    sprintf(
                        /* translators: %s = italic phrase styled by .coupon__title em */
                        __( 'Get the %s.', 'rucktalk-minimal' ),
                        '<em>' . esc_html__( '8-week RuckTalk plan', 'rucktalk-minimal' ) . '</em>'
                    )
                );
                ?>
            </h2>
            <p class="coupon__sub">
                <?php esc_html_e( "What I'd do in your first eight weeks if I were starting over. Sent by email after you verify.", 'rucktalk-minimal' ); ?>
            </p>
            <?php
            echo do_shortcode(
                '[rt_signup placement="inline" button="' . esc_attr__( 'Send it to me', 'rucktalk-minimal' ) . '"]'
            );
            ?>
        </div>
    </div>
</section>
