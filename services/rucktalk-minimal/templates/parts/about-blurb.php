<?php
/**
 * About pulled-quote — rucktalk-minimal homepage partial.
 *
 * Mirrors mockup .about block. Two-column grid: terracotta-accented
 * label on the left, italic blockquote + signature + link on the
 * right. Quote copy is LOCKED per design-language §6c — do not edit
 * without Mike's sign-off.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

$blurb = apply_filters(
    'rucktalk_about_blurb',
    "I'm not an expert in anything. I just look at the world and give my honest take — and then I want to hear yours. RuckTalk is a conversation for the go-getters: real life out loud, decisions worth getting right, and the stuff everybody else is also going through."
);
?>
<section class="about">
    <div class="wrap about__inner">
        <div class="about__label"><?php esc_html_e( 'About RuckTalk', 'rucktalk-minimal' ); ?></div>
        <div>
            <blockquote class="about__body">
                &ldquo;<?php echo esc_html( $blurb ); ?>&rdquo;
            </blockquote>
            <p class="about__sig">&mdash; <?php esc_html_e( 'Mike, host', 'rucktalk-minimal' ); ?></p>
            <a class="about__link" href="<?php echo esc_url( home_url( '/about/' ) ); ?>">
                <?php esc_html_e( 'More about the show', 'rucktalk-minimal' ); ?> &rarr;
            </a>
        </div>
    </div>
</section>
