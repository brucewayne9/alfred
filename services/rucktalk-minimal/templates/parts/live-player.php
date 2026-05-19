<?php
/**
 * Dedicated LoovaCast player block — rucktalk-minimal homepage partial.
 *
 * Renders the mockup .live block: dark full-width band between hero
 * and pillars with an 88px play button. Track / episode metadata is
 * filterable so the actual LoovaCast feed payload (Task 25b) can swap
 * in via `rucktalk_radio_now_playing` + co. without editing this file.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

$now_playing = apply_filters(
    'rucktalk_live_now_playing',
    'How to find an extra hour you didn\'t know you had'
);
$episode_no  = (int) apply_filters( 'rucktalk_live_episode_number', 0 );
$time_in     = apply_filters( 'rucktalk_live_time_in', '' );
$up_next     = apply_filters( 'rucktalk_live_up_next', '' );
?>
<section class="live">
    <div class="live__inner">

        <div class="live__art" aria-hidden="true">
            <span class="live__art-text">RuckTalk<br>Radio</span>
        </div>

        <div class="live__copy">
            <div class="live__eyebrow">
                <span class="radio__on" aria-hidden="true"></span>
                <?php esc_html_e( 'Tune in live · 24/7', 'rucktalk-minimal' ); ?>
            </div>
            <h2 class="live__title">
                <?php esc_html_e( 'Now playing:', 'rucktalk-minimal' ); ?>
                <em>&ldquo;<?php echo esc_html( $now_playing ); ?>&rdquo;</em>
            </h2>
            <div class="live__meta">
                <?php if ( $episode_no > 0 ) : ?>
                    <span><?php printf( esc_html__( 'Episode %d', 'rucktalk-minimal' ), $episode_no ); ?></span>
                <?php endif; ?>
                <?php if ( '' !== $time_in ) : ?>
                    <span class="live__meta-sep">&middot;</span>
                    <span><?php echo esc_html( $time_in ); ?></span>
                <?php endif; ?>
                <?php if ( '' !== $up_next ) : ?>
                    <span class="live__meta-sep">&middot;</span>
                    <span><?php printf( esc_html__( 'Next up: %s', 'rucktalk-minimal' ), esc_html( $up_next ) ); ?></span>
                <?php endif; ?>
            </div>
            <div class="live__attrib">
                <?php esc_html_e( 'Streaming on', 'rucktalk-minimal' ); ?>
                <span class="live__attrib-logo">LoovaCast</span>
            </div>
        </div>

        <div class="live__controls">
            <button class="live__btn" type="button" aria-label="<?php esc_attr_e( 'Play live stream', 'rucktalk-minimal' ); ?>">&#9654;</button>
            <span class="live__sub"><?php esc_html_e( 'Press play. Always streaming.', 'rucktalk-minimal' ); ?></span>
        </div>

    </div>
</section>
