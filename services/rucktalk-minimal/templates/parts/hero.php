<?php
/**
 * Hero — rucktalk-minimal homepage partial.
 *
 * Mirrors mockup .hero block verbatim. Copy is locked:
 *   - Tagline: "Notes from a guy figuring it out." (italic em on
 *     "a guy figuring it out", terracotta via .hero__title em)
 *   - Strap: see design-language §6b
 *   - Eyebrow: "A conversation for the go-getters"
 *
 * Hero photo lives at assets/img/mike-hero.jpg. Until Mike provides
 * the real shot, we render the mockup's placeholder block instead so
 * QA can compare layout side-by-side with the locked mockup.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

$hero_img_path = get_stylesheet_directory() . '/assets/img/mike-hero.jpg';
$hero_img_uri  = get_stylesheet_directory_uri() . '/assets/img/mike-hero.jpg';
$has_hero_img  = file_exists( $hero_img_path );

$tagline_html  = apply_filters(
    'rucktalk_hero_tagline_html',
    'Notes from <em>a guy figuring it out.</em>'
);
$strap_text    = apply_filters(
    'rucktalk_hero_strap',
    "One go-getter's running commentary on health, business, family, strength, and what everybody else is going through. Tell me what I'm missing."
);
?>
<section class="hero">
    <div class="wrap hero__inner">

        <div class="hero__media reveal reveal--1">
            <?php if ( $has_hero_img ) : ?>
                <img src="<?php echo esc_url( $hero_img_uri ); ?>"
                     alt="<?php esc_attr_e( 'Mike Johnson — host of RuckTalk', 'rucktalk-minimal' ); ?>"
                     class="hero__photo"
                     loading="eager"
                     decoding="async">
            <?php else : ?>
                <span class="hero__media-label"><?php esc_html_e( 'Photo: Mike', 'rucktalk-minimal' ); ?></span>
                <span class="hero__media-meta"><?php esc_html_e( '4 × 5 portrait · placeholder', 'rucktalk-minimal' ); ?></span>
            <?php endif; ?>
        </div>

        <div class="hero__copy">
            <div class="hero__eyebrow reveal reveal--1"><?php esc_html_e( 'A conversation for the go-getters', 'rucktalk-minimal' ); ?></div>

            <h1 class="hero__title reveal reveal--2" id="rt-tagline">
                <?php
                // wp_kses_post strips dangerous tags but allows <em> — copy
                // is owned by Mike + the filter above, never user input.
                echo wp_kses_post( $tagline_html );
                ?>
            </h1>

            <p class="hero__strap reveal reveal--3">
                <?php echo esc_html( $strap_text ); ?>
            </p>

            <div class="hero__cta reveal reveal--4">
                <a class="btn btn--primary" href="#rt-hero-signup" id="hero-signup">
                    <?php esc_html_e( 'Get the free 8-week plan', 'rucktalk-minimal' ); ?>
                </a>
                <a class="btn btn--secondary" href="<?php echo esc_url( home_url( '/podcast/' ) ); ?>">
                    <?php echo wp_kses_post( '&#9654;&nbsp;&nbsp;' . esc_html__( "Today's episode", 'rucktalk-minimal' ) ); ?>
                </a>
            </div>

            <div class="hero__listen reveal reveal--5">
                <span class="hero__listen-label"><?php esc_html_e( 'Listen on', 'rucktalk-minimal' ); ?></span>
                <a class="platform" href="<?php echo esc_url( apply_filters( 'rucktalk_listen_url_spotify', '#' ) ); ?>" rel="noopener" target="_blank">
                    <span class="platform__dot" style="background:#1DB954"></span>Spotify
                </a>
                <a class="platform" href="<?php echo esc_url( apply_filters( 'rucktalk_listen_url_apple', '#' ) ); ?>" rel="noopener" target="_blank">
                    <span class="platform__dot" style="background:#FB5BC5"></span>Apple Podcasts
                </a>
                <a class="platform" href="<?php echo esc_url( apply_filters( 'rucktalk_listen_url_youtube', '#' ) ); ?>" rel="noopener" target="_blank">
                    <span class="platform__dot" style="background:#FF0000"></span>YouTube Music
                </a>
            </div>
        </div>

    </div>
</section>
