<?php
/**
 * Header — rucktalk-minimal.
 *
 * Renders: doctype, head, opening body, floating LoovaCast radio bar,
 * sticky site header (brand + primary nav), and the opening <main>.
 *
 * Class names + structure are a verbatim port of the approved homepage
 * mockup at https://aialfred.groundrushcloud.com/static/drafts/rucktalk-
 * homepage-mockup.html. CSS lives in:
 *   - style.css                 (design tokens)
 *   - assets/css/rucktalk.css   (layout primitives + .head)
 *   - assets/css/player.css     (.radio floating bar)
 *
 * Sonaar parent already registers main-menu + responsive-menu locations,
 * and Mike's existing Main Menu (term 6) is attached to both. We render
 * `main-menu` here — do NOT register a new theme_location for it.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}
?><!DOCTYPE html>
<html <?php language_attributes(); ?>>
<head>
    <meta charset="<?php bloginfo( 'charset' ); ?>">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="profile" href="https://gmpg.org/xfn/11">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <?php wp_head(); ?>
</head>
<body <?php body_class(); ?>>
<?php wp_body_open(); ?>

<!-- ─── FLOATING LOOVACAST RADIO BAR ─── -->
<?php
$rt_lc_public = (string) get_option( 'rt_loovacast_stream_url', '' );
$rt_lc_listen = '';
if ( preg_match( '#^(https?://[^/]+)/public/([A-Za-z0-9_-]+)/?$#', $rt_lc_public, $rt_lc_m ) ) {
    $rt_lc_listen = $rt_lc_m[1] . '/listen/' . $rt_lc_m[2] . '/radio.mp3';
}
?>
<div class="radio" id="rt-radio-bar"
     data-station-url="<?php echo esc_attr( $rt_lc_public ); ?>"
     data-station-id="<?php echo esc_attr( get_option( 'rt_loovacast_station_id', '' ) ); ?>"
     data-listen-url="<?php echo esc_attr( $rt_lc_listen ); ?>">
    <span class="radio__on" aria-hidden="true"></span>
    <span class="radio__label">Live</span>
    <span class="radio__sep"></span>
    <span class="radio__track"><?php echo esc_html( apply_filters( 'rucktalk_radio_now_playing', 'RuckTalk Radio · Tune in any time' ) ); ?></span>
    <button class="radio__play" type="button" aria-label="<?php esc_attr_e( 'Play live radio', 'rucktalk-minimal' ); ?>">&#9654;</button>
    <span class="radio__attrib">Powered by <a href="https://loovacast.com" rel="noopener" target="_blank">LoovaCast</a></span>
</div>

<!-- ─── SITE HEADER (sticky, sits below radio bar) ─── -->
<header class="head">
    <div class="wrap head__inner">
        <a class="brand" href="<?php echo esc_url( home_url( '/' ) ); ?>" rel="home">
            <?php
            $rt_logo_id  = get_theme_mod( 'custom_logo' );
            $rt_logo_src = $rt_logo_id ? wp_get_attachment_image_src( $rt_logo_id, 'medium' ) : false;
            if ( $rt_logo_src ) {
                // Render the logo ourselves (no nested <a>, no 1400×671 raw dims).
                printf(
                    '<img class="brand__logo" src="%s" alt="%s" width="%d" height="%d" decoding="async">',
                    esc_url( $rt_logo_src[0] ),
                    esc_attr( get_bloginfo( 'name' ) ),
                    intval( $rt_logo_src[1] ),
                    intval( $rt_logo_src[2] )
                );
            } else {
                ?>
                <span class="brand__placeholder">RuckTalk <small>logo</small></span>
                <?php
            }
            ?>
        </a>
        <nav class="nav nav--desktop" aria-label="<?php esc_attr_e( 'Primary', 'rucktalk-minimal' ); ?>">
            <?php
            wp_nav_menu( array(
                'theme_location' => 'main-menu',
                'container'      => false,
                'items_wrap'     => '%3$s',
                'fallback_cb'    => 'rucktalk_default_nav',
                'depth'          => 1,
            ) );
            ?>
        </nav>

        <button class="nav-burger" type="button"
                aria-label="<?php esc_attr_e( 'Open menu', 'rucktalk-minimal' ); ?>"
                aria-expanded="false"
                aria-controls="rt-mobile-nav">
            <span class="nav-burger__bar"></span>
            <span class="nav-burger__bar"></span>
            <span class="nav-burger__bar"></span>
        </button>
    </div>

    <!-- Mobile nav drawer — toggled by nav-burger above. Slides down from header. -->
    <div id="rt-mobile-nav" class="nav-drawer" data-open="0" aria-hidden="true">
        <nav class="nav nav--mobile" aria-label="<?php esc_attr_e( 'Mobile navigation', 'rucktalk-minimal' ); ?>">
            <?php
            wp_nav_menu( array(
                'theme_location' => 'main-menu',
                'container'      => false,
                'items_wrap'     => '%3$s',
                'fallback_cb'    => 'rucktalk_default_nav',
                'depth'          => 1,
            ) );
            ?>
        </nav>
    </div>
</header>

<main class="rt-main" id="rt-main">
<?php
/**
 * Fallback nav when Mike's Main Menu isn't assigned. Mirrors the mockup
 * link list so the layout doesn't collapse during early QA.
 */
if ( ! function_exists( 'rucktalk_default_nav' ) ) {
    function rucktalk_default_nav() {
        $items = array(
            'podcast'  => __( 'Podcast',  'rucktalk-minimal' ),
            'watch'    => __( 'Watch',    'rucktalk-minimal' ),
            'blog'     => __( 'Blog',     'rucktalk-minimal' ),
            'training' => __( 'Training', 'rucktalk-minimal' ),
            'shop'     => __( 'Shop',     'rucktalk-minimal' ),
            'about'    => __( 'About',    'rucktalk-minimal' ),
        );
        foreach ( $items as $slug => $label ) {
            printf(
                '<a href="%1$s">%2$s</a>',
                esc_url( home_url( '/' . $slug . '/' ) ),
                esc_html( $label )
            );
        }
        printf(
            '<a href="%1$s" class="nav__cta">%2$s</a>',
            esc_url( home_url( '/training/free/' ) ),
            esc_html__( 'Free plan', 'rucktalk-minimal' )
        );
    }
}
