<?php
/**
 * rucktalk-minimal — child theme bootstrap.
 *
 * Enqueues parent (Sonaar) and child assets, declares theme supports,
 * loads modular feature files from inc/.
 *
 * Parent handle is `iron-master` (Sonaar's own enqueue handle), confirmed
 * via the 2026-05-19 audit. Using the wrong handle here results in the
 * parent stylesheet getting double-loaded, which hurts page weight.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

define( 'RUCKTALK_THEME_VERSION', '1.2.0' );

/**
 * Enqueue parent + child assets.
 */
function rucktalk_enqueue_assets() {
    $v = RUCKTALK_THEME_VERSION;

    // Google Fonts — Archivo Black + Archivo + Bricolage Grotesque (locked design).
    wp_enqueue_style(
        'rucktalk-fonts',
        'https://fonts.googleapis.com/css2?family=Archivo+Black&family=Archivo:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500;1,600&family=Bricolage+Grotesque:opsz,wght@12..96,300;12..96,400;12..96,500;12..96,600;12..96,700&display=swap',
        array(),
        null
    );

    // Parent (Sonaar) stylesheet — handle `iron-master` per parent's own enqueue.
    // Loading explicitly here so child styles can declare it as a dependency.
    wp_enqueue_style(
        'iron-master',
        get_template_directory_uri() . '/style.css',
        array(),
        wp_get_theme( 'sonaar' )->get( 'Version' )
    );

    // Child design tokens (this file's style.css).
    wp_enqueue_style(
        'rucktalk-tokens',
        get_stylesheet_directory_uri() . '/style.css',
        array( 'iron-master' ),
        $v
    );

    // Structural / component CSS — lands in Task 14 (Wave 3).
    if ( file_exists( get_stylesheet_directory() . '/assets/css/rucktalk.css' ) ) {
        wp_enqueue_style(
            'rucktalk-structure',
            get_stylesheet_directory_uri() . '/assets/css/rucktalk.css',
            array( 'rucktalk-tokens' ),
            $v
        );
    }

    // LoovaCast floating radio bar CSS + JS — landing in Task 25 (Wave 6).
    if ( file_exists( get_stylesheet_directory() . '/assets/css/player.css' ) ) {
        wp_enqueue_style(
            'rucktalk-player',
            get_stylesheet_directory_uri() . '/assets/css/player.css',
            array( 'rucktalk-tokens' ),
            $v
        );
    }
    if ( file_exists( get_stylesheet_directory() . '/assets/js/player.js' ) ) {
        wp_enqueue_script(
            'rucktalk-player-js',
            get_stylesheet_directory_uri() . '/assets/js/player.js',
            array(),
            $v,
            true
        );
    }

    // Newsletter popup CSS + JS — landing in Task 21 (Wave 4).
    if ( file_exists( get_stylesheet_directory() . '/assets/css/popup.css' ) ) {
        wp_enqueue_style(
            'rucktalk-popup',
            get_stylesheet_directory_uri() . '/assets/css/popup.css',
            array( 'rucktalk-tokens' ),
            $v
        );
    }
    if ( file_exists( get_stylesheet_directory() . '/assets/js/popup.js' ) ) {
        wp_enqueue_script(
            'rucktalk-popup-js',
            get_stylesheet_directory_uri() . '/assets/js/popup.js',
            array(),
            $v,
            true
        );
    }
    if ( file_exists( get_stylesheet_directory() . '/assets/js/signup.js' ) ) {
        wp_enqueue_script(
            'rucktalk-signup-js',
            get_stylesheet_directory_uri() . '/assets/js/signup.js',
            array( 'rucktalk-popup-js' ),
            $v,
            true
        );
        wp_localize_script( 'rucktalk-signup-js', 'RuckTalkSignup', array(
            'restUrl' => esc_url_raw( rest_url( 'rucktalk/v1/signup' ) ),
            'nonce'   => wp_create_nonce( 'wp_rest' ),
        ) );
    }

    // /training + /training/free structural CSS — only on those templates.
    // Both page templates ship as part of Wave 4 (Tasks 17-18). We check
    // both is_page_template() variants because WP returns the *filename*
    // for the active template.
    if (
        file_exists( get_stylesheet_directory() . '/assets/css/training.css' )
        && ( is_page_template( 'page-training.php' ) || is_page_template( 'page-training-free.php' ) )
    ) {
        wp_enqueue_style(
            'rucktalk-training',
            get_stylesheet_directory_uri() . '/assets/css/training.css',
            array( 'rucktalk-tokens', 'rucktalk-structure' ),
            $v
        );
    }

    // Ecosystem strip + AIROI block — landing in Tasks 27-28 (Wave 6).
    if ( file_exists( get_stylesheet_directory() . '/assets/css/ecosystem.css' ) ) {
        wp_enqueue_style(
            'rucktalk-ecosystem',
            get_stylesheet_directory_uri() . '/assets/css/ecosystem.css',
            array( 'rucktalk-tokens' ),
            $v
        );
    }
    if ( file_exists( get_stylesheet_directory() . '/assets/js/airoi-block.js' ) ) {
        wp_enqueue_script(
            'rucktalk-airoi-js',
            get_stylesheet_directory_uri() . '/assets/js/airoi-block.js',
            array(),
            $v,
            true
        );
    }
}
add_action( 'wp_enqueue_scripts', 'rucktalk_enqueue_assets', 20 );

// Load modules. Each file checks ABSPATH; stubs are safe to require.
require_once get_stylesheet_directory() . '/inc/theme-supports.php';
require_once get_stylesheet_directory() . '/inc/menu-locations.php';
require_once get_stylesheet_directory() . '/inc/shortcodes.php';
require_once get_stylesheet_directory() . '/inc/rest-signup.php';
require_once get_stylesheet_directory() . '/inc/airoi-tagger.php';
require_once get_stylesheet_directory() . '/inc/sonaar-overrides.php';

// Pillar snippet system (Task 13b) — only required if file exists yet.
if ( file_exists( get_stylesheet_directory() . '/inc/pillar-snippets.php' ) ) {
    require_once get_stylesheet_directory() . '/inc/pillar-snippets.php';
}
