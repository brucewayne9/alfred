<?php
/**
 * Plugin Name: Alfred SEO
 * Plugin URI: https://aialfred.groundrushcloud.com
 * Description: SEO foundation — schema, OG, meta, sitemap, alt text, internal linking.
 *              Controlled remotely by Alfred orchestrator on 105.
 * Version: 0.1.0
 * Requires PHP: 8.0
 * Requires at least: 6.0
 * Author: Alfred Labs
 * License: Proprietary
 */

if ( ! defined( 'ABSPATH' ) ) { exit; }

define( 'ALFRED_SEO_VERSION', '0.1.0' );
define( 'ALFRED_SEO_DIR', plugin_dir_path( __FILE__ ) );
define( 'ALFRED_SEO_URL', plugin_dir_url( __FILE__ ) );

// Bootstrap: include modules. Each module registers its own hooks.
require_once ALFRED_SEO_DIR . 'inc/settings.php';
require_once ALFRED_SEO_DIR . 'inc/schema/validator.php';
require_once ALFRED_SEO_DIR . 'inc/schema/builder.php';
require_once ALFRED_SEO_DIR . 'inc/schema/product.php';
require_once ALFRED_SEO_DIR . 'inc/schema/article.php';
require_once ALFRED_SEO_DIR . 'inc/schema/organization.php';
require_once ALFRED_SEO_DIR . 'inc/schema/breadcrumb.php';
require_once ALFRED_SEO_DIR . 'inc/schema/faq.php';
require_once ALFRED_SEO_DIR . 'inc/schema/website.php';
require_once ALFRED_SEO_DIR . 'inc/schema/collection.php';
require_once ALFRED_SEO_DIR . 'inc/open-graph.php';
require_once ALFRED_SEO_DIR . 'inc/meta.php';
require_once ALFRED_SEO_DIR . 'inc/sitemap.php';
// Modules added in subsequent tasks register here.

register_activation_hook( __FILE__, function () {
    // Ensure default options exist on activation.
    if ( false === get_option( 'alfred_seo_settings' ) ) {
        add_option( 'alfred_seo_settings', alfred_seo_default_settings() );
    }
});
