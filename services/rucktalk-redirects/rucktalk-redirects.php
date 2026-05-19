<?php
/**
 * Plugin Name: RuckTalk Legacy Redirects
 * Description: 301-redirects defensive legacy fitasruck.com-style paths that may resolve on rucktalk.com. Defensive only — primary 301s live at the fitasruck.com Cloudflare layer (see Plan 1A Task 23). This mu-plugin catches direct hits (typed-in URLs, stale bookmarks, internal links) that bypass the apex redirect.
 * Version: 1.0.0
 * Author: Ground Rush Labs
 */

if ( ! defined( 'ABSPATH' ) ) {
	exit;
}

/**
 * On every front-end request, check the incoming path against the legacy
 * map and 301-redirect if the request begins with one of the old paths.
 *
 * Prefix match (strpos === 0) is intentional — query strings, trailing
 * fragments, and child slugs all collapse to the canonical destination.
 */
add_action(
	'template_redirect',
	function () {
		$path = isset( $_SERVER['REQUEST_URI'] ) ? (string) $_SERVER['REQUEST_URI'] : '/';

		$map = array(
			'/8-week-plan/' => '/training/8-week-plan/',
			'/8-week-plan'  => '/training/8-week-plan/',
			'/free-plan/'   => '/training/free/',
			'/free-plan'    => '/training/free/',
			'/checkout/'    => '/training/',
			'/checkout-1/'  => '/training/',
		);

		foreach ( $map as $old => $new ) {
			if ( strpos( $path, $old ) === 0 ) {
				wp_safe_redirect( esc_url_raw( home_url( $new ) ), 301 );
				exit;
			}
		}
	}
);
