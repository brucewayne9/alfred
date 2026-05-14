<?php
// services/alfred-seo/inc/rest/auth.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_rest_permission_check( WP_REST_Request $req ) {
    // WP's REST API handles application-password auth automatically when
    // the Authorization: Basic header is present. We just require an authed
    // user with manage_options.
    if ( ! is_user_logged_in() ) {
        return new WP_Error( 'rest_unauthorized', 'Authentication required', array( 'status' => 401 ) );
    }
    if ( ! current_user_can( 'manage_options' ) ) {
        return new WP_Error( 'rest_forbidden', 'Insufficient permissions', array( 'status' => 403 ) );
    }
    return true;
}
