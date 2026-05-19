<?php
/**
 * REST signup endpoint — rucktalk-minimal.
 *
 * POST /wp-json/rucktalk/v1/signup
 *   Body:  { email: "...", placement: "hero|inline|footer|popup|training-free" }
 *   200:   { ok: true }
 *   400:   { ok: false, error: "invalid_email" }
 *   500:   { ok: false, error: "config_missing" }
 *   502:   { ok: false, error: "brevo_unreachable" }
 *   502:   { ok: false, error: "brevo_rejected", status: <code>, body: "..." }
 *
 * Subscribes the contact to Brevo's RuckTalk list with double opt-in
 * (Brevo's /v3/contacts/doubleOptinConfirmation endpoint). Brevo sends
 * the verify email; on confirm, the Brevo automation drops the user
 * back at /training/free/?confirmed=1 (handled by page-training-free.php)
 * and the n8n weekly-newsletter pipeline picks them up via webhook
 * (workflow o9cIjGWj8z9pwknY — Phase 0 spec §5c).
 *
 * WP options consumed (all required):
 *   rt_brevo_api_key             — Brevo v3 api-key
 *   rt_brevo_list_id             — int list id for "RuckTalk"
 *   rt_brevo_opt_in_template_id  — int template id for the confirm email
 *
 * Also registers a non-JS fallback at admin-post.php?action=rt_signup
 * so the [rt_signup] form keeps working when JS is disabled / blocked.
 * The admin-post handler verifies the form nonce; the REST endpoint is
 * CSRF-protected by WP's standard rest nonce (x-wp-nonce header).
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

const RT_BREVO_API_BASE = 'https://api.brevo.com/v3';

/**
 * Register POST /wp-json/rucktalk/v1/signup.
 */
function rt_register_signup_route() {
    register_rest_route(
        'rucktalk/v1',
        '/signup',
        array(
            'methods'             => 'POST',
            'callback'            => 'rt_handle_signup',
            'permission_callback' => '__return_true',
            'args'                => array(
                'email'     => array(
                    'required' => true,
                    'type'     => 'string',
                ),
                'placement' => array(
                    'required' => false,
                    'type'     => 'string',
                ),
            ),
        )
    );
}
add_action( 'rest_api_init', 'rt_register_signup_route' );

/**
 * Core signup handler — used by both the REST route and the admin-post
 * fallback. Returns a WP_REST_Response in all paths so the REST route
 * can `return` it directly and the admin-post wrapper can inspect status.
 *
 * @param WP_REST_Request $r Request with `email` + `placement` params.
 * @return WP_REST_Response
 */
function rt_handle_signup( WP_REST_Request $r ) {
    $email_raw = (string) $r->get_param( 'email' );
    $email     = sanitize_email( $email_raw );
    $placement = sanitize_text_field( (string) $r->get_param( 'placement' ) );

    if ( '' === $placement ) {
        $placement = 'unknown';
    }

    if ( '' === $email || ! is_email( $email ) ) {
        return new WP_REST_Response(
            array(
                'ok'    => false,
                'error' => 'invalid_email',
            ),
            400
        );
    }

    $api_key     = (string) get_option( 'rt_brevo_api_key', '' );
    $list_id     = (int) get_option( 'rt_brevo_list_id', 0 );
    $template_id = (int) get_option( 'rt_brevo_opt_in_template_id', 0 );

    if ( '' === $api_key || $list_id <= 0 || $template_id <= 0 ) {
        return new WP_REST_Response(
            array(
                'ok'    => false,
                'error' => 'config_missing',
            ),
            500
        );
    }

    $body = array(
        'email'          => $email,
        'includeListIds' => array( $list_id ),
        'templateId'     => $template_id,
        'redirectionUrl' => home_url( '/training/free/?confirmed=1' ),
        'attributes'     => array(
            'SIGNUP_PLACEMENT' => $placement,
            'SIGNUP_SOURCE'    => 'rucktalk.com',
        ),
    );

    $resp = wp_remote_post(
        RT_BREVO_API_BASE . '/contacts/doubleOptinConfirmation',
        array(
            'headers' => array(
                'accept'       => 'application/json',
                'content-type' => 'application/json',
                'api-key'      => $api_key,
            ),
            'body'    => wp_json_encode( $body ),
            'timeout' => 15,
        )
    );

    if ( is_wp_error( $resp ) ) {
        return new WP_REST_Response(
            array(
                'ok'    => false,
                'error' => 'brevo_unreachable',
            ),
            502
        );
    }

    $code = (int) wp_remote_retrieve_response_code( $resp );
    if ( $code >= 400 ) {
        return new WP_REST_Response(
            array(
                'ok'     => false,
                'error'  => 'brevo_rejected',
                'status' => $code,
                'body'   => wp_remote_retrieve_body( $resp ),
            ),
            502
        );
    }

    return new WP_REST_Response( array( 'ok' => true ), 200 );
}

/**
 * Non-JS fallback: admin-post.php?action=rt_signup
 *
 * The [rt_signup] form posts here with a `rt_signup` nonce in the
 * `rt_signup_nonce` field. We verify the nonce, run the same handler,
 * then redirect back to the referer with `?signup=ok` (or `?signup=err`
 * plus a coarse reason code for the page to surface).
 */
function rt_admin_post_signup() {
    $nonce = isset( $_POST['rt_signup_nonce'] ) ? sanitize_text_field( wp_unslash( $_POST['rt_signup_nonce'] ) ) : '';
    if ( ! wp_verify_nonce( $nonce, 'rt_signup' ) ) {
        $back = wp_get_referer() ? wp_get_referer() : home_url( '/' );
        wp_safe_redirect( add_query_arg( array( 'signup' => 'err', 'reason' => 'bad_nonce' ), $back ) );
        exit;
    }

    $email     = isset( $_POST['email'] ) ? sanitize_email( wp_unslash( $_POST['email'] ) ) : '';
    $placement = isset( $_POST['placement'] ) ? sanitize_text_field( wp_unslash( $_POST['placement'] ) ) : '';

    $r = new WP_REST_Request( 'POST', '/rucktalk/v1/signup' );
    $r->set_param( 'email', $email );
    $r->set_param( 'placement', $placement );

    $resp = rt_handle_signup( $r );
    $data = $resp->get_data();
    $ok   = is_array( $data ) && ! empty( $data['ok'] );

    $back  = wp_get_referer() ? wp_get_referer() : home_url( '/' );
    $args  = $ok
        ? array( 'signup' => 'ok' )
        : array( 'signup' => 'err', 'reason' => isset( $data['error'] ) ? sanitize_key( $data['error'] ) : 'unknown' );

    wp_safe_redirect( add_query_arg( $args, $back ) );
    exit;
}
add_action( 'admin_post_nopriv_rt_signup', 'rt_admin_post_signup' );
add_action( 'admin_post_rt_signup',        'rt_admin_post_signup' );
