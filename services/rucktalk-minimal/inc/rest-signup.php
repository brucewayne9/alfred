<?php
/**
 * REST signup endpoint — rucktalk-minimal.
 *
 * POST /wp-json/rucktalk/v1/signup
 *   Body:  { email: "...", placement: "hero|inline|footer|popup|training-free" }
 *   200:   { ok: true }
 *   400:   { ok: false, error: "invalid_email" }
 *   500:   { ok: false, error: "config_missing" }
 *   502:   { ok: false, error: "brevo_unreachable" | "brevo_rejected" }
 *
 * Single opt-in flow (2026-05-20 onward):
 *   1. Validate + sanitize email
 *   2. Add contact to Brevo list 6 via /v3/contacts (single opt-in)
 *   3. Send beautifully-designed welcome email from info@rucktalk.com
 *      with the 8-week PDF attached
 *   4. Fire webhook to n8n for the newsletter automation pipeline
 *   5. Toolkit account provisioning (Phase C.2) hooks here later
 *
 * WP options consumed:
 *   rt_brevo_api_key             — Brevo v3 api-key
 *   rt_brevo_list_id             — int list id for "RuckTalk"
 *   rt_n8n_signup_webhook_url    — n8n webhook (optional; skip if blank)
 *
 * Welcome email template at services/rucktalk-minimal/templates/email/welcome.{html,txt}
 * PDF lives at wp-content/uploads/2026/05/fitasruck-8week.pdf
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

const RT_BREVO_API_BASE     = 'https://api.brevo.com/v3';
const RT_BREVO_LIST_ID      = 6;
const RT_PDF_ABS_PATH       = WP_CONTENT_DIR . '/uploads/2026/05/fitasruck-8week.pdf';
const RT_WELCOME_TPL_HTML   = '/templates/email/welcome.html';
const RT_WELCOME_TPL_TXT    = '/templates/email/welcome.txt';

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
                'email'     => array( 'required' => true,  'type' => 'string' ),
                'placement' => array( 'required' => false, 'type' => 'string' ),
                'name'      => array( 'required' => false, 'type' => 'string' ),
            ),
        )
    );
}
add_action( 'rest_api_init', 'rt_register_signup_route' );

/**
 * Core signup handler.
 */
function rt_handle_signup( WP_REST_Request $r ) {
    $email     = sanitize_email( (string) $r->get_param( 'email' ) );
    $placement = sanitize_text_field( (string) $r->get_param( 'placement' ) );
    $name_raw  = sanitize_text_field( (string) $r->get_param( 'name' ) );

    if ( '' === $placement ) {
        $placement = 'unknown';
    }
    if ( '' === $email || ! is_email( $email ) ) {
        return new WP_REST_Response( array( 'ok' => false, 'error' => 'invalid_email' ), 400 );
    }

    // ---- 1) Add to Brevo list (single opt-in) -----------------------
    $api_key = (string) get_option( 'rt_brevo_api_key', '' );
    if ( '' === $api_key ) {
        return new WP_REST_Response( array( 'ok' => false, 'error' => 'config_missing' ), 500 );
    }

    $first_name = '' !== $name_raw ? $name_raw : '';
    $brevo_body = array(
        'email'          => $email,
        'listIds'        => array( RT_BREVO_LIST_ID ),
        'updateEnabled'  => true,
        'attributes'     => array(
            'SIGNUP_PLACEMENT' => $placement,
            'SIGNUP_SOURCE'    => 'rucktalk.com',
            'SIGNUP_DATE'      => gmdate( 'Y-m-d' ),
        ),
    );
    if ( '' !== $first_name ) {
        $brevo_body['attributes']['FIRSTNAME'] = $first_name;
    }

    $resp = wp_remote_post(
        RT_BREVO_API_BASE . '/contacts',
        array(
            'headers' => array(
                'accept'       => 'application/json',
                'content-type' => 'application/json',
                'api-key'      => $api_key,
            ),
            'body'    => wp_json_encode( $brevo_body ),
            'timeout' => 15,
        )
    );

    if ( is_wp_error( $resp ) ) {
        // Network blip — don't block the welcome email. Log + continue.
        error_log( '[rt-signup] brevo_unreachable: ' . $resp->get_error_message() );
    } else {
        $code = (int) wp_remote_retrieve_response_code( $resp );
        // 201 = created, 204 = updated. 400 = "Contact already exist" is OK
        // (we set updateEnabled=true so Brevo updates instead, but older
        // contacts may still 400). Treat <500 as non-fatal.
        if ( $code >= 500 ) {
            error_log( '[rt-signup] brevo_rejected status=' . $code . ' body=' . wp_remote_retrieve_body( $resp ) );
        }
    }

    // ---- 2) Provision toolkit account (Phase C.2) -------------------
    // Fires the toolkit's provision_from_rucktalk endpoint and stashes
    // the set-password URL so the welcome email can include it.
    $toolkit_url = rt_provision_toolkit_account( $email, $first_name );

    // ---- 3) Send welcome email --------------------------------------
    $emailed = rt_send_welcome_email( $email, $first_name, $placement, $toolkit_url );

    // ---- 4) Fire n8n webhook (best-effort) --------------------------
    rt_fire_n8n_signup_webhook( $email, $placement, $first_name );

    return new WP_REST_Response(
        array(
            'ok'       => true,
            'emailed'  => $emailed,
            'toolkit'  => ! empty( $toolkit_url ),
        ),
        200
    );
}

/**
 * Build the {{TOOLKIT_BLOCK}} (HTML + plain text) for the welcome email.
 * Returns [html, txt]. If $toolkit_url is empty, returns ['', ''] so the
 * email renders with no toolkit section (graceful — PDF still delivered).
 */
function rt_build_toolkit_block( $toolkit_url ) {
    if ( '' === $toolkit_url ) {
        return array( '', '' );
    }
    $url = esc_url( $toolkit_url );

    $html = '<!-- Divider -->'
        . '<tr><td style="padding:24px 36px 0;"><div style="height:1px;background:rgba(212,103,63,0.25);"></div></td></tr>'
        . '<!-- Toolkit -->'
        . '<tr>'
        . '<td style="padding:24px 36px 8px;">'
        . '<div style="font-family:\'Helvetica Neue\',Arial,sans-serif;font-size:11px;letter-spacing:0.22em;text-transform:uppercase;color:#D4673F;font-weight:700;margin:0 0 10px;">№ 2 — Bonus, free for the family</div>'
        . '<h2 style="margin:0 0 12px;font-family:\'Helvetica Black\',\'Arial Black\',Arial,sans-serif;font-weight:900;font-size:22px;line-height:1.2;color:#ECE4D2;">'
        . 'Your Nervous System Toolkit'
        . '</h2>'
        . '<p style="margin:0 0 16px;font-family:\'Helvetica Neue\',Arial,sans-serif;font-size:15px;line-height:1.65;color:rgba(236,228,210,0.85);">'
        . 'We built a free app suite for nervous-system regulation — six tools (Drift, Hum, Still, Pulse, Cold, Obsidian Mirror) for breathing, vagal tone, HRV biofeedback, and progressive relaxation. Pick a 2-minute reset that works for the moment.'
        . '</p>'
        . '<p style="margin:0 0 18px;font-family:\'Helvetica Neue\',Arial,sans-serif;font-size:14px;line-height:1.6;color:rgba(236,228,210,0.7);">'
        . 'Click below to set your password and unlock it. One account, all six tools, forever.'
        . '</p>'
        . '<table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr><td>'
        . '<a href="' . $url . '" style="display:inline-block;background:#D4673F;color:#36302A;font-family:\'Helvetica Neue\',Arial,sans-serif;font-weight:700;font-size:14px;text-decoration:none;padding:12px 22px;border-radius:4px;letter-spacing:0.04em;">Set password &amp; open the toolkit</a>'
        . '</td></tr></table>'
        . '</td></tr>';

    $txt = "\n---\n\n"
        . "№ 2 — BONUS, FREE FOR THE FAMILY: YOUR NERVOUS SYSTEM TOOLKIT\n\n"
        . "We built a free app suite for nervous-system regulation — six tools (Drift, Hum, Still, Pulse, Cold, Obsidian Mirror) for breathing, vagal tone, HRV biofeedback, and progressive relaxation. Pick a 2-minute reset that works for the moment.\n\n"
        . "Click below to set your password and unlock it. One account, all six tools, forever:\n\n"
        . $toolkit_url . "\n\n";

    return array( $html, $txt );
}

/**
 * Server-to-server: ask the toolkit on tech.groundrushlabs.com to
 * provision a pre-verified account + return the set-password URL.
 * Returns the magic-link URL on success, '' on any failure (we still
 * proceed with PDF delivery — toolkit is bonus, not blocker).
 */
function rt_provision_toolkit_account( $email, $first_name ) {
    if ( ! get_option( 'rt_toolkit_provision_enabled', 0 ) ) {
        return '';
    }
    $base   = (string) get_option( 'rt_toolkit_base_url', '' );
    $secret = (string) get_option( 'rt_toolkit_provision_secret', '' );
    if ( '' === $base || '' === $secret ) {
        return '';
    }

    $resp = wp_remote_post(
        rtrim( $base, '/' ) . '/api/auth.php',
        array(
            'headers' => array( 'content-type' => 'application/json' ),
            'body'    => wp_json_encode( array(
                'action' => 'provision_from_rucktalk',
                'secret' => $secret,
                'email'  => $email,
                'name'   => $first_name,
            ) ),
            'timeout' => 8,
        )
    );

    if ( is_wp_error( $resp ) ) {
        error_log( '[rt-signup] toolkit provision unreachable: ' . $resp->get_error_message() );
        return '';
    }
    $body = json_decode( (string) wp_remote_retrieve_body( $resp ), true );
    if ( ! is_array( $body ) || empty( $body['ok'] ) || empty( $body['set_password_url'] ) ) {
        error_log( '[rt-signup] toolkit provision rejected: ' . wp_remote_retrieve_body( $resp ) );
        return '';
    }
    return (string) $body['set_password_url'];
}

/**
 * Render + send the welcome email with the 8-week PDF attached.
 * Returns true if wp_mail() succeeded.
 *
 * @param string $email        Recipient
 * @param string $first_name   First name (may be '')
 * @param string $placement    Where they signed up
 * @param string $toolkit_url  Set-password magic link from provision
 *                             endpoint, '' if provisioning skipped/failed
 */
function rt_send_welcome_email( $email, $first_name, $placement, $toolkit_url = '' ) {
    $theme_dir = get_stylesheet_directory();
    $tpl_html_path = $theme_dir . RT_WELCOME_TPL_HTML;
    $tpl_txt_path  = $theme_dir . RT_WELCOME_TPL_TXT;

    $html = file_exists( $tpl_html_path ) ? (string) file_get_contents( $tpl_html_path ) : '';
    $txt  = file_exists( $tpl_txt_path )  ? (string) file_get_contents( $tpl_txt_path )  : '';

    if ( '' === $html ) {
        error_log( '[rt-signup] welcome.html template missing at ' . $tpl_html_path );
        return false;
    }

    $first = '' !== $first_name ? $first_name : 'ruck talker';
    $unsub = home_url( '/unsubscribe/?e=' . rawurlencode( $email ) );

    list( $tk_block_html, $tk_block_txt ) = rt_build_toolkit_block( $toolkit_url );

    $vars  = array(
        '{{FIRST_NAME}}'        => esc_html( $first ),
        '{{SIGNUP_DATE}}'       => gmdate( 'F j, Y' ),
        '{{UNSUBSCRIBE_URL}}'   => esc_url( $unsub ),
        '{{YEAR}}'              => gmdate( 'Y' ),
        '{{TOOLKIT_BLOCK}}'     => $tk_block_html,
        '{{TOOLKIT_BLOCK_TXT}}' => $tk_block_txt,
    );

    /**
     * Filter the welcome-email substitution map so Phase C.2 can inject
     * the toolkit-credentials block without rewriting this function.
     *
     * @param array  $vars       Substitution map ({{TOKEN}} => string)
     * @param string $email      Recipient email
     * @param string $first_name First name (may be '')
     * @param string $placement  Signup placement
     */
    $vars = apply_filters( 'rt_welcome_email_vars', $vars, $email, $first_name, $placement );

    $html = strtr( $html, $vars );
    $txt  = strtr( $txt,  $vars );

    $subject = 'Welcome to RuckTalk — your 8-week plan is attached';

    $headers = array(
        'Content-Type: text/html; charset=UTF-8',
        'From: RuckTalk <info@rucktalk.com>',
        'Reply-To: Mike Johnson <mjohnson@groundrushinc.com>',
        'X-RuckTalk-Placement: ' . sanitize_text_field( $placement ),
    );

    $attachments = array();
    if ( file_exists( RT_PDF_ABS_PATH ) ) {
        $attachments[] = RT_PDF_ABS_PATH;
    } else {
        error_log( '[rt-signup] PDF missing at ' . RT_PDF_ABS_PATH );
    }

    // Plain-text alt-body via phpmailer hook so HTML clients get HTML
    // and text-only clients get the plain template.
    add_action( 'phpmailer_init', $alt_body = function ( $phpmailer ) use ( $txt ) {
        if ( '' !== $txt ) { $phpmailer->AltBody = $txt; }
    }, 100 );

    $ok = wp_mail( $email, $subject, $html, $headers, $attachments );

    remove_action( 'phpmailer_init', $alt_body, 100 );

    if ( ! $ok ) {
        error_log( '[rt-signup] wp_mail returned false for ' . $email );
    }
    return (bool) $ok;
}

/**
 * Fire the n8n newsletter-automation webhook (best-effort, fire-and-forget).
 * URL configured via wp_option rt_n8n_signup_webhook_url; skipped if blank.
 */
function rt_fire_n8n_signup_webhook( $email, $placement, $first_name ) {
    $url = (string) get_option( 'rt_n8n_signup_webhook_url', '' );
    if ( '' === $url ) {
        return;
    }
    $payload = array(
        'email'       => $email,
        'first_name'  => $first_name,
        'placement'   => $placement,
        'source'      => 'rucktalk.com',
        'signup_at'   => gmdate( 'c' ),
        'ip'          => isset( $_SERVER['REMOTE_ADDR'] ) ? sanitize_text_field( wp_unslash( $_SERVER['REMOTE_ADDR'] ) ) : '',
        'user_agent'  => isset( $_SERVER['HTTP_USER_AGENT'] ) ? sanitize_text_field( wp_unslash( $_SERVER['HTTP_USER_AGENT'] ) ) : '',
    );

    // 5-second timeout — don't block the user's signup flow.
    wp_remote_post(
        $url,
        array(
            'headers'  => array( 'content-type' => 'application/json' ),
            'body'     => wp_json_encode( $payload ),
            'timeout'  => 5,
            'blocking' => false, // fire-and-forget
        )
    );
}

/**
 * Non-JS fallback: admin-post.php?action=rt_signup
 */
function rt_admin_post_signup() {
    $nonce = isset( $_POST['rt_signup_nonce'] ) ? sanitize_text_field( wp_unslash( $_POST['rt_signup_nonce'] ) ) : '';
    if ( ! wp_verify_nonce( $nonce, 'rt_signup' ) ) {
        $back = wp_get_referer() ? wp_get_referer() : home_url( '/' );
        wp_safe_redirect( add_query_arg( array( 'signup' => 'err', 'reason' => 'bad_nonce' ), $back ) );
        exit;
    }

    $email     = isset( $_POST['email'] )     ? sanitize_email(      wp_unslash( $_POST['email'] ) )     : '';
    $placement = isset( $_POST['placement'] ) ? sanitize_text_field( wp_unslash( $_POST['placement'] ) ) : '';
    $name      = isset( $_POST['name'] )      ? sanitize_text_field( wp_unslash( $_POST['name'] ) )      : '';

    $r = new WP_REST_Request( 'POST', '/rucktalk/v1/signup' );
    $r->set_param( 'email', $email );
    $r->set_param( 'placement', $placement );
    if ( '' !== $name ) { $r->set_param( 'name', $name ); }

    $resp = rt_handle_signup( $r );
    $data = $resp->get_data();
    $ok   = is_array( $data ) && ! empty( $data['ok'] );

    $back = wp_get_referer() ? wp_get_referer() : home_url( '/' );
    $args = $ok
        ? array( 'signup' => 'ok' )
        : array( 'signup' => 'err', 'reason' => isset( $data['error'] ) ? sanitize_key( $data['error'] ) : 'unknown' );

    wp_safe_redirect( add_query_arg( $args, $back ) );
    exit;
}
add_action( 'admin_post_nopriv_rt_signup', 'rt_admin_post_signup' );
add_action( 'admin_post_rt_signup',        'rt_admin_post_signup' );
