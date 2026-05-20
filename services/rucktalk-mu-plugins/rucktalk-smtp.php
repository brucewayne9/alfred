<?php
/**
 * Plugin Name: RuckTalk SMTP
 * Description: Routes wp_mail() through info@rucktalk.com via Mailcow (mail.doowoprnb.com). Credentials read from WP options (rt_smtp_*). No plugin UI — config lives in wp_options.
 * Version: 1.0.0
 * Author: Alfred
 *
 * Drop into wp-content/mu-plugins/ on rt-wordpress.
 *
 * Required options (set via wp-cli, never via the admin UI):
 *   rt_smtp_host        — e.g. mail.doowoprnb.com
 *   rt_smtp_port        — 465 (smtps) or 587 (starttls)
 *   rt_smtp_user        — info@rucktalk.com
 *   rt_smtp_pass        — app passwd (plaintext, single-purpose)
 *   rt_smtp_from        — info@rucktalk.com
 *   rt_smtp_from_name   — RuckTalk
 *   rt_smtp_encryption  — "ssl" (smtps/465) or "tls" (starttls/587)
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

add_action( 'phpmailer_init', function ( $phpmailer ) {
    $host = (string) get_option( 'rt_smtp_host', '' );
    $user = (string) get_option( 'rt_smtp_user', '' );
    $pass = (string) get_option( 'rt_smtp_pass', '' );
    if ( '' === $host || '' === $user || '' === $pass ) {
        return; // fall back to default PHP mail() if config missing
    }
    $port  = (int) get_option( 'rt_smtp_port', 465 );
    $enc   = (string) get_option( 'rt_smtp_encryption', 'ssl' );
    $from  = (string) get_option( 'rt_smtp_from', $user );
    $fname = (string) get_option( 'rt_smtp_from_name', 'RuckTalk' );

    $phpmailer->isSMTP();
    $phpmailer->Host       = $host;
    $phpmailer->Port       = $port;
    $phpmailer->SMTPAuth   = true;
    $phpmailer->Username   = $user;
    $phpmailer->Password   = $pass;
    $phpmailer->SMTPSecure = $enc;     // 'ssl' or 'tls'
    $phpmailer->From       = $from;
    $phpmailer->FromName   = $fname;
    $phpmailer->Sender     = $from;
    $phpmailer->XMailer    = 'RuckTalk SMTP (Mailcow)';

    // CharSet utf-8 is safer for HTML emails with em-dashes etc.
    $phpmailer->CharSet    = 'UTF-8';
    $phpmailer->Encoding   = 'base64';
}, 99 );

// Default From: header — wp_mail() applies this only if not overridden.
add_filter( 'wp_mail_from', function ( $email ) {
    $from = (string) get_option( 'rt_smtp_from', '' );
    return $from !== '' ? $from : $email;
}, 99 );

add_filter( 'wp_mail_from_name', function ( $name ) {
    $fname = (string) get_option( 'rt_smtp_from_name', '' );
    return $fname !== '' ? $fname : $name;
}, 99 );
