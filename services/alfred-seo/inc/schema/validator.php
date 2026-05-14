<?php
// services/alfred-seo/inc/schema/validator.php

if ( ! defined( 'ABSPATH' ) ) { exit; }

/**
 * Validate a JSON-LD payload before injection.
 * Returns true if structurally valid, false otherwise. Logs failures.
 */
function alfred_seo_validate_jsonld( $payload ) {
    if ( ! is_array( $payload ) ) { return false; }
    // Top-level OR @graph wrapper must declare schema.org context.
    if ( ! isset( $payload['@context'] ) && ! isset( $payload['@graph'] ) ) {
        alfred_seo_log_schema_failure( 'missing @context', $payload );
        return false;
    }
    if ( isset( $payload['@context'] ) && 'https://schema.org' !== $payload['@context'] ) {
        alfred_seo_log_schema_failure( 'bad @context', $payload );
        return false;
    }
    // If single payload (no @graph), must have @type.
    if ( ! isset( $payload['@graph'] ) && ! isset( $payload['@type'] ) ) {
        alfred_seo_log_schema_failure( 'missing @type', $payload );
        return false;
    }
    // Round-trip JSON encode to catch bad UTF-8 or non-encodable values.
    $encoded = wp_json_encode( $payload );
    if ( false === $encoded ) {
        alfred_seo_log_schema_failure( 'json_encode failed', $payload );
        return false;
    }
    return true;
}

function alfred_seo_log_schema_failure( $reason, $payload ) {
    $log_dir = wp_upload_dir()['basedir'] . '/alfred-seo-logs';
    if ( ! is_dir( $log_dir ) ) { wp_mkdir_p( $log_dir ); }
    $line = sprintf(
        "[%s] schema rejected (%s): %s\n",
        gmdate( 'c' ),
        $reason,
        substr( wp_json_encode( $payload ), 0, 500 )
    );
    @file_put_contents( $log_dir . '/schema-failures.log', $line, FILE_APPEND | LOCK_EX );
}
