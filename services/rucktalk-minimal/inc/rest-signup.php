<?php
/**
 * REST signup endpoint — rucktalk-minimal.
 *
 * POST /wp-json/rucktalk/v1/signup { email, placement }
 *   → Brevo double-opt-in subscribe (config via wp options
 *      rt_brevo_api_key / rt_brevo_list_id / rt_brevo_opt_in_template_id)
 *   → Brevo confirmation webhook fires n8n workflow o9cIjGWj8z9pwknY
 *      for the weekly newsletter list (configured in Brevo dashboard).
 *
 * Also wraps a non-JS admin-post fallback for accessibility.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

// Task 19 fills this in.
