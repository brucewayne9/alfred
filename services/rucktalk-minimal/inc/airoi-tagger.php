<?php
/**
 * AIROI auto-tagger — rucktalk-minimal.
 *
 * On post save, scans content for AI/business keywords. If matched, sets
 * post meta `rt_show_airoi=1`. The_content filter then injects the AIROI
 * contextual CTA block at the end of tagged posts.
 *
 * Only applies to standard `post` type — not pages, not `podcast` CPT.
 * Stays subtle per Phase 0 §12c: contextual, not blanket.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

// Task 27 fills this in.
