<?php
/**
 * Sonaar overrides + ecosystem mounts — rucktalk-minimal.
 *
 * Surgical additions to parent behavior. CRITICAL preservation rules:
 *   - NEVER touch these Sonaar filters (they back the podcast RSS feed
 *     subscribed to by Spotify / Apple / YouTube Music):
 *       sonaar_feed_slug
 *       sonaar_helper_feed_home_url
 *       sonaar_podcast_feed_query_args
 *   - Any feed-affecting change is T3 and requires a feed-validator test
 *     (castfeedvalidator.com) before merge.
 *
 * Also: injects the LumaBot embed script (Task 26).
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

// Task 26 fills this in.
