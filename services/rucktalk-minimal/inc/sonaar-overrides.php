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
 * Also injects the LumaBot chat widget embed before </body>. Config lives
 * in three WP options:
 *   rt_lumabot_script_url  — embed script src (e.g. chatui-dev.groundrushlabs.com/embed/chat-widget.js)
 *   rt_lumabot_tenant_id   — Luma tenant id
 *   rt_lumabot_base_url    — Luma API base url
 *
 * If any of the three is empty, the embed is skipped silently so the page
 * still renders cleanly during config gaps.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

/**
 * Emit the LumaBot embed script. Bound to wp_footer so it lands just
 * before </body>, after the chat mount div in footer.php.
 */
function rt_lumabot_embed() {
    $script_url = (string) get_option( 'rt_lumabot_script_url', '' );
    $tenant_id  = (string) get_option( 'rt_lumabot_tenant_id', '' );
    $base_url   = (string) get_option( 'rt_lumabot_base_url', '' );

    if ( '' === $script_url || '' === $tenant_id || '' === $base_url ) {
        return; // graceful skip — no widget renders
    }
    ?>
    <!-- Chat Widget by Luma — rucktalk-minimal -->
    <script
        src="<?php echo esc_url( $script_url ); ?>"
        data-tenant-id="<?php echo esc_attr( $tenant_id ); ?>"
        data-base-url="<?php echo esc_url( $base_url ); ?>"
        defer></script>
    <?php
}
add_action( 'wp_footer', 'rt_lumabot_embed', 99 );
