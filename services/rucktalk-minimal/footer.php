<?php
/**
 * Footer — rucktalk-minimal.
 *
 * Renders: closes <main>, dark footer block with Sunday-email signup,
 * "Part of the Ground Rush ecosystem" strip (via [rt_ecosystem_strip]),
 * legal/meta row, LumaBot chat mount, and the newsletter popup HTML.
 *
 * Mockup parity: this footer's class names + structure mirror the
 * `.footer` block from rucktalk-homepage-mockup.html verbatim.
 *
 * The popup partial is included here (not in front-page.php) so it is
 * available on every page — popup.js controls visibility via scroll
 * depth / exit intent + a 14-day dismiss cookie.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}
?>
</main><!-- .rt-main -->

<!-- ─── SITE FOOTER ─── -->
<footer class="footer" role="contentinfo">
    <div class="footer__inner">

        <div class="footer__signup">
            <div class="footer__signup-copy">
                <h3>One Sunday email. <em>That's it.</em></h3>
                <p>Best post of the week + the episode worth your time. 8 AM ET.</p>
            </div>
            <?php echo do_shortcode( '[rt_signup placement="footer"]' ); ?>
        </div>

        <?php echo do_shortcode( '[rt_ecosystem_strip]' ); ?>

        <div class="footer__meta">
            <p>&copy; <?php echo esc_html( gmdate( 'Y' ) ); ?> RuckTalk &middot; <a href="https://groundrushlabs.com" rel="noopener" target="_blank">Ground Rush Labs</a></p>
            <ul class="footer__links">
                <?php
                if ( has_nav_menu( 'footer' ) ) {
                    wp_nav_menu( array(
                        'theme_location' => 'footer',
                        'container'      => false,
                        'items_wrap'     => '%3$s',
                        'fallback_cb'    => false,
                        'depth'          => 1,
                    ) );
                } else {
                    ?>
                    <li><a href="<?php echo esc_url( home_url( '/terms/' ) ); ?>"><?php esc_html_e( 'Terms', 'rucktalk-minimal' ); ?></a></li>
                    <li><a href="<?php echo esc_url( home_url( '/privacy/' ) ); ?>"><?php esc_html_e( 'Privacy', 'rucktalk-minimal' ); ?></a></li>
                    <li><a href="<?php echo esc_url( home_url( '/feed/podcast/' ) ); ?>"><?php esc_html_e( 'RSS', 'rucktalk-minimal' ); ?></a></li>
                    <li><a href="<?php echo esc_url( home_url( '/contact/' ) ); ?>"><?php esc_html_e( 'Contact', 'rucktalk-minimal' ); ?></a></li>
                    <?php
                }
                ?>
            </ul>
        </div>

    </div>
</footer>

<!-- ─── LUMABOT CHAT MOUNT ─── -->
<div id="rt-lumabot-mount" data-bot-id="rucktalk" aria-hidden="true">
    <!-- LumaBot widget embed is appended here by inc/sonaar-overrides.php (Task 26) -->
</div>

<!-- ─── NEWSLETTER POPUP (hidden by default; shown via popup.js) ─── -->
<?php get_template_part( 'templates/parts/popup-signup' ); ?>

<?php wp_footer(); ?>
</body>
</html>
