<?php
/**
 * roen-minimal footer
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}
?>
</main><?php // close .roen-main from header.php ?>

<footer class="roen-footer">
    <div class="roen-container roen-footer__inner">
        <div class="roen-footer__brand">
            <?php echo roen_wordmark_svg(); // phpcs:ignore WordPress.Security.EscapeOutput.OutputNotEscaped -- SVG asset shipped with theme ?>
            <p class="roen-footer__legal">© <?php echo (int) date( 'Y' ); ?> Roen Handmade.</p>
        </div>

        <nav class="roen-footer__col" aria-label="<?php esc_attr_e( 'Social', 'roen-minimal' ); ?>">
            <h4 class="roen-footer__heading">follow</h4>
            <a href="https://www.instagram.com/roenhandmade/" rel="noopener" target="_blank">instagram</a>
            <a href="https://www.facebook.com/roenhandmade/" rel="noopener" target="_blank">facebook</a>
        </nav>

        <nav class="roen-footer__col" aria-label="<?php esc_attr_e( 'Help', 'roen-minimal' ); ?>">
            <h4 class="roen-footer__heading">help</h4>
            <a href="<?php echo esc_url( home_url( '/about/' ) ); ?>">about</a>
            <a href="mailto:mjohnson@groundrushinc.com">contact</a>
            <a href="<?php echo esc_url( home_url( '/privacy-policy/' ) ); ?>">privacy</a>
            <a href="<?php echo esc_url( home_url( '/refund_returns/' ) ); ?>">returns</a>
        </nav>
    </div>
</footer>

<?php wp_footer(); ?>
</body>
</html>
