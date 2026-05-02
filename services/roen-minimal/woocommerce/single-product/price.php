<?php
/**
 * roen-minimal price markup on single product.
 * Inter 300, no extra chrome.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

global $product;
?>
<p class="roen-single__price price"><?php echo $product->get_price_html(); // phpcs:ignore ?></p>
