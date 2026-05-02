<?php
/**
 * roen-minimal simple-product add-to-cart form.
 *
 * Override of woocommerce/templates/single-product/add-to-cart/simple.php
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

global $product;

if ( ! $product->is_purchasable() ) {
    return;
}

echo wc_get_stock_html( $product ); // phpcs:ignore

if ( $product->is_in_stock() ) :
    do_action( 'woocommerce_before_add_to_cart_form' );
    ?>
    <form class="cart roen-atc-form" action="<?php echo esc_url( apply_filters( 'woocommerce_add_to_cart_form_action', $product->get_permalink() ) ); ?>" method="post" enctype="multipart/form-data">
        <?php do_action( 'woocommerce_before_add_to_cart_button' ); ?>
        <?php do_action( 'woocommerce_before_add_to_cart_quantity' ); ?>

        <?php
        woocommerce_quantity_input( array(
            'min_value'   => apply_filters( 'woocommerce_quantity_input_min', $product->get_min_purchase_quantity(), $product ),
            'max_value'   => apply_filters( 'woocommerce_quantity_input_max', $product->get_max_purchase_quantity(), $product ),
            'input_value' => isset( $_POST['quantity'] ) ? wc_stock_amount( wp_unslash( $_POST['quantity'] ) ) : $product->get_min_purchase_quantity(),
        ) );
        ?>

        <?php do_action( 'woocommerce_after_add_to_cart_quantity' ); ?>

        <button type="submit"
                name="add-to-cart"
                value="<?php echo esc_attr( $product->get_id() ); ?>"
                class="single_add_to_cart_button button alt roen-atc-btn">
            add to cart
        </button>

        <?php do_action( 'woocommerce_after_add_to_cart_button' ); ?>
    </form>
    <?php
    do_action( 'woocommerce_after_add_to_cart_form' );
endif;
