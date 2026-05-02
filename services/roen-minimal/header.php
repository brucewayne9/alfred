<?php
/**
 * roen-minimal header
 *
 * Replaces Storefront's default header entirely.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}
?><!doctype html>
<html <?php language_attributes(); ?>>
<head>
    <meta charset="<?php bloginfo( 'charset' ); ?>" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="profile" href="https://gmpg.org/xfn/11" />
    <?php wp_head(); ?>
</head>

<body <?php body_class(); ?>>
<?php wp_body_open(); ?>

<header class="roen-header" role="banner">
    <div class="roen-container roen-header__inner">
        <a class="roen-header__brand" href="<?php echo esc_url( home_url( '/' ) ); ?>" aria-label="<?php esc_attr_e( 'Roen home', 'roen-minimal' ); ?>">
            <?php echo file_get_contents( get_stylesheet_directory() . '/assets/img/roen-wordmark.svg' ); // phpcs:ignore -- SVG inline ?>
        </a>

        <nav class="roen-header__nav" role="navigation" aria-label="<?php esc_attr_e( 'Primary', 'roen-minimal' ); ?>">
            <a href="<?php echo esc_url( get_permalink( wc_get_page_id( 'shop' ) ) ); ?>">shop</a>
            <a href="<?php echo esc_url( home_url( '/about/' ) ); ?>">about</a>
            <a class="roen-header__cart" href="<?php echo esc_url( wc_get_cart_url() ); ?>">
                cart (<span class="roen-cart-count"><?php echo (int) WC()->cart->get_cart_contents_count(); ?></span>)
            </a>
        </nav>
    </div>
</header>

<main id="content" class="roen-main">
