<?php
/**
 * roen-minimal single-product gallery.
 *
 * Override of woocommerce/templates/single-product/product-image.php
 *
 * Layout: main image (left) + vertical thumbnail column (right).
 * Click a thumb -> swap main image. No Flexslider, no Zoom, no Lightbox.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

global $product;

$post_thumbnail_id = $product->get_image_id();
$gallery_ids       = $product->get_gallery_image_ids();

// All image IDs in order: featured first, then the rest.
$all_ids = array();
if ( $post_thumbnail_id ) {
    $all_ids[] = (int) $post_thumbnail_id;
}
foreach ( $gallery_ids as $id ) {
    $all_ids[] = (int) $id;
}

if ( empty( $all_ids ) ) {
    echo '<div class="roen-gallery roen-gallery--empty">';
    echo wc_placeholder_img( 'woocommerce_single' ); // phpcs:ignore
    echo '</div>';
    return;
}

$first_full = wp_get_attachment_image_src( $all_ids[0], 'full' );
$first_alt  = get_post_meta( $all_ids[0], '_wp_attachment_image_alt', true );
?>

<div class="roen-gallery" data-gallery>
    <div class="roen-gallery__main">
        <img
            class="roen-gallery__main-img"
            src="<?php echo esc_url( $first_full[0] ); ?>"
            alt="<?php echo esc_attr( $first_alt ?: $product->get_name() ); ?>"
            data-main
            <?php echo $first_full ? 'width="' . esc_attr( $first_full[1] ) . '" height="' . esc_attr( $first_full[2] ) . '"' : ''; ?>
        />
    </div>

    <?php if ( count( $all_ids ) > 1 ) : ?>
    <div class="roen-gallery__thumbs" role="list">
        <?php foreach ( $all_ids as $i => $id ) :
            $thumb = wp_get_attachment_image_src( $id, 'woocommerce_thumbnail' );
            $full  = wp_get_attachment_image_src( $id, 'full' );
            $alt   = get_post_meta( $id, '_wp_attachment_image_alt', true );
            if ( ! $thumb || ! $full ) {
                continue;
            }
            ?>
            <button
                type="button"
                class="roen-gallery__thumb<?php echo 0 === $i ? ' is-active' : ''; ?>"
                data-thumb
                data-full="<?php echo esc_url( $full[0] ); ?>"
                data-alt="<?php echo esc_attr( $alt ?: $product->get_name() ); ?>"
                aria-label="<?php echo esc_attr( sprintf( __( 'View image %d', 'roen-minimal' ), $i + 1 ) ); ?>"
            >
                <img src="<?php echo esc_url( $thumb[0] ); ?>" alt="" loading="lazy" />
            </button>
        <?php endforeach; ?>
    </div>
    <?php endif; ?>
</div>
