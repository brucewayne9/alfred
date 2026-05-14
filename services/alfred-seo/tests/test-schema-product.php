<?php
// services/alfred-seo/tests/test-schema-product.php

class Test_Schema_Product extends WP_UnitTestCase {
    private $product_id;

    public function set_up() {
        parent::set_up();
        if ( ! class_exists( 'WC_Product_Simple' ) ) {
            $this->markTestSkipped( 'WooCommerce not loaded in this test bootstrap.' );
        }
        $product = new WC_Product_Simple();
        $product->set_name( 'Red Bead Toggle Necklace' );
        $product->set_regular_price( '65.00' );
        $product->set_short_description( 'Faceted red beads on a knotted cord.' );
        $product->set_sku( 'roen-757' );
        $product->set_stock_status( 'instock' );
        $this->product_id = $product->save();
    }

    public function test_builder_returns_product_schema() {
        $result = alfred_seo_schema_product( $this->product_id );
        $this->assertEquals( 'Product', $result['@type'] );
        $this->assertEquals( 'Red Bead Toggle Necklace', $result['name'] );
        $this->assertEquals( 'roen-757', $result['sku'] );
        $this->assertEquals( 'Offer', $result['offers']['@type'] );
        $this->assertEquals( '65.00', $result['offers']['price'] );
        $this->assertEquals( 'https://schema.org/InStock', $result['offers']['availability'] );
    }

    public function test_builder_returns_null_for_unpurchasable() {
        $product = wc_get_product( $this->product_id );
        $product->set_status( 'draft' );
        $product->save();
        $this->assertNull( alfred_seo_schema_product( $this->product_id ) );
    }
}
