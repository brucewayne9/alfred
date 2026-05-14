<?php
class Test_Schema_Collection extends WP_UnitTestCase {
    public function test_collection_for_product_category() {
        if ( ! taxonomy_exists( 'product_cat' ) ) { $this->markTestSkipped(); }
        $term_id = $this->factory->term->create( array(
            'taxonomy' => 'product_cat',
            'name'     => 'Bracelets',
        ) );
        $term   = get_term( $term_id );
        $result = alfred_seo_schema_collection( $term );
        $this->assertEquals( 'CollectionPage', $result['@type'] );
        $this->assertEquals( 'Bracelets', $result['name'] );
    }
}
