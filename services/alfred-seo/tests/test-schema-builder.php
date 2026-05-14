<?php
class Test_Schema_Builder extends WP_UnitTestCase {
    public function test_dispatch_returns_array_for_homepage() {
        $this->go_to( home_url( '/' ) );
        $result = alfred_seo_build_schema_for_current_page();
        $this->assertIsArray( $result );
        $this->assertArrayHasKey( '@context', $result );
        $this->assertEquals( 'https://schema.org', $result['@context'] );
    }

    public function test_validator_rejects_missing_context() {
        $bad = array( '@type' => 'Product', 'name' => 'X' );
        $this->assertFalse( alfred_seo_validate_jsonld( $bad ) );
    }

    public function test_validator_accepts_valid_payload() {
        $good = array( '@context' => 'https://schema.org', '@type' => 'Product', 'name' => 'X' );
        $this->assertTrue( alfred_seo_validate_jsonld( $good ) );
    }
}
