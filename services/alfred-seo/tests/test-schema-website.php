<?php
class Test_Schema_Website extends WP_UnitTestCase {
    public function test_website_has_searchaction() {
        $result = alfred_seo_schema_website();
        $this->assertEquals( 'WebSite', $result['@type'] );
        $this->assertEquals( 'SearchAction', $result['potentialAction']['@type'] );
        $this->assertStringContainsString( '/?s={search_term_string}', $result['potentialAction']['target']['urlTemplate'] );
    }
}
