<?php
class Test_Robots extends WP_UnitTestCase {
    public function test_sitemap_line_added() {
        $out = apply_filters( 'robots_txt', "User-agent: *\nDisallow: /wp-admin/\n", true );
        $this->assertStringContainsString( 'Sitemap:', $out );
        $this->assertStringContainsString( '/alfred-sitemap.xml', $out );
    }
}
