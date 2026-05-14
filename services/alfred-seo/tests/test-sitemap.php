<?php
class Test_Sitemap extends WP_UnitTestCase {
    public function test_sitemap_index_has_urls() {
        $xml = alfred_seo_render_sitemap_index();
        $this->assertStringContainsString( '<?xml version="1.0"', $xml );
        $this->assertStringContainsString( '<sitemapindex', $xml );
        $this->assertStringContainsString( '/alfred-sitemap-pages.xml', $xml );
        $this->assertStringContainsString( '/alfred-sitemap-posts.xml', $xml );
    }

    public function test_pages_sitemap_includes_published_pages() {
        $page_id = $this->factory->post->create( array( 'post_type' => 'page', 'post_status' => 'publish' ) );
        $xml = alfred_seo_render_sitemap_for( 'pages' );
        $this->assertStringContainsString( get_permalink( $page_id ), $xml );
    }
}
