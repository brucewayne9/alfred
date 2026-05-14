<?php
class Test_Open_Graph extends WP_UnitTestCase {
    public function test_og_tags_emitted_on_home() {
        $this->go_to( home_url( '/' ) );
        ob_start();
        alfred_seo_render_open_graph();
        $out = ob_get_clean();
        $this->assertStringContainsString( 'property="og:title"', $out );
        $this->assertStringContainsString( 'property="og:url"', $out );
        $this->assertStringContainsString( 'name="twitter:card"', $out );
    }
}
