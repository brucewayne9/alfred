<?php
class Test_Internal_Links extends WP_UnitTestCase {
    public function test_phrase_replaced_with_link() {
        alfred_seo_update_settings( array( 'internal_links' => array(
            'evil eye bracelet' => 'https://example.com/products/evil-eye/',
        ) ) );
        $out = alfred_seo_apply_internal_links( '<p>Our evil eye bracelet is popular.</p>' );
        $this->assertStringContainsString( 'href="https://example.com/products/evil-eye/"', $out );
        $this->assertStringContainsString( '>evil eye bracelet</a>', $out );
    }

    public function test_does_not_link_inside_existing_anchor() {
        alfred_seo_update_settings( array( 'internal_links' => array( 'evil eye' => 'https://x.com/' ) ) );
        $out = alfred_seo_apply_internal_links( '<a href="/foo">evil eye</a>' );
        $this->assertStringContainsString( '<a href="/foo">evil eye</a>', $out );
        $this->assertStringNotContainsString( 'https://x.com/', $out );
    }

    public function test_only_links_first_occurrence_per_phrase() {
        alfred_seo_update_settings( array( 'internal_links' => array( 'bracelet' => 'https://x.com/' ) ) );
        $out = alfred_seo_apply_internal_links( 'A bracelet and another bracelet.' );
        $this->assertEquals( 1, substr_count( $out, 'href="https://x.com/"' ) );
    }
}
