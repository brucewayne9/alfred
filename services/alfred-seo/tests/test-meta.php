<?php
class Test_Meta_Description extends WP_UnitTestCase {
    public function test_custom_field_overrides_excerpt() {
        $post_id = $this->factory->post->create( array( 'post_excerpt' => 'Auto excerpt.' ) );
        update_post_meta( $post_id, '_alfred_seo_meta_description', 'Alfred-pushed description.' );
        $this->go_to( get_permalink( $post_id ) );
        ob_start(); alfred_seo_render_meta_description(); $out = ob_get_clean();
        $this->assertStringContainsString( 'Alfred-pushed description.', $out );
    }

    public function test_excerpt_fallback() {
        $post_id = $this->factory->post->create( array( 'post_excerpt' => 'Auto excerpt.' ) );
        $this->go_to( get_permalink( $post_id ) );
        ob_start(); alfred_seo_render_meta_description(); $out = ob_get_clean();
        $this->assertStringContainsString( 'Auto excerpt.', $out );
    }
}
