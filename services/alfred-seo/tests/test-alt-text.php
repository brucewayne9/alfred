<?php
class Test_Alt_Text extends WP_UnitTestCase {
    public function test_fallback_to_filename_when_alfred_down() {
        $attachment_id = $this->factory->attachment->create_object(
            'red-bead-necklace.jpg',
            0,
            array( 'post_mime_type' => 'image/jpeg', 'post_status' => 'inherit' )
        );
        // Force Alfred lookup to fail by setting bad endpoint.
        alfred_seo_update_settings( array( 'alfred_endpoint' => 'http://127.0.0.1:1' ) );
        alfred_seo_fill_alt_text_on_upload( $attachment_id );
        $alt = get_post_meta( $attachment_id, '_wp_attachment_image_alt', true );
        $this->assertNotEmpty( $alt );
        $this->assertStringContainsString( 'red bead necklace', strtolower( $alt ) );
    }
}
