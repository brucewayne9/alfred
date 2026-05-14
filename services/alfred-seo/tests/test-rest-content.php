<?php
// services/alfred-seo/tests/test-rest-content.php
class Test_REST_Content extends WP_UnitTestCase {
    public function test_content_creates_draft() {
        wp_set_current_user( $this->factory->user->create( array( 'role' => 'administrator' ) ) );
        $req = new WP_REST_Request( 'POST', '/alfred-seo/v1/content' );
        $req->set_body_params( array(
            'title'             => 'Evil Eye Jewelry Guide',
            'content'           => '<p>Content body</p>',
            'meta_description'  => 'A guide to evil eye jewelry.',
            'slug'              => 'evil-eye-guide',
            'post_type'         => 'post',
            'status'            => 'draft',
        ) );
        $resp = rest_do_request( $req );
        $this->assertEquals( 201, $resp->get_status() );
        $data = $resp->get_data();
        $this->assertArrayHasKey( 'post_id', $data );
        $post = get_post( $data['post_id'] );
        $this->assertEquals( 'draft', $post->post_status );
    }
}
