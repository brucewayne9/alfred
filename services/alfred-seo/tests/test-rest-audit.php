<?php
// services/alfred-seo/tests/test-rest-audit.php
class Test_REST_Audit extends WP_UnitTestCase {
    public function test_audit_returns_page_list() {
        wp_set_current_user( $this->factory->user->create( array( 'role' => 'administrator' ) ) );
        $req = new WP_REST_Request( 'GET', '/alfred-seo/v1/audit' );
        $resp = rest_do_request( $req );
        $this->assertEquals( 200, $resp->get_status() );
        $data = $resp->get_data();
        $this->assertArrayHasKey( 'pages', $data );
        $this->assertArrayHasKey( 'missing_meta', $data );
    }

    public function test_audit_rejects_unauthorized() {
        wp_set_current_user( 0 );
        $req = new WP_REST_Request( 'GET', '/alfred-seo/v1/audit' );
        $resp = rest_do_request( $req );
        $this->assertEquals( 401, $resp->get_status() );
    }
}
