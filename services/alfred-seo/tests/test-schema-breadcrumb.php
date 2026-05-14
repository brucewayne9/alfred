<?php
class Test_Schema_Breadcrumb extends WP_UnitTestCase {
    public function test_breadcrumb_for_post() {
        $cat_id = $this->factory->category->create( array( 'name' => 'Style Notes' ) );
        $post_id = $this->factory->post->create( array(
            'post_title' => 'Caring for beads',
            'post_category' => array( $cat_id ),
        ) );
        $this->go_to( get_permalink( $post_id ) );
        $result = alfred_seo_schema_breadcrumb();
        $this->assertEquals( 'BreadcrumbList', $result['@type'] );
        $this->assertGreaterThanOrEqual( 2, count( $result['itemListElement'] ) );
        $this->assertEquals( 'Home', $result['itemListElement'][0]['name'] );
    }
}
