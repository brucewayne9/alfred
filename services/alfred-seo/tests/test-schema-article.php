<?php
class Test_Schema_Article extends WP_UnitTestCase {
    public function test_article_schema_for_post() {
        $author_id = $this->factory->user->create( array( 'display_name' => 'Roen' ) );
        $post_id = $this->factory->post->create( array(
            'post_title' => 'How to care for handmade jewelry',
            'post_content' => 'Some content here.',
            'post_author' => $author_id,
        ) );
        $result = alfred_seo_schema_article( $post_id );
        $this->assertEquals( 'Article', $result['@type'] );
        $this->assertEquals( 'How to care for handmade jewelry', $result['headline'] );
        $this->assertEquals( 'Person', $result['author']['@type'] );
    }
}
