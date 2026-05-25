<?php
// services/alfred-seo/tests/test-schema-audio-object.php
class Test_Schema_AudioObject extends WP_UnitTestCase {

    public function test_returns_null_when_no_audio_url_in_meta() {
        $post_id = $this->factory->post->create( array( 'post_type' => 'podcast' ) );
        $this->assertNull( alfred_seo_schema_audio_object( $post_id ) );
    }

    public function test_builds_audio_object_from_post_meta() {
        $post_id = $this->factory->post->create( array(
            'post_type'  => 'podcast',
            'post_title' => 'The Presence Protocol',
        ) );
        update_post_meta( $post_id, '_rt_audio_url',      'https://rucktalk.com/audio/ep-42.mp3' );
        update_post_meta( $post_id, '_rt_audio_duration', 'PT42M18S' );
        update_post_meta( $post_id, '_rt_audio_bytes',    '40432101' );

        $schema = alfred_seo_schema_audio_object( $post_id );

        $this->assertIsArray( $schema );
        $this->assertSame( 'AudioObject', $schema['@type'] );
        $this->assertSame( 'https://rucktalk.com/audio/ep-42.mp3', $schema['contentUrl'] );
        $this->assertSame( 'PT42M18S', $schema['duration'] );
        $this->assertSame( '40432101', (string) $schema['contentSize'] );
        $this->assertSame( 'audio/mpeg', $schema['encodingFormat'] );
    }
}
