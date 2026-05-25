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

    public function test_reads_sonaar_track_mp3_podcast_attachment_when_rt_url_absent() {
        // Create the audio attachment with realistic Sonaar metadata.
        $attachment_id = $this->factory->attachment->create_object(
            'rucktalk_ep210.mp3',
            0,
            array(
                'post_mime_type' => 'audio/mpeg',
                'post_type'      => 'attachment',
            )
        );
        update_post_meta( $attachment_id, '_wp_attached_file', '2026/05/rucktalk_ep210.mp3' );
        update_post_meta( $attachment_id, '_wp_attachment_metadata', array(
            'filesize'  => 28914669,
            'length'    => 1205,           // seconds — Sonaar stores integer
            'mime_type' => 'audio/mpeg',
        ) );

        $post_id = $this->factory->post->create( array( 'post_type' => 'podcast' ) );
        update_post_meta( $post_id, 'track_mp3_podcast', $attachment_id );

        $schema = alfred_seo_schema_audio_object( $post_id );

        $this->assertIsArray( $schema );
        $this->assertSame( 'AudioObject', $schema['@type'] );
        $this->assertStringContainsString( 'rucktalk_ep210.mp3', $schema['contentUrl'] );
        $this->assertSame( 'PT20M5S', $schema['duration'] );       // 1205s = 20:05
        $this->assertSame( 28914669, $schema['contentSize'] );
        $this->assertSame( 'audio/mpeg', $schema['encodingFormat'] );
    }

    public function test_rt_audio_url_takes_precedence_over_sonaar_meta() {
        // If a site explicitly sets _rt_audio_url, that wins (gives Mike an
        // override hatch without uninstalling Sonaar).
        $attachment_id = $this->factory->attachment->create_object( 'a.mp3', 0, array() );
        $post_id = $this->factory->post->create( array( 'post_type' => 'podcast' ) );
        update_post_meta( $post_id, 'track_mp3_podcast', $attachment_id );
        update_post_meta( $post_id, '_rt_audio_url', 'https://example.com/override.mp3' );

        $schema = alfred_seo_schema_audio_object( $post_id );
        $this->assertSame( 'https://example.com/override.mp3', $schema['contentUrl'] );
    }

    public function test_returns_null_when_neither_rt_nor_sonaar_meta_present() {
        $post_id = $this->factory->post->create( array( 'post_type' => 'podcast' ) );
        $this->assertNull( alfred_seo_schema_audio_object( $post_id ) );
    }
}
