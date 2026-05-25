<?php
// services/alfred-seo/tests/test-schema-podcast-episode.php
class Test_Schema_PodcastEpisode extends WP_UnitTestCase {

    public function test_returns_null_for_non_podcast_post_type() {
        $post_id = $this->factory->post->create( array( 'post_type' => 'post' ) );
        $this->assertNull( alfred_seo_schema_podcast_episode( $post_id ) );
    }

    public function test_emits_required_episode_fields() {
        update_option( 'alfred_seo_settings', array(
            'business_name'   => 'RuckTalk',
            'audio_post_type' => 'podcast',
        ) );
        $post_id = $this->factory->post->create( array(
            'post_type'    => 'podcast',
            'post_title'   => 'The Presence Protocol',
            'post_excerpt' => 'Five tactical shifts for the entrepreneur dad.',
            'post_status'  => 'publish',
        ) );
        update_post_meta( $post_id, '_rt_audio_url',      'https://rucktalk.com/audio/ep-42.mp3' );
        update_post_meta( $post_id, '_rt_audio_duration', 'PT42M18S' );
        update_post_meta( $post_id, '_rt_episode_number', '42' );

        $schema = alfred_seo_schema_podcast_episode( $post_id );

        $this->assertIsArray( $schema );
        $this->assertSame( 'PodcastEpisode', $schema['@type'] );
        $this->assertSame( 'The Presence Protocol', $schema['name'] );
        $this->assertSame( '42', (string) $schema['episodeNumber'] );
        $this->assertSame( 'PT42M18S', $schema['timeRequired'] );
        $this->assertArrayHasKey( 'partOfSeries', $schema );
        $this->assertSame( 'PodcastSeries', $schema['partOfSeries']['@type'] );
        $this->assertArrayHasKey( 'associatedMedia', $schema );
        $this->assertSame( 'AudioObject', $schema['associatedMedia']['@type'] );
    }
}
