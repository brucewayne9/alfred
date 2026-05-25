<?php
// services/alfred-seo/tests/test-schema-builder-podcast.php
class Test_Schema_Builder_Podcast extends WP_UnitTestCase {

    public function test_homepage_includes_podcast_series_when_flag_on() {
        update_option( 'alfred_seo_settings', array(
            'business_name'    => 'RuckTalk',
            'is_podcast_site'  => true,
            'audio_post_type'  => 'podcast',
        ) );
        $this->go_to( home_url( '/' ) );
        $payload = alfred_seo_build_schema_for_current_page();

        $types = array_column( $payload['@graph'] ?? array( $payload ), '@type' );
        $this->assertContains( 'PodcastSeries', $types );
    }

    public function test_single_podcast_post_emits_podcast_episode() {
        update_option( 'alfred_seo_settings', array(
            'business_name'    => 'RuckTalk',
            'is_podcast_site'  => true,
            'audio_post_type'  => 'podcast',
        ) );
        $post_id = $this->factory->post->create( array(
            'post_type'   => 'podcast',
            'post_status' => 'publish',
            'post_title'  => 'Test Episode',
        ) );
        update_post_meta( $post_id, '_rt_audio_url', 'https://rucktalk.com/audio/test.mp3' );

        $this->go_to( get_permalink( $post_id ) );
        $payload = alfred_seo_build_schema_for_current_page();

        $types = array_column( $payload['@graph'] ?? array( $payload ), '@type' );
        $this->assertContains( 'PodcastEpisode', $types );
    }
}
