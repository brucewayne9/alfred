<?php
// services/alfred-seo/tests/test-schema-podcast-series.php
class Test_Schema_PodcastSeries extends WP_UnitTestCase {

    public function test_emits_required_fields_on_home() {
        update_option( 'alfred_seo_settings', array(
            'business_name'  => 'RuckTalk',
            'site_tagline'   => 'Exploring together the art of tactical living.',
            'audio_post_type'=> 'podcast',
        ) );

        $schema = alfred_seo_schema_podcast_series();

        $this->assertIsArray( $schema );
        $this->assertSame( 'PodcastSeries', $schema['@type'] );
        $this->assertSame( 'RuckTalk', $schema['name'] );
        $this->assertSame( home_url( '/' ), $schema['url'] );
        $this->assertArrayHasKey( 'description', $schema );
        $this->assertArrayHasKey( 'webFeed', $schema );
        $this->assertArrayHasKey( 'inLanguage', $schema );
    }

    public function test_returns_null_when_no_business_name() {
        update_option( 'alfred_seo_settings', array() );
        $this->assertNull( alfred_seo_schema_podcast_series() );
    }
}
