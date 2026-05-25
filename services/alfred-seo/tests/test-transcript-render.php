<?php
class Test_Transcript_Render extends WP_UnitTestCase {

    public function test_appends_transcript_block_on_singular_podcast_post() {
        update_option( 'alfred_seo_settings', array(
            'audio_post_type'           => 'podcast',
            'transcript_post_meta_key'  => '_rt_transcript',
        ) );
        $post_id = $this->factory->post->create( array(
            'post_type'    => 'podcast',
            'post_status'  => 'publish',
            'post_content' => '<p>Episode body</p>',
        ) );
        update_post_meta( $post_id, '_rt_transcript', "Hello world.\n\nSecond paragraph." );

        $this->go_to( get_permalink( $post_id ) );
        the_post();
        $rendered = apply_filters( 'the_content', get_the_content() );

        $this->assertStringContainsString( '<details class="alfred-seo-transcript"', $rendered );
        $this->assertStringContainsString( 'Transcript (5 words)', $rendered );
        $this->assertStringContainsString( '<p>Hello world.</p>', $rendered );
        $this->assertStringContainsString( '<p>Second paragraph.</p>', $rendered );
    }

    public function test_does_not_append_on_non_audio_post_type() {
        update_option( 'alfred_seo_settings', array( 'audio_post_type' => 'podcast' ) );
        $post_id = $this->factory->post->create( array( 'post_type' => 'post' ) );
        update_post_meta( $post_id, '_rt_transcript', 'Should not appear.' );

        $this->go_to( get_permalink( $post_id ) );
        the_post();
        $rendered = apply_filters( 'the_content', get_the_content() );

        $this->assertStringNotContainsString( 'alfred-seo-transcript', $rendered );
    }

    public function test_does_not_append_when_meta_empty() {
        update_option( 'alfred_seo_settings', array( 'audio_post_type' => 'podcast' ) );
        $post_id = $this->factory->post->create( array(
            'post_type'   => 'podcast',
            'post_status' => 'publish',
        ) );
        $this->go_to( get_permalink( $post_id ) );
        the_post();
        $rendered = apply_filters( 'the_content', get_the_content() );

        $this->assertStringNotContainsString( 'alfred-seo-transcript', $rendered );
    }
}
