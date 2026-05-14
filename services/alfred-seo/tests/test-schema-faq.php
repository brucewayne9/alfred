<?php
class Test_Schema_FAQ extends WP_UnitTestCase {
    public function test_faq_extracted_from_h2_question_pattern() {
        $content = "<h2>What is a toggle clasp?</h2><p>A toggle clasp has a bar and a ring.</p>"
                 . "<h2>How do I size this?</h2><p>Standard 7 inches; ask for resize.</p>";
        $result = alfred_seo_schema_faq_from_content( $content );
        $this->assertEquals( 'FAQPage', $result['@type'] );
        $this->assertCount( 2, $result['mainEntity'] );
        $this->assertEquals( 'What is a toggle clasp?', $result['mainEntity'][0]['name'] );
    }

    public function test_returns_null_when_no_faqs() {
        $this->assertNull( alfred_seo_schema_faq_from_content( '<p>just prose</p>' ) );
    }
}
