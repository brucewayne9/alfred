<?php
class Test_Schema_Organization extends WP_UnitTestCase {
    public function test_basic_organization() {
        alfred_seo_update_settings( array(
            'business_name' => 'Roen',
            'business_type' => 'Organization',
        ) );
        $result = alfred_seo_schema_organization();
        $this->assertEquals( 'Organization', $result['@type'] );
        $this->assertEquals( 'Roen', $result['name'] );
    }

    public function test_localbusiness_includes_address() {
        alfred_seo_update_settings( array(
            'business_name' => 'Roen',
            'business_type' => 'LocalBusiness',
            'local_address' => array(
                'streetAddress'   => '123 Atlanta Ave',
                'addressLocality' => 'Atlanta',
                'addressRegion'   => 'GA',
                'postalCode'      => '30303',
                'addressCountry'  => 'US',
            ),
        ) );
        $result = alfred_seo_schema_organization();
        $this->assertEquals( 'LocalBusiness', $result['@type'] );
        $this->assertEquals( 'Atlanta', $result['address']['addressLocality'] );
    }
}
