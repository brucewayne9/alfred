<?php
// services/alfred-seo/inc/schema/organization.php
if ( ! defined( 'ABSPATH' ) ) { exit; }

function alfred_seo_schema_organization() {
    $settings = alfred_seo_get_settings();
    $type     = in_array( $settings['business_type'], array( 'Organization', 'LocalBusiness', 'SoftwareApplication', 'Service' ), true )
        ? $settings['business_type']
        : 'Organization';

    $schema = array(
        '@context' => 'https://schema.org',
        '@type'    => $type,
        'name'     => $settings['business_name'] ?: get_bloginfo( 'name' ),
        'url'      => home_url( '/' ),
    );

    $logo_url = get_site_icon_url( 512 );
    if ( $logo_url ) {
        $schema['logo'] = $logo_url;
    }

    if ( ! empty( $settings['social_handles'] ) ) {
        $sames = array();
        foreach ( $settings['social_handles'] as $platform => $handle ) {
            $handle = ltrim( $handle, '@' );
            switch ( strtolower( $platform ) ) {
                case 'twitter':   $sames[] = 'https://twitter.com/' . $handle; break;
                case 'instagram': $sames[] = 'https://instagram.com/' . $handle; break;
                case 'facebook':  $sames[] = 'https://facebook.com/' . $handle; break;
                case 'pinterest': $sames[] = 'https://pinterest.com/' . $handle; break;
                case 'youtube':   $sames[] = 'https://youtube.com/@' . $handle; break;
                case 'linkedin':  $sames[] = 'https://linkedin.com/in/' . $handle; break;
            }
        }
        if ( $sames ) { $schema['sameAs'] = $sames; }
    }

    if ( 'LocalBusiness' === $type && ! empty( $settings['local_address'] ) ) {
        $schema['address'] = array_merge(
            array( '@type' => 'PostalAddress' ),
            array_filter( $settings['local_address'] )
        );
    }

    return $schema;
}
