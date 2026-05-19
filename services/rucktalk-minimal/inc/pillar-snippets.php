<?php
/**
 * Pillar snippet manager — rucktalk-minimal.
 *
 * Stores 5 pools of one-line "today's take" advice snippets (one pool
 * per pillar) in a single WP option `rt_pillar_snippets`. Each homepage
 * render picks ONE snippet per pillar deterministically keyed by
 * (pillar, date) — all visitors today see the same snippet, but it
 * rotates tomorrow.
 *
 * Admin UI: Settings → "RuckTalk Pillars" — five textareas, one snippet
 * per line. Mike pastes new snippets, hits Save, done.
 *
 * Source spec: docs/superpowers/specs/2026-05-19-rucktalk-design-
 * language.md §5e (starter pool included there).
 *
 * Note on the constants: `const` at file scope is intentional — these
 * pin the canonical pillar order so the deterministic index calculation
 * (day_of_year + pillar_idx) stays stable across releases.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

if ( ! defined( 'RT_PILLAR_OPTION' ) ) {
    define( 'RT_PILLAR_OPTION', 'rt_pillar_snippets' );
}
if ( ! defined( 'RT_PILLAR_OPTION_GROUP' ) ) {
    define( 'RT_PILLAR_OPTION_GROUP', 'rt_pillars' );
}

/**
 * Canonical pillar order — used to compute the per-pillar rotation
 * offset. Order matters; do not reorder without a one-time migration.
 *
 * @return string[]
 */
function rt_pillars_list() {
    return array( 'health', 'business', 'family', 'strength', 'shared' );
}

/**
 * Get today's snippet for a given pillar. Returns empty string if the
 * pool is empty (template hides the "Today's take" block when empty).
 *
 * @param string $pillar Pillar slug.
 * @return string
 */
function rt_pillar_snippet_today( $pillar ) {
    $pillar = (string) $pillar;
    $pools  = get_option( RT_PILLAR_OPTION, array() );
    if ( ! is_array( $pools ) ) {
        return '';
    }
    $pool = isset( $pools[ $pillar ] ) && is_array( $pools[ $pillar ] ) ? $pools[ $pillar ] : array();
    if ( empty( $pool ) ) {
        return '';
    }

    $pillars     = rt_pillars_list();
    $pillar_idx  = array_search( $pillar, $pillars, true );
    if ( false === $pillar_idx ) {
        $pillar_idx = 0;
    }
    // gmdate('z') is 0-indexed day of year in UTC. Combined with the
    // pillar offset, this means each pillar advances daily but on a
    // staggered schedule so the row doesn't all change in lockstep.
    $day_of_year = (int) gmdate( 'z' );
    $idx         = ( $day_of_year + (int) $pillar_idx ) % count( $pool );

    return (string) $pool[ $idx ];
}

/**
 * Register the option + schema with WP Settings API.
 */
function rt_pillar_settings_init() {
    register_setting(
        RT_PILLAR_OPTION_GROUP,
        RT_PILLAR_OPTION,
        array(
            'type'              => 'array',
            'description'       => 'RuckTalk five-pillar daily snippet pools.',
            'sanitize_callback' => 'rt_pillar_sanitize',
            'default'           => array(),
        )
    );
}
add_action( 'admin_init', 'rt_pillar_settings_init' );

/**
 * Sanitize the option payload. Accepts a map of pillar => textarea
 * string OR pillar => array, returns a map of pillar => array<string>
 * with empties / HTML stripped.
 *
 * @param mixed $input Raw POSTed value.
 * @return array<string,string[]>
 */
function rt_pillar_sanitize( $input ) {
    if ( ! is_array( $input ) ) {
        return array();
    }
    $clean = array();
    foreach ( rt_pillars_list() as $p ) {
        $raw   = isset( $input[ $p ] ) ? $input[ $p ] : '';
        $lines = is_string( $raw ) ? preg_split( "/\r\n|\r|\n/", $raw ) : (array) $raw;
        $clean[ $p ] = array_values(
            array_filter(
                array_map(
                    static function ( $line ) {
                        return trim( wp_strip_all_tags( (string) $line ) );
                    },
                    $lines
                ),
                'strlen'
            )
        );
    }
    return $clean;
}

/**
 * Register the admin settings page under Settings → RuckTalk Pillars.
 */
function rt_pillar_menu() {
    add_options_page(
        __( 'RuckTalk Pillars', 'rucktalk-minimal' ),
        __( 'RuckTalk Pillars', 'rucktalk-minimal' ),
        'manage_options',
        'rt-pillars',
        'rt_pillar_settings_page'
    );
}
add_action( 'admin_menu', 'rt_pillar_menu' );

/**
 * Render the settings page.
 */
function rt_pillar_settings_page() {
    if ( ! current_user_can( 'manage_options' ) ) {
        return;
    }
    $pools = get_option( RT_PILLAR_OPTION, array() );
    if ( ! is_array( $pools ) ) {
        $pools = array();
    }
    ?>
    <div class="wrap">
        <h1><?php esc_html_e( "RuckTalk Pillars — Today's Take", 'rucktalk-minimal' ); ?></h1>
        <p>
            <?php
            esc_html_e(
                'One snippet per line. The homepage cycles through them in order, one per day per pillar. All visitors on a given day see the same snippet; tomorrow it rotates.',
                'rucktalk-minimal'
            );
            ?>
        </p>
        <form method="post" action="options.php">
            <?php settings_fields( RT_PILLAR_OPTION_GROUP ); ?>
            <?php foreach ( rt_pillars_list() as $p ) :
                $pool  = isset( $pools[ $p ] ) && is_array( $pools[ $p ] ) ? $pools[ $p ] : array();
                $value = implode( "\n", $pool );
                $today = rt_pillar_snippet_today( $p );
                ?>
                <h2><?php echo esc_html( ucfirst( $p ) ); ?></h2>
                <?php if ( '' !== $today ) : ?>
                    <p style="color:#555;font-style:italic;margin:0 0 6px;">
                        <strong><?php esc_html_e( "Today's pick:", 'rucktalk-minimal' ); ?></strong>
                        <?php echo esc_html( $today ); ?>
                    </p>
                <?php endif; ?>
                <textarea name="<?php echo esc_attr( RT_PILLAR_OPTION ); ?>[<?php echo esc_attr( $p ); ?>]"
                          rows="6"
                          style="width:100%;max-width:780px;font-family:Menlo,Consolas,monospace;font-size:13px;line-height:1.5;"><?php
                    echo esc_textarea( $value );
                ?></textarea>
            <?php endforeach; ?>
            <?php submit_button(); ?>
        </form>
    </div>
    <?php
}
