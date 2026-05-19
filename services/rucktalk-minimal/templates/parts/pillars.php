<?php
/**
 * Five Pillars grid — rucktalk-minimal homepage partial.
 *
 * Mirrors mockup .pillars block. Per-card "Today's take" snippet is
 * pulled from rt_pillar_snippet_today() (inc/pillar-snippets.php) and
 * is server-rendered — production does NOT use the mockup's 5s JS
 * rotation. The deterministic per-day picker means all visitors today
 * see the same line, and tomorrow it rotates.
 *
 * Each pillar links to /blog/category/<slug>/ where Phase 1B's SEO
 * work will land the pillar archives.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

$pillars = array(
    array(
        'slug' => 'health',
        'num'  => '01',
        'name' => 'Health',
        'what' => 'Keeping a body that holds up to the rest of life.',
    ),
    array(
        'slug' => 'business',
        'num'  => '02',
        'name' => 'Business',
        'what' => 'Building something while everything else is also on fire.',
    ),
    array(
        'slug' => 'family',
        'num'  => '03',
        'name' => 'Family',
        'what' => 'Showing up at home like you mean it.',
    ),
    array(
        'slug' => 'strength',
        'num'  => '04',
        'name' => 'Strength',
        'what' => 'Mind, body, the way you carry yourself through the week.',
    ),
    array(
        'slug' => 'shared',
        'num'  => '05',
        'name' => 'Shared',
        'what' => 'What everybody else is going through, said plainly.',
    ),
);

$snippet_fn_available = function_exists( 'rt_pillar_snippet_today' );
?>
<section class="pillars">
    <div class="wrap">

        <div class="pillars__head">
            <span class="section__eyebrow"><?php esc_html_e( "What we're talking about this week", 'rucktalk-minimal' ); ?></span>
            <h2 class="section__title">
                <?php
                echo wp_kses_post(
                    sprintf(
                        /* translators: %s = italic phrase styled by .section__title em */
                        __( 'Five things %s', 'rucktalk-minimal' ),
                        '<em>' . esc_html__( 'worth getting right.', 'rucktalk-minimal' ) . '</em>'
                    )
                );
                ?>
            </h2>
        </div>

        <div class="pillars__grid">
            <?php foreach ( $pillars as $p ) :
                $snippet = $snippet_fn_available ? rt_pillar_snippet_today( $p['slug'] ) : '';
                $url     = home_url( '/blog/category/' . $p['slug'] . '/' );
                ?>
                <a class="pillar" href="<?php echo esc_url( $url ); ?>" data-pillar="<?php echo esc_attr( $p['slug'] ); ?>">
                    <span class="pillar__num"><?php echo esc_html( $p['num'] ); ?></span>
                    <h3 class="pillar__name"><?php echo esc_html( $p['name'] ); ?></h3>
                    <p class="pillar__what"><?php echo esc_html( $p['what'] ); ?></p>
                    <?php if ( '' !== $snippet ) : ?>
                        <div class="pillar__latest">
                            <p class="pillar__latest-label"><?php esc_html_e( "Today's take", 'rucktalk-minimal' ); ?></p>
                            <p class="pillar__latest-snippet" data-pillar="<?php echo esc_attr( $p['slug'] ); ?>">
                                <?php echo esc_html( $snippet ); ?>
                            </p>
                        </div>
                    <?php endif; ?>
                    <span class="pillar__arrow" aria-hidden="true">&rarr;</span>
                </a>
            <?php endforeach; ?>
        </div>

    </div>
</section>
