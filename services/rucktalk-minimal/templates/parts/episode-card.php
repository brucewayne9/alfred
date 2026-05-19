<?php
/**
 * Latest podcast episode card — rucktalk-minimal homepage partial.
 *
 * Mirrors mockup .episode block. Queries the Sonaar `podcast` CPT
 * (which lives in the parent's bundled sonaar-music suite — keep
 * parent active). If no episodes exist yet, the partial renders an
 * empty state instead of erroring.
 *
 * The Listen / Watch toggle is a visual pill only — wiring an actual
 * Watch view lands in Phase 5 per the design-language overlay. For
 * v1 we display both but only Listen is active.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

$q = new WP_Query(
    array(
        'post_type'           => 'podcast',
        'posts_per_page'      => 1,
        'post_status'         => 'publish',
        'orderby'             => 'date',
        'order'               => 'DESC',
        'ignore_sticky_posts' => true,
        'no_found_rows'       => true,
    )
);

if ( ! $q->have_posts() ) {
    // Empty state — keep the section but show a holding card so the
    // homepage layout doesn't collapse during initial content seeding.
    ?>
    <section class="episode">
        <div class="wrap" style="display: contents;">
            <div class="episode__art" aria-hidden="true">
                <span class="episode__ep"><?php esc_html_e( 'Coming soon', 'rucktalk-minimal' ); ?></span>
                <h3 class="episode__title-on-art"><?php echo wp_kses_post( __( 'First episode <em>drops shortly</em>', 'rucktalk-minimal' ) ); ?></h3>
                <span class="episode__date">&nbsp;</span>
            </div>
            <div class="episode__copy">
                <span class="episode__num">&mdash; <?php esc_html_e( 'Pilot episode pending', 'rucktalk-minimal' ); ?></span>
                <h2 class="episode__title"><?php esc_html_e( 'The first episode is on its way.', 'rucktalk-minimal' ); ?></h2>
                <p class="episode__excerpt"><?php esc_html_e( 'Subscribe below and we\'ll send you a note the moment it drops.', 'rucktalk-minimal' ); ?></p>
                <div class="episode__actions">
                    <a class="btn btn--primary" href="#rt-hero-signup"><?php esc_html_e( 'Notify me', 'rucktalk-minimal' ); ?></a>
                </div>
            </div>
        </div>
    </section>
    <?php
    return;
}

$q->the_post();
$post_id     = get_the_ID();
$permalink   = get_permalink();
$title       = get_the_title();
$excerpt     = wp_strip_all_tags( get_the_excerpt() );
$published   = get_the_date( 'M j' );
$published_full = get_the_date( 'M j, Y' );

// Sonaar stores episode number on the post — fall back gracefully if
// the meta key isn't set.
$episode_num = (int) get_post_meta( $post_id, 'podcast_itunes_episode_number', true );
if ( ! $episode_num ) {
    $episode_num = (int) get_post_meta( $post_id, 'episode_number', true );
}
$duration = get_post_meta( $post_id, 'podcast_itunes_duration', true );
if ( ! $duration ) {
    $duration = get_post_meta( $post_id, 'duration', true );
}

$cover_url = get_the_post_thumbnail_url( $post_id, 'large' );

// First category (lowercased, used as a label) — defaults to "Episode".
$cats     = get_the_category();
$cat_name = ! empty( $cats ) ? $cats[0]->name : __( 'Episode', 'rucktalk-minimal' );
?>
<section class="episode">
    <div class="wrap" style="display: contents;">
        <div class="episode__art" style="<?php echo $cover_url ? 'background-image:url(' . esc_url( $cover_url ) . ');background-size:cover;background-position:center;' : ''; ?>">
            <span class="episode__ep">
                <?php
                if ( $episode_num ) {
                    printf( esc_html__( 'Ep. %1$d · %2$s', 'rucktalk-minimal' ), $episode_num, esc_html( $published ) );
                } else {
                    echo esc_html( $published );
                }
                ?>
            </span>
            <h3 class="episode__title-on-art"><?php echo esc_html( $title ); ?></h3>
            <span class="episode__date">
                <?php
                if ( $duration ) {
                    printf( esc_html__( '%1$s · %2$s', 'rucktalk-minimal' ), esc_html( $published_full ), esc_html( $duration ) );
                } else {
                    echo esc_html( $published_full );
                }
                ?>
            </span>
        </div>
        <div class="episode__copy">
            <?php if ( $episode_num ) : ?>
                <span class="episode__num">&mdash; <?php printf( esc_html__( 'Episode No. %d', 'rucktalk-minimal' ), $episode_num ); ?></span>
            <?php endif; ?>
            <h2 class="episode__title"><a href="<?php echo esc_url( $permalink ); ?>" style="color:inherit;text-decoration:none;"><?php echo esc_html( $title ); ?></a></h2>
            <?php if ( '' !== $excerpt ) : ?>
                <p class="episode__excerpt"><?php echo esc_html( wp_trim_words( $excerpt, 40 ) ); ?></p>
            <?php endif; ?>

            <div class="episode__formats" role="tablist" aria-label="<?php esc_attr_e( 'Episode format', 'rucktalk-minimal' ); ?>">
                <span class="episode__fmt episode__fmt--on" role="tab" aria-selected="true">&#9654; <?php esc_html_e( 'Listen', 'rucktalk-minimal' ); ?></span>
                <span class="episode__fmt" role="tab" aria-selected="false">&#9654; <?php esc_html_e( 'Watch', 'rucktalk-minimal' ); ?></span>
            </div>

            <div class="episode__meta">
                <?php if ( $duration ) : ?>
                    <span><?php echo esc_html( $duration ); ?></span>
                    <span>&middot;</span>
                <?php endif; ?>
                <span><?php echo esc_html( $cat_name ); ?></span>
                <span>&middot;</span>
                <span><?php echo esc_html( $published_full ); ?></span>
            </div>

            <div class="episode__actions">
                <a class="btn btn--primary" href="<?php echo esc_url( $permalink ); ?>">&#9654; <?php esc_html_e( 'Play episode', 'rucktalk-minimal' ); ?></a>
                <a class="btn btn--ghost" href="<?php echo esc_url( home_url( '/podcast/' ) ); ?>"><?php esc_html_e( 'All episodes', 'rucktalk-minimal' ); ?></a>
            </div>
        </div>
    </div>
</section>
<?php
wp_reset_postdata();
