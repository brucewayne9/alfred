<?php
/**
 * Latest 3 blog posts — rucktalk-minimal homepage partial.
 *
 * Mirrors mockup .blog block. Three standard posts, recency order.
 * Each card: terracotta category eyebrow (.card__cat) + Archivo Black
 * headline (.card__title) + body excerpt + meta row.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

$q = new WP_Query(
    array(
        'post_type'           => 'post',
        'posts_per_page'      => 3,
        'post_status'         => 'publish',
        'orderby'             => 'date',
        'order'               => 'DESC',
        'ignore_sticky_posts' => true,
        'no_found_rows'       => true,
    )
);

if ( ! $q->have_posts() ) {
    return;
}
?>
<section class="blog">
    <div class="wrap">

        <div class="blog__head">
            <span class="section__eyebrow"><?php esc_html_e( 'All recent posts', 'rucktalk-minimal' ); ?></span>
            <a class="btn btn--ghost" href="<?php echo esc_url( home_url( '/blog/' ) ); ?>"><?php esc_html_e( 'Read the archive', 'rucktalk-minimal' ); ?> &rarr;</a>
        </div>

        <div class="blog__grid">
            <?php while ( $q->have_posts() ) :
                $q->the_post();
                $cats     = get_the_category();
                $cat_name = ! empty( $cats ) ? $cats[0]->name : '';
                $excerpt  = wp_strip_all_tags( get_the_excerpt() );
                $read_min = (int) ceil( str_word_count( wp_strip_all_tags( get_post_field( 'post_content', get_the_ID() ) ) ) / 220 );
                $read_min = max( 1, $read_min );
                ?>
                <article class="card">
                    <?php if ( '' !== $cat_name ) : ?>
                        <span class="card__cat"><?php echo esc_html( $cat_name ); ?></span>
                    <?php endif; ?>
                    <h3 class="card__title">
                        <a href="<?php echo esc_url( get_permalink() ); ?>"><?php the_title(); ?></a>
                    </h3>
                    <?php if ( '' !== $excerpt ) : ?>
                        <p class="card__excerpt"><?php echo esc_html( wp_trim_words( $excerpt, 30 ) ); ?></p>
                    <?php endif; ?>
                    <div class="card__meta">
                        <span><?php printf( esc_html__( '%d min read', 'rucktalk-minimal' ), $read_min ); ?></span>
                        <span class="card__meta-sep">&middot;</span>
                        <span><?php echo esc_html( get_the_date( 'M j' ) ); ?></span>
                    </div>
                </article>
            <?php endwhile; ?>
        </div>

    </div>
</section>
<?php
wp_reset_postdata();
