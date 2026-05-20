<?php
/**
 * Blog archive (/blog/) — rucktalk-minimal child theme override.
 *
 * Renders the page assigned to `page_for_posts` (id 8701 / "Blog") as a
 * card grid of recent posts instead of Sonaar's default full-content
 * stream. Each card: featured image + cat eyebrow + title + excerpt +
 * read-time/date meta. Card click → permalink to single post.
 *
 * Pagination via paginate_links() at the bottom.
 *
 * Sonaar's archive templates handle podcast/album/video CPTs separately;
 * this file targets the default `post` archive only.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

get_header();
?>
<main class="rt-main rt-blog-archive">

    <header class="rt-blog-archive__head">
        <div class="wrap">
            <span class="section__eyebrow"><?php esc_html_e( 'The RuckTalk journal', 'rucktalk-minimal' ); ?></span>
            <h1 class="rt-blog-archive__title">
                <?php
                $blog_id = (int) get_option( 'page_for_posts' );
                $blog_title = $blog_id ? get_the_title( $blog_id ) : __( 'Blog', 'rucktalk-minimal' );
                echo esc_html( $blog_title );
                ?>
            </h1>
            <p class="rt-blog-archive__strap">
                <?php esc_html_e( "Notes from a guy figuring it out. Pick a story.", 'rucktalk-minimal' ); ?>
            </p>
        </div>
    </header>

    <section class="rt-blog-archive__grid-section">
        <div class="wrap">
            <?php if ( have_posts() ) : ?>

                <div class="rt-blog-archive__grid">
                    <?php
                    while ( have_posts() ) :
                        the_post();
                        $cats     = get_the_category();
                        $cat_name = ! empty( $cats ) ? $cats[0]->name : '';
                        $excerpt  = wp_strip_all_tags( get_the_excerpt() );
                        $read_min = (int) ceil( str_word_count( wp_strip_all_tags( get_post_field( 'post_content', get_the_ID() ) ) ) / 220 );
                        $read_min = max( 1, $read_min );
                        $thumb_url = get_the_post_thumbnail_url( get_the_ID(), 'medium_large' );
                        ?>
                        <article class="rt-card">
                            <a class="rt-card__media" href="<?php the_permalink(); ?>" aria-label="<?php echo esc_attr( get_the_title() ); ?>"
                               <?php if ( $thumb_url ) : ?>style="background-image:url('<?php echo esc_url( $thumb_url ); ?>');"<?php endif; ?>>
                                <?php if ( ! $thumb_url ) : ?>
                                    <span class="rt-card__media-fallback" aria-hidden="true">RuckTalk</span>
                                <?php endif; ?>
                            </a>
                            <div class="rt-card__body">
                                <?php if ( '' !== $cat_name ) : ?>
                                    <span class="rt-card__cat"><?php echo esc_html( $cat_name ); ?></span>
                                <?php endif; ?>
                                <h2 class="rt-card__title">
                                    <a href="<?php the_permalink(); ?>"><?php the_title(); ?></a>
                                </h2>
                                <?php if ( '' !== $excerpt ) : ?>
                                    <p class="rt-card__excerpt"><?php echo esc_html( wp_trim_words( $excerpt, 28 ) ); ?></p>
                                <?php endif; ?>
                                <div class="rt-card__meta">
                                    <span><?php printf( esc_html__( '%d min read', 'rucktalk-minimal' ), $read_min ); ?></span>
                                    <span class="rt-card__meta-sep">&middot;</span>
                                    <span><?php echo esc_html( get_the_date( 'M j, Y' ) ); ?></span>
                                </div>
                            </div>
                        </article>
                    <?php endwhile; ?>
                </div>

                <nav class="rt-blog-archive__pagination" role="navigation" aria-label="<?php esc_attr_e( 'Posts pagination', 'rucktalk-minimal' ); ?>">
                    <?php
                    echo paginate_links( array(
                        'mid_size'           => 1,
                        'prev_text'          => __( '&larr; Newer', 'rucktalk-minimal' ),
                        'next_text'          => __( 'Older &rarr;', 'rucktalk-minimal' ),
                    ) );
                    ?>
                </nav>

            <?php else : ?>

                <div class="rt-blog-archive__empty">
                    <h2><?php esc_html_e( 'Stories incoming.', 'rucktalk-minimal' ); ?></h2>
                    <p><?php esc_html_e( 'The first posts drop shortly — subscribe below and we\'ll send a note when they do.', 'rucktalk-minimal' ); ?></p>
                </div>

            <?php endif; ?>
        </div>
    </section>

</main>
<?php
get_footer();
