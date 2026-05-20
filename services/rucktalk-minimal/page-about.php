<?php
/**
 * Template Name: RuckTalk About
 *
 * Editorial-magazine About page for rucktalk.com.
 *
 * Replaces the legacy /about-mike-johnson/ Elementor render (which inherited
 * dark-mode backgrounds against black Elementor text — unreadable). Image-led
 * scroll, dark warm slate palette, hand-set typography to match the homepage.
 *
 * Content + photography lifted from the FaR site (single-source for now):
 *   - mj-hero / mj-gear / mj-posing  → 1080×1920 portraits of Mike
 *   - field-ruck-kb / field-recovery / field-show-up  → "Field Tape" magazine cuts
 * Both come from fitasruck.com/wp-content/uploads/fitasruck/. Re-compressed
 * to ~430 KB (Mike shots) and ~120 KB (field tapes) at q85, stripped EXIF,
 * stored locally in assets/img/about/ so the page is self-contained.
 *
 * Voice: per docs/superpowers/specs/2026-05-19-rucktalk-design-language.md §6
 *   - "Go-getters" not "guys" (the FaR source uses "guys" — translated here)
 *   - Practical not aspirational
 *   - Fellow-traveler, not coach posture
 *   - Quotes from Mike's own copy preserved verbatim where they fit the rules
 *
 * Visual scope: all class names prefixed `.about-page__*` (BEM-ish) so the
 * page-only stylesheet at assets/css/about.css can't bleed elsewhere.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

get_header();

// Resolve asset URIs once so the template stays readable.
$img_base       = get_stylesheet_directory_uri() . '/assets/img/about/';
$hero_src       = $img_base . 'mj-hero.jpg';
$gear_src       = $img_base . 'mj-gear.jpg';
$posing_src     = $img_base . 'mj-posing.jpg';
$ruck_kb_src    = $img_base . 'field-ruck-kb.jpg';
$recovery_src   = $img_base . 'field-recovery.jpg';
$show_up_src    = $img_base . 'field-show-up.jpg';

/**
 * YouTube channel URL for the watch CTA. Override via the
 * `rucktalk_youtube_url` filter when the real channel handle lands.
 * Defaults to a YouTube search for "rucktalk podcast" so the link is
 * never dead.
 */
$youtube_url = apply_filters(
    'rucktalk_youtube_url',
    'https://www.youtube.com/results?search_query=rucktalk+podcast+mike+johnson'
);
?>
<main class="about-page" id="about-main">

    <?php /* ───────────────── 1. HERO ───────────────── */ ?>
    <section class="about-page__hero" aria-labelledby="about-hero-title">
        <img src="<?php echo esc_url( $hero_src ); ?>"
             alt=""
             class="about-page__hero-photo"
             loading="eager"
             decoding="async"
             aria-hidden="true">
        <div class="about-page__hero-inner">
            <p class="about-page__hero-folio">Vol. I &middot; The Host</p>
            <p class="about-page__hero-eyebrow"><?php esc_html_e( 'About the host', 'rucktalk-minimal' ); ?></p>
            <h1 class="about-page__hero-title" id="about-hero-title">
                <?php
                echo wp_kses_post(
                    /* translators: italic phrase rendered as emphasized terracotta */
                    sprintf( __( 'Mike Johnson. %s', 'rucktalk-minimal' ), '<em>One go-getter, figuring it out.</em>' )
                );
                ?>
            </h1>
            <p class="about-page__hero-strap">
                <?php esc_html_e( '51. Husband. Father. Atlanta. Built and sold companies, gained the weight back in his 40s, then spent two years getting it off the right way. Now hosts a conversation about the rest of it.', 'rucktalk-minimal' ); ?>
            </p>
        </div>
    </section>


    <?php /* ───────────────── 2. PULLED QUOTE ───────────────── */ ?>
    <section class="about-page__quote" aria-label="Pulled quote">
        <div class="about-page__quote-inner">
            <span class="about-page__quote-mark" aria-hidden="true">&ldquo;</span>
            <p class="about-page__quote-body">
                <?php
                echo wp_kses_post(
                    __( 'I&rsquo;m not an expert in anything. I just look at the world and give my honest take &mdash; <em>and then I want to hear yours.</em>', 'rucktalk-minimal' )
                );
                ?>
            </p>
            <p class="about-page__quote-sig">&mdash; <?php esc_html_e( 'Mike, host', 'rucktalk-minimal' ); ?></p>
        </div>
    </section>


    <?php /* ───────────────── 3. THE STORY ───────────────── */ ?>
    <section class="about-page__story" aria-labelledby="about-story-label">
        <div class="about-page__story-inner">
            <p class="about-page__story-label" id="about-story-label"><?php esc_html_e( 'The Story', 'rucktalk-minimal' ); ?></p>
            <div class="about-page__story-prose">
                <p>
                    <?php
                    echo wp_kses_post(
                        __( 'A few years back I woke up after about an hour of sleep and felt like my heart was all over the place. A full-blown panic attack at three in the morning. In that moment I knew I&rsquo;d changed &mdash; <em>and not in a good way.</em>', 'rucktalk-minimal' )
                    );
                    ?>
                </p>
                <p>
                    <?php esc_html_e( 'Bad sleep. Drinking. A desk all day. Forty-something body that didn&rsquo;t bounce back like it used to. I knew if I wanted to enjoy the rest of my life I had to do a complete 180. So I spent eighteen months working out what actually holds up for someone with a job, a family, and a body that needs a day to recover from a day. Outside. Calisthenics. Pull-ups, push-ups. A lot of rucking. I got on a schedule. I started showing up.', 'rucktalk-minimal' ); ?>
                </p>
                <p>
                    <?php esc_html_e( 'RuckTalk is the conversation around the rest of it. The work part. The marriage part. The being-a-decent-friend part. The picking-up-your-kids-when-they&rsquo;re-thirty part. Five pillars: health, business, family, strength, and what everybody else is going through. I run two ventures, raise a family in Atlanta, and try to do both without falling apart. The show is me talking to other go-getters about how to do all of that on purpose &mdash; and asking them what I&rsquo;m missing.', 'rucktalk-minimal' ); ?>
                </p>
                <p>
                    <strong><?php esc_html_e( 'No expert posture. No soapbox. No 25-year-old influencer energy.', 'rucktalk-minimal' ); ?></strong>
                    <?php esc_html_e( 'Just a regular guy talking to other regular go-getters about real life out loud.', 'rucktalk-minimal' ); ?>
                </p>
            </div>
        </div>
    </section>


    <?php /* §4 visual grid removed 2026-05-19 — Mike kept only the guest video. */ ?>


    <?php /* ───────────────── 4. ON OTHER PEOPLE'S MICS (guest embed) ───────────────── */ ?>
    <section class="about-page__guest" aria-labelledby="about-guest-title">
        <div class="about-page__guest-inner">
            <p class="about-page__guest-eyebrow"><?php esc_html_e( "Other people's mics", 'rucktalk-minimal' ); ?></p>
            <h2 class="about-page__guest-title" id="about-guest-title">
                <?php
                echo wp_kses_post(
                    sprintf( __( 'Mike on the road — %s', 'rucktalk-minimal' ), '<em>"Learning from Failure."</em>' )
                );
                ?>
            </h2>
            <p class="about-page__guest-sub">
                <?php esc_html_e( "A long-form conversation about building Ground Rush, getting punched in the face by reality, and what got him back up. Worth the hour if you've ever shipped something and watched it land sideways.", 'rucktalk-minimal' ); ?>
            </p>
            <div class="about-page__guest-video">
                <iframe
                    src="https://www.youtube-nocookie.com/embed/8Vvi-TII8yo?rel=0&modestbranding=1"
                    title="<?php esc_attr_e( 'Mike Johnson — Learning from Failure (guest podcast)', 'rucktalk-minimal' ); ?>"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                    referrerpolicy="strict-origin-when-cross-origin"
                    allowfullscreen
                    loading="lazy"></iframe>
            </div>
            <p class="about-page__guest-caption">
                <?php esc_html_e( "Catch Mike elsewhere? Send the link — we'll add it here.", 'rucktalk-minimal' ); ?>
            </p>
        </div>
    </section>


    <?php /* ───────────────── 5. WATCH / VIDEO ───────────────── */ ?>
    <section class="about-page__watch" aria-labelledby="about-watch-title">
        <div class="about-page__watch-inner">
            <p class="about-page__watch-eyebrow"><?php esc_html_e( 'On camera', 'rucktalk-minimal' ); ?></p>
            <h2 class="about-page__watch-title" id="about-watch-title">
                <?php
                echo wp_kses_post(
                    sprintf( __( 'Watch the show %s', 'rucktalk-minimal' ), '<em>on YouTube.</em>' )
                );
                ?>
            </h2>
            <p class="about-page__watch-sub">
                <?php esc_html_e( 'Episodes drop as long-form video too. Less polish, more signal. If you’d rather see a conversation than just hear it, the channel’s where to go.', 'rucktalk-minimal' ); ?>
            </p>
            <a class="about-page__watch-cta"
               href="<?php echo esc_url( $youtube_url ); ?>"
               rel="noopener"
               target="_blank">
                <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                    <path d="M23.5 6.2a3 3 0 0 0-2.1-2.1C19.6 3.6 12 3.6 12 3.6s-7.6 0-9.4.5A3 3 0 0 0 .5 6.2 31.4 31.4 0 0 0 0 12a31.4 31.4 0 0 0 .5 5.8 3 3 0 0 0 2.1 2.1c1.8.5 9.4.5 9.4.5s7.6 0 9.4-.5a3 3 0 0 0 2.1-2.1A31.4 31.4 0 0 0 24 12a31.4 31.4 0 0 0-.5-5.8zM9.6 15.6V8.4l6.3 3.6-6.3 3.6z"/>
                </svg>
                <?php esc_html_e( 'Watch on YouTube', 'rucktalk-minimal' ); ?>
            </a>
        </div>
    </section>


    <?php /* ───────────────── 6. FIVE PILLARS (numbered list) ───────────────── */ ?>
    <section class="about-page__pillars" aria-labelledby="about-pillars-title">
        <div class="about-page__pillars-head">
            <span class="about-page__pillars-eyebrow"><?php esc_html_e( 'What the show is about', 'rucktalk-minimal' ); ?></span>
            <h2 class="about-page__pillars-title" id="about-pillars-title">
                <?php
                echo wp_kses_post(
                    sprintf( __( 'Five things %s', 'rucktalk-minimal' ), '<em>worth getting right.</em>' )
                );
                ?>
            </h2>
        </div>
        <ol class="about-page__pillars-list">
            <li class="about-page__pillar">
                <span class="about-page__pillar-num">01</span>
                <div>
                    <h3 class="about-page__pillar-name"><?php esc_html_e( 'Health', 'rucktalk-minimal' ); ?></h3>
                    <p class="about-page__pillar-def"><?php esc_html_e( 'Keeping a body that holds up to the rest of life. Sleep, food, hips, breath. The unglamorous stuff that decides everything else.', 'rucktalk-minimal' ); ?></p>
                </div>
            </li>
            <li class="about-page__pillar">
                <span class="about-page__pillar-num">02</span>
                <div>
                    <h3 class="about-page__pillar-name"><?php esc_html_e( 'Business', 'rucktalk-minimal' ); ?></h3>
                    <p class="about-page__pillar-def"><?php esc_html_e( 'Building something while everything else is also on fire. Pricing, customers, the hour after lunch when you actually think.', 'rucktalk-minimal' ); ?></p>
                </div>
            </li>
            <li class="about-page__pillar">
                <span class="about-page__pillar-num">03</span>
                <div>
                    <h3 class="about-page__pillar-name"><?php esc_html_e( 'Family', 'rucktalk-minimal' ); ?></h3>
                    <p class="about-page__pillar-def"><?php esc_html_e( 'Showing up at home like you mean it. Twenty minutes Sunday night with your partner. Phones in another room at dinner. The walks.', 'rucktalk-minimal' ); ?></p>
                </div>
            </li>
            <li class="about-page__pillar">
                <span class="about-page__pillar-num">04</span>
                <div>
                    <h3 class="about-page__pillar-name"><?php esc_html_e( 'Strength', 'rucktalk-minimal' ); ?></h3>
                    <p class="about-page__pillar-def"><?php esc_html_e( 'Mind and body, the way you carry yourself through a week. Train so you can pick up your kids when they’re thirty.', 'rucktalk-minimal' ); ?></p>
                </div>
            </li>
            <li class="about-page__pillar">
                <span class="about-page__pillar-num">05</span>
                <div>
                    <h3 class="about-page__pillar-name"><?php esc_html_e( 'Shared', 'rucktalk-minimal' ); ?></h3>
                    <p class="about-page__pillar-def"><?php esc_html_e( 'What everybody else is going through, said plainly. Most people are way more tired than they let on. That includes you.', 'rucktalk-minimal' ); ?></p>
                </div>
            </li>
        </ol>
    </section>


    <?php /* ───────────────── 7. CTA — newsletter signup ───────────────── */ ?>
    <section class="about-page__cta" aria-labelledby="about-cta-title">
        <div class="about-page__cta-inner">
            <p class="about-page__cta-eyebrow"><?php esc_html_e( 'The free plan', 'rucktalk-minimal' ); ?></p>
            <h2 class="about-page__cta-title" id="about-cta-title">
                <?php
                echo wp_kses_post(
                    sprintf( __( 'Get the free %s', 'rucktalk-minimal' ), '<em>8-week plan.</em>' )
                );
                ?>
            </h2>
            <p class="about-page__cta-sub">
                <?php esc_html_e( 'What I’d do in your first eight weeks if I were starting over. Sent to your inbox after you verify.', 'rucktalk-minimal' ); ?>
            </p>
            <?php echo do_shortcode( '[rt_signup placement="about"]' ); ?>
        </div>
    </section>

</main>
<?php
get_footer();
