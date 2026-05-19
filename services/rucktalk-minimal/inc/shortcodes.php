<?php
/**
 * Shortcodes — rucktalk-minimal.
 *
 * Registers:
 *   [rt_signup placement="hero|inline|footer|popup|training-free"]
 *     — Brevo double-opt-in email-capture form (Task 11).
 *       Markup matches the .signup block in the approved homepage mockup.
 *   [rt_ecosystem_strip]
 *     — Sister-brand wordmark row (Task 11). Renders the .eco block from
 *       the mockup. Lives inside footer.php; safe to drop on any page.
 *   [rt_pillars]
 *     — Five-pillar grid with daily-rotating "Today's take" snippets.
 *       Defers to templates/parts/pillars.php which calls
 *       rt_pillar_snippet_today() from inc/pillar-snippets.php (Task 13b).
 *
 * All output is escaped at the leaves (esc_html / esc_url / esc_attr).
 * Form submission flow:
 *   - Client JS (assets/js/signup.js, Task 19) POSTs to rest URL provided
 *     via wp_localize_script as `RuckTalkSignup.restUrl` with the nonce.
 *   - No-JS fallback: form posts to admin-post.php?action=rt_signup which
 *     inc/rest-signup.php (Task 19) will also handle.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

/**
 * [rt_signup placement="..."]
 *
 * @param array $atts Shortcode attributes.
 * @return string Rendered HTML.
 */
function rt_signup_shortcode( $atts ) {
    $a = shortcode_atts(
        array(
            'placement'   => 'inline',
            'button'      => 'Send it to me',
            'placeholder' => 'you@example.com',
            'micro'       => "We'll email to confirm. No spam, unsubscribe any time.",
        ),
        $atts,
        'rt_signup'
    );

    $placement   = sanitize_html_class( $a['placement'] );
    $button      = sanitize_text_field( $a['button'] );
    $placeholder = sanitize_text_field( $a['placeholder'] );
    $micro       = sanitize_text_field( $a['micro'] );
    $field_id    = 'rt-signup-email-' . $placement;

    ob_start();
    ?>
    <form class="signup rt-signup-form rt-signup-form--<?php echo esc_attr( $placement ); ?>"
          data-placement="<?php echo esc_attr( $placement ); ?>"
          method="post"
          action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>">
        <input type="hidden" name="action" value="rt_signup">
        <input type="hidden" name="placement" value="<?php echo esc_attr( $placement ); ?>">
        <?php wp_nonce_field( 'rt_signup', 'rt_signup_nonce' ); ?>
        <label class="screen-reader-text" for="<?php echo esc_attr( $field_id ); ?>">
            <?php esc_html_e( 'Email address', 'rucktalk-minimal' ); ?>
        </label>
        <input class="signup__input rt-signup-form__email"
               id="<?php echo esc_attr( $field_id ); ?>"
               type="email"
               name="email"
               placeholder="<?php echo esc_attr( $placeholder ); ?>"
               autocomplete="email"
               required>
        <button class="signup__btn rt-signup-form__submit" type="submit">
            <?php echo esc_html( $button ); ?>
        </button>
        <?php if ( '' !== $micro ) : ?>
            <p class="coupon__micro rt-signup-form__micro"><?php echo esc_html( $micro ); ?></p>
        <?php endif; ?>
        <div class="rt-signup-form__status" role="status" aria-live="polite"></div>
    </form>
    <?php
    return ob_get_clean();
}
add_shortcode( 'rt_signup', 'rt_signup_shortcode' );

/**
 * [rt_ecosystem_strip]
 *
 * Sister-brand link row, mockup .eco block. Renders inside the dark
 * footer; the typographic treatment (wordmark + small descriptor)
 * lives in assets/css/ecosystem.css.
 *
 * @return string Rendered HTML.
 */
function rt_ecosystem_strip_shortcode() {
    $brands = array(
        array(
            'slug'    => 'loovacast',
            'name'    => 'LoovaCast',
            'tagline' => 'Radio for creators',
            'url'     => 'https://loovacast.com',
        ),
        array(
            'slug'    => 'lumabot',
            'name'    => 'LumaBot',
            'tagline' => 'AI chat for your site',
            'url'     => 'https://lumabot.com',
        ),
        array(
            'slug'    => 'airoi',
            'name'    => 'AIROI',
            'tagline' => 'AI savings calc',
            'url'     => 'https://aialfred.groundrushcloud.com/static/ai-savings-calc/',
        ),
        array(
            'slug'    => 'roen',
            'name'    => 'Roen Handmade',
            'tagline' => 'Handmade jewelry',
            'url'     => 'https://roenhandmade.com',
        ),
        array(
            'slug'    => 'grl',
            'name'    => 'Ground Rush Labs',
            'tagline' => 'The studio',
            'url'     => 'https://groundrushlabs.com',
        ),
    );

    /**
     * Filter: rucktalk_ecosystem_brands
     * Allow runtime mutation of the ecosystem strip (e.g. retiring or
     * re-ordering brands without redeploying the theme).
     */
    $brands = apply_filters( 'rucktalk_ecosystem_brands', $brands );

    ob_start();
    ?>
    <div class="eco" aria-label="<?php esc_attr_e( 'Part of the Ground Rush ecosystem', 'rucktalk-minimal' ); ?>">
        <p class="eco__label"><?php esc_html_e( 'Part of the Ground Rush ecosystem', 'rucktalk-minimal' ); ?></p>
        <ul class="eco__list">
            <?php foreach ( $brands as $b ) : ?>
                <li class="eco__item">
                    <a href="<?php echo esc_url( $b['url'] ); ?>" rel="noopener" target="_blank">
                        <span class="eco__name"><?php echo esc_html( $b['name'] ); ?></span>
                        <span class="eco__tag"><?php echo esc_html( $b['tagline'] ); ?></span>
                    </a>
                </li>
            <?php endforeach; ?>
        </ul>
    </div>
    <?php
    return ob_get_clean();
}
add_shortcode( 'rt_ecosystem_strip', 'rt_ecosystem_strip_shortcode' );

/**
 * [rt_pillars]
 *
 * Renders the dynamic 5-pillar grid via templates/parts/pillars.php,
 * which calls rt_pillar_snippet_today() per pillar. Wrap in ob_start
 * because get_template_part echoes — we want the shortcode to return.
 *
 * @return string Rendered HTML.
 */
function rt_pillars_shortcode() {
    ob_start();
    get_template_part( 'templates/parts/pillars' );
    return ob_get_clean();
}
add_shortcode( 'rt_pillars', 'rt_pillars_shortcode' );
