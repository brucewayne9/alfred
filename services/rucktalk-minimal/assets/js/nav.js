/* rucktalk-minimal — mobile nav drawer toggle.
 *
 * Binds the .nav-burger button to the #rt-mobile-nav drawer. On click:
 * toggles drawer visibility, swaps aria-expanded, locks body scroll
 * while open, and rotates the burger bars into an X icon (CSS handles
 * the visual via [aria-expanded="true"] selectors).
 *
 * Closes on: same-button second click, ESC keydown, drawer link click,
 * window resize > 920 (drawer becomes desktop nav).
 */
(function () {
    'use strict';

    function init() {
        var burger = document.querySelector('.nav-burger');
        var drawer = document.getElementById('rt-mobile-nav');
        if (!burger || !drawer) { return; }

        function setOpen(open) {
            drawer.dataset.open = open ? '1' : '0';
            drawer.setAttribute('aria-hidden', open ? 'false' : 'true');
            burger.setAttribute('aria-expanded', open ? 'true' : 'false');
            burger.setAttribute(
                'aria-label',
                open ? 'Close menu' : 'Open menu'
            );
            document.body.classList.toggle('rt-nav-open', open);
        }

        burger.addEventListener('click', function () {
            setOpen(drawer.dataset.open !== '1');
        });

        // Close drawer when any link inside is tapped (the visitor navigated).
        drawer.addEventListener('click', function (e) {
            if (e.target.tagName === 'A') { setOpen(false); }
        });

        // ESC closes the drawer.
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && drawer.dataset.open === '1') {
                setOpen(false);
                burger.focus();
            }
        });

        // If the window grows back past the mobile breakpoint, ensure the
        // drawer state doesn't leak into desktop layout.
        var mql = window.matchMedia('(min-width: 921px)');
        function onMqlChange() { if (mql.matches) { setOpen(false); } }
        if (mql.addEventListener) {
            mql.addEventListener('change', onMqlChange);
        } else if (mql.addListener) {
            mql.addListener(onMqlChange);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
