/* rucktalk-minimal — LoovaCast player controls
 *
 * Ported from the approved homepage mockup. Two surfaces:
 *   1. .radio__play — slim floating radio bar play toggle
 *   2. .live__btn   — dedicated player block play toggle
 *
 * Both swap the visible glyph between ▶ (U+25B6) and ⏸ (U+23F8) on
 * click. The actual stream wiring lands in Task 23 (LoovaCast station
 * URL is read from data-station-url on #rt-radio-bar) — for now this
 * keeps the visual contract from the mockup.
 *
 * No external deps. Defers to DOMContentLoaded so the script tag can be
 * loaded in the footer with `in_footer => true`.
 */
(function () {
    'use strict';

    var PLAY  = '▶';      // ▶
    var PAUSE = '⏸';      // ⏸

    function bindToggle(selector) {
        var nodes = document.querySelectorAll(selector);
        if (!nodes || !nodes.length) { return; }
        Array.prototype.forEach.call(nodes, function (btn) {
            btn.addEventListener('click', function (e) {
                e.preventDefault();
                var current = (btn.textContent || '').trim();
                if (current === PLAY) {
                    btn.textContent = PAUSE;
                    btn.setAttribute('aria-pressed', 'true');
                } else {
                    btn.textContent = PLAY;
                    btn.setAttribute('aria-pressed', 'false');
                }
            });
        });
    }

    function init() {
        bindToggle('.radio__play');
        bindToggle('.live__btn');

        // Episode Listen/Watch format toggle from the mockup — visual
        // segmented control on .episode__formats.
        var fmtNodes = document.querySelectorAll('.episode__fmt');
        if (fmtNodes && fmtNodes.length) {
            Array.prototype.forEach.call(fmtNodes, function (fmt) {
                fmt.addEventListener('click', function () {
                    Array.prototype.forEach.call(fmtNodes, function (x) {
                        x.classList.remove('episode__fmt--on');
                        x.setAttribute('aria-selected', 'false');
                    });
                    fmt.classList.add('episode__fmt--on');
                    fmt.setAttribute('aria-selected', 'true');
                });
            });
        }

        // Hero primary CTA — smooth-scrolls to the inline coupon signup.
        var heroCta = document.getElementById('hero-signup');
        if (heroCta) {
            heroCta.addEventListener('click', function (e) {
                var target = document.getElementById('rt-hero-signup') || document.querySelector('.coupon');
                if (!target) { return; }
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                setTimeout(function () {
                    var input = target.querySelector('input[type="email"]');
                    if (input) { input.focus(); }
                }, 600);
            });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
