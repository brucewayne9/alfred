/**
 * roen-minimal product gallery — click thumbnail to swap main image.
 * No deps. Idempotent. Bails if no gallery on the page.
 */
(function () {
    'use strict';

    function init() {
        var galleries = document.querySelectorAll('[data-gallery]');
        if (galleries.length === 0) return;

        galleries.forEach(function (gallery) {
            var main = gallery.querySelector('[data-main]');
            var thumbs = gallery.querySelectorAll('[data-thumb]');
            if (!main || thumbs.length < 2) return;

            thumbs.forEach(function (thumb) {
                thumb.addEventListener('click', function () {
                    var nextSrc = thumb.getAttribute('data-full');
                    var nextAlt = thumb.getAttribute('data-alt');
                    if (!nextSrc || nextSrc === main.getAttribute('src')) return;

                    main.classList.add('is-swapping');
                    var pre = new Image();
                    pre.onload = function () {
                        main.setAttribute('src', nextSrc);
                        if (nextAlt) main.setAttribute('alt', nextAlt);
                        main.classList.remove('is-swapping');
                    };
                    pre.onerror = function () {
                        main.classList.remove('is-swapping');
                    };
                    pre.src = nextSrc;

                    thumbs.forEach(function (t) { t.classList.remove('is-active'); });
                    thumb.classList.add('is-active');
                });
            });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
