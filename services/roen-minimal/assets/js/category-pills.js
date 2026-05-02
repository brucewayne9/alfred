(function () {
    'use strict';

    function init() {
        var pills = document.querySelectorAll('.roen-pill');
        var cards = document.querySelectorAll('.roen-card');
        if (pills.length === 0 || cards.length === 0) {
            return;
        }

        function applyFilter(slug) {
            cards.forEach(function (card) {
                var cats = (card.getAttribute('data-cats') || '').split(/\s+/);
                var match = (slug === 'all') || cats.indexOf(slug) !== -1;
                card.classList.toggle('is-hidden', !match);
            });
        }

        pills.forEach(function (pill) {
            pill.addEventListener('click', function () {
                pills.forEach(function (p) { p.classList.remove('is-active'); });
                pill.classList.add('is-active');
                applyFilter(pill.getAttribute('data-cat'));
            });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
}());
