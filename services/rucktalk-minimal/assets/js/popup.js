/* rucktalk-minimal — newsletter popup trigger
 *
 * The ONE allowed popup per design-language §5l / Phase 0 §12e.
 *
 * Trigger rules:
 *   1. First page visit: open at 50% scroll depth.
 *   2. Second page visit: open on exit-intent (mouse leaves viewport top).
 *   3. After the user closes or signs up, set a 14-day cookie so we
 *      don't pester them again.
 *
 * No external deps. Defers to DOMContentLoaded.
 */
(function () {
    'use strict';

    var COOKIE_NAME = 'rt_popup_dismiss';
    var COOKIE_DAYS = 14;
    var VISIT_KEY   = 'rt_popup_visits';
    var SCROLL_THRESHOLD = 0.5;   // 50% of scroll height

    function getCookie(name) {
        var match = document.cookie.match(new RegExp('(?:^|; )' + name + '=([^;]*)'));
        return match ? decodeURIComponent(match[1]) : null;
    }

    function setCookie(name, value, days) {
        var expires = new Date(Date.now() + days * 864e5).toUTCString();
        document.cookie = name + '=' + encodeURIComponent(value) + '; expires=' + expires + '; path=/; SameSite=Lax';
    }

    function bumpVisitCount() {
        var n = 0;
        try {
            n = parseInt(window.sessionStorage.getItem(VISIT_KEY) || '0', 10) || 0;
            n += 1;
            window.sessionStorage.setItem(VISIT_KEY, String(n));
        } catch (err) {
            n = 1; // private browsing — treat as first visit, fail safe
        }
        return n;
    }

    function openPopup(popup) {
        if (!popup || popup.dataset.open === '1') { return; }
        popup.dataset.open = '1';
        popup.setAttribute('aria-hidden', 'false');
        var firstInput = popup.querySelector('input[type="email"]');
        if (firstInput) {
            setTimeout(function () { firstInput.focus(); }, 50);
        }
    }

    function closePopup(popup, persist) {
        if (!popup || popup.dataset.open !== '1') { return; }
        popup.dataset.open = '0';
        popup.setAttribute('aria-hidden', 'true');
        if (persist) {
            setCookie(COOKIE_NAME, '1', COOKIE_DAYS);
        }
    }

    function bindClose(popup) {
        var triggers = popup.querySelectorAll('[data-rt-popup-close]');
        Array.prototype.forEach.call(triggers, function (t) {
            t.addEventListener('click', function (e) {
                e.preventDefault();
                closePopup(popup, true);
            });
        });
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && popup.dataset.open === '1') {
                closePopup(popup, true);
            }
        });
        // Treat a successful signup as a permanent dismiss.
        var form = popup.querySelector('form');
        if (form) {
            form.addEventListener('submit', function () {
                // Don't block the submission; just persist the cookie.
                setCookie(COOKIE_NAME, '1', COOKIE_DAYS);
            });
        }
    }

    function bindScrollTrigger(popup) {
        var fired = false;
        function onScroll() {
            if (fired) { return; }
            var doc = document.documentElement;
            var scrollable = (doc.scrollHeight - doc.clientHeight) || 1;
            var pct = (window.scrollY || doc.scrollTop) / scrollable;
            if (pct >= SCROLL_THRESHOLD) {
                fired = true;
                openPopup(popup);
                window.removeEventListener('scroll', onScroll);
            }
        }
        window.addEventListener('scroll', onScroll, { passive: true });
    }

    function bindExitIntent(popup) {
        function onLeave(e) {
            // Fire only when the cursor exits via the top of the viewport.
            if (e.clientY > 5) { return; }
            if (!e.relatedTarget && !e.toElement) {
                openPopup(popup);
                document.removeEventListener('mouseout', onLeave);
            }
        }
        document.addEventListener('mouseout', onLeave);
    }

    function init() {
        var popup = document.getElementById('rt-popup');
        if (!popup) { return; }
        bindClose(popup);

        if (getCookie(COOKIE_NAME)) {
            return; // user previously dismissed — respect for 14d
        }

        var visit = bumpVisitCount();
        if (visit >= 2) {
            bindExitIntent(popup);
        } else {
            bindScrollTrigger(popup);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
