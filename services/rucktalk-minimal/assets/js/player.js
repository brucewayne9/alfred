/* rucktalk-minimal — LoovaCast + episode player controls + mini-player
 *
 * One <audio> element. Three trigger surfaces:
 *   1. .radio__play       — slim radio bar (header)        → radio
 *   2. .live__btn         — dedicated LoovaCast block      → radio
 *   3. .btn--episode-play — "Today's episode" / cards CTAs → episode
 *
 * Plus a docked mini-player (#rt-mp) that surfaces when audio starts.
 * Mini-player provides: title, scrubber, current/total time, play/pause,
 * close. For radio (live stream) the scrubber is hidden.
 *
 * Position memory (episodes only): localStorage key
 *   `rt-pos:<mp3-url>` = currentTime seconds. Saved every 5s while
 *   playing. Restored on next play of the same episode. Cleared on
 *   ended event. Radio is exempt (live stream — no position).
 *
 * Now-playing label for radio polls /api/nowplaying/<station_id>
 * every 30s. Mini-player title reflects this when source is radio.
 *
 * Progressive enhancement: episode buttons are <a href=permalink>;
 * we preventDefault only when data-episode-mp3 is present. Missing
 * MP3 → anchor navigates to the episode permalink instead.
 */
(function () {
    'use strict';

    var PLAY  = '▶';
    var PAUSE = '⏸';
    var POLL_MS = 30000;
    var POS_SAVE_MS = 5000;

    var audio = null;
    var currentSource = null;       // 'radio' | 'episode' | null
    var currentEpMp3 = '';
    var currentEpTitle = '';
    var currentEpPermalink = '';
    var pollTimer = null;
    var posSaveTimer = null;
    var apiOrigin = '';
    var stationId = '';
    var fallbackRadioTrack = '';
    var seekingActive = false;

    /* ---- helpers --------------------------------------------------- */

    function $(sel, root) { return (root || document).querySelector(sel); }
    function $$(sel, root) { return (root || document).querySelectorAll(sel); }

    function fmtTime(secs) {
        secs = Number(secs);
        if (!isFinite(secs) || secs < 0) { return '0:00'; }
        var m = Math.floor(secs / 60);
        var s = Math.floor(secs % 60);
        return m + ':' + (s < 10 ? '0' : '') + s;
    }

    function getRadioListenUrl() {
        var bar = document.getElementById('rt-radio-bar');
        return bar ? (bar.getAttribute('data-listen-url') || '') : '';
    }

    function posKey(mp3) { return 'rt-pos:' + mp3; }

    function loadPos(mp3) {
        if (!mp3) { return 0; }
        try {
            var v = parseFloat(localStorage.getItem(posKey(mp3)) || '0');
            return isFinite(v) && v > 0 ? v : 0;
        } catch (e) { return 0; }
    }

    function savePos(mp3, secs) {
        if (!mp3) { return; }
        try {
            // Don't save trivially-small positions or end-of-episode.
            if (secs < 5) { return; }
            if (audio && audio.duration && secs >= audio.duration - 10) {
                localStorage.removeItem(posKey(mp3));
                return;
            }
            localStorage.setItem(posKey(mp3), String(secs));
        } catch (e) { /* storage full / disabled — silent */ }
    }

    function clearPos(mp3) {
        if (!mp3) { return; }
        try { localStorage.removeItem(posKey(mp3)); } catch (e) { /* noop */ }
    }

    /* ---- mini-player UI -------------------------------------------- */

    var mp = {
        root: null, play: null, glyph: null, kicker: null,
        title: null, sourceLabel: null, range: null, cur: null, tot: null, close: null
    };

    function mpInit() {
        mp.root        = document.getElementById('rt-mp');
        if (!mp.root) { return; }
        mp.play        = $('.mp__play', mp.root);
        mp.glyph       = $('.mp__glyph', mp.root);
        mp.kicker      = $('.mp__kicker', mp.root);
        mp.sourceLabel = $('.mp__source-label', mp.root);
        mp.title       = $('.mp__title', mp.root);
        mp.range       = $('.mp__range', mp.root);
        mp.cur         = $('.mp__time--cur', mp.root);
        mp.tot         = $('.mp__time--tot', mp.root);
        mp.close       = $('.mp__close', mp.root);

        if (mp.play) { mp.play.addEventListener('click', mpOnPlayClick); }
        if (mp.close) { mp.close.addEventListener('click', mpOnCloseClick); }
        if (mp.range) {
            mp.range.addEventListener('input', mpOnSeekInput);
            mp.range.addEventListener('change', mpOnSeekCommit);
            mp.range.addEventListener('mousedown', function () { seekingActive = true; });
            mp.range.addEventListener('touchstart', function () { seekingActive = true; }, { passive: true });
            mp.range.addEventListener('mouseup', function () { seekingActive = false; });
            mp.range.addEventListener('touchend', function () { seekingActive = false; });
        }
    }

    function mpShow() {
        if (!mp.root) { return; }
        mp.root.setAttribute('data-open', '1');
        mp.root.setAttribute('aria-hidden', 'false');
    }

    function mpHide() {
        if (!mp.root) { return; }
        mp.root.setAttribute('data-open', '0');
        mp.root.setAttribute('aria-hidden', 'true');
    }

    function mpSetSource(source) {
        if (!mp.root) { return; }
        mp.root.setAttribute('data-source', source || '');
        if (mp.sourceLabel) {
            mp.sourceLabel.textContent = source === 'episode' ? 'Now playing' : 'Live radio';
        }
    }

    function mpSetTitle(title, href) {
        if (mp.title) {
            mp.title.textContent = title || '—';
            mp.title.setAttribute('href', href || '#');
        }
    }

    function mpSetProgress() {
        if (!mp.root || !audio) { return; }
        var cur = audio.currentTime || 0;
        var tot = audio.duration || 0;
        if (mp.cur) { mp.cur.textContent = fmtTime(cur); }
        if (mp.tot) {
            mp.tot.textContent = (currentSource === 'radio' || !isFinite(tot) || tot <= 0)
                ? 'LIVE' : fmtTime(tot);
        }
        if (mp.range && !seekingActive) {
            if (currentSource === 'radio' || !isFinite(tot) || tot <= 0) {
                mp.range.value = '0';
                mp.root.style.setProperty('--rt-progress', '0%');
            } else {
                var pct = Math.max(0, Math.min(100, (cur / tot) * 100));
                mp.range.value = String(pct);
                mp.root.style.setProperty('--rt-progress', pct + '%');
            }
        }
    }

    function mpSetGlyph(playing) {
        if (mp.glyph) { mp.glyph.textContent = playing ? PAUSE : PLAY; }
        if (mp.play) { mp.play.setAttribute('aria-pressed', playing ? 'true' : 'false'); }
    }

    function mpOnPlayClick() {
        if (!audio) { return; }
        if (audio.paused) {
            var p = audio.play();
            if (p && typeof p.catch === 'function') { p.catch(function () {}); }
        } else {
            audio.pause();
        }
    }

    function mpOnCloseClick() {
        if (audio) {
            audio.pause();
            try { audio.currentTime = 0; } catch (e) {}
        }
        currentSource = null;
        mpHide();
        updateAllGlyphs();
    }

    function mpOnSeekInput() {
        if (!audio || !audio.duration || currentSource === 'radio') { return; }
        var pct = parseFloat(mp.range.value);
        if (mp.root) { mp.root.style.setProperty('--rt-progress', pct + '%'); }
        if (mp.cur) { mp.cur.textContent = fmtTime((pct / 100) * audio.duration); }
    }

    function mpOnSeekCommit() {
        if (!audio || !audio.duration || currentSource === 'radio') {
            seekingActive = false;
            return;
        }
        var pct = parseFloat(mp.range.value);
        try { audio.currentTime = (pct / 100) * audio.duration; } catch (e) { /* noop */ }
        seekingActive = false;
    }

    /* ---- glyph sync across all trigger surfaces -------------------- */

    function setBtnGlyph(btn, playing) {
        var glyphEl = btn.querySelector('.btn__glyph');
        if (glyphEl) {
            glyphEl.textContent = playing ? PAUSE : PLAY;
        } else {
            btn.textContent = playing ? PAUSE : PLAY;
        }
        btn.setAttribute('aria-pressed', playing ? 'true' : 'false');
    }

    function updateAllGlyphs() {
        var playing = !!(audio && !audio.paused && currentSource);
        Array.prototype.forEach.call($$('.radio__play, .live__btn'), function (b) {
            setBtnGlyph(b, playing && currentSource === 'radio');
        });
        Array.prototype.forEach.call($$('.btn--episode-play'), function (b) {
            var mp3 = b.getAttribute('data-episode-mp3') || '';
            var match = playing && currentSource === 'episode' && mp3 && mp3 === currentEpMp3;
            setBtnGlyph(b, match);
        });
        mpSetGlyph(playing);
    }

    /* ---- now-playing label (radio) --------------------------------- */

    function setRadioTrackText(title) {
        if (!title) { return; }
        var trackEl = $('.radio__track');
        if (trackEl) { trackEl.textContent = title; }
        var liveTitleEm = $('.live__title em');
        if (liveTitleEm) { liveTitleEm.textContent = '“' + title + '”'; }
        if (currentSource === 'radio') {
            mpSetTitle(title, getRadioListenUrl() || '#');
        }
    }

    function refreshNowPlaying() {
        if (!apiOrigin || !stationId || typeof fetch !== 'function') { return; }
        var url = apiOrigin + '/api/nowplaying/' + encodeURIComponent(stationId);
        fetch(url, { cache: 'no-store', credentials: 'omit' })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (data) {
                if (!data) { return; }
                var title = '';
                if (data.now_playing && data.now_playing.song) {
                    var s = data.now_playing.song;
                    if (s.artist && s.title) {
                        title = s.artist + ' — ' + s.title;
                    } else if (s.text) {
                        title = s.text;
                    } else if (s.title) {
                        title = s.title;
                    }
                }
                if (title) {
                    setRadioTrackText(title);
                } else if (fallbackRadioTrack) {
                    setRadioTrackText(fallbackRadioTrack);
                }
            })
            .catch(function () { /* swallow — keep existing label */ });
    }

    /* ---- audio core ------------------------------------------------ */

    function ensureAudio() {
        if (audio) { return; }
        audio = new Audio();
        audio.preload = 'none';
        audio.crossOrigin = 'anonymous';
        audio.addEventListener('loadedmetadata', mpSetProgress);
        audio.addEventListener('durationchange', mpSetProgress);
        audio.addEventListener('timeupdate', mpSetProgress);
        audio.addEventListener('playing', function () {
            updateAllGlyphs();
            if (currentSource === 'episode') {
                if (posSaveTimer) { clearInterval(posSaveTimer); }
                posSaveTimer = setInterval(function () {
                    savePos(currentEpMp3, audio.currentTime);
                }, POS_SAVE_MS);
            }
        });
        audio.addEventListener('pause', function () {
            updateAllGlyphs();
            if (currentSource === 'episode') {
                savePos(currentEpMp3, audio.currentTime);
            }
            if (posSaveTimer) { clearInterval(posSaveTimer); posSaveTimer = null; }
        });
        audio.addEventListener('ended', function () {
            if (currentSource === 'episode') {
                clearPos(currentEpMp3);
            }
            currentSource = null;
            if (posSaveTimer) { clearInterval(posSaveTimer); posSaveTimer = null; }
            updateAllGlyphs();
        });
        audio.addEventListener('error', function () {
            currentSource = null;
            if (posSaveTimer) { clearInterval(posSaveTimer); posSaveTimer = null; }
            updateAllGlyphs();
        });
    }

    function playRadio() {
        var url = getRadioListenUrl();
        if (!url) { return; }
        ensureAudio();
        if (currentSource === 'radio' && !audio.paused) {
            audio.pause();
            return;
        }
        currentSource = 'radio';
        if (audio.src !== url) { audio.src = url; }
        mpSetSource('radio');
        mpSetTitle(($('.radio__track') && $('.radio__track').textContent.trim()) || 'RuckTalk Radio',
                   getRadioListenUrl() || '#');
        mpShow();
        try { audio.load(); } catch (e) { /* noop */ }
        var p = audio.play();
        if (p && typeof p.catch === 'function') {
            p.catch(function () { currentSource = null; updateAllGlyphs(); });
        }
    }

    function playEpisode(mp3, title, permalink) {
        if (!mp3) { return; }
        ensureAudio();
        if (currentSource === 'episode' && currentEpMp3 === mp3 && !audio.paused) {
            audio.pause();
            return;
        }
        var resuming = (currentEpMp3 === mp3 && audio.paused && currentSource === 'episode');
        currentSource = 'episode';
        currentEpMp3 = mp3;
        currentEpTitle = title || 'Episode';
        currentEpPermalink = permalink || '#';
        if (audio.src !== mp3) {
            audio.src = mp3;
        }
        mpSetSource('episode');
        mpSetTitle(currentEpTitle, currentEpPermalink);
        mpShow();
        try { audio.load(); } catch (e) { /* noop */ }
        // Restore saved position if any. Apply after metadata loads so
        // we don't fight the browser's own seek logic.
        var savedPos = resuming ? 0 : loadPos(mp3);
        if (savedPos > 0) {
            var applyPos = function () {
                try { audio.currentTime = savedPos; } catch (e) {}
                audio.removeEventListener('loadedmetadata', applyPos);
            };
            audio.addEventListener('loadedmetadata', applyPos);
        }
        var p = audio.play();
        if (p && typeof p.catch === 'function') {
            p.catch(function () { currentSource = null; updateAllGlyphs(); });
        }
    }

    /* ---- trigger handlers ----------------------------------------- */

    function onRadioBtnClick(e) {
        e.preventDefault();
        playRadio();
    }

    function onEpisodeBtnClick(e) {
        var btn = e.currentTarget;
        var mp3 = btn.getAttribute('data-episode-mp3') || '';
        if (!mp3) { return; /* fall through — anchor navigates */ }
        e.preventDefault();
        var title = btn.getAttribute('data-episode-title') || btn.textContent.trim();
        var permalink = btn.getAttribute('data-episode-permalink') || btn.getAttribute('href') || '#';
        playEpisode(mp3, title, permalink);
    }

    /* ---- init ----------------------------------------------------- */

    function initRadio() {
        var bar = document.getElementById('rt-radio-bar');
        if (!bar) { return; }

        stationId = bar.getAttribute('data-station-id') || '';
        try {
            var u = bar.getAttribute('data-station-url') || '';
            if (u) { apiOrigin = new URL(u).origin; }
        } catch (e) { apiOrigin = ''; }

        var trackEl = $('.radio__track');
        if (trackEl) { fallbackRadioTrack = (trackEl.textContent || '').trim(); }

        Array.prototype.forEach.call($$('.radio__play, .live__btn'), function (btn) {
            btn.addEventListener('click', onRadioBtnClick);
        });

        refreshNowPlaying();
        if (!pollTimer) { pollTimer = setInterval(refreshNowPlaying, POLL_MS); }
    }

    function initEpisode() {
        Array.prototype.forEach.call($$('.btn--episode-play'), function (btn) {
            btn.addEventListener('click', onEpisodeBtnClick);
        });
    }

    function initEpisodeFormatToggle() {
        var fmtNodes = $$('.episode__fmt');
        if (!fmtNodes || !fmtNodes.length) { return; }
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

    function initHeroCta() {
        var heroCta = document.getElementById('hero-signup');
        if (!heroCta) { return; }
        heroCta.addEventListener('click', function (e) {
            var target = document.getElementById('rt-hero-signup') || $('.coupon');
            if (!target) { return; }
            e.preventDefault();
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            setTimeout(function () {
                var input = target.querySelector('input[type="email"]');
                if (input) { input.focus(); }
            }, 600);
        });
    }

    function init() {
        mpInit();
        initRadio();
        initEpisode();
        initEpisodeFormatToggle();
        initHeroCta();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
