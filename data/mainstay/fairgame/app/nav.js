/* Fans First — shared top navigation + mobile drawer.
 * Single source of truth: drops into any page that includes
 *   <script src="nav.js" defer></script>
 * Replaces the first <nav> on the page (acts as SSR placeholder) and injects
 * its own scoped CSS, so the chrome is identical everywhere regardless of the
 * page's own styles. Auth-aware (reads localStorage.fairgame_session).
 *
 * Per-page hooks (optional, set on <body>):
 *   data-ffnav="hero"        -> transparent over a dark hero, tints on scroll
 *   data-ffnav-active="rod|discover|contact|tickets|account"
 * Defaults: solid light bar; active inferred from the URL.
 */
(function () {
  'use strict';

  // ----- session -----
  function session() {
    try { return JSON.parse(localStorage.getItem('fairgame_session') || 'null'); }
    catch (e) { return null; }
  }
  var authed = !!(session() && session().token);

  // ----- which page are we on -----
  var file = (location.pathname.split('/').pop() || 'index.html').toLowerCase();
  var active = (document.body && document.body.getAttribute('data-ffnav-active')) || ({
    'index.html': 'rod', '': 'rod',
    'discover.html': 'discover',
    'contact.html': 'contact',
    'mytickets.html': 'tickets',
    'account.html': 'account'
  })[file] || '';

  var hero = document.body && document.body.getAttribute('data-ffnav') === 'hero';

  // ----- primary links (left of the auth area) -----
  var LINKS = [
    { key: 'rod',      label: 'Rod Wave',  href: 'index.html' },
    { key: 'discover', label: 'Discover',  href: 'discover.html' },
    { key: 'contact',  label: 'Contact',   href: 'contact.html' }
  ];

  function linkHTML(l, cls) {
    var a = active === l.key ? ' active' : '';
    return '<a class="' + cls + a + '" href="' + l.href + '">' + l.label + '</a>';
  }

  // auth area markup (desktop)
  var rightHTML = authed
    ? '<a class="ffn-link" href="mytickets.html">My Tickets</a>' +
      '<a class="ffn-btn ffn-btn-gold" href="account.html">My Account</a>'
    : '<a class="ffn-link" href="account.html">Sign in</a>' +
      '<a class="ffn-btn ffn-btn-gold" href="account.html">Sign up</a>';

  var burgerSVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 6h18M3 12h18M3 18h18"/></svg>';
  var closeSVG  = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>';

  // ----- build nav -----
  var nav = document.createElement('nav');
  nav.className = 'ffnav' + (hero ? '' : ' solid');
  nav.id = 'ffnav';
  nav.innerHTML =
    '<div class="ffn-in">' +
      '<a class="ffn-brand" href="index.html" aria-label="Fans First home"><span class="mk"><img src="img/ff-mark.png" alt="" draggable="false"></span>FANS&nbsp;FIRST</a>' +
      '<div class="ffn-links">' + LINKS.map(function (l) { return linkHTML(l, 'ffn-link'); }).join('') + '</div>' +
      '<div class="ffn-spacer"></div>' +
      '<div class="ffn-right">' + rightHTML + '</div>' +
      '<button class="ffn-burger" type="button" aria-label="Open menu" aria-expanded="false">' + burgerSVG + '</button>' +
    '</div>';

  // ----- build drawer -----
  var drawerAuth = authed
    ? '<a class="ffn-d-link" href="mytickets.html">My Tickets</a>' +
      '<a class="ffn-d-link" href="account.html">My Account</a>'
    : '<a class="ffn-d-link" href="account.html">Sign in</a>';
  var drawerCTA = authed
    ? ''
    : '<a class="ffn-btn ffn-btn-gold ffn-d-cta" href="account.html">Get verified — Sign up</a>';

  var drawer = document.createElement('div');
  drawer.className = 'ffn-drawer';
  drawer.innerHTML =
    '<div class="ffn-scrim" data-close></div>' +
    '<aside class="ffn-panel" role="dialog" aria-label="Menu">' +
      '<div class="ffn-panel-top">' +
        '<span class="ffn-brand dark"><span class="mk"><img src="img/ff-mark.png" alt="" draggable="false"></span>FANS&nbsp;FIRST</span>' +
        '<button class="ffn-x" type="button" aria-label="Close menu" data-close>' + closeSVG + '</button>' +
      '</div>' +
      '<nav class="ffn-d-links">' +
        LINKS.map(function (l) {
          var a = active === l.key ? ' active' : '';
          return '<a class="ffn-d-link' + a + '" href="' + l.href + '">' + l.label + '</a>';
        }).join('') +
        '<div class="ffn-d-sep"></div>' +
        drawerAuth +
      '</nav>' +
      drawerCTA +
      '<p class="ffn-d-foot">All sales final · Verified fans only · Don’t Look Down Tour 2026</p>' +
    '</aside>';

  // ----- styles -----
  var css = document.createElement('style');
  css.id = 'ffnav-css';
  css.textContent = [
    ".ffnav{position:sticky;top:0;z-index:60;height:64px;display:flex;align-items:center;border-bottom:1px solid transparent;transition:background .3s,border-color .3s;font-family:'Figtree',system-ui,sans-serif}",
    ".ffnav .ffn-in{max-width:1200px;margin:0 auto;padding:0 24px;width:100%;display:flex;align-items:center;gap:14px}",
    ".ffnav .ffn-brand{display:flex;align-items:center;gap:8px;font-weight:900;font-size:19px;letter-spacing:-.02em;color:#fff;transition:color .3s;text-decoration:none;white-space:nowrap}",
    ".ffnav .ffn-brand .mk{width:32px;height:32px;border-radius:8px;background:#0b0a09;display:grid;place-items:center;overflow:hidden;flex:0 0 32px}",
    ".ffnav .ffn-brand .mk img{width:23px;height:23px;object-fit:contain;display:block}",
    ".ffnav .ffn-links{display:flex;align-items:center;gap:2px;margin-left:10px}",
    ".ffnav .ffn-spacer{margin-left:auto}",
    ".ffnav .ffn-right{display:flex;align-items:center;gap:10px}",
    ".ffnav a.ffn-link{position:relative;font-weight:600;font-size:14px;color:rgba(255,255,255,.82);padding:8px 12px;border-radius:9px;text-decoration:none;transition:color .2s,background .2s}",
    ".ffnav a.ffn-link:hover{color:#fff;background:rgba(255,255,255,.08)}",
    ".ffnav a.ffn-link.active{color:#fff}",
    ".ffnav a.ffn-link.active::after{content:'';position:absolute;left:12px;right:12px;bottom:2px;height:2px;border-radius:2px;background:#82bcc4}",
    ".ffnav .ffn-btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;font-family:inherit;font-weight:700;font-size:13px;border-radius:11px;padding:10px 16px;text-decoration:none;border:1px solid transparent;cursor:pointer;transition:transform .14s,background .14s,border-color .14s}",
    ".ffnav .ffn-btn-gold{background:#5fa8b3;color:#1a1206}",
    ".ffnav .ffn-btn-gold:hover{background:#82bcc4;transform:translateY(-2px)}",
    ".ffnav .ffn-burger{display:none;align-items:center;justify-content:center;width:42px;height:42px;margin-left:auto;border:1px solid rgba(255,255,255,.24);background:transparent;border-radius:11px;color:#fff;cursor:pointer}",
    ".ffnav .ffn-burger svg{width:21px;height:21px}",
    // solid / scrolled (light) variant
    ".ffnav.solid,.ffnav.scrolled{background:rgba(245,241,234,.94);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-bottom-color:#e4dccc}",
    ".ffnav.solid .ffn-brand,.ffnav.scrolled .ffn-brand{color:#17140d}",
    ".ffnav.solid a.ffn-link,.ffnav.scrolled a.ffn-link{color:#67604f}",
    ".ffnav.solid a.ffn-link:hover,.ffnav.scrolled a.ffn-link:hover{color:#17140d;background:rgba(0,0,0,.04)}",
    ".ffnav.solid a.ffn-link.active,.ffnav.scrolled a.ffn-link.active{color:#17140d}",
    ".ffnav.solid .ffn-burger,.ffnav.scrolled .ffn-burger{border-color:#e4dccc;color:#17140d}",
    "@media(max-width:860px){.ffnav .ffn-links,.ffnav .ffn-right{display:none}.ffnav .ffn-burger{display:inline-flex}}",
    // drawer
    ".ffn-drawer{position:fixed;inset:0;z-index:120;visibility:hidden}",
    ".ffn-drawer.open{visibility:visible}",
    ".ffn-scrim{position:absolute;inset:0;background:rgba(7,6,5,.55);opacity:0;transition:opacity .3s}",
    ".ffn-drawer.open .ffn-scrim{opacity:1}",
    ".ffn-panel{position:absolute;top:0;right:0;height:100%;width:min(87vw,360px);background:#0b0a09;color:#fff;display:flex;flex-direction:column;padding:18px 22px 26px;transform:translateX(102%);transition:transform .32s cubic-bezier(.4,0,.2,1);box-shadow:-22px 0 60px rgba(0,0,0,.5)}",
    ".ffn-drawer.open .ffn-panel{transform:none}",
    ".ffn-panel-top{display:flex;align-items:center;justify-content:space-between;height:46px}",
    ".ffn-brand.dark{color:#fff;font-weight:900;font-size:18px;letter-spacing:-.02em;display:flex;align-items:center;gap:8px}",
    ".ffn-brand.dark .mk{width:30px;height:30px;border-radius:8px;background:#0b0a09;display:grid;place-items:center;overflow:hidden;flex:0 0 30px}",
    ".ffn-brand.dark .mk img{width:22px;height:22px;object-fit:contain;display:block}",
    ".ffn-x{width:40px;height:40px;display:grid;place-items:center;border:1px solid rgba(255,255,255,.2);background:transparent;border-radius:11px;color:#fff;cursor:pointer}",
    ".ffn-x svg{width:20px;height:20px}",
    ".ffn-d-links{display:flex;flex-direction:column;margin-top:26px;gap:2px}",
    ".ffn-d-link{font-weight:700;font-size:1.45rem;letter-spacing:-.01em;color:#f3efe6;text-decoration:none;padding:12px 4px;transition:color .2s}",
    ".ffn-d-link:hover{color:#82bcc4}",
    ".ffn-d-link.active{color:#82bcc4}",
    ".ffn-d-sep{height:1px;background:#211e1b;margin:14px 0}",
    ".ffn-d-cta{margin-top:auto;width:100%;font-size:15px;padding:14px}",
    ".ffn-d-foot{margin-top:22px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#6f6a60;line-height:1.6}"
  ].join('\n');

  // ----- mount -----
  function mount() {
    document.head.appendChild(css);
    // Hide (don't remove) the page's own <nav> so any legacy ids it carries
    // (#navRight, #nav, #mainNav) stay in the DOM for that page's own JS —
    // prevents null-deref in async renderBadge()/scroll handlers.
    var old = document.querySelector('nav');
    if (old && old.parentNode) {
      old.style.display = 'none';
      old.setAttribute('aria-hidden', 'true');
      old.parentNode.insertBefore(nav, old);
    } else {
      document.body.insertBefore(nav, document.body.firstChild);
    }
    document.body.appendChild(drawer);

    // scroll tint (hero pages only)
    if (hero) {
      var onScroll = function () { nav.classList.toggle('scrolled', window.scrollY > 40); };
      window.addEventListener('scroll', onScroll, { passive: true });
      onScroll();
    }

    // drawer open/close
    var burger = nav.querySelector('.ffn-burger');
    function setOpen(open) {
      drawer.classList.toggle('open', open);
      burger.setAttribute('aria-expanded', open ? 'true' : 'false');
      document.documentElement.style.overflow = open ? 'hidden' : '';
    }
    burger.addEventListener('click', function () { setOpen(true); });
    drawer.addEventListener('click', function (e) {
      if (e.target.hasAttribute('data-close') || e.target.closest('[data-close]')) setOpen(false);
    });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') setOpen(false); });
  }

  if (document.body) mount();
  else document.addEventListener('DOMContentLoaded', mount);
})();
