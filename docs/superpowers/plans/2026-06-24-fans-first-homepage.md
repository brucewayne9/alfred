# Fans First — Homepage Redesign Implementation Plan (Plan 1 of 7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the live Fans First homepage (`data/mainstay/fairgame/app/index.html`) to the approved DESIGN.md, wired to the real `/fairgame/api/shows` data, with the Rod photo flood and the *Don't Look Down* album-trailer moment.

**Architecture:** The approved static mock at `~/.openclaw/workspace/static/drafts/fansfirst-mock/index.html` (live: https://aialfred.groundrushcloud.com/drafts/fansfirst-mock/) is the **canonical implementation** of the look. This plan (a) retokens the shared `fairgame.css` to the DESIGN.md palette/type so every page inherits the new system, (b) ports the mock's homepage markup into `index.html`, and (c) replaces the mock's static tour cards with the existing `loadShows()` fetch against `GET /fairgame/api/shows`. No backend changes.

**Tech Stack:** Static HTML/CSS/vanilla JS (no framework), shared `fairgame.css`, FastAPI backend already serving `/fairgame/api/*`, served from `data/mainstay/fairgame/app/`. Fonts: Figtree (Google). Images: pre-optimized JPGs.

## Global Constraints (from DESIGN.md — apply to every task)
- One accent only: gold `--gold #e6bd72` (bright/dark), `--gold-deep #9c6f23` (on light), buttons `--gold-btn #d8a84c` with text `#1a1206`. Seat-map green/red/grey are the ONLY other colors and only inside the map.
- Palette: dark `--bg #0b0a09` (hero/footer/album), light `--paper #f5f1ea` / `--paper-alt #ece5d7`, card `#fff`, ink `#17140d`/`#67604f`/`#988f7c`, lines `#e4dccc` (light) / `#2a2724` (dark).
- Type: **Figtree** only (400–900). No Anton/Sora carryover on the homepage.
- Buttons **squared** `border-radius:11px` (NOT pills). Cards 14px, tags 6px.
- **Line SVG icons only — NO emoji anywhere.** (Grep gate in Task 9.)
- Copy: "lowest **verified** price, guaranteed" — never "lowest anywhere". Final-sale line in footer.
- Fully responsive, web-only. No image >500KB shipped to mobile.
- Reference mock is the source of truth for markup/CSS blocks; port verbatim, then apply the deltas each task names. Do not hand-reinvent styles already in the mock.

## File Structure
- Modify: `data/mainstay/fairgame/app/fairgame.css` — swap `:root` tokens + base nav/button/section primitives to DESIGN.md (shared; benefits all pages).
- Rewrite: `data/mainstay/fairgame/app/index.html` — new homepage markup + `loadShows()` restyle + album modal.
- Create (copy): `data/mainstay/fairgame/app/img/` — `rod-hero.jpg`, `rod-live-1.jpg`, `rod-live-2.jpg`, `rod-live-3.jpg`, `rod-fans.jpg`, `rod-arena.jpg`, `rod-portrait.jpg`.
- Unchanged: backend (`core/api/fairgame.py`, `core/fairgame/*`), other pages (reskinned in later plans).

> NOTE: retokening shared `fairgame.css` will shift the *other* pages toward the new palette before they're individually reskinned (Plans 2–6). That is expected and acceptable — they stay functional; only their polish lands later.

---

### Task 1: Bring optimized image assets into the app

**Files:**
- Create: `data/mainstay/fairgame/app/img/*` (7 jpgs)

- [ ] **Step 1: Copy the approved, pre-optimized images from the mock into the app**

```bash
cd /home/aialfred/alfred
mkdir -p data/mainstay/fairgame/app/img
SRC=/home/aialfred/.openclaw/workspace/static/drafts/fansfirst-mock
cp "$SRC"/rod-hero.jpg "$SRC"/rod-live-1.jpg "$SRC"/rod-live-2.jpg "$SRC"/rod-live-3.jpg \
   "$SRC"/rod-fans.jpg "$SRC"/rod-arena.jpg "$SRC"/rod-portrait.jpg \
   data/mainstay/fairgame/app/img/
```

- [ ] **Step 2: Verify all 7 present and each <520KB**

Run:
```bash
cd /home/aialfred/alfred && ls -l data/mainstay/fairgame/app/img/ && \
  find data/mainstay/fairgame/app/img -size +520k
```
Expected: 7 files listed; the `find` prints nothing (none over 520KB).

- [ ] **Step 3: Commit**

```bash
git add data/mainstay/fairgame/app/img
git commit -m "feat(fairgame): add optimized Rod photo assets for homepage"
```

---

### Task 2: Retoken `fairgame.css` to the DESIGN.md system

**Files:**
- Modify: `data/mainstay/fairgame/app/fairgame.css` (the `:root{...}` block, and the `@import`/font link if present)

**Interfaces:**
- Produces: CSS variables consumed by every page — `--bg --paper --paper-alt --card --ink --ink-2 --ink-3 --line-l --line-d --gold --gold-deep --gold-btn --green --red --grey --r-btn --r-card --r-tag`, font family `Figtree`.

- [ ] **Step 1: Replace the `:root` token block** in `fairgame.css` with the DESIGN.md tokens (copy verbatim from the reference mock's `:root`, which already encodes them). Keep any non-color utility tokens the other pages need (`--maxw`, `--ease`, `--shadow`) but point colors at the new values:

```css
:root{
  --bg:#0b0a09; --bg2:#121110;
  --paper:#f5f1ea; --paper-alt:#ece5d7; --card:#ffffff;
  --ink:#17140d; --ink-2:#67604f; --ink-3:#988f7c;
  --line-l:#e4dccc; --line-d:#2a2724;
  --silver:#aaa49a; --mute:#6f6a60;
  --gold:#e6bd72; --gold-deep:#9c6f23; --gold-btn:#d8a84c;
  --green:#13bd57; --red:#e23b3b; --grey:#322f2a;
  --r-btn:11px; --r-card:14px; --r-tag:6px;
  --maxw:1200px; --ease:cubic-bezier(.2,.7,.2,1);
  --sh-l:0 14px 38px rgba(22,18,8,.10); --sh-d:0 22px 60px rgba(0,0,0,.6);
  --font-display:'Figtree',system-ui,sans-serif; --font-body:'Figtree',system-ui,sans-serif;
}
```

- [ ] **Step 2: Swap the font import** — ensure the Figtree `<link>` is loaded by pages. In `fairgame.css`, if fonts are `@import`ed, replace the Anton/Sora import with:

```css
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800;900&display=swap');
```

- [ ] **Step 3: Set `body` to the light canvas** (was dark). Update the existing `body{...}` rule:

```css
body{font-family:var(--font-body);background:var(--paper);color:var(--ink);-webkit-font-smoothing:antialiased;line-height:1.5}
```

- [ ] **Step 4: Verify other pages still serve (no CSS parse break)**

Run:
```bash
curl -s -o /dev/null -w "%{http_code}\n" https://aialfred.groundrushcloud.com/fairgame/app/show.html
```
Expected: `200`. Open it; confirm the page renders (palette shifted, layout intact — full reskin lands in a later plan).

- [ ] **Step 5: Commit**

```bash
git add data/mainstay/fairgame/app/fairgame.css
git commit -m "feat(fairgame): retoken shared CSS to Fans First DESIGN.md (gold, Figtree, squared)"
```

---

### Task 3: Port the homepage shell — head, announce bar, nav, hero

**Files:**
- Rewrite: `data/mainstay/fairgame/app/index.html` (head + `<body>` through the hero)

**Interfaces:**
- Consumes: `img/rod-hero.jpg` (Task 1), tokens + Figtree (Task 2).
- Produces: `nav.scrolled` toggle behavior; hero markup with co-brand lockup.

- [ ] **Step 1: Replace `index.html` head + announce + nav + hero** by porting those sections verbatim from the reference mock (`fansfirst-mock/index.html`), with these deltas: image `src="img/rod-hero.jpg"`; keep `<link rel="stylesheet" href="fairgame.css">`; page-local `<style>` only for homepage-unique sections (hero/gallery/album) — shared primitives come from `fairgame.css`. Hero copy stays: eyebrow co-brand `Fans First × Rod Wave · Official Partner`, H1 "The best seats / Rod ever held. / Only here.", the trust row (three checks).

- [ ] **Step 2: Keep the scroll-aware nav script** (port from mock):

```javascript
const nav=document.querySelector('nav');
const onScroll=()=>nav.classList.toggle('scrolled',window.scrollY>40);
window.addEventListener('scroll',onScroll,{passive:true});onScroll();
```

- [ ] **Step 3: Verify hero serves with image**

Run:
```bash
curl -s -o /dev/null -w "%{http_code}\n" https://aialfred.groundrushcloud.com/fairgame/app/img/rod-hero.jpg
curl -s https://aialfred.groundrushcloud.com/fairgame/app/index.html | grep -c 'Rod ever held'
```
Expected: `200`, then `1`. Open the page on desktop + a phone-width window: hero photo loads, nav transparent→solid on scroll.

- [ ] **Step 4: Commit**

```bash
git add data/mainstay/fairgame/app/index.html
git commit -m "feat(fairgame): homepage shell — announce, co-brand nav, cinematic hero"
```

---

### Task 4: Photo gallery flood band

**Files:**
- Modify: `data/mainstay/fairgame/app/index.html` (insert gallery band after `</header>`)

- [ ] **Step 1: Insert the gallery band** verbatim from the mock (the `.gallery > .gal-rail > figure` block), pointing images at `img/rod-live-1.jpg`, `img/rod-live-2.jpg`, `img/rod-fans.jpg`, `img/rod-live-3.jpg`, `img/rod-arena.jpg` with their captions.

- [ ] **Step 2: Verify**

Run:
```bash
curl -s https://aialfred.groundrushcloud.com/fairgame/app/index.html | grep -c 'gal-rail'
```
Expected: `1`. On a phone width, the strip scrolls horizontally.

- [ ] **Step 3: Commit**

```bash
git add data/mainstay/fairgame/app/index.html
git commit -m "feat(fairgame): Rod photo gallery flood band"
```

---

### Task 5: Tour rail wired to live `/shows`

**Files:**
- Modify: `data/mainstay/fairgame/app/index.html` (the "Pick your city" section + the `showCard()`/`loadShows()` JS)

**Interfaces:**
- Consumes: `GET /fairgame/api/shows` → `{shows:[{id,show_date,city,venue,min_price_cents,resale_from_cents,active_listings,remaining}]}` (existing). Helpers `money()`, `esc()`, `$()`, `API` (existing in mock/app JS).
- Produces: `.tcard` markup matching DESIGN.md.

- [ ] **Step 1: Add the section shell** "Pick your city" (eyebrow `25 dates · Sep–Nov 2026`, H2, lead) from the mock, with an empty `<div class="rail" id="showsGrid"></div>`.

- [ ] **Step 2: Replace `showCard(s)`** so it emits the new `.tcard` (link to `show.html?id=`), using live fields. Sold-out when `remaining<=0`:

```javascript
function tag(s){return s.remaining!=null && s.remaining<=0
  ? '<span class="tag" style="color:var(--red);border-color:#f0c6c6;background:#fbeded">Sold out</span>'
  : '<span class="tag">Rod-held</span>';}
function avail(s){
  if(s.remaining==null) return '';
  if(s.remaining<=0) return '<span class="avail"><span class="dot r"></span>0 seats</span>';
  return '<span class="avail"><span class="dot"></span>'+s.remaining.toLocaleString()+' seats</span>';}
function showCard(s){
  const sold = s.remaining!=null && s.remaining<=0;
  const price = s.min_price_cents;
  return '<a class="tcard'+(sold?' sold':'')+'" href="show.html?id='+encodeURIComponent(s.id)+'">'+
    tag(s)+
    '<div class="date">'+esc(s.show_date||'Dates TBA')+'</div>'+
    '<div class="city">'+esc(s.city||'TBA')+'</div>'+
    '<div class="ven">'+esc(s.venue||'')+'</div>'+
    '<div class="row"><div class="from">'+(price!=null?money(price)+' <small>+ fee</small>':'<small>TBA</small>')+'</div>'+
    avail(s)+'</div></a>';
}
```

- [ ] **Step 3: Keep `loadShows()`** fetching `API + '/shows'` and `grid.innerHTML = shows.map(showCard).join('')` (already in the app file). Ensure `#showsGrid` is the `.rail`. Keep the existing empty/error states (restyle copy only).

- [ ] **Step 4: Verify against the live API**

Run:
```bash
curl -s https://aialfred.groundrushcloud.com/fairgame/api/shows | head -c 400
```
Expected: JSON with a `shows` array. Open the homepage: real dates render as `.tcard`s in the rail; a sold-out show shows the red dot.

- [ ] **Step 5: Commit**

```bash
git add data/mainstay/fairgame/app/index.html
git commit -m "feat(fairgame): live tour rail (/shows) restyled to DESIGN.md cards"
```

---

### Task 6: Seat-map preview + Fans First promise (SVG icons)

**Files:**
- Modify: `data/mainstay/fairgame/app/index.html`

- [ ] **Step 1: Port the seat-map preview band** (dark `.map-viz` panel + `.bowl` JS + green/red/grey legend) and the **promise** 4-card grid from the mock — including the inline line-SVG icons (ticket/price/phone/shield). The "Safe delivery" card text reads: "Sent to your Ticketmaster account a few days before the show. A TM account is required."

- [ ] **Step 2: Port the bowl-fill script** verbatim:

```javascript
const bowl=document.getElementById('bowl');
const p=[2,2,1,1,1,1,1,2,2, 2,1,1,0,0,0,1,1,2, 1,1,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,1,1, 2,1,1,0,0,0,1,1,2, 2,2,1,1,1,1,1,2,2];
p.forEach((v,i)=>{const d=document.createElement('div');d.className='seat';if(v===1)d.classList.add('g');else if(v===2&&i%3===0)d.classList.add('r');bowl.appendChild(d);});
```

- [ ] **Step 3: Verify**

Run:
```bash
curl -s https://aialfred.groundrushcloud.com/fairgame/app/index.html | grep -c 'class="bowl"'
```
Expected: `1`. Page shows the dark seat panel with green/red/grey dots and the legend.

- [ ] **Step 4: Commit**

```bash
git add data/mainstay/fairgame/app/index.html
git commit -m "feat(fairgame): seat-map preview + Fans First promise cards"
```

---

### Task 7: Album announcement + *Don't Look Down* trailer modal

**Files:**
- Modify: `data/mainstay/fairgame/app/index.html`

**Interfaces:**
- Consumes: `img/rod-portrait.jpg`; YouTube id `VpMAwDs3oI0`.

- [ ] **Step 1: Port the `.album` section** from the mock (eyebrow "New album · out August 28", H2 "Don't Look / Down. / The new album.", "Watch the trailer" `[data-yt]` button + "Notify me", portrait with `.play [data-yt]` and "Official album trailer" label) using `img/rod-portrait.jpg`.

- [ ] **Step 2: Port the modal markup + script** (the `#ytmodal` block and the open/close JS):

```javascript
const YT='VpMAwDs3oI0';
const modal=document.getElementById('ytmodal'),frame=document.getElementById('ytframe');
const openYT=e=>{e.preventDefault();frame.innerHTML='<iframe width="100%" height="100%" src="https://www.youtube.com/embed/'+YT+'?autoplay=1&rel=0" title="Rod Wave — Don\'t Look Down" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>';modal.style.display='flex';};
const closeYT=()=>{modal.style.display='none';frame.innerHTML='';};
document.querySelectorAll('[data-yt]').forEach(el=>el.addEventListener('click',openYT));
document.getElementById('ytclose').addEventListener('click',closeYT);
modal.addEventListener('click',e=>{if(e.target===modal)closeYT();});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeYT();});
```

- [ ] **Step 3: Verify the trailer opens**

Run:
```bash
curl -s https://aialfred.groundrushcloud.com/fairgame/app/index.html | grep -c 'VpMAwDs3oI0'
```
Expected: `1`. Click "Watch the trailer" / the play button → lightbox plays the YouTube trailer; Esc/backdrop closes it.

- [ ] **Step 4: Commit**

```bash
git add data/mainstay/fairgame/app/index.html
git commit -m "feat(fairgame): Don't Look Down album section + trailer lightbox"
```

---

### Task 8: Discover teaser + footer

**Files:**
- Modify: `data/mainstay/fairgame/app/index.html`

- [ ] **Step 1: Port the Discover teaser** (`Discover · Fans First guarantee`, H2 "Not here for Rod? / We'll find you the best deal.", the $1 guarantee lead, the search field, and the blurred-listing reveal card with the gold "cheapest verified" winner + `$1` unlock bar) and the **footer** (3 link columns + the final-sale line) verbatim from the mock. Link "Unlock the deals" / "Search" to `discover.html`.

- [ ] **Step 2: Verify copy compliance**

Run:
```bash
curl -s https://aialfred.groundrushcloud.com/fairgame/app/index.html | grep -Ec 'lowest verified|All sales final'
```
Expected: `>=1` for the verified-price phrasing and the final-sale line both present (run each grep separately to confirm both).

- [ ] **Step 3: Commit**

```bash
git add data/mainstay/fairgame/app/index.html
git commit -m "feat(fairgame): Discover $1 teaser + footer with final-sale line"
```

---

### Task 9: Responsive, no-emoji, and performance gate

**Files:**
- Modify: `data/mainstay/fairgame/app/index.html` (mobile CSS only if gaps found)

- [ ] **Step 1: Emoji gate** — there must be ZERO emoji in the homepage:

Run:
```bash
cd /home/aialfred/alfred
grep -P '[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}]' data/mainstay/fairgame/app/index.html
```
Expected: no output (exit 1). If any match, replace with a line SVG or remove.

- [ ] **Step 2: Port the mock's mobile breakpoints** (`@media(max-width:860px)` and `(max-width:560px)`) into the page-local `<style>`: hero stacks (photo on top), full-width CTAs, album reflow, gallery scroll, reduced section padding.

- [ ] **Step 3: Verify responsive at 3 widths** — load the homepage at 1280px, 768px, and 380px (browser devtools or a headless screenshot). Confirm: no horizontal overflow, hero readable, tour rail + gallery scroll, album stacks, buttons reachable (≥44px touch targets).

- [ ] **Step 4: Performance check** — total homepage image weight stays lean:

Run:
```bash
cd /home/aialfred/alfred && du -ch data/mainstay/fairgame/app/img/*.jpg | tail -1
```
Expected: total under ~2.6MB (lazy-load below-the-fold `<img>` already via `loading="lazy"` — add it to gallery/album/promise imgs if missing).

- [ ] **Step 5: Commit**

```bash
git add data/mainstay/fairgame/app/index.html
git commit -m "feat(fairgame): responsive polish, lazy images, no-emoji gate"
```

---

### Task 10: Final visual parity check vs the approved mock

- [ ] **Step 1: Side-by-side** — open the live app homepage and the approved mock (`/drafts/fansfirst-mock/`) at desktop + phone widths. Confirm parity: dark hero + light body, single gold accent, squared buttons, no emoji, Rod flood, album trailer, $1 Discover. Note any drift.

- [ ] **Step 2: Fix any drift** found inline, commit per fix.

- [ ] **Step 3: Final commit / mark Plan 1 done**

```bash
git add -A data/mainstay/fairgame/app
git commit -m "feat(fairgame): homepage redesign complete — parity with approved mock (Plan 1)"
```

---

## Self-Review
- **Spec coverage (this plan = homepage only):** design language ✓ (Tasks 2–10), co-brand/partnership ✓ (T3), Rod photo flood ✓ (T1,T4,T7), live tour data ✓ (T5), seat-map preview + legend ✓ (T6), album promo + trailer ✓ (T7), Discover $1 teaser + verified-price copy ✓ (T8), no-emoji + responsive ✓ (T9). Out of this plan (own plans next): full seat-map page, checkout + triple no-refund, Discover engine + $1 paywall, accounts/My Tickets, admin CMS, delivery assist tool.
- **Placeholders:** none — new logic (token block, `showCard` restyle, album modal, gates) is shown in full; verbatim-port instructions point at the canonical mock rather than duplicating 300 lines of approved CSS (DRY).
- **Type consistency:** `showCard(s)`/`loadShows()`/`money()`/`esc()`/`API` names match the existing app JS; `/shows` field names (`remaining`, `min_price_cents`, `show_date`, `city`, `venue`) match `core/fairgame/events.py` + the live endpoint.

## Next plans (build order)
2. Seat-map page (`show.html`) · 3. Checkout (TM email + triple no-refund) · 4. Discover $1 paywall (`discover.html` + `/discover`) · 5. Accounts / My Tickets · 6. Admin CMS (`admin.html` + `/admin`) · 7. Delivery assist tool (`tm_transfer.py`, throttled + human-approved).
