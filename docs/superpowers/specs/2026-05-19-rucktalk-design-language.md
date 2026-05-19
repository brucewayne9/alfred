# RuckTalk Design Language — Locked

**Date approved:** 2026-05-19
**Approved by:** Mike Johnson
**Parent spec:** [`2026-05-19-rucktalk-rebuild-phase-0.md`](./2026-05-19-rucktalk-rebuild-phase-0.md)
**Visual reference (live mockup):** https://aialfred.groundrushcloud.com/static/drafts/rucktalk-homepage-mockup.html
**Mockup source:** `~/.openclaw/workspace/static/drafts/rucktalk-homepage-mockup.html` (symlinked at `static/drafts/`)
**Implementation plan:** [`docs/superpowers/plans/2026-05-19-rucktalk-phase-1a-site-redesign.md`](../plans/2026-05-19-rucktalk-phase-1a-site-redesign.md)

This document is the source of truth for the visual + copy language of rucktalk.com after Mike's design approval. Any task in Plan 1A or later that touches CSS, copy, or layout MUST pull from this doc.

---

## 1. Aesthetic positioning

**"Editorial Workman."** Magazine-style editorial layout with industrial type — feels like an indie magazine for builders and providers, not a SaaS landing page or a wellness lifestyle brand.

- Type-led above the photo
- Generous whitespace
- Asymmetric layouts
- Warm paper-cream surfaces, not white
- Single accent (terracotta) doing most of the work, with rare forest-green and mustard moments

## 2. Type stack

| Role | Font | Weights | Notes |
|------|------|---------|-------|
| Display (headlines, brand) | **Archivo Black** | 900 only (one weight) | Heavy industrial grotesque. Freight-truck weight. Use for hero h1, section big titles, brand mark. |
| Display secondary (italic, sub-heads, italic emphasis) | **Archivo** | 400, 500, 600, 700 (italic + roman) | Companion to Archivo Black. All italics + pulled quotes + sub-heads. |
| Body | **Bricolage Grotesque** | 300, 400, 500, 600, 700 | Modern grotesque with character (angled cuts on a/g/c). All paragraphs, UI text, captions. |

**Google Fonts URL:**
```html
<link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=Archivo:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500;1,600&family=Bricolage+Grotesque:opsz,wght@12..96,300;12..96,400;12..96,500;12..96,600;12..96,700&display=swap" rel="stylesheet">
```

**CSS variables:**
```css
--f-display:   "Archivo Black", "Helvetica Neue", Impact, sans-serif;
--f-display-2: "Archivo", "Helvetica Neue", sans-serif;
--f-body:      "Bricolage Grotesque", -apple-system, BlinkMacSystemFont, sans-serif;
```

**Rejected:** Fraunces (too literary/decorative — Mike: "goofy"). Inter (banned per skill — too generic). Anything condensed-stencil — too "stadium signage."

## 3. Color palette

| Token | Hex | Use |
|-------|-----|-----|
| `--paper` | `#F4F0E6` | Body background — warm cream |
| `--paper-deep` | `#ECE5D3` | Darker cream — sectional surfaces, shop bg |
| `--paper-soft` | `#FBF8F0` | Lightest cream — popup panel, hover surfaces |
| `--ink` | `#1B1815` | Deep warm ink (NOT pure black) — body text, brand |
| `--ink-2` | `#4A4540` | Mid-ink — body copy, sub-headings |
| `--ink-mute` | `#8C857B` | Muted — captions, metadata |
| `--rule` | `#C8C2B4` | Warm rule lines — borders, dividers |
| `--clay` | `#B85432` | **Primary accent** — terracotta, CTAs, italic emphasis |
| `--clay-deep` | `#8B3A1F` | Hover/active terracotta |
| `--clay-soft` | `#E8C9B8` | Surface tint, drop shadows |
| `--mark` | `#2D6741` | **Secondary accent** — forest green, badges, "today's take" indicators |
| `--mark-soft` | `#C8DCC3` | Light forest tint (rare) |
| `--ink-radio` | `#131110` | Slightly darker than ink — floating radio bar + LoovaCast block |
| `--warn` | `#C9A227` | Mustard — used sparingly, NOT in v1 surfaces |

**Hard rules:**
- ❌ No pure white anywhere — always `--paper-soft` or lighter `--paper`
- ❌ No pure black — always `--ink`
- ❌ No blue (kills the warm tone)
- ✅ Body always has subtle SVG-noise paper grain texture (see code in mockup)

## 4. Spacing + layout tokens

```css
--s-1: 4px;  --s-2: 8px;  --s-3: 12px; --s-4: 16px;
--s-5: 24px; --s-6: 32px; --s-7: 48px; --s-8: 64px;
--s-9: 96px; --s-10: 128px;

--max-w: 1240px;
--gutter: 32px;
--radio-h: 54px;
```

## 5. Component patterns

### 5a. Floating LoovaCast radio bar
Persistent top bar, full width, `--ink-radio` background, slim (`--radio-h` = 54px). Live indicator (pulsing terracotta dot), label "Live", current track scrolling, terracotta play button (32px circle), "Powered by **LoovaCast**" attribution.

### 5b. Site header
Sits below radio bar, sticky on scroll. Brand on left (logo placeholder for now — Mike to provide real logo). Primary nav on right: **Podcast · Watch · Blog · Training · Shop · About · Free plan** (Free plan is the CTA pill, dark `--ink` background).

### 5c. Hero (asymmetric editorial)
- Grid: 5fr (photo) | 7fr (copy)
- Photo: 4:5 aspect, terracotta drop-shadow offset 12px
- Copy block:
  - Eyebrow: "A conversation for the go-getters" (terracotta, uppercase, letter-spacing 0.18em, leading dash)
  - **H1 tagline:** `Notes from <em>a guy figuring it out.</em>` (Archivo Black 88px, italic "a guy figuring it out" in Archivo italic 600 terracotta)
  - Strap line: italic Archivo (`--f-display-2`), `--fs-lg`, 44ch max width
  - Two CTAs: primary "Get the free 8-week plan" (`--ink` bg, `--paper-soft` text), secondary "▶ Today's episode" (outlined `--ink`)
  - Listen-on row: "Listen on" label + 3 platform pills (Spotify · Apple Podcasts · YouTube Music) — colored dots + rounded border + hover state
- Folio number top-right: "Vol. I · No. 01" (italic, muted)

### 5d. Dedicated LoovaCast player block
Full-width `--ink-radio` dark section between hero and pillars. Grid: 180px art tile | flex copy column | controls.
- Art tile: 180×180, terracotta→clay-deep→forest gradient, white concentric-circle decoration, "RuckTalk Radio" italic
- Eyebrow: pulsing live dot + "Tune in live · 24/7"
- Title: "Now playing: [italic terracotta-soft track name]"
- Meta: episode # · time in · next up
- Attribution: "Streaming on **LoovaCast**" pill
- Controls: 88px terracotta circle play button + "Press play. Always streaming." sub

### 5e. Five Pillars (dynamic)
**Visual pattern:** 5-column grid, no card chrome, vertical rules between. Each pillar:
- Number (01-05, italic terracotta Archivo)
- Pillar name (Archivo Black UPPERCASE)
- "What it stands for" — one-line definition (body sans, muted ink)
- Dashed rule
- **"Today's take" label** (forest-green pulsing dot + 10px uppercase letter-spaced)
- **Rotating snippet** (italic Archivo 500, 1-2 sentences of practical advice that changes daily)

**Snippet rotation strategy (implementation):**
- WP option `rt_pillar_snippets` holds a JSON object: `{health: [str, ...], business: [str, ...], family: [str, ...], strength: [str, ...], shared: [str, ...]}`
- Front-end template picks one snippet per pillar deterministically keyed by `(pillar, date)` so all visitors today see the same snippet, but tomorrow's render is different
- Admin UI (settings page) has 5 textareas — Mike pastes one snippet per line, save → updates the option
- Mockup shows JS rotation every 5s for visual demo; production is server-side daily rotation

**Starter pool (Mike-voiced, ready to seed):**
- Health: "Stretch your hips before coffee. Your lower back's been asking nicely." · "Heavy carries beat cardio for fat loss and look better at the playground." · "Sleep before midnight or you're chasing yesterday's tired all day." · "Three deep breaths before you check your phone. That's the difference."
- Business: "Block one hour after lunch. No phone. Ask: what would I do differently if I started today?" · "Raise your prices. The right people will pay; the wrong ones were always going to leave." · "If you can't explain it to your spouse, you can't sell it." · "Fire the customer that's costing you sleep. They're never worth it."
- Family: "Twenty minutes Sunday night. Counter, notebook, your spouse. Prevents 80% of the dumb arguments." · "Pick up the phone when your mom calls. You don't get those calls forever." · "Eat dinner together. Phones in the other room. Five days a week minimum." · "Walk with one of the kids alone. They tell you things in motion they won't sitting down."
- Strength: "Tuesdays make you. Mondays anyone can do." · "Stop training to look strong. Train so you can pick up your kids when they're 30." · "Sit ups don't fix your back. Picking things up does." · "The hardest set is the one after you wanted to stop."
- Shared: "Most people are way more tired than they let on. Includes you." · "Nobody's got it figured out. They just stopped saying so out loud." · "Being kind is free and almost always the right call." · "Ask better questions. Listen longer than feels comfortable."

### 5f. Latest episode (magazine spread)
2-column grid: 1fr art | 1.4fr copy.
- Art: 1:1 square, `--ink` background, forest-green offset shadow (14px), vinyl-record concentric-circle decoration, episode # + title + date inside the tile
- Copy: italic episode number (Archivo italic terracotta), big Archivo Black headline, italic excerpt, **Listen/Watch toggle pill** (segmented control), meta row, two action buttons

### 5g. Blog grid (recency)
Standard 3-column editorial grid. Each card: terracotta category eyebrow + Archivo Black headline + body excerpt + meta row. Complements the pillar grid (which is curated); this one is raw recency for completeness.

### 5h. Shop teaser
Full-bleed `--paper-deep` background section. 3 product cards with placeholder visuals (double-frame border treatment), product title (Archivo bold), italic terracotta price.

### 5i. Newsletter coupon (the one allowed popup pattern, inline version)
"Clip out" coupon style: 2px dashed `--ink` border, scissors `✂` emoji in top-left corner (rotated -20°), italic forest-green "Free · Email-gated" stamp top-right (rotated 8°). Centered headline + italic sub + signup form + micro-copy.

### 5j. About pulled quote
2-column grid: `1fr | 2fr`. Left: "About RuckTalk" label with terracotta accent rule. Right: italic Archivo blockquote (Mike's voice, ~50ch), "— Mike, host" signature, "More about the show →" link.

### 5k. Footer
Full `--ink` dark section. Footer signup (echoes coupon styling but on dark), ecosystem strip (5 brand wordmarks with italic names + uppercase tag lines), legal meta row.

### 5l. Newsletter popup
Centered modal, paper-cream panel, PDF cover image (REAL cover at launch, not placeholder), italic terracotta-em headline, sub, signup form. Triggers: scroll-depth 50% on first page OR exit-intent on second page; 14-day dismiss cookie.

## 6. Copy voice

### 6a. The locked tagline
> **"Notes from a guy figuring it out."**

The phrase "a guy figuring it out" is italicized and terracotta. Treat as a single unit — don't break across lines mid-phrase.

### 6b. The locked hero strap
> "One go-getter's running commentary on health, business, family, strength, and what everybody else is going through. Tell me what I'm missing."

Captures: observational stance (running commentary), inclusive ("go-getters" not "guys"), audience exchange ("tell me what I'm missing").

### 6c. The locked About blurb
> "I'm not an expert in anything. I just look at the world and give my honest take — and then I want to hear yours. RuckTalk is a conversation for the go-getters: real life out loud, decisions worth getting right, and the stuff everybody else is also going through."

### 6d. Voice rules (apply everywhere)
- **"Go-getters" not "guys"** — site is for builders and providers, not gender-coded
- **Conversational, not curated** — "Notes from" not "Insights into." "Tell me what I'm missing" not "Subscribe for updates."
- **Practical not aspirational** — "Stretch your hips before coffee" not "Optimize your morning routine for peak performance"
- **No guru posture** — Mike's not a coach, he's a fellow-traveler observing the world. Avoid "tools," "frameworks," "blueprints," "protocols." Use "the way I'd do it," "what works for me," "one trick that helped."
- **No "real talk"** — Mike specifically rejected it; implies others are bullshitting. Mike isn't framing himself as truth-teller.
- **Inclusive without being preachy** — copy should welcome women, single people, non-parents, but it doesn't have to advertise that fact. Just don't write copy that excludes them.

## 7. Motion + interaction

- Hero load: staggered text reveal (5 elements, 0.05s → 0.55s delays, cubic-bezier ease-out, 8px translate)
- Radio bar live dot: pulse 2.2s infinite (terracotta box-shadow expand)
- "Today's take" green dot: same pulse pattern but forest-green
- Pillar hover: background tint to `--paper-soft`, arrow `→` slides in from left
- Episode format toggle (Listen/Watch): instant pill swap, no animation
- Tagline picker (mockup-only): hero updates live on click — proves picker pattern for future copy A/B
- Pillar snippets: server-rendered in prod (daily rotation), no client motion needed except optional fade-in-fade-out on cron-pull AJAX (not v1)

## 8. Outstanding visual asks

| # | Item | Owner | Blocks |
|---|------|-------|--------|
| 1 | Real RuckTalk logo file (SVG preferred, PNG @2x acceptable) | Mike | Header brand rendering — placeholder block in until provided |
| 2 | Mike's hero portrait (4:5, 1600px+ wide, JPG q82, no gym props) | Mike | Hero photo block |
| 3 | Real PDF cover image (for popup + /training/free) | Mike (or repurpose existing FaR cover) | Popup visual + training page visual |
| 4 | 5 ecosystem brand logos (SVG wordmarks): LoovaCast, LumaBot, AIROI, Roen, GRL | Mike or generate | Footer ecosystem strip — text-only fallback works for v1 |
| 5 | Shop product photography (3 SKUs minimum to fill homepage teaser) | Mike (Phase 2) | Shop teaser remains placeholder until products exist |

## 9. Implementation pointers

- All design tokens above are encoded in the mockup HTML at `https://aialfred.groundrushcloud.com/static/drafts/rucktalk-homepage-mockup.html` — Plan 1A's CSS task (Task 14) lifts the `:root` token block and component styles directly from there
- Mockup uses class names that match the BEM-ish `rt-` prefix in Plan 1A's CSS spec — names should be preserved
- The mockup's section ordering IS the prod page ordering (Plan 1A Wave 3-6 maps 1:1)
- JS in mockup is illustrative only — production wires real WP REST + Brevo + daily snippet rotation via PHP

---

## Self-review

**Placeholder scan:** All TBDs are in §8 (Outstanding visual asks), each clearly owner-tagged. No silent placeholders.

**Internal consistency:** Color tokens, font stack, spacing scale all line up with the mockup at the URL above. Pillar snippet pattern matches mockup's JS rotation logic. Tagline + strap + About copy match what's currently rendered in the mockup.

**Scope check:** This is purely a visual + copy reference. Implementation tasks live in Plan 1A.
