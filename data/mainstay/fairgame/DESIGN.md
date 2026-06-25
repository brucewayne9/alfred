# Fans First — DESIGN.md

The design contract for the Fans First web app (Rod Wave official resale + Discover).
Every screen is built to this file. Reference mock: `drafts/fansfirst-mock/` →
https://aialfred.groundrushcloud.com/drafts/fansfirst-mock/

> Inspired by the Spotify design language (music-native, near-black, content-as-color)
> but **lightened**: a cinematic dark hero/footer with a warm light editorial body.

## 1. Voice & Atmosphere
- **Premium but cool.** Reads like a real partnership between an artist and a serious
  company — not a loud "AI" marketplace. Confident, restrained, fan-first.
- **Rod's photography is the color.** The UI stays neutral; imagery carries the emotion.
- **Tone of copy:** warm, direct, Gen-Z-fluent, never corporate-stiff, never cutesy.
  e.g. *"Real seats. Real fans. No scalper games."*
- **NO emoji anywhere.** Iconography is line SVG only. Emoji read as slop.

## 2. Color
| Token | Hex | Role |
|-------|-----|------|
| `--bg` | `#0b0a09` | Dark hero / footer / album / seat-map panel |
| `--bg2` | `#121110` | Dark gallery band |
| `--paper` | `#f5f1ea` | Warm light canvas (body default) |
| `--paper-alt` | `#ece5d7` | Alternate light band |
| `--card` | `#ffffff` | Cards on light |
| `--ink` | `#17140d` | Primary text on light |
| `--ink-2` | `#67604f` | Secondary text on light |
| `--ink-3` | `#988f7c` | Tertiary / muted on light |
| `--silver` | `#aaa49a` | Secondary text on dark |
| `--line-l` | `#e4dccc` | Borders on light |
| `--line-d` | `#2a2724` | Borders on dark |
| `--teal` | `#82bcc4` | **The one accent** — bright teal (on dark) |
| `--teal-deep` | `#2c6f7b` | teal text/eyebrows on light (contrast) |
| `--teal-btn` | `#5fa8b3` | Button fill (always dark text `#1a1206`) |

**Accent discipline:** teal is the *only* brand color. The seat-map semantics are the
**only** other colors allowed, and only inside the map:
`--green #13bd57` (available · ours), `--red #e23b3b` (sold · ours), `--grey #322f2a` (not our inventory).

## 3. Typography
- **One family: `Figtree`** (Google Fonts), weights 400/500/600/700/800/900.
- Display/H1: 900, `letter-spacing:-.04em`, line-height ~.98.
- Section H2: 800, `-.03em`.
- Eyebrow/label: 12px, 700, `letter-spacing:.18em`, UPPERCASE, teal-deep (light) / teal (dark).
- Body: 16–17px, 400, `--ink-2`. Lead max-width ~560px.
- No condensed/novelty display faces. Restraint over flair.

## 4. Shape & Components
- **Buttons are squared, not pills:** `--r-btn:11px`. Icon-driven (lead/trail line SVG).
  - `.btn-teal` = primary (teal fill, dark text). `.btn-line` = secondary (1px border).
  - Hover: 2px lift, no color circus.
- Cards: `--r-card:14px`, 1px `--line-l`, soft shadow `0 14px 38px rgba(22,18,8,.10)`.
- Status tags: `--r-tag:6px` small caps ("Rod-held", "Sold out").
- Search inputs: squared 11px, line-icon left.
- Depth on dark = heavy shadows; depth on light = hairline border + soft shadow.

## 5. Layout
- Container `max-width:1200px`, gutters 24px (18px mobile).
- Section padding 88px (60px mobile). Alternate `paper` / `paper-alt` bands.
- **Structure of a marketing page:** dark hero → photo gallery band → light content
  sections → one dark "moment" (album promo) → light → dark footer. Dark is for
  *imagery moments*, never for whole walls of UI.

## 6. Signature Moves (keep these)
1. **Cinematic dark hero** with Rod's photo, gradient fade into `--bg`, floating stat chip.
2. **Co-brand lockup:** `FANS FIRST × ROD WAVE · Official Partner` (announcement bar + hero).
3. **Photo gallery flood** — horizontal scroll of live/fan shots under the hero.
4. **Seat map = dark panel inside a light band** so green/red/grey pop.
5. **$1 reveal** — blurred competitor prices, teal "cheapest verified" winner, unlock bar.
6. **Sticky nav** transparent over the hero, solidifies to paper-blur on scroll.

## 7. Do / Don't
**Do:** let photos carry color · one teal accent · squared buttons · line icons · generous
light space · honest copy ("lowest *verified* price", "all sales final, worded warmly").
**Don't:** multiple bright accents · emoji · pill buttons · novelty fonts · dark-on-dark walls ·
overstate claims ("lowest anywhere" is not defensible — say "lowest verified, guaranteed").

## 8. Responsive
Web-only, fully responsive (no native app this phase). Breakpoints 860px (stack) and 560px
(single column, full-width CTAs, scrolling rails). Optimize hero/section imagery for mobile
(<500KB) — never ship multi-MB PNGs to phones.
