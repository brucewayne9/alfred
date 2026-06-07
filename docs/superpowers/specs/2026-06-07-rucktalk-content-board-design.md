# RuckTalk Content Board — Design Spec

**Date:** 2026-06-07
**Owner:** Mike Johnson (sir)
**Built by:** Alfred
**Status:** Approved (design), pending implementation plan

## Purpose

A phone-friendly web board that turns Mike's social brain-dumps into shoot-ready
Instagram Reel cards, tracks each reel from idea → posted, and shows the real
Instagram numbers (reach · saves · shares) on posted reels so winners are obvious.

This is the **execution surface for the "Regulated Builder" RuckTalk Instagram
strategy** (see memory `project_rucktalk_instagram_strategy.md`). Mike is a solo
operator who gets off course; this board is how he sees his content pipeline at a
glance and how Alfred holds him accountable.

**Scope:** RuckTalk Instagram (@rucktalk) only. Roen is shelved. Radio dumps are
NOT on this board — they keep their existing flow.

## Success Criteria

- Mike can dump a social idea from his phone and within ~60s see a card on the board.
- Every card carries a full shoot blueprint he can read and film from.
- Mike tracks a reel through To Shoot → Shot → Posted with one tap.
- Posted reels show live IG stats so the winning format is obvious.
- Mike is reliably reminded to paste each reel's link (the one manual step).

## The Three Lanes (dump routing)

Brain-dumps already capture via the Telegram bot (`interfaces/telegram/bot.py`,
`_extract_brain_dump` + `handle_message`). Extend routing by lane tag:

| Dump opens with… | Routes to | Behavior |
|---|---|---|
| `brain dump radio …` | Radio journal | Existing flow (source=radio). NOT on the board. |
| `brain dump social …` / `brain dump instagram …` | Content Board | Save to journal AND create a rough card in "To Shoot". |
| `brain dump …` (no lane) | General journal | Alfred triages into radio or social later. |

Bot reply on a social dump: **"🎬 Card's on the board."**

## How a Thought Becomes a Card

1. **Instant rough card** (bot, automatic): on a `social` dump the bot appends the
   raw text to the journal and POSTs a rough card to the board API. Rough card =
   title (first ~6 words), the raw dump as the script seed, placeholder hook,
   `status=to_shoot`, `polished=false`. This is the "instant draft" half of the
   approved "instant draft, I polish later" model.
2. **Polish pass** (Alfred, in working sessions): upgrade the card to the full
   blueprint — sharpen hook, write shot, script, caption, hashtags, save/share
   line; set `polished=true` (shows a ✨ marker). Same card, upgraded in place.

### Card fields

- `id`, `created_at`
- `title` — short label
- `raw` — the original dump text (script seed)
- `hook` — first-3-seconds line + on-screen text
- `shot` — what to film (e.g. "talk to camera mid-ruck", "at the studio board")
- `script` — ~30s talking points / loose script
- `caption` — full caption with keywords baked in
- `hashtags` — 3–5
- `cta` — the save/share closer line
- `status` — `to_shoot` | `shot` | `posted`
- `polished` — bool (✨ when true)
- `reel_url` — IG link, pasted when moved to Posted (empty until then)
- `stats` — `{reach, saves, shares, fetched_at}` (null until link + pull)

## The Board (frontend)

- **3 columns:** To Shoot → Shot → Posted. Tap/drag a card to advance it.
- **Tap a card** to open the full blueprint; every field editable inline.
- **Mobile-first**, behind Caddy basic_auth (same as the Rollout board), own cred.
- Path: `aialfred.groundrushcloud.com/rt-board/`.
- Posted cards render `reach · saves · shares`; a 🔥 marker on standout reels.

## The Scoreboard (Posted column)

- Moving a card **Shot → Posted** triggers an inline prompt: *"Paste the reel link
  to turn on stats 👇"*. The move IS the reminder.
- A Posted card with no `reel_url` shows a yellow **"⚠️ link needed"** badge until
  filled — a persistent visual nag.
- Alfred sweeps stragglers in the weekly sync: flags any Posted card missing a link.
- Once a link exists, stats pull from the Meta Graph API (`META_ACCESS_TOKEN`,
  RuckTalk IG id `17841461784057534`) via media insights (reach, saves, shares).

## Architecture (reuse the Rollout-board pattern)

Mirror `core/api/rollout_board.py` (single JSON-blob state, last-write-wins,
gated at the edge by Caddy basic_auth):

- **New backend:** `core/api/rucktalk_content_board.py`
  - `GET  /rt-board/api/state` → current board (seeds empty columns on first run)
  - `PUT  /rt-board/api/state` → overwrite (card edits / column moves)
  - `POST /rt-board/api/card` → append a rough card (called by the bot)
  - `POST /rt-board/api/refresh-stats` → pull IG stats for posted cards w/ links
- **State file:** `data/rucktalk/content_board/board_state.json`
- **Frontend:** static HTML/JS page (Kanban, inline edit, mobile-first), served at
  `/rt-board/`, Caddy basic_auth.
- **Capture routing:** extend `interfaces/telegram/bot.py` brain-dump handler to
  detect the `radio` / `social` / `instagram` lane and POST social dumps to the
  board API.
- **Stats:** reuse existing Meta token plumbing; map `reel_url` → IG media id →
  insights. (Worst case, resolve media id from the account's recent media list.)

## The One Manual Step

Instagram does not link a card to a post automatically. When Mike marks a reel
**Posted**, he pastes its link once. Three-layer reminder (move-prompt + nag badge
+ weekly sweep) ensures he never forgets. Everything else is automatic.

## Out of Scope (v1 — YAGNI)

- Auto-posting reels (intentional: reels need Mike's real face/voice — the whole
  authenticity play; he films, never AI-posts).
- Roen / any second account.
- Auto-matching posts to cards without a pasted link.
- Drag-reorder within a column, multi-user, comments.

## Open Questions

- Reel link → media id resolution: prefer Mike pasting the canonical reel URL;
  confirm the URL form Instagram gives him and that it maps to a media id via the
  Graph API. Validate during implementation.
