# Central Casting — Design Spec

**Date:** 2026-06-06
**Owner:** Mike Johnson
**Status:** Approved (design) → ready for implementation plan

## One-liner

A private studio app for creating AI radio personalities ("DJs") — record or pick a
voice, draft a personality, and schedule them onto AzuraCast stations on demand.
Build the library once; swap hosts (and drop in guests) like cartridges.

## Goals

- Turn the current **manual** workflow (Mike emails recordings → Alfred hand-places wavs,
  hand-writes personas, hand-inserts DB rows) into a **repeatable pipeline**.
- Push Qwen3-TTS to its limit so every personality is **distinct and recognizable** when
  browsing the library.
- Let Mike **swap DJs in and out** of station slots on a schedule, without rebuilding anything.
- Support **diverse personalities and ethnicities** at roster scale (many DJs, fast).

## Non-goals (explicitly out of scope)

- Multi-tenant / client logins / billing. Single operator (Mike); agents may drive the API.
- Replacing AzuraCast. Central Casting is the *studio*; AzuraCast is one *stage* it deploys to.
- Real-time DRM, payment, or public-facing UI. Internal admin tool only.

## Users & access

- **Primary user:** Mike only. Login-gated.
- **Secondary:** Lucius / Oracle / Alfred may drive it via a clean API (no extra human users).
- Lives as a **module inside Alfred Labs** (FastAPI + React on 105), reusing existing auth.

## Why this architecture

- Reuses the **Qwen3-TTS server** (105:7860) for cloning/synthesis.
- Reuses the **mood-routing logic** already proven in `TtsService.php`.
- Reuses the **duo conversation engine** (host + second voice) for guests.
- Reuses the **proven PDO deploy path** to `station_ai_dj_breaks` + wav placement on 98.
- Standalone service or AzuraCast-native were rejected: standalone re-plumbs auth + the
  AzuraCast bridge for no gain; AzuraCast-native traps each DJ inside one station and can't
  model "voice + mood set + persona + daypart" as one swappable unit.

## The library lives outside AzuraCast

Its own store: a DB (personalities, voices, assignments, bookings) + the wav files + persona
text. "Deploy to AzuraCast" is a **button/scheduler action** that pushes a chosen DJ onto a
chosen station slot. Same DJ is reusable across stations.

## Data model

### Personality (a "DJ")
- `name`, `avatar`, `role` = **host** | **guest**
- `persona_prompt` (the full character/show prompt used by the brain)
- `archetype_tags` (e.g. firebrand, wellness-host, strategist)
- `expertise` (guests only — e.g. "exercise physiologist, recovery science")
- `status` = draft | ready | live
- `voice_source` = recorded | stock-shelf | generated

### Voice
- The **8 standard mood clips** + a **laugh** asset.
- Stored as `<Name>_<mood>.wav` (matches the existing engine convention).

### Assignment (hosts)
- `station`, `slot/daypart`, `personality (host)`, `effective_datetime`.
- Scheduled: flips itself at the slot boundary.

### Booking (guests)
- `station`, `host`, `personality (guest)`, `topic_brief`, and a visit window expressed as
  **either** a number of breaks **or** a time window (operator chooses per booking).
- During the visit, breaks render as **host + guest conversations** (duo engine); after the
  visit the host returns to solo.
- Future: recurring contributor bookings; a guest across multiple hosts' shows.

## The standard Mood Pack (intake)

Every recorded voice is captured on the **same 8 reads**, same booth / mic / distance /
session, ~20–35s each (~3–3½ min total). Consistency comes from same voice + same booth,
not same words — so each read's copy is written to *pull* its emotion (authentic exemplars,
even from non-actors). The 8 reads are enshrined on the record screen:

1. **Neutral** (warm baseline / anchor)
2. **Fired up** (high energy)
3. **Serious** (grounded, sincere)
4. **Amused** (light, lands on a laugh)
5. **Thoughtful** (reflective, slower, lower)
6. **Reactions** (short ad-libs + the laugh — connective tissue)
7. **Wry / Sarcastic** (dry, deadpan)
8. **Intimate / Late-night** (hushed, close-mic, slow)

Canonical script copy for all 8 reads is stored with the app (drafted 2026-06-06; see
conversation / to be embedded in the record screen).

### How moods are used at airtime (existing behavior, preserved)

Kimi K2.6 writes the break with inline delivery tags (`[neutral]`, `[fired]`, `[serious]`,
`[amused]`, `[thoughtful]`, `[wry]`, `[intimate]`) + `{laugh}`. The engine splits by tag,
synthesizes each block with the **matching mood clip** as the reference, stitches with breath
gaps, splices the laugh, level-matches, renders one mp3 → on air with the bed ducked under it.
Fallback chain: requested mood → `_neutral` → default.

## Screens / flows

1. **Create a DJ**
   - Choose voice source: **Record** · **Stock Shelf** · *(v3)* **Generate from description**.
   - *Record:* on-screen 8-read Mood Pack + same-booth checklist; capture/upload one clip per
     mood; drag-and-drop upload supported (replaces emailing files).
   - *Persona:* type a one-line brief **or** pick an archetype → app drafts the full persona via
     Kimi/Claude → operator edits/approves.
   - Save → auto-process (trim, level-match, register clones) → DJ shows **Ready**.
2. **Library** — grid of DJ cards (avatar, name, archetype, ▶ "hear them" preview, status, edit).
3. **Deploy / Schedule** — pick DJ → station → slot → effective time → confirm. Per-station
   lineup timeline shows queued swaps. Self-flips at the boundary (copy wavs to 98, write the
   break row, live engine picks up next break). Guests use the **Booking** flow (breaks or window).
4. **Stock Shelf** — bank of ready-made diverse reference voices for in-a-pinch creation; grows over time.

## Voice sources

- **Recorded (A-tier, default):** the Mood Pack capture above.
- **Stock Shelf (v2):** pick the closest ready-made reference voice (spanning ethnicities,
  genders, ages, accents), name it, give it a persona. ~60 seconds, no recording.
- **Generate from description (v3):** type a voice description → a voice-design model invents a
  novel reference clip → cloned. Deferred (extra model, less deterministic, can't always re-summon
  the exact voice).

## Safety rails

- **Preview before air:** every DJ renders a *fresh sample break* in their voice **without**
  touching a live station. Audition first. (Hard rule: never test on a live station.)
- **Processing errors** (clipping, too short, bad audio) flagged on the card with what to re-record.
- **Deploy errors** (98 unreachable, wav copy fail) roll back cleanly + Telegram alert. Never a
  half-deployed jock.

## Build order

- **v1 (MVP):** Library + record/upload intake + persona drafting + Preview + scheduled host
  deploy to AzuraCast. Fully replaces the manual email workflow.
- **v1.5:** Guest role + bookings (duo-engine wiring into the live break scheduler).
- **v2:** Stock voice shelf.
- **v3:** Generate-a-voice-from-description.

## Testing

- Preview-render path is the primary functional check (audition without airing).
- Deploy path verified against a **non-live / scratch station or fake slot** before touching a
  real daypart.
- Processing pipeline validated on known-good and known-bad audio (short, clipped, silent).

## Open items to confirm during planning

- Exact Alfred Labs module boundary (API routes, where the React screens mount).
- Storage location for library wavs on 105 vs. the existing `qwen3-tts/resources/` convention.
- Scheduler mechanism for self-flipping assignments/bookings (cron vs. in-app worker).
