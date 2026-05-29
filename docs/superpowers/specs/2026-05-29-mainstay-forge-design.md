# Mainstay Forge — Design Spec

**Date:** 2026-05-29
**Working name:** Mainstay Forge (rename pending Mike)
**Owner:** Mike Johnson (President, Mainstay Tech) · Built by Alfred
**Primary users:** Mainstay social team — Jordan, Dharmic, Mello (+ Markion, Chris on the thread)
**Hard deadline:** Rod Wave *Don't Look Down* drops **June 19, 2026**. Teaser/leak campaign runs up to it.

---

## 1. Purpose

A self-serve, mobile-first dashboard that lets the Mainstay social team mass-produce and distribute short-form viral content for artist rollouts — first use case being Rod Wave's *Don't Look Down*. "Opus Clip for us." First product under Mainstay Tech.

The team's real job is **seeding sounds**: get a song's hook into ears across decentralized fan/burner pages before release so the sound is familiar at drop. The tool exists to make that *fast, organized, and suppression-resistant*.

This is **not** just a clipper. It is a **hook-seeding machine**: Creation × Multiplication, with a Distribution war-room and an Intelligence scoreboard around it.

## 2. Reference material (what the team actually makes)

Studied from Mike's sample links. Three confirmed content formats:

1. **Emotional film montage** — 5–6 borrowed emotional movie/TV clips (~5s each) stitched and synced to an unreleased hook, zero captions, ends on a branded "follow @handle" TikTok-style outro card. (e.g. @diariesofateengal — Euphoria/Twilight/K-drama crying clips.)
2. **Kinetic-lyric vertical** — an aesthetic vessel (night-drive POV, etc.) with the lyric building **word-by-word, synced to the vocal**, in an elegant non-default font (yellow serif, two-tier sizing). (e.g. @sleepalatte/CJ — "I ain't / mad / at you".)
3. **Leak graphic** — a single **still image** post (TikTok photo mode): fabricated album cover + fake tracklist, distressed blackletter title, artist photo, Parental Advisory tag. Pure engagement bait. (56.1K likes on a still.) FlyerDrop's wheelhouse.

Common threads: the **audio is the star, the visual is a mood vessel** (usually borrowed); text is either **none** or a **styled/animated lyric**, never the default platform font; the **@handle is watermarked in**; register is heartbreak/longing (Rod Wave's lane).

## 3. Scope

### In scope — v1 (by June 19)
Four subsystems:

- **(1) Creation** — the three formats above.
- **(2) Multiplication** — 1 master → 10–30 non-duplicate variants (anti-suppression).
- **(4) Distribution** — war-room: account groups, posting calendar, content assignment, post-ready packages. **Human posts; no auto-post in v1.**
- **(5) Intelligence** — scoreboard: per-sound and per-variant performance, winning-sound detection, funnel recommendation.

### Out of scope — later phases
- **Phase 2 — Moment Detection** (AI clipping of interviews/lives/music videos: auto-find hooks, quotes, crowd reactions, suggest cut points). The "Opus Clip" half. Deferred — it isn't what gets the hook into ears by drop day.
- **v1.x fast-follow** — true platform-API **auto-posting** and **deep API analytics**, added per-platform where ToS-safe and approvals land.

### Explicit non-goals
- No hands-off auto-posting in v1 (it is the fastest way to get burner pages banned — see §9).
- No holding of unreleased masters on our servers (audio is user-provided per job; not vaulted).

## 4. Architecture — modules

Each module is independently understandable, testable, and communicates through a defined interface.

### 4.1 Dashboard (web app)
- **Stack:** FastAPI backend + React frontend (matches Alfred Labs). Mobile-first/responsive (hard requirement from Markion). Reuse FlyerDrop's auth + app-shell patterns.
- **Tabs:** Create · Library · Queue · Distribution · Intelligence.
- **Depends on:** all services below via the backend API.

### 4.2 Asset Library service
- **Job:** source, store, tag, and serve **mood vessels**.
- **Sources:** clean/commercial-safe APIs (Pexels, Pixabay, Mixkit, Coverr, Archive.org public-domain film) auto-pulled and tagged; a **fan-only Giphy/Tenor "cinematic GIF" tab** (clearly walled — gray-area copyrighted clips, never for official channels); **user uploads**; **Higgsfield-generated** vessels.
- **Tagging:** mood taxonomy (heartbreak, night-drive, crying, city-dusk, empty-room, rain…) for search.
- **Storage:** files on Nextcloud / local; metadata + tags in DB.
- **Interface:** `search(mood, source) -> [vessel]`, `upload(file) -> vessel`, `generate(prompt) -> vessel` (Higgsfield).

### 4.3 Creation engine
Three renderers sharing primitives (1080×1920 output, auto @handle watermark, optional branded outro card):
- **Montage renderer** — adapts `rucktalk_thesis_montage.py`: stitch N vessels, sentence/beat-boundary cuts synced to the hook, append follow-card outro.
- **Kinetic-lyric renderer** — Whisper-transcribe the hook → word-level timings → Remotion typesets the lyric building word-by-word, synced to vocal. **Caption preset system: 8–12 styles** (elegant serif, bold condensed, handwritten script, clean sans, two-tier setup/punchline layouts…) **+ one-tap "no caption."** User types lyric or accepts the Whisper suggestion; swaps font/position freely.
- **Leak-graphic renderer** — ComfyUI art + templated layouts (album cover + editable tracklist, blackletter title, advisory tag). Single-image output for photo-mode posts.
- **Interface:** `render(format, inputs) -> master_asset`.

### 4.4 Multiplication engine
- **Job:** the anti-suppression core. One master → **10–30 non-duplicate variants**.
- **Transforms:** crops, mirror/flip, speed ±, frame shifts, LUT/filter variants, caption/subtitle variants, alternate outro, micro-zoom/pan.
- **Guarantee:** each variant gets a perceptual hash; reject/regenerate any variant too close to a sibling, so they read as genuinely distinct to platform dedup.
- **Interface:** `multiply(master, count, transform_profile) -> [variant]`.

### 4.5 Distribution war-room
- **Job:** organize seeding without pulling the trigger.
- **Data:** accounts (handle, platform, group), **account groups** (burner/pilot vs. main — walled apart), posting **calendar/schedule**, **assignment** (variant → account → time).
- **Output:** a **post-ready package** per assignment (the variant file + pre-written caption + hashtags) delivered to the operator (download / Nextcloud / phone). **Human taps post.**
- **Page-integrity guard:** warn before assigning untested content to a "main" page; encourage burner-first testing.
- **Interface:** `assign(variant, account, time)`, `export_package(assignment)`.

### 4.6 Intelligence scoreboard
- **Job:** sound-testing + funneling analytics.
- **Ingest:** operator pastes the post URL back (v1); optional public-count polling (e.g. TikTok oEmbed/public metrics) where available.
- **Metrics:** views, watch-time/retention proxy, saves, shares, likes — grouped by **sound** and by **clip variant**.
- **Output:** winning-sound + winning-variant detection; **funnel recommendation** ("push the big pages onto Variant #7 / Sound B").
- **Interface:** `ingest(post_url | metrics)`, `report(by=sound|variant)`.

### 4.7 Render workers / job queue
- **Job:** run GPU-heavy work off the request path on **105's GPU** (Whisper, ComfyUI, Remotion, ffmpeg).
- **Queue:** background worker (RQ/Redis or DB-backed queue); live progress to the dashboard Queue tab.
- **Interface:** `enqueue(job) -> job_id`, `status(job_id)`.

## 5. Data flow

```
Create (format + vessel(s) + audio hook)
  -> enqueue render job (105 GPU)
  -> Creation engine renders master
  -> Multiplication engine -> N non-duplicate variants
  -> variants land in Library + Nextcloud delivery folders
  -> Distribution: assign variants to accounts on the calendar
  -> operator posts from phone (post-ready package)
  -> Intelligence ingests post links + metrics
  -> winning sound/variant -> funnel recommendation
```

## 6. Data model (initial)

`projects` · `assets` (vessels: source, mood_tags, path) · `audio_hooks` (per-job, not vaulted) · `renders` (masters) · `variations` (parent render, transform set, phash) · `accounts` · `account_groups` · `schedules` · `assignments` (variation → account → time) · `posts` (assignment → url) · `metrics` (post → views/saves/shares/retention, sound_id).

## 7. Tech stack & reuse

- **Web:** FastAPI + React (reuse Alfred Labs patterns + FlyerDrop shell/auth).
- **Render:** Remotion (kinetic lyric, montage compositing, outro card), ffmpeg (cut/transform/multiplication), Whisper (lyric transcription + word timings), ComfyUI on 105 (leak-graphic art, vessel stills).
- **AI vessel gen:** Higgsfield via its **MCP server** (30+ models; agent-native; clean integration — no scraping).
- **Stock vessels:** Pexels / Pixabay / Mixkit / Coverr / Archive.org APIs (clean) + Giphy / Tenor (fan tab).
- **Storage/delivery:** Nextcloud (`groundrushcloud.com`) — folders already provisioned at `/Content/Mainstay-RodWave/`.
- **DB:** SQLite to start (migrate to Postgres if concurrency demands).
- **Host:** server **105**, GPU-local. Proposed subdomain `forge.groundrushcloud.com`.

## 8. Integrations

| Integration | Use | Notes |
|---|---|---|
| Higgsfield (MCP) | Generate mood vessels | Has official MCP + API; Creator plan includes API access |
| Whisper (105) | Lyric transcription + word timing | Local, GPU |
| ComfyUI (105) | Leak-graphic art, vessel stills | Local, GPU; via `comfyui_gen.py` |
| Remotion | Compositing / typesetting / outro | Reuse RuckTalk rigs |
| Pexels/Pixabay/Mixkit/Coverr/Archive.org | Clean vessel library | APIs, auto-tag |
| Giphy/Tenor | Fan-tab cinematic GIFs | Walled; not for official channels |
| Nextcloud | Delivery | Folders provisioned |
| Platform public metrics | Scoreboard ingest | oEmbed/public counts where available |

## 9. Risk & honesty notes

- **Auto-posting is the page-killer.** One tool, one server/IP, auto-firing into many burner accounts is the exact footprint platform spam-detection hunts for — it would get the pages banned, defeating the whole purpose. v1 keeps humans on the "post" button; the Multiplication engine is the algorithmic defense. True auto-posting is a hardened fast-follow, per-platform, only where ToS-safe.
- **Copyright tiering.** Borrowed movie/TV clips (montage, Giphy/Tenor) live on **fan/burner accounts only**, never an official Mainstay-branded channel. Clean stock + public-domain + AI-gen are safe everywhere. The library enforces this split.
- **Unreleased material.** Audio hooks are user-provided per job and not stored/vaulted; the Nextcloud delivery link is password-gated.

## 10. Success criteria (v1)

- An operator can, on a phone, produce a finished 1080×1920 post in any of the 3 formats in **< 5 minutes**.
- One master yields **10–30 variants that pass a perceptual-distinctness check**.
- Distribution can assign variants to accounts on a calendar and export post-ready packages.
- Intelligence shows per-sound and per-variant performance and names a winner.
- Delivered to the team's Nextcloud folders. Mobile-first throughout.

## 11. Open questions

- Final name + subdomain (`forge.groundrushcloud.com` proposed).
- Caption preset count — 8–12 confirmed as a target; final list TBD with the team.
- Whisper lyric-suggestion: in v1 or fast-follow? (Currently: in v1 as an editable suggestion.)
- Which platforms first for public-metric ingest (TikTok confirmed primary; IG/FB next).

## 12. Phasing within v1 (for the implementation plan)

Build order anchored to June 19:
1. **Spine** — dashboard shell + auth + job queue + Nextcloud delivery.
2. **Creation** — kinetic-lyric first (most reuse from RuckTalk), then montage, then leak-graphic.
3. **Multiplication** — wire onto every master.
4. **Distribution** — accounts, calendar, assignment, packages.
5. **Intelligence** — ingest + scoreboard + funnel call.
6. **Higgsfield** vessel generation wired into the Library.

Phase 2 (Moment Detection) and v1.x (auto-posting, deep analytics) follow after launch.
