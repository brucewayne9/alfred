# Remotion Template Library — Design Spec

**Date:** 2026-04-17
**Owner:** Alfred (on behalf of Mike Johnson)
**Project path:** `/home/aialfred/remotion/`
**Reference corpus:** `/home/aialfred/remotion/references/` (21 reels + `PATTERN_INVENTORY.md`)

## Goal

Replace the current one-off Remotion templates with a coherent, multi-brand video template library that matches Mike's "agency-level" quality bar (reverse-engineered from 21 hand-picked Instagram Reels) and serves both:

- **Auto tier** — daily social / episode cron engines that render without human involvement
- **Hero tier** — hand-authored launch, campaign, and brand pieces

A shared design system (primitives, rigs, brand theme tokens, motion tokens, grade presets) ensures auto output looks like a scaled-down version of hero output — not a parallel second-tier brand.

## Context

- Existing Remotion project at `/home/aialfred/remotion/` has 6 one-off templates (`RuckTalkClip`, `RuckTalkShort`, `RuckTalkPromo`, `RuckTalkGrit`, `LoovaCastPromo`, `BrandPromo`) with no shared abstractions.
- Mike flagged the current output as "remedial." The reference corpus defines the target ceiling.
- Daily engines (`rucktalk_daily_social.py`, `rucktalk_episode_pipeline.py`) invoke Remotion via `npx remotion render` subprocess + props JSON. That pattern is preserved.

## Decisions Made During Brainstorm

- **Scope:** Ship RuckTalk first; architect all code to accept LoovaCast and Ground Rush Labs without a rewrite.
- **Tiers:** One shared design system, two template tiers — auto (daily cron) and hero (hand-authored). Same primitives, different prop builders.
- **Aesthetic direction — three rigs:** grit-doc (primary) + kinetic-type (secondary) + magazine-podcast (the RuckTalk editorial format Mike didn't initially name but actually relies on). An additional four specialized rigs promoted from outliers in the reference corpus: speed-ramp, beat-montage, vfx-flex, cinematic.
- **Reference corpus cleanup:** Reels 09, 10, 11 (CapCut/AE tutorials) and Reel 13 (AI illustration) dropped from the reference set. Reel 18 kept as inspiration only.
- **Architecture:** Tiered hybrid — atomic primitives + opinionated rigs + brand theme tokens + tier-specific prop builders. No schema-driven composer; hero work is not gated by JSON.

## Section 1 — Directory Structure + Brand Theme

```
/home/aialfred/remotion/
├─ src/
│   ├─ primitives/              atomic components, no brand knowledge
│   ├─ rigs/                    opinionated layouts, read brand from context
│   ├─ theme/
│   │   ├─ tokens.ts            motion tokens, grade tokens, caption timing
│   │   ├─ BrandContext.tsx     React context provider
│   │   ├─ rucktalk.ts
│   │   ├─ loovacast.ts
│   │   └─ grl.ts
│   ├─ engine/
│   │   ├─ autoProps.ts         daily engines → rig props
│   │   └─ heroProps.ts         hand-authored → rig props
│   ├─ Root.tsx                 Remotion compositions (one per rig)
│   └─ index.ts
├─ public/                       static assets (logos, LUTs, bg audio)
├─ references/                   reel corpus + PATTERN_INVENTORY.md
└─ package.json
```

**Brand theme shape** — every rig and primitive reads from `BrandContext`:

```ts
export type BrandId = "rucktalk" | "loovacast" | "grl";

export type TtsProviderId = "kokoro" | "qwen3";
export type ImageProviderId = "comfyui-local" | "comfyui-cloud" | "higgsfield";
export type VideoProviderId = "ltx-cloud" | "higgsfield";

export type BrandTheme = {
  id: BrandId;
  colors: { primary: string; accent: string; bgDark: string; fgLight: string };
  fonts: { headline: string; body: string; display?: string };
  assets: { logo: string; watermark?: string };
  meta: { url: string; handle: string; tagline: string };
  defaults: {
    grade: GradeId;
    captionStyle: CaptionStyleId;
    tts: { provider: TtsProviderId; voice: string };  // voice string is provider-specific
    image: ImageProviderId;
    video: VideoProviderId;
  };
};
```

## Section 2 — The 8 Primitives

| # | Name | What it does | Key props | Source reels |
|---|------|-------------|-----------|--------------|
| 1 | `<BigWordCaption />` | Center ALL-CAPS word, hard cut in, holds, swaps. Variants: single, stacked, scale-on-beat | `word`, `variant`, `startFrame`, `endFrame` | 03, 05, 07 |
| 2 | `<KaraokeCaptionLine />` | 2–5 word line, ALL CAPS, one word highlighted in accent, advances word-by-word | `words`, `highlightColor`, `timing[]` | 14, 15 |
| 3 | `<LowerThirdBrandBar />` | Angled black ribbon with logo tile + name + role | `name`, `role`, `logoAsset` | 15, 21 |
| 4 | `<EpisodeBadge />` | Accent-color pill: "EP N" / "EPISODE 100" | `number`, `variant: "pill" \| "ribbon"` | 14, 21 |
| 5 | `<AudioWaveformStrip />` | Audio-reactive bar visualizer tied to composition track | `barCount`, `color`, `audioSrc` | 21 |
| 6 | `<BrandLockup />` | Persistent corner mark: logo + handle | `position`, `opacity` | 14, 15, 21 |
| 7 | `<GradeOverlay />` | LUT + grain + vignette + optional letterbox, preset-selectable | `preset: GradeId` | all grit-doc |
| 8 | `<EndCard />` | Terminal card — centered logo + URL, or portrait + "MEET NAME" | `variant`, `duration` | 14, 15, 16, 21 |

**Clip utilities (supporting, not rigs):**
- `<BrollClip />` — wraps Remotion `<Video />` with optional speed-ramp curve + auto-grade.
- `useBeatMap(audio)` — hook returning beat frames for cut syncing.

## Section 3 — The 7 Rigs

Each rig is an opinionated composition with fixed layout, known timing signature, and a default grade preset. All read brand from `BrandContext`.

1. **`<GritDocRig />`** — documentary B-roll montage. Sequence of `<BrollClip />` cut on beats, sparse `<BigWordCaption />`, `teal-orange-crushed` grade, `<BrandLockup />`. Duration 15–30s. Cuts medium–fast. **Auto + Hero.**
2. **`<KineticTypeRig />`** — text drives the frame. Slow B-roll underneath, stacked `<BigWordCaption />` hits on beats, optional `<KaraokeCaptionLine />`. Duration 8–15s. Cuts slow. **Auto + Hero.**
3. **`<MagazineRig />`** ★ — RuckTalk podcast/clip format. `<EpisodeBadge />` top-left, `<BrandLockup />` top-right, full-bleed `<BrollClip />`, `<AudioWaveformStrip />`, `<KaraokeCaptionLine />` lower-third, credit footer. Duration 30–60s. Cuts ≤1. **Auto + Hero.** Replaces current `RuckTalkClip.tsx` as the daily clip engine's target. Single most load-bearing rig.
4. **`<SpeedRampRig />`** — single-hero speed-ramp. One `<BrollClip />` with ramp curve, one `<BigWordCaption />` on impact, `<GradeOverlay />`, `<EndCard />`. Duration 6–12s. **Hero only.**
5. **`<BeatMontageRig />`** — MV-style rapid cut montage. 40–100 `<BrollClip />` fragments on beat markers, optional flicker, minimal typography. Duration 10–20s. Cuts 3+/s. **Hero only.**
6. **`<VfxFlexRig />`** — transition-heavy. 2–4 clips with whip/morph/light-leak, one `<BigWordCaption />` sign-off. Duration 8–15s. **Hero only.**
7. **`<CinematicRig />`** — 2.35:1 letterbox, 24fps feel, color-graded B-roll with Foley, minimal text, `<EndCard />` fade. Duration 15–45s. **Hero only.**

Auto tier uses rigs 1–3. Hero tier uses any of the 7.

## Section 4 — Input Contract

**Auto tier** (`engine/autoProps.ts`):

```ts
type AutoBrief = {
  brand: BrandId;
  date: string;
  rotation: number;          // day-of-year % N — picks rig
  episode?: EpisodeData;     // from NextCloud queue
  clip?: ClipData;           // from clip queue
  audio?: AudioData;
  assets?: AssetRefs;
  tts?: { provider: TtsProviderId; voice: string };  // overrides brand default
};

type AutoOutput =
  | { rig: "MagazineRig";    props: MagazineRigProps }
  | { rig: "GritDocRig";     props: GritDocRigProps }
  | { rig: "KineticTypeRig"; props: KineticTypeRigProps };
```

Guardrail rule: `MagazineRig` requires a real episode clip; if missing, engine falls back to `KineticTypeRig`.

**Hero tier** (`engine/heroProps.ts`):

```ts
type HeroBrief = {
  rig: RigName;         // any of 7
  brand: BrandId;
  props: RigProps;      // typed per rig
  assets: AssetManifest;
  tts: { provider: TtsProviderId; voice: string };  // always explicit
};
```

**TTS provider pool + rotation:**

```ts
// theme/providers.ts

export const ttsRotation = {
  kokoro: ["am_adam", "am_michael", "am_eric", "af_sarah", "af_sky", "af_alloy"],  // 3M / 3F
  qwen3:  ["Barbra_Gordon", "Brenda_Walker", "JAYDEE", "Louis_Lane"],              // cloned voices
};

// Reserved: not in any rotation. Must be set explicitly on a brief (hero tier).
export const ttsReserved = {
  qwen3: ["MJ"],  // Mike's own cloned voice — only on opt-in hero briefs
};
```

Auto tier draws from `ttsRotation[provider]` by `(rotation % pool.length)`. Hero briefs can set any voice including reserved ones.

Hero briefs live as JSON at `briefs/YYYY-MM-DD-topic.json` or as direct TSX entries in `Root.tsx`.

**Data sources (auto tier):**

| Source | Provides | Used by |
|---|---|---|
| `rucktalk_clip_queue` JSON | episode clips + transcripts + timestamps | MagazineRig |
| Kokoro TTS | narrated voiceovers | KineticTypeRig, GritDocRig |
| ComfyUI (local/cloud) | bg images, LTX-2 short clips | KineticTypeRig bg, GritDocRig fallback B-roll |
| NotebookLM | 2-person podcast audio | GritDocRig variant |
| Grey Matter recall | topic ideas, quotes | caption text |

**Voice selection:**
- Permanent brand default → `theme/<brand>.ts` → `defaults.tts`
- Per-render override → set `tts` on the brief
- Helper: `npm run voices` prints Kokoro + Qwen3 voice pools (pulled live from both providers) with rotation and reserved-voice annotations

**Asset provider abstraction (Python side, `scripts/providers/`):**

```
tts/
├─ base.py            TtsProvider interface: synth(text, voice) → audio path
├─ kokoro.py          wraps localhost:8880 (OpenAI-compatible)
└─ qwen3.py           wraps localhost:7860 (/synthesize_speech/)

image/
├─ base.py            ImageProvider: gen(prompt) → image path
├─ comfyui_local.py
├─ comfyui_cloud.py
└─ higgsfield.py      stub — raises NotImplementedError until Mike has API access

video/
├─ base.py            VideoProvider: gen(prompt, duration) → video path
├─ ltx_cloud.py
└─ higgsfield.py      stub
```

Adding a provider (e.g., filling in Higgsfield) = one new file implementing the interface + flipping a default in `theme/<brand>.ts`. Rigs never change.

## Section 5 — Motion + Grade Tokens

One shared design system file. Every rig and primitive reads from it — no magic numbers in components.

```ts
// theme/tokens.ts

export const motion = {
  ease: {
    snap:    [0.8, 0, 0.1, 1],
    smooth:  [0.33, 1, 0.68, 1],
    ramp:    [0.9, 0, 0.1, 1],
    natural: [0.25, 0.1, 0.25, 1],
  },
  caption: {
    popIn: 3,
    hold: 45,
    popOut: 4,
    karaokeAdvance: 15,
  },
  cadence: {
    slow:    [0.0, 0.5],
    medium:  [0.5, 1.5],
    fast:    [1.5, 3.0],
    extreme: [3.0, 8.0],
  },
  ramp: {
    holdRip: { pre: 0.35, rip: 2.5, post: 0.4 },
    dropIn:  { pre: 0.15, rip: 2.0, post: 1.0 },
    rampOut: { pre: 1.0,  rip: 2.0, post: 0.2 },
  },
};

export const grade = {
  "teal-orange-crushed": { lut: "teal-orange.cube", grain: 0.15, vignette: 0.25, contrast: 1.1 },
  "bw-film":             { lut: "bw-film.cube",     grain: 0.25, vignette: 0.15 },
  "warm-studio":         { lut: "warm-studio.cube", grain: 0.05, vignette: 0.10 },
  "product-black":       { lut: null,               grain: 0.0,  vignette: 0.30 },
};
```

Changing one ease curve updates every rig. No hunting through templates.

## Section 6 — Migration + Render Pipeline

**Root.tsx — parameterized, not exploded.** 7 compositions total (one per rig). Brand passed via props → `<BrandProvider>` wraps the rig internally. Not one composition per rig × brand.

```ts
<Composition
  id="MagazineRig"
  component={MagazineRig}
  width={1080} height={1920} fps={30}
  durationInFrames={1800}
  defaultProps={{ brand: "rucktalk", /* rig defaults */ }}
/>
```

Render command unchanged: `npx remotion render src/index.ts MagazineRig out.mp4 --props=props.json`.

**Migration — four-step cutover, no outage:**

1. **Phase 1 (non-breaking):** Build `primitives/`, `rigs/`, `theme/`, `engine/` alongside existing `templates/`. Nothing renamed or deleted. Old cron keeps working.
2. **Phase 2:** Cut `rucktalk_episode_pipeline.py` over from `RuckTalkClip` → `MagazineRig`. Render side-by-side and compare before shipping.
3. **Phase 3:** Migrate `rucktalk_daily_social.py` from current template mix → `GritDocRig` / `KineticTypeRig` / `MagazineRig` via `autoProps()`.
4. **Phase 4:** Delete old templates and their Root.tsx entries.

Old templates live in `src/templates/_deprecated/` during phases 1-3 so the cutover is safe to revert.

**Final render entry points:**

| Caller | Command | Inputs |
|---|---|---|
| `rucktalk_episode_pipeline.py` | `npx remotion render MagazineRig` | episode props JSON |
| `rucktalk_daily_social.py` | `npx remotion render <rig>` | `autoProps()` output |
| Hero / ad-hoc | `npm run hero <rig> <brief.json>` (new helper) | `HeroBrief` JSON |

## Section 7 — Testing

1. **Prop contract** — every rig exports a Zod schema next to its TS props. `autoProps()` validates before returning; on failure, engine falls back to previous day's known-good rig and logs to Telegram.
2. **Render smoke** — `tests/smoke.sh` renders one fixture per rig at 480p, checks output exists, non-zero, duration matches. Runs pre-commit.
3. **Brand-swap** — render `MagazineRig` under `rucktalk` and `loovacast`, sample pixel at known logo position to verify brand color appears.
4. **Visual diff (opt-in)** — `tests/golden/<rig>/<brand>/frame-XX.png` with `npm run test:visual`. ~2% threshold. `npm run test:visual:approve` regenerates goldens on intentional design changes.
5. **Cutover gate (manual)** — before Phase 2 and 3 cutovers, render same source through old and new → Telegram to Mike → eyeball approval required before flipping cron.

**Not in scope:**
- No component-level React unit tests.
- No Remotion internal mocking.
- No CI — pre-commit on server 105 only.

## Out of Scope

- Real-world B-roll footage generation (the template rigs assume curated footage is supplied via asset pipeline).
- Beat detection of music from arbitrary tracks — auto tier uses Kokoro/NotebookLM audio where beat structure is known; hero tier gets hand-authored `beatMap[]`.
- Text-behind-subject segmentation effects (hero only, flagged as high-cost).
- Render queue orchestration — stays in the existing daily engines.

## References

- Pattern inventory: `/home/aialfred/remotion/references/PATTERN_INVENTORY.md`
- Reference reels: `/home/aialfred/remotion/references/reels/` (21 mp4 + info.json)
- Extracted keyframes: `/home/aialfred/remotion/references/frames/<reel-id>/frame-01..06.jpg`
- Memory of reference list: `~/.claude/projects/-home-aialfred-alfred/memory/project_remotion_references.md`
