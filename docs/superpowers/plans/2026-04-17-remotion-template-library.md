# Remotion Template Library — Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-brand Remotion template library (8 primitives + 7 rigs + brand theme + provider abstraction) that lives alongside existing templates, renders all 7 rigs end-to-end from fixtures, and is ready to replace the existing daily and episode pipelines in a follow-up plan.

**Architecture:** Tiered hybrid. Atomic primitives (brand-agnostic React components reading from `BrandContext`) compose into opinionated rigs (full-frame layouts). Brand tokens (color, fonts, assets, TTS provider+voice, grade preset) live in `src/theme/`. Shared motion/grade tokens in `src/theme/tokens.ts`. Python-side provider abstraction (`scripts/providers/`) isolates Kokoro/Qwen3 TTS and ComfyUI/Higgsfield image+video so swaps require no Remotion code changes.

**Tech Stack:** Remotion 4.0.438, React 19, TypeScript 6, Zod 3.x (new), Node 22.22.0, Python 3.11 (provider side), ffprobe/ffmpeg for smoke tests.

**Source spec:** `docs/superpowers/specs/2026-04-17-remotion-template-library-design.md`

**Scope boundaries:** Phase 1 ONLY — build alongside existing templates, do NOT modify `rucktalk_episode_pipeline.py` or `rucktalk_daily_social.py`, do NOT delete old templates. Migration is a separate plan.

**Repo:** The Remotion project lives at `/home/aialfred/remotion/` (branch `master`, separate repo from `/home/aialfred/alfred/`). Python providers live at `/home/aialfred/alfred/scripts/providers/` (branch `main`). Commits go to their respective repos.

---

## File Structure

**`/home/aialfred/remotion/` (Remotion repo):**

Create:
- `src/theme/tokens.ts` — motion + grade tokens
- `src/theme/providers.ts` — provider IDs + voice rotation pools
- `src/theme/BrandContext.tsx` — React context + provider
- `src/theme/rucktalk.ts`, `loovacast.ts`, `grl.ts` — brand themes
- `src/theme/index.ts` — barrel export + types
- `src/primitives/{BigWordCaption,KaraokeCaptionLine,LowerThirdBrandBar,EpisodeBadge,AudioWaveformStrip,BrandLockup,GradeOverlay,EndCard,BrollClip}.tsx`
- `src/primitives/useBeatMap.ts`
- `src/primitives/index.ts`
- `src/rigs/{GritDocRig,KineticTypeRig,MagazineRig,SpeedRampRig,BeatMontageRig,VfxFlexRig,CinematicRig}.tsx`
- `src/rigs/index.ts`
- `src/engine/schemas.ts` — Zod schemas
- `src/engine/autoProps.ts` — daily-engine prop builder
- `src/engine/heroProps.ts` — hand-authored prop loader
- `tests/fixtures/{grit-doc,kinetic-type,magazine,speed-ramp,beat-montage,vfx-flex,cinematic}.json`
- `tests/smoke.sh` — render smoke per rig
- `tests/brand-swap.sh` — brand-swap sampler
- `tests/golden/.gitkeep` — opt-in visual diff goldens (empty)
- `scripts/voices.mjs` — CLI lists Kokoro + Qwen3 voices
- `scripts/hero.mjs` — CLI renders a hero brief
- `public/luts/.gitkeep` — LUT CUBE files go here (placeholder LUTs shipped with plan)

Modify:
- `src/Root.tsx` — add 7 new compositions (keep existing 6)
- `package.json` — add `zod`, add `npm run voices`, `npm run hero`, `npm run smoke`, `npm run test:brand-swap`

Move (deprecate, do not delete):
- `src/templates/` → `src/templates/_deprecated/` (all 6 existing templates)
- `src/components/` → keep; individual components retired case-by-case in later tasks

**`/home/aialfred/alfred/` (Alfred repo, Python providers):**

Create:
- `scripts/providers/__init__.py`
- `scripts/providers/tts/{__init__,base,kokoro,qwen3}.py`
- `scripts/providers/image/{__init__,base,comfyui_local,comfyui_cloud,higgsfield}.py`
- `scripts/providers/video/{__init__,base,ltx_cloud,higgsfield}.py`
- `tests/providers/{test_tts_base,test_tts_kokoro,test_tts_qwen3,test_image_base,test_video_base}.py`
- `scripts/voices.py` — Python helper used by provider tests

---

# PHASE A — Scaffolding & Theme (Tasks 1-6)

---

### Task 1: Scaffold directories and deprecate existing templates

**Files:**
- Create: `/home/aialfred/remotion/src/{theme,primitives,rigs,engine,templates/_deprecated}/`
- Create: `/home/aialfred/remotion/{tests/fixtures,tests/golden,scripts,public/luts}/`
- Move: `/home/aialfred/remotion/src/templates/*.tsx` → `/home/aialfred/remotion/src/templates/_deprecated/`
- Modify: `/home/aialfred/remotion/src/Root.tsx` — update import paths

- [ ] **Step 1: Create directories**

```bash
cd /home/aialfred/remotion
mkdir -p src/theme src/primitives src/rigs src/engine
mkdir -p src/templates/_deprecated
mkdir -p tests/fixtures tests/golden scripts public/luts
touch tests/golden/.gitkeep public/luts/.gitkeep
```

- [ ] **Step 2: Move existing templates to _deprecated**

```bash
cd /home/aialfred/remotion
git mv src/templates/BrandPromo.tsx         src/templates/_deprecated/
git mv src/templates/LoovaCastPromo.tsx     src/templates/_deprecated/
git mv src/templates/RuckTalkClip.tsx       src/templates/_deprecated/
git mv src/templates/RuckTalkGrit.tsx       src/templates/_deprecated/
git mv src/templates/RuckTalkPromo.tsx      src/templates/_deprecated/
git mv src/templates/RuckTalkShort.tsx      src/templates/_deprecated/
```

- [ ] **Step 3: Fix Root.tsx imports to point at _deprecated paths**

Replace each existing import line in `src/Root.tsx`:

```ts
// Before
import { LoovaCastPromo } from "./templates/LoovaCastPromo";
// After
import { LoovaCastPromo } from "./templates/_deprecated/LoovaCastPromo";
```

Apply the same pattern to `RuckTalkPromo`, `RuckTalkShort`, `RuckTalkClip`, `RuckTalkGrit`, `BrandPromo`. Do NOT remove any `<Composition/>` blocks — the old templates must still render so the existing cron keeps working.

- [ ] **Step 4: Verify Remotion still bundles**

```bash
cd /home/aialfred/remotion
npx remotion lint 2>&1 | tail -5 || true
npx tsc --noEmit 2>&1 | tail -20
```

Expected: no TypeScript errors from the moved files (clean compile). Any errors from the new empty directories are expected and OK — they're not imported yet.

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/remotion
git add -A
git commit -m "chore(remotion): scaffold new library dirs; deprecate old templates

Move src/templates/*.tsx to src/templates/_deprecated/ so they keep
rendering for existing cron while the new library is being built.
Create empty src/{theme,primitives,rigs,engine}/ and tests/ dirs."
```

---

### Task 2: Add zod dependency and new npm scripts

**Files:**
- Modify: `/home/aialfred/remotion/package.json`

- [ ] **Step 1: Install zod**

```bash
cd /home/aialfred/remotion
npm install zod@3
```

Expected: `zod` added to dependencies, `package-lock.json` updated.

- [ ] **Step 2: Add npm scripts**

Edit `/home/aialfred/remotion/package.json` — replace the entire `"scripts"` block with:

```json
"scripts": {
  "studio":         "remotion studio src/index.ts",
  "render":         "remotion render src/index.ts",
  "still":          "remotion still src/index.ts",
  "voices":         "node scripts/voices.mjs",
  "hero":           "node scripts/hero.mjs",
  "smoke":          "bash tests/smoke.sh",
  "test:brand-swap":"bash tests/brand-swap.sh",
  "test:visual":    "node scripts/visual-diff.mjs",
  "test:visual:approve": "node scripts/visual-diff.mjs --approve",
  "typecheck":      "tsc --noEmit",
  "test":           "echo \"Error: no test specified\" && exit 1"
}
```

- [ ] **Step 3: Verify**

```bash
cd /home/aialfred/remotion
npm run typecheck 2>&1 | tail -5
node -e "require('zod')"
```

Expected: typecheck clean, zod requires without error.

- [ ] **Step 4: Commit**

```bash
cd /home/aialfred/remotion
git add package.json package-lock.json
git commit -m "deps(remotion): add zod + scripts for voices, hero, smoke, typecheck"
```

---

### Task 3: Create motion + grade tokens

**Files:**
- Create: `/home/aialfred/remotion/src/theme/tokens.ts`

- [ ] **Step 1: Write the file**

```ts
// src/theme/tokens.ts
// Single source of truth for all motion timing, easing, and grade presets.
// Rigs and primitives import from here instead of hardcoding values.

export type EaseTuple = readonly [number, number, number, number];

export const motion = {
  ease: {
    snap:    [0.8, 0, 0.1, 1]       as EaseTuple,
    smooth:  [0.33, 1, 0.68, 1]     as EaseTuple,
    ramp:    [0.9, 0, 0.1, 1]       as EaseTuple,
    natural: [0.25, 0.1, 0.25, 1]   as EaseTuple,
  },
  caption: {
    popIn: 3,
    hold: 45,
    popOut: 4,
    karaokeAdvance: 15,
  },
  cadence: {
    slow:    [0.0, 0.5] as const,
    medium:  [0.5, 1.5] as const,
    fast:    [1.5, 3.0] as const,
    extreme: [3.0, 8.0] as const,
  },
  ramp: {
    holdRip: { pre: 0.35, rip: 2.5, post: 0.4 },
    dropIn:  { pre: 0.15, rip: 2.0, post: 1.0 },
    rampOut: { pre: 1.0,  rip: 2.0, post: 0.2 },
  },
} as const;

export type GradeId =
  | "teal-orange-crushed"
  | "bw-film"
  | "warm-studio"
  | "product-black";

export type GradeSpec = {
  lut: string | null;    // filename under public/luts/, or null if no LUT
  grain: number;         // 0.0-1.0
  vignette: number;      // 0.0-1.0
  contrast?: number;     // 1.0 = neutral
};

export const grade: Record<GradeId, GradeSpec> = {
  "teal-orange-crushed": { lut: "teal-orange.cube", grain: 0.15, vignette: 0.25, contrast: 1.1 },
  "bw-film":             { lut: "bw-film.cube",     grain: 0.25, vignette: 0.15 },
  "warm-studio":         { lut: "warm-studio.cube", grain: 0.05, vignette: 0.10 },
  "product-black":       { lut: null,               grain: 0.0,  vignette: 0.30 },
};
```

- [ ] **Step 2: Typecheck**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
cd /home/aialfred/remotion
git add src/theme/tokens.ts
git commit -m "feat(theme): motion + grade tokens

Shared design tokens — ease curves, caption timing, cut-cadence
buckets, speed-ramp profiles, and grade presets. Single source of
truth for every rig and primitive."
```

---

### Task 4: Create provider types and voice rotation pools

**Files:**
- Create: `/home/aialfred/remotion/src/theme/providers.ts`

- [ ] **Step 1: Write the file**

```ts
// src/theme/providers.ts

export type TtsProviderId = "kokoro" | "qwen3";
export type ImageProviderId = "comfyui-local" | "comfyui-cloud" | "higgsfield";
export type VideoProviderId = "ltx-cloud" | "higgsfield";

export type TtsVoice = {
  provider: TtsProviderId;
  voice: string;  // provider-specific id
};

// Default rotation pools. Auto tier picks via (rotation % pool.length).
export const ttsRotation: Record<TtsProviderId, string[]> = {
  kokoro: ["am_adam", "am_michael", "am_eric", "af_sarah", "af_sky", "af_alloy"],
  qwen3:  ["Barbra_Gordon", "Brenda_Walker", "JAYDEE", "Louis_Lane"],
};

// Reserved voices are never drawn by auto rotation. Hero briefs can request them.
export const ttsReserved: Partial<Record<TtsProviderId, string[]>> = {
  qwen3: ["MJ"],   // Mike's own cloned voice
};

// Helper: pick a rotation voice by index (wrap-around).
export function pickRotationVoice(provider: TtsProviderId, index: number): TtsVoice {
  const pool = ttsRotation[provider];
  if (pool.length === 0) throw new Error(`No rotation voices for ${provider}`);
  return { provider, voice: pool[Math.abs(index) % pool.length] };
}

// Helper: validate a voice is permitted on a given tier.
export function isRotationVoice(v: TtsVoice): boolean {
  return ttsRotation[v.provider]?.includes(v.voice) ?? false;
}
export function isReservedVoice(v: TtsVoice): boolean {
  return ttsReserved[v.provider]?.includes(v.voice) ?? false;
}
```

- [ ] **Step 2: Typecheck**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
cd /home/aialfred/remotion
git add src/theme/providers.ts
git commit -m "feat(theme): TTS/image/video provider types + voice rotation pools"
```

---

### Task 5: Create BrandContext + BrandProvider

**Files:**
- Create: `/home/aialfred/remotion/src/theme/BrandContext.tsx`

- [ ] **Step 1: Write the file**

```tsx
// src/theme/BrandContext.tsx
import React, { createContext, useContext } from "react";
import type { GradeId } from "./tokens";
import type { TtsVoice, ImageProviderId, VideoProviderId } from "./providers";

export type BrandId = "rucktalk" | "loovacast" | "grl";
export type CaptionStyleId = "karaoke-yellow" | "hero-white" | "plain";

export type BrandTheme = {
  id: BrandId;
  colors: {
    primary: string;
    accent: string;
    bgDark: string;
    fgLight: string;
  };
  fonts: {
    headline: string;
    body: string;
    display?: string;
  };
  assets: {
    logo: string;        // relative path under public/
    watermark?: string;
  };
  meta: {
    url: string;
    handle: string;
    tagline: string;
  };
  defaults: {
    grade: GradeId;
    captionStyle: CaptionStyleId;
    tts: TtsVoice;
    image: ImageProviderId;
    video: VideoProviderId;
  };
};

const BrandContext = createContext<BrandTheme | null>(null);

export const BrandProvider: React.FC<{
  brand: BrandTheme;
  children: React.ReactNode;
}> = ({ brand, children }) => (
  <BrandContext.Provider value={brand}>{children}</BrandContext.Provider>
);

export function useBrand(): BrandTheme {
  const ctx = useContext(BrandContext);
  if (!ctx) {
    throw new Error(
      "useBrand() called outside <BrandProvider>. Every rig must be wrapped."
    );
  }
  return ctx;
}
```

- [ ] **Step 2: Typecheck**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
cd /home/aialfred/remotion
git add src/theme/BrandContext.tsx
git commit -m "feat(theme): BrandContext + BrandProvider + useBrand hook"
```

---

### Task 6: Create the three brand theme files + barrel export

**Files:**
- Create: `/home/aialfred/remotion/src/theme/rucktalk.ts`
- Create: `/home/aialfred/remotion/src/theme/loovacast.ts`
- Create: `/home/aialfred/remotion/src/theme/grl.ts`
- Create: `/home/aialfred/remotion/src/theme/index.ts`

- [ ] **Step 1: Write rucktalk.ts**

```ts
// src/theme/rucktalk.ts
import type { BrandTheme } from "./BrandContext";

export const rucktalk: BrandTheme = {
  id: "rucktalk",
  colors: {
    primary: "#f97316",   // orange
    accent:  "#dc2626",   // crimson accent (caption highlight, waveform)
    bgDark:  "#0a0a0a",
    fgLight: "#ffffff",
  },
  fonts: {
    headline: "Montserrat",
    body: "Inter",
  },
  assets: {
    logo: "rucktalk-logo.png",
  },
  meta: {
    url: "rucktalk.com",
    handle: "@rucktalk",
    tagline: "Tactical Ideas for Real Life",
  },
  defaults: {
    grade: "teal-orange-crushed",
    captionStyle: "karaoke-yellow",
    tts: { provider: "kokoro", voice: "am_adam" },
    image: "comfyui-cloud",
    video: "ltx-cloud",
  },
};
```

- [ ] **Step 2: Write loovacast.ts**

```ts
// src/theme/loovacast.ts
import type { BrandTheme } from "./BrandContext";

export const loovacast: BrandTheme = {
  id: "loovacast",
  colors: {
    primary: "#8b5cf6",   // purple
    accent:  "#c4b5fd",   // lighter purple
    bgDark:  "#0a0a0a",
    fgLight: "#ffffff",
  },
  fonts: {
    headline: "Montserrat",
    body: "Inter",
  },
  assets: {
    logo: "loovacast-logo.png",   // drop into public/ when available
  },
  meta: {
    url: "loovacast.com",
    handle: "@loovacast",
    tagline: "Your Voice. Your Station. Your Way.",
  },
  defaults: {
    grade: "warm-studio",
    captionStyle: "hero-white",
    tts: { provider: "kokoro", voice: "af_sarah" },
    image: "comfyui-cloud",
    video: "ltx-cloud",
  },
};
```

- [ ] **Step 3: Write grl.ts**

```ts
// src/theme/grl.ts
import type { BrandTheme } from "./BrandContext";

export const grl: BrandTheme = {
  id: "grl",
  colors: {
    primary: "#E8650A",   // tactical orange
    accent:  "#f97316",
    bgDark:  "#0a0a0a",
    fgLight: "#ffffff",
  },
  fonts: {
    headline: "Montserrat",
    body: "Inter",
  },
  assets: {
    logo: "grl-logo.png",         // drop into public/ when available
  },
  meta: {
    url: "groundrushlabs.com",
    handle: "@groundrushlabs",
    tagline: "We build things that work.",
  },
  defaults: {
    grade: "teal-orange-crushed",
    captionStyle: "hero-white",
    tts: { provider: "kokoro", voice: "am_michael" },
    image: "comfyui-cloud",
    video: "ltx-cloud",
  },
};
```

- [ ] **Step 4: Write barrel index.ts**

```ts
// src/theme/index.ts
export * from "./BrandContext";
export * from "./tokens";
export * from "./providers";
export { rucktalk } from "./rucktalk";
export { loovacast } from "./loovacast";
export { grl } from "./grl";

import type { BrandId, BrandTheme } from "./BrandContext";
import { rucktalk } from "./rucktalk";
import { loovacast } from "./loovacast";
import { grl } from "./grl";

export const brands: Record<BrandId, BrandTheme> = {
  rucktalk,
  loovacast,
  grl,
};

export function getBrand(id: BrandId): BrandTheme {
  const b = brands[id];
  if (!b) throw new Error(`Unknown brand: ${id}`);
  return b;
}
```

- [ ] **Step 5: Typecheck**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
cd /home/aialfred/remotion
git add src/theme/
git commit -m "feat(theme): rucktalk + loovacast + grl brand themes, barrel export"
```

---

# PHASE B — Python Provider Abstraction (Tasks 7-12)

---

### Task 7: TTS provider base class + tests

**Files:**
- Create: `/home/aialfred/alfred/scripts/providers/__init__.py`
- Create: `/home/aialfred/alfred/scripts/providers/tts/__init__.py`
- Create: `/home/aialfred/alfred/scripts/providers/tts/base.py`
- Create: `/home/aialfred/alfred/tests/providers/__init__.py`
- Create: `/home/aialfred/alfred/tests/providers/test_tts_base.py`

- [ ] **Step 1: Write the failing test**

`/home/aialfred/alfred/tests/providers/test_tts_base.py`:

```python
import pytest
from pathlib import Path
from scripts.providers.tts.base import TtsProvider, TtsRequest, TtsResult

class FakeTts(TtsProvider):
    name = "fake"
    def list_voices(self): return ["v1", "v2"]
    def synth(self, req: TtsRequest) -> TtsResult:
        return TtsResult(audio_path=Path("/tmp/fake.wav"), duration_s=1.0, voice=req.voice)

def test_tts_provider_contract():
    p = FakeTts()
    assert p.name == "fake"
    assert p.list_voices() == ["v1", "v2"]
    result = p.synth(TtsRequest(text="hello", voice="v1"))
    assert result.voice == "v1"
    assert result.audio_path == Path("/tmp/fake.wav")

def test_tts_provider_is_abstract():
    with pytest.raises(TypeError):
        TtsProvider()  # abstract, cannot instantiate
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/aialfred/alfred
pytest tests/providers/test_tts_base.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'scripts.providers.tts'`.

- [ ] **Step 3: Create package __init__ files**

```bash
cd /home/aialfred/alfred
mkdir -p scripts/providers/tts tests/providers
touch scripts/providers/__init__.py scripts/providers/tts/__init__.py tests/providers/__init__.py
```

- [ ] **Step 4: Implement base.py**

`/home/aialfred/alfred/scripts/providers/tts/base.py`:

```python
"""TTS provider abstract interface.

Every TTS backend (Kokoro, Qwen3, future providers) implements this.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TtsRequest:
    text: str
    voice: str
    speed: float = 1.0
    output_path: Path | None = None  # None => provider picks a temp path


@dataclass(frozen=True)
class TtsResult:
    audio_path: Path
    duration_s: float
    voice: str


class TtsProvider(ABC):
    """Abstract TTS provider. Subclasses MUST implement name, list_voices, synth."""

    name: str  # short id: "kokoro", "qwen3"

    @abstractmethod
    def list_voices(self) -> Sequence[str]:
        """Return the voices this provider exposes."""

    @abstractmethod
    def synth(self, req: TtsRequest) -> TtsResult:
        """Synthesize speech. Return path to a .wav file on disk."""
```

- [ ] **Step 5: Run test to verify pass**

```bash
cd /home/aialfred/alfred
pytest tests/providers/test_tts_base.py -v 2>&1 | tail -10
```

Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/aialfred/alfred
git add scripts/providers/__init__.py scripts/providers/tts/__init__.py scripts/providers/tts/base.py tests/providers/__init__.py tests/providers/test_tts_base.py
git commit -m "feat(providers): TTS provider abstract base + contract tests"
```

---

### Task 8: Kokoro TTS provider implementation

**Files:**
- Create: `/home/aialfred/alfred/scripts/providers/tts/kokoro.py`
- Create: `/home/aialfred/alfred/tests/providers/test_tts_kokoro.py`

- [ ] **Step 1: Write the failing test**

`/home/aialfred/alfred/tests/providers/test_tts_kokoro.py`:

```python
import pytest
from pathlib import Path
from scripts.providers.tts.kokoro import KokoroTts
from scripts.providers.tts.base import TtsRequest

@pytest.mark.integration
def test_kokoro_lists_voices():
    """Integration: requires Kokoro service on localhost:8880."""
    p = KokoroTts()
    voices = p.list_voices()
    assert isinstance(voices, list)
    assert "am_adam" in voices
    assert "af_sarah" in voices

@pytest.mark.integration
def test_kokoro_synthesizes_short_clip(tmp_path: Path):
    p = KokoroTts()
    out = tmp_path / "test.wav"
    result = p.synth(TtsRequest(text="Testing one two three.", voice="am_adam", output_path=out))
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 1000
    assert result.voice == "am_adam"
    assert result.duration_s > 0.5
```

- [ ] **Step 2: Run tests to verify fail**

```bash
cd /home/aialfred/alfred
pytest tests/providers/test_tts_kokoro.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'scripts.providers.tts.kokoro'`.

- [ ] **Step 3: Implement kokoro.py**

`/home/aialfred/alfred/scripts/providers/tts/kokoro.py`:

```python
"""Kokoro TTS provider. Wraps the local OpenAI-compatible service at :8880."""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Sequence

import requests

from .base import TtsProvider, TtsRequest, TtsResult

KOKORO_URL = "http://localhost:8880"


class KokoroTts(TtsProvider):
    name = "kokoro"

    def __init__(self, base_url: str = KOKORO_URL, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_voices(self) -> Sequence[str]:
        r = requests.get(f"{self.base_url}/v1/audio/voices", timeout=5)
        r.raise_for_status()
        return r.json()["voices"]

    def synth(self, req: TtsRequest) -> TtsResult:
        out_path = req.output_path or Path(tempfile.mkstemp(suffix=".wav")[1])
        r = requests.post(
            f"{self.base_url}/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": req.text,
                "voice": req.voice,
                "response_format": "wav",
                "speed": req.speed,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        out_path.write_bytes(r.content)

        # Read duration from wav header
        with wave.open(str(out_path), "rb") as w:
            duration = w.getnframes() / float(w.getframerate())

        return TtsResult(audio_path=out_path, duration_s=duration, voice=req.voice)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /home/aialfred/alfred
pytest tests/providers/test_tts_kokoro.py -v 2>&1 | tail -10
```

Expected: both tests PASS (integration tests hit live Kokoro at :8880).

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add scripts/providers/tts/kokoro.py tests/providers/test_tts_kokoro.py
git commit -m "feat(providers): Kokoro TTS provider + live integration tests"
```

---

### Task 9: Qwen3 TTS provider implementation

**Files:**
- Create: `/home/aialfred/alfred/scripts/providers/tts/qwen3.py`
- Create: `/home/aialfred/alfred/tests/providers/test_tts_qwen3.py`

- [ ] **Step 1: Write the failing test**

`/home/aialfred/alfred/tests/providers/test_tts_qwen3.py`:

```python
import pytest
from pathlib import Path
from scripts.providers.tts.qwen3 import Qwen3Tts
from scripts.providers.tts.base import TtsRequest

@pytest.mark.integration
def test_qwen3_lists_voices():
    """Integration: requires Qwen3-TTS service on localhost:7860."""
    p = Qwen3Tts()
    voices = p.list_voices()
    assert isinstance(voices, list)
    # Expect both cloned and designed voices merged:
    assert "MJ" in voices
    assert "Lois_Lane" in voices

@pytest.mark.integration
def test_qwen3_synthesizes_short_clip(tmp_path: Path):
    p = Qwen3Tts()
    out = tmp_path / "test.wav"
    result = p.synth(TtsRequest(text="Testing one two three.", voice="JAYDEE", output_path=out))
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 1000
    assert result.voice == "JAYDEE"
```

- [ ] **Step 2: Run tests to verify fail**

```bash
cd /home/aialfred/alfred
pytest tests/providers/test_tts_qwen3.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement qwen3.py**

`/home/aialfred/alfred/scripts/providers/tts/qwen3.py`:

```python
"""Qwen3 TTS provider. Wraps the FastAPI service at :7860.

Merges cloned voices (/cloned_voices) and designed voices (/voice_design/voices)
into a single voice pool. Synthesis goes through /synthesize_speech/ for known
voices or /voice_design/synthesize for designed voices.
"""
from __future__ import annotations

import tempfile
import wave
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import requests

from .base import TtsProvider, TtsRequest, TtsResult

QWEN3_URL = "http://localhost:7860"


class Qwen3Tts(TtsProvider):
    name = "qwen3"

    def __init__(self, base_url: str = QWEN3_URL, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _fetch_voice_catalogs(self) -> tuple[list[str], list[str]]:
        """Return (cloned_names, designed_names)."""
        cloned = requests.get(f"{self.base_url}/cloned_voices", timeout=5).json()
        designed = requests.get(f"{self.base_url}/voice_design/voices", timeout=5).json()
        cloned_names = [v["name"] for v in cloned.get("cloned_voices", [])]
        designed_names = [v["name"] for v in designed.get("voices", [])]
        return cloned_names, designed_names

    def list_voices(self) -> Sequence[str]:
        cloned, designed = self._fetch_voice_catalogs()
        return cloned + designed

    def synth(self, req: TtsRequest) -> TtsResult:
        out_path = req.output_path or Path(tempfile.mkstemp(suffix=".wav")[1])
        cloned, designed = self._fetch_voice_catalogs()

        if req.voice in cloned:
            endpoint = "/synthesize_speech/"
        elif req.voice in designed:
            endpoint = "/voice_design/synthesize"
        else:
            raise ValueError(f"Unknown Qwen3 voice: {req.voice}")

        r = requests.get(
            f"{self.base_url}{endpoint}",
            params={"text": req.text, "voice": req.voice, "speed": req.speed},
            timeout=self.timeout,
        )
        r.raise_for_status()
        out_path.write_bytes(r.content)

        with wave.open(str(out_path), "rb") as w:
            duration = w.getnframes() / float(w.getframerate())

        return TtsResult(audio_path=out_path, duration_s=duration, voice=req.voice)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /home/aialfred/alfred
pytest tests/providers/test_tts_qwen3.py -v 2>&1 | tail -10
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add scripts/providers/tts/qwen3.py tests/providers/test_tts_qwen3.py
git commit -m "feat(providers): Qwen3 TTS provider (cloned + designed voice pools)"
```

---

### Task 10: Image provider base + ComfyUI adapters + Higgsfield stub

**Files:**
- Create: `/home/aialfred/alfred/scripts/providers/image/{__init__,base,comfyui_local,comfyui_cloud,higgsfield}.py`
- Create: `/home/aialfred/alfred/tests/providers/test_image_base.py`

- [ ] **Step 1: Write the failing test**

`/home/aialfred/alfred/tests/providers/test_image_base.py`:

```python
import pytest
from pathlib import Path
from scripts.providers.image.base import ImageProvider, ImageRequest, ImageResult
from scripts.providers.image.higgsfield import HiggsfieldImage

class FakeImage(ImageProvider):
    name = "fake"
    def gen(self, req: ImageRequest) -> ImageResult:
        return ImageResult(image_path=Path("/tmp/fake.png"), width=req.width, height=req.height)

def test_image_provider_contract():
    p = FakeImage()
    res = p.gen(ImageRequest(prompt="a test", width=1024, height=1024))
    assert res.image_path == Path("/tmp/fake.png")
    assert res.width == 1024

def test_higgsfield_stub_raises():
    p = HiggsfieldImage()
    with pytest.raises(NotImplementedError):
        p.gen(ImageRequest(prompt="x"))
```

- [ ] **Step 2: Run tests to verify fail**

```bash
cd /home/aialfred/alfred
pytest tests/providers/test_image_base.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create package + base.py**

```bash
cd /home/aialfred/alfred
mkdir -p scripts/providers/image
touch scripts/providers/image/__init__.py
```

`/home/aialfred/alfred/scripts/providers/image/base.py`:

```python
"""Image provider abstract interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    width: int = 1080
    height: int = 1920
    seed: int | None = None
    output_path: Path | None = None


@dataclass(frozen=True)
class ImageResult:
    image_path: Path
    width: int
    height: int


class ImageProvider(ABC):
    name: str

    @abstractmethod
    def gen(self, req: ImageRequest) -> ImageResult: ...
```

- [ ] **Step 4: Implement ComfyUI local + cloud adapters**

`/home/aialfred/alfred/scripts/providers/image/comfyui_local.py`:

```python
"""ComfyUI local GPU provider. Delegates to existing comfyui_gen.py script."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .base import ImageProvider, ImageRequest, ImageResult

COMFYUI_GEN_SCRIPT = Path("/home/aialfred/alfred/scripts/comfyui_gen.py")


class ComfyUiLocal(ImageProvider):
    name = "comfyui-local"

    def gen(self, req: ImageRequest) -> ImageResult:
        out = req.output_path or Path(tempfile.mkstemp(suffix=".png")[1])
        cmd = [
            "python3", str(COMFYUI_GEN_SCRIPT),
            "generate", req.prompt,
            "--output", str(out),
            "--width", str(req.width),
            "--height", str(req.height),
        ]
        if req.seed is not None:
            cmd.extend(["--seed", str(req.seed)])
        subprocess.run(cmd, check=True, timeout=300)
        return ImageResult(image_path=out, width=req.width, height=req.height)
```

`/home/aialfred/alfred/scripts/providers/image/comfyui_cloud.py`:

```python
"""ComfyUI Cloud provider. Delegates to comfyui_gen.py --cloud flag."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .base import ImageProvider, ImageRequest, ImageResult

COMFYUI_GEN_SCRIPT = Path("/home/aialfred/alfred/scripts/comfyui_gen.py")


class ComfyUiCloud(ImageProvider):
    name = "comfyui-cloud"

    def gen(self, req: ImageRequest) -> ImageResult:
        out = req.output_path or Path(tempfile.mkstemp(suffix=".png")[1])
        cmd = [
            "python3", str(COMFYUI_GEN_SCRIPT),
            "generate", req.prompt,
            "--cloud",
            "--output", str(out),
            "--width", str(req.width),
            "--height", str(req.height),
        ]
        if req.seed is not None:
            cmd.extend(["--seed", str(req.seed)])
        subprocess.run(cmd, check=True, timeout=300)
        return ImageResult(image_path=out, width=req.width, height=req.height)
```

`/home/aialfred/alfred/scripts/providers/image/higgsfield.py`:

```python
"""Higgsfield image provider — stub until API access arrives."""
from __future__ import annotations

from .base import ImageProvider, ImageRequest, ImageResult


class HiggsfieldImage(ImageProvider):
    name = "higgsfield"

    def gen(self, req: ImageRequest) -> ImageResult:
        raise NotImplementedError(
            "Higgsfield image provider not implemented yet. "
            "Fill in when API credentials are available."
        )
```

- [ ] **Step 5: Run tests to verify pass**

```bash
cd /home/aialfred/alfred
pytest tests/providers/test_image_base.py -v 2>&1 | tail -10
```

Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/aialfred/alfred
git add scripts/providers/image/ tests/providers/test_image_base.py
git commit -m "feat(providers): image provider base + ComfyUI local/cloud + Higgsfield stub"
```

---

### Task 11: Video provider base + LTX Cloud + Higgsfield stub

**Files:**
- Create: `/home/aialfred/alfred/scripts/providers/video/{__init__,base,ltx_cloud,higgsfield}.py`
- Create: `/home/aialfred/alfred/tests/providers/test_video_base.py`

- [ ] **Step 1: Write the failing test**

`/home/aialfred/alfred/tests/providers/test_video_base.py`:

```python
import pytest
from pathlib import Path
from scripts.providers.video.base import VideoProvider, VideoRequest, VideoResult
from scripts.providers.video.higgsfield import HiggsfieldVideo

class FakeVideo(VideoProvider):
    name = "fake"
    def gen(self, req: VideoRequest) -> VideoResult:
        return VideoResult(video_path=Path("/tmp/fake.mp4"), duration_s=req.duration_s)

def test_video_provider_contract():
    p = FakeVideo()
    res = p.gen(VideoRequest(prompt="test", duration_s=5.0))
    assert res.video_path == Path("/tmp/fake.mp4")
    assert res.duration_s == 5.0

def test_higgsfield_video_stub_raises():
    with pytest.raises(NotImplementedError):
        HiggsfieldVideo().gen(VideoRequest(prompt="x", duration_s=3))
```

- [ ] **Step 2: Run tests to verify fail**

```bash
cd /home/aialfred/alfred
pytest tests/providers/test_video_base.py -v 2>&1 | tail -10
```

- [ ] **Step 3: Create package + base.py**

```bash
cd /home/aialfred/alfred
mkdir -p scripts/providers/video
touch scripts/providers/video/__init__.py
```

`/home/aialfred/alfred/scripts/providers/video/base.py`:

```python
"""Video provider abstract interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoRequest:
    prompt: str
    duration_s: float
    width: int = 1080
    height: int = 1920
    seed: int | None = None
    output_path: Path | None = None


@dataclass(frozen=True)
class VideoResult:
    video_path: Path
    duration_s: float


class VideoProvider(ABC):
    name: str

    @abstractmethod
    def gen(self, req: VideoRequest) -> VideoResult: ...
```

- [ ] **Step 4: Implement LTX Cloud adapter**

`/home/aialfred/alfred/scripts/providers/video/ltx_cloud.py`:

```python
"""LTX-2 Cloud video provider. Delegates to ComfyUI cloud LTX workflow."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .base import VideoProvider, VideoRequest, VideoResult

# LTX-2 is invoked via the same comfyui_gen.py with a --video flag.
COMFYUI_GEN_SCRIPT = Path("/home/aialfred/alfred/scripts/comfyui_gen.py")


class LtxCloud(VideoProvider):
    name = "ltx-cloud"

    def gen(self, req: VideoRequest) -> VideoResult:
        out = req.output_path or Path(tempfile.mkstemp(suffix=".mp4")[1])
        cmd = [
            "python3", str(COMFYUI_GEN_SCRIPT),
            "generate-video", req.prompt,
            "--cloud",
            "--duration", str(req.duration_s),
            "--output", str(out),
            "--width", str(req.width),
            "--height", str(req.height),
        ]
        subprocess.run(cmd, check=True, timeout=600)
        return VideoResult(video_path=out, duration_s=req.duration_s)
```

- [ ] **Step 5: Implement Higgsfield stub**

`/home/aialfred/alfred/scripts/providers/video/higgsfield.py`:

```python
"""Higgsfield video provider — stub until API access arrives."""
from __future__ import annotations

from .base import VideoProvider, VideoRequest, VideoResult


class HiggsfieldVideo(VideoProvider):
    name = "higgsfield"

    def gen(self, req: VideoRequest) -> VideoResult:
        raise NotImplementedError(
            "Higgsfield video provider not implemented yet. "
            "Fill in when API credentials are available."
        )
```

- [ ] **Step 6: Run tests to verify pass**

```bash
cd /home/aialfred/alfred
pytest tests/providers/test_video_base.py -v 2>&1 | tail -10
```

- [ ] **Step 7: Commit**

```bash
cd /home/aialfred/alfred
git add scripts/providers/video/ tests/providers/test_video_base.py
git commit -m "feat(providers): video provider base + LTX Cloud + Higgsfield stub"
```

---

### Task 12: Python voice-list helper + npm voices bridge

**Files:**
- Create: `/home/aialfred/alfred/scripts/voices.py`
- Create: `/home/aialfred/remotion/scripts/voices.mjs`

- [ ] **Step 1: Write voices.py**

`/home/aialfred/alfred/scripts/voices.py`:

```python
"""List Kokoro + Qwen3 voices grouped by provider + rotation/reserved tier."""
from __future__ import annotations

import json
import sys

from scripts.providers.tts.kokoro import KokoroTts
from scripts.providers.tts.qwen3 import Qwen3Tts

# Mirror of theme/providers.ts rotation pools — keep in sync manually.
ROTATION = {
    "kokoro": ["am_adam", "am_michael", "am_eric", "af_sarah", "af_sky", "af_alloy"],
    "qwen3":  ["Barbra_Gordon", "Brenda_Walker", "JAYDEE", "Louis_Lane"],
}
RESERVED = {
    "qwen3": ["MJ"],
}


def annotate(provider: str, voice: str) -> str:
    if voice in RESERVED.get(provider, []): return "[RESERVED]"
    if voice in ROTATION.get(provider, []): return "[rotation]"
    return ""


def main(json_output: bool = False):
    all_voices = {
        "kokoro": list(KokoroTts().list_voices()),
        "qwen3":  list(Qwen3Tts().list_voices()),
    }
    if json_output:
        print(json.dumps(all_voices, indent=2))
        return

    for provider, voices in all_voices.items():
        print(f"\n=== {provider.upper()} ({len(voices)} voices) ===")
        for v in sorted(voices):
            tag = annotate(provider, v)
            print(f"  {v:20} {tag}")


if __name__ == "__main__":
    main(json_output="--json" in sys.argv)
```

- [ ] **Step 2: Write voices.mjs (npm bridge)**

`/home/aialfred/remotion/scripts/voices.mjs`:

```javascript
#!/usr/bin/env node
// Prints Kokoro + Qwen3 voice pools. Shells out to Alfred's Python helper.
import { execSync } from "node:child_process";

const script = "/home/aialfred/alfred/scripts/voices.py";
try {
  const out = execSync(`python3 ${script}`, { encoding: "utf-8", stdio: ["ignore", "pipe", "inherit"] });
  process.stdout.write(out);
} catch (e) {
  console.error("Failed to list voices:", e.message);
  process.exit(1);
}
```

- [ ] **Step 3: Verify both work**

```bash
cd /home/aialfred/alfred
python3 scripts/voices.py 2>&1 | head -20
python3 scripts/voices.py --json 2>&1 | head -5

cd /home/aialfred/remotion
npm run voices 2>&1 | head -20
```

Expected: both print a formatted voice list with `[rotation]` tags on the 10 pooled voices and `[RESERVED]` on `MJ`.

- [ ] **Step 4: Commit Python side**

```bash
cd /home/aialfred/alfred
git add scripts/voices.py
git commit -m "feat(providers): voices.py CLI lists Kokoro + Qwen3 with rotation tags"
```

- [ ] **Step 5: Commit Remotion side**

```bash
cd /home/aialfred/remotion
git add scripts/voices.mjs
git commit -m "feat(scripts): npm run voices bridges to Alfred python voices helper"
```

---

# PHASE C — Primitives (Tasks 13-22)

Each primitive has: fixture render test, implementation, typecheck, commit. For brevity each task shows just the component code — all primitives live under `src/primitives/` and all read from `useBrand()` for color/font tokens unless noted.

---

### Task 13: BigWordCaption primitive

**Files:**
- Create: `/home/aialfred/remotion/src/primitives/BigWordCaption.tsx`

- [ ] **Step 1: Write the component**

```tsx
// src/primitives/BigWordCaption.tsx
import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { useBrand } from "../theme";
import { motion } from "../theme/tokens";

export type BigWordVariant = "single" | "stacked" | "scaleOnBeat";

export interface BigWordCaptionProps {
  word: string;                 // the caption text (may contain newlines for stacked)
  startFrame: number;
  endFrame: number;
  variant?: BigWordVariant;     // default "single"
  sizeRatio?: number;           // 0.0-1.0 of frame height; default 0.40
  align?: "center" | "left";    // default center
}

export const BigWordCaption: React.FC<BigWordCaptionProps> = ({
  word,
  startFrame,
  endFrame,
  variant = "single",
  sizeRatio = 0.40,
  align = "center",
}) => {
  const brand = useBrand();
  const frame = useCurrentFrame();
  const { height, fps } = useVideoConfig();

  if (frame < startFrame || frame > endFrame) return null;

  const localFrame = frame - startFrame;
  const fadeInFrames = motion.caption.popIn;
  const fadeOutFrames = motion.caption.popOut;
  const duration = endFrame - startFrame;

  const opacity = interpolate(
    localFrame,
    [0, fadeInFrames, duration - fadeOutFrames, duration],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const scale = variant === "scaleOnBeat"
    ? spring({ frame: localFrame, fps, config: { damping: 14, stiffness: 200, mass: 0.6 } })
    : 1;

  const fontSize = height * sizeRatio;
  const lines = variant === "stacked" ? word.split("\n") : [word];

  return (
    <div style={{
      position: "absolute",
      inset: 0,
      display: "flex",
      alignItems: "center",
      justifyContent: align === "center" ? "center" : "flex-start",
      paddingLeft: align === "left" ? 60 : 0,
      textAlign: align,
      opacity,
      transform: `scale(${scale})`,
    }}>
      <div style={{
        fontFamily: `${brand.fonts.headline}, sans-serif`,
        fontWeight: 900,
        color: brand.colors.fgLight,
        textTransform: "uppercase",
        lineHeight: 0.95,
        letterSpacing: 2,
        textShadow: "0 4px 24px rgba(0,0,0,0.9)",
        fontSize,
      }}>
        {lines.map((l, i) => <div key={i}>{l}</div>)}
      </div>
    </div>
  );
};
```

- [ ] **Step 2: Typecheck**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
```

Expected: no errors (note: import of `useBrand` from `../theme` resolves via `src/theme/index.ts`).

- [ ] **Step 3: Commit**

```bash
cd /home/aialfred/remotion
git add src/primitives/BigWordCaption.tsx
git commit -m "feat(primitives): BigWordCaption — hero word pop-in/hold/swap"
```

---

### Task 14: KaraokeCaptionLine primitive

**Files:**
- Create: `/home/aialfred/remotion/src/primitives/KaraokeCaptionLine.tsx`

- [ ] **Step 1: Write the component**

```tsx
// src/primitives/KaraokeCaptionLine.tsx
import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { useBrand } from "../theme";

export interface KaraokeWord {
  text: string;
  startFrame: number;  // when this word becomes highlighted
  endFrame: number;
}

export interface KaraokeCaptionLineProps {
  words: KaraokeWord[];
  bottomPx?: number;       // distance from bottom, default 280
  fontSize?: number;       // default 72
  highlightColor?: string; // defaults to brand.colors.accent
}

export const KaraokeCaptionLine: React.FC<KaraokeCaptionLineProps> = ({
  words,
  bottomPx = 280,
  fontSize = 72,
  highlightColor,
}) => {
  const brand = useBrand();
  const frame = useCurrentFrame();

  // Find which word is currently highlighted (if any)
  const active = words.find(w => frame >= w.startFrame && frame <= w.endFrame);
  if (!words.length) return null;

  // Line visible from first word start to last word end + fade
  const firstStart = words[0].startFrame;
  const lastEnd = words[words.length - 1].endFrame;
  const lineOpacity = interpolate(
    frame,
    [firstStart - 5, firstStart, lastEnd, lastEnd + 5],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const color = highlightColor ?? brand.colors.accent;

  return (
    <div style={{
      position: "absolute",
      bottom: bottomPx,
      left: 40,
      right: 40,
      textAlign: "center",
      opacity: lineOpacity,
      fontFamily: `${brand.fonts.headline}, sans-serif`,
      fontWeight: 800,
      fontSize,
      textTransform: "uppercase",
      color: brand.colors.fgLight,
      textShadow: "0 3px 18px rgba(0,0,0,0.9)",
      letterSpacing: 1.5,
      lineHeight: 1.1,
    }}>
      {words.map((w, i) => (
        <React.Fragment key={i}>
          {i > 0 && " "}
          <span style={{ color: active === w ? color : brand.colors.fgLight }}>
            {w.text}
          </span>
        </React.Fragment>
      ))}
    </div>
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/primitives/KaraokeCaptionLine.tsx
git commit -m "feat(primitives): KaraokeCaptionLine — word-by-word highlight"
```

---

### Task 15: LowerThirdBrandBar primitive

**Files:**
- Create: `/home/aialfred/remotion/src/primitives/LowerThirdBrandBar.tsx`

- [ ] **Step 1: Write the component**

```tsx
// src/primitives/LowerThirdBrandBar.tsx
import React from "react";
import { Img, staticFile, useCurrentFrame, interpolate } from "remotion";
import { useBrand } from "../theme";

export interface LowerThirdBrandBarProps {
  name: string;
  role?: string;
  appearFrame?: number;       // default 10
  bottomPx?: number;          // default 80
  logoOverride?: string;      // override brand.assets.logo
}

export const LowerThirdBrandBar: React.FC<LowerThirdBrandBarProps> = ({
  name,
  role,
  appearFrame = 10,
  bottomPx = 80,
  logoOverride,
}) => {
  const brand = useBrand();
  const frame = useCurrentFrame();

  const opacity = interpolate(frame, [appearFrame, appearFrame + 12], [0, 1], {
    extrapolateRight: "clamp",
  });
  const translateX = interpolate(frame, [appearFrame, appearFrame + 15], [-50, 0], {
    extrapolateRight: "clamp",
  });

  const logoSrc = logoOverride ?? brand.assets.logo;

  return (
    <div style={{
      position: "absolute",
      bottom: bottomPx,
      left: 0,
      opacity,
      transform: `translateX(${translateX}px) skewX(-12deg)`,
      display: "flex",
      alignItems: "center",
      gap: 18,
      background: brand.colors.bgDark,
      padding: "14px 28px 14px 22px",
      borderLeft: `6px solid ${brand.colors.accent}`,
    }}>
      <div style={{ transform: "skewX(12deg)", display: "flex", alignItems: "center", gap: 14 }}>
        <div style={{
          width: 44, height: 44,
          background: brand.colors.primary,
          borderRadius: 6,
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <Img src={staticFile(logoSrc)} style={{ width: 34, height: 34, objectFit: "contain" }} />
        </div>
        <div>
          <div style={{
            fontFamily: `${brand.fonts.headline}, sans-serif`,
            fontWeight: 800,
            color: brand.colors.fgLight,
            fontSize: 20,
            letterSpacing: 1.5,
            textTransform: "uppercase",
            lineHeight: 1,
          }}>{name}</div>
          {role && (
            <div style={{
              fontFamily: `${brand.fonts.body}, sans-serif`,
              fontWeight: 500,
              color: brand.colors.fgLight,
              opacity: 0.6,
              fontSize: 14,
              letterSpacing: 1,
              textTransform: "uppercase",
              marginTop: 4,
            }}>{role}</div>
          )}
        </div>
      </div>
    </div>
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/primitives/LowerThirdBrandBar.tsx
git commit -m "feat(primitives): LowerThirdBrandBar — angled logo+name+role ribbon"
```

---

### Task 16: EpisodeBadge primitive

**Files:**
- Create: `/home/aialfred/remotion/src/primitives/EpisodeBadge.tsx`

- [ ] **Step 1: Write the component**

```tsx
// src/primitives/EpisodeBadge.tsx
import React from "react";
import { useBrand } from "../theme";

export interface EpisodeBadgeProps {
  number: number;
  variant?: "pill" | "ribbon";   // default "pill"
  prefix?: string;                // default "EP"
  position?: {
    top?: number;                 // default 40
    left?: number;                // default 44
  };
}

export const EpisodeBadge: React.FC<EpisodeBadgeProps> = ({
  number,
  variant = "pill",
  prefix = "EP",
  position = {},
}) => {
  const brand = useBrand();
  const top = position.top ?? 40;
  const left = position.left ?? 44;

  const sharedText = {
    fontFamily: `${brand.fonts.headline}, sans-serif`,
    fontWeight: 900 as const,
    textTransform: "uppercase" as const,
    color: brand.colors.fgLight,
    letterSpacing: 2,
  };

  if (variant === "pill") {
    return (
      <div style={{
        position: "absolute",
        top, left,
        background: brand.colors.primary,
        padding: "8px 16px",
        borderRadius: 999,
        ...sharedText,
        fontSize: 16,
      }}>
        {prefix} {number}
      </div>
    );
  }

  // ribbon
  return (
    <div style={{
      position: "absolute",
      top, left,
      display: "flex",
      flexDirection: "column",
    }}>
      <div style={{ ...sharedText, fontSize: 14, opacity: 0.7 }}>
        EPISODE
      </div>
      <div style={{ ...sharedText, fontSize: 38, marginTop: 2, color: brand.colors.primary }}>
        {number}
      </div>
    </div>
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/primitives/EpisodeBadge.tsx
git commit -m "feat(primitives): EpisodeBadge — pill or ribbon episode number"
```

---

### Task 17: AudioWaveformStrip primitive (replaces WaveformBar)

**Files:**
- Create: `/home/aialfred/remotion/src/primitives/AudioWaveformStrip.tsx`

- [ ] **Step 1: Write the component (synthesized waveform; real audio-reactive variant is follow-up)**

```tsx
// src/primitives/AudioWaveformStrip.tsx
// Procedural waveform animation — does not sample real audio yet; matches
// WaveformBar (deprecated) behavior but reads color from brand context.
import React from "react";
import { useCurrentFrame, interpolate } from "remotion";
import { useBrand } from "../theme";

export interface AudioWaveformStripProps {
  barCount?: number;     // default 20
  barWidth?: number;     // default 4
  barGap?: number;       // default 3
  maxHeight?: number;    // default 28
  minHeight?: number;    // default 3
  color?: string;        // defaults to brand.colors.accent
  fadeInFrames?: number; // default 20
}

export const AudioWaveformStrip: React.FC<AudioWaveformStripProps> = ({
  barCount = 20,
  barWidth = 4,
  barGap = 3,
  maxHeight = 28,
  minHeight = 3,
  color,
  fadeInFrames = 20,
}) => {
  const brand = useBrand();
  const frame = useCurrentFrame();
  const fill = color ?? brand.colors.accent;

  const bars = Array.from({ length: barCount }, (_, i) => {
    const p1 = Math.sin(frame * 0.15 + i * 1.3) * 0.5 + 0.5;
    const p2 = Math.sin(frame * 0.22 + i * 0.7 + 2.1) * 0.5 + 0.5;
    const p3 = Math.sin(frame * 0.08 + i * 2.1 + 4.3) * 0.5 + 0.5;
    const combined = p1 * 0.5 + p2 * 0.3 + p3 * 0.2;
    const centerBias = 1 - (Math.abs(i - barCount / 2) / (barCount / 2)) * 0.4;
    return minHeight + (maxHeight - minHeight) * combined * centerBias;
  });

  const opacity = interpolate(frame, [0, fadeInFrames], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: barGap,
      opacity,
      height: maxHeight,
    }}>
      {bars.map((h, i) => (
        <div key={i} style={{
          width: barWidth,
          height: h,
          backgroundColor: fill,
          borderRadius: barWidth / 2,
        }} />
      ))}
    </div>
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/primitives/AudioWaveformStrip.tsx
git commit -m "feat(primitives): AudioWaveformStrip — brand-aware waveform"
```

---

### Task 18: BrandLockup primitive

**Files:**
- Create: `/home/aialfred/remotion/src/primitives/BrandLockup.tsx`

- [ ] **Step 1: Write the component**

```tsx
// src/primitives/BrandLockup.tsx
import React from "react";
import { Img, staticFile, useCurrentFrame, interpolate } from "remotion";
import { useBrand } from "../theme";

export type LockupPosition =
  | "top-left" | "top-right"
  | "bottom-left" | "bottom-right";

export interface BrandLockupProps {
  position?: LockupPosition;      // default "top-right"
  opacity?: number;                // default 1
  showHandle?: boolean;            // default true
  size?: number;                   // logo side in px, default 64
  fadeInFrames?: number;           // default 15
}

export const BrandLockup: React.FC<BrandLockupProps> = ({
  position = "top-right",
  opacity = 1,
  showHandle = true,
  size = 64,
  fadeInFrames = 15,
}) => {
  const brand = useBrand();
  const frame = useCurrentFrame();

  const alpha = interpolate(frame, [0, fadeInFrames], [0, opacity], {
    extrapolateRight: "clamp",
  });

  const corner: Record<LockupPosition, React.CSSProperties> = {
    "top-left":     { top: 36, left: 36,  alignItems: "flex-start" },
    "top-right":    { top: 36, right: 36, alignItems: "flex-end" },
    "bottom-left":  { bottom: 36, left: 36, alignItems: "flex-start" },
    "bottom-right": { bottom: 36, right: 36, alignItems: "flex-end" },
  };

  return (
    <div style={{
      position: "absolute",
      ...corner[position],
      display: "flex",
      flexDirection: "column",
      gap: 6,
      opacity: alpha,
    }}>
      <Img src={staticFile(brand.assets.logo)} style={{
        width: size, height: size, objectFit: "contain",
      }} />
      {showHandle && (
        <div style={{
          fontFamily: `${brand.fonts.body}, sans-serif`,
          fontWeight: 600,
          fontSize: 13,
          color: brand.colors.fgLight,
          opacity: 0.6,
          letterSpacing: 1,
        }}>
          {brand.meta.handle}
        </div>
      )}
    </div>
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/primitives/BrandLockup.tsx
git commit -m "feat(primitives): BrandLockup — persistent corner logo + handle"
```

---

### Task 19: GradeOverlay primitive

**Files:**
- Create: `/home/aialfred/remotion/src/primitives/GradeOverlay.tsx`

> **Note for implementer:** CSS-only LUT application is approximate — a true LUT pipeline requires a render-time post-process. This component ships the grain + vignette + contrast layers inline; LUT file presence is tracked in props for future wire-up when a proper post-pipeline exists. Letterbox bars are optional.

- [ ] **Step 1: Write the component**

```tsx
// src/primitives/GradeOverlay.tsx
import React from "react";
import { AbsoluteFill } from "remotion";
import { grade as gradeMap, type GradeId } from "../theme/tokens";

export interface GradeOverlayProps {
  preset: GradeId;
  letterbox?: boolean;            // 2.35:1 bars
  letterboxOpacity?: number;      // default 1
}

export const GradeOverlay: React.FC<GradeOverlayProps> = ({
  preset,
  letterbox = false,
  letterboxOpacity = 1,
}) => {
  const spec = gradeMap[preset];
  if (!spec) throw new Error(`Unknown grade preset: ${preset}`);

  return (
    <>
      {/* Vignette */}
      {spec.vignette > 0 && (
        <AbsoluteFill style={{
          background: `radial-gradient(ellipse at center, transparent 45%, rgba(0,0,0,${spec.vignette}) 100%)`,
          pointerEvents: "none",
        }} />
      )}

      {/* Grain — css noise via svg data uri */}
      {spec.grain > 0 && (
        <AbsoluteFill style={{
          backgroundImage:
            "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9'/></filter><rect width='100%' height='100%' filter='url(%23n)' opacity='0.5'/></svg>\")",
          backgroundSize: "200px 200px",
          mixBlendMode: "overlay",
          opacity: spec.grain,
          pointerEvents: "none",
        }} />
      )}

      {/* Contrast is baked into the video; if we want to push it, we'd need
          a filter chain on the <Video/>. For now, track the spec.contrast value. */}

      {letterbox && (
        <>
          <div style={{
            position: "absolute", top: 0, left: 0, right: 0,
            height: "15%",
            background: "#000",
            opacity: letterboxOpacity,
          }} />
          <div style={{
            position: "absolute", bottom: 0, left: 0, right: 0,
            height: "15%",
            background: "#000",
            opacity: letterboxOpacity,
          }} />
        </>
      )}
    </>
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/primitives/GradeOverlay.tsx
git commit -m "feat(primitives): GradeOverlay — grain + vignette + letterbox"
```

---

### Task 20: EndCard primitive

**Files:**
- Create: `/home/aialfred/remotion/src/primitives/EndCard.tsx`

- [ ] **Step 1: Write the component**

```tsx
// src/primitives/EndCard.tsx
import React from "react";
import { Img, staticFile, useCurrentFrame, interpolate } from "remotion";
import { useBrand } from "../theme";

export type EndCardVariant = "logo-url" | "portrait-meet";

export interface EndCardProps {
  variant?: EndCardVariant;       // default "logo-url"
  startFrame: number;
  endFrame: number;
  portraitName?: string;           // required when variant="portrait-meet"
  portraitAsset?: string;          // path under public/, required when variant="portrait-meet"
}

export const EndCard: React.FC<EndCardProps> = ({
  variant = "logo-url",
  startFrame,
  endFrame,
  portraitName,
  portraitAsset,
}) => {
  const brand = useBrand();
  const frame = useCurrentFrame();

  if (frame < startFrame || frame > endFrame) return null;

  const local = frame - startFrame;
  const opacity = interpolate(local, [0, 15, (endFrame - startFrame) - 8, (endFrame - startFrame)], [0, 1, 1, 0]);

  if (variant === "logo-url") {
    return (
      <div style={{
        position: "absolute", inset: 0,
        background: brand.colors.bgDark,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 24,
        opacity,
      }}>
        <Img src={staticFile(brand.assets.logo)} style={{
          width: 220, height: 220, objectFit: "contain",
        }} />
        <div style={{
          fontFamily: `${brand.fonts.headline}, sans-serif`,
          fontWeight: 800,
          fontSize: 36,
          color: brand.colors.fgLight,
          letterSpacing: 3,
          textTransform: "uppercase",
        }}>
          {brand.meta.url}
        </div>
      </div>
    );
  }

  // portrait-meet
  if (!portraitName || !portraitAsset) {
    throw new Error("EndCard variant=portrait-meet requires portraitName and portraitAsset");
  }
  return (
    <div style={{
      position: "absolute", inset: 0,
      background: brand.colors.bgDark,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      gap: 32,
      opacity,
    }}>
      <Img src={staticFile(portraitAsset)} style={{
        width: 480, height: 640, objectFit: "cover", borderRadius: 8,
      }} />
      <div style={{
        fontFamily: `${brand.fonts.headline}, sans-serif`,
        fontWeight: 900,
        fontSize: 72,
        color: brand.colors.fgLight,
        letterSpacing: 2,
        textTransform: "uppercase",
      }}>
        Meet {portraitName}
      </div>
    </div>
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/primitives/EndCard.tsx
git commit -m "feat(primitives): EndCard — logo-url or portrait-meet terminal card"
```

---

### Task 21: BrollClip (speed-ramp-capable video component)

**Files:**
- Create: `/home/aialfred/remotion/src/primitives/BrollClip.tsx`

- [ ] **Step 1: Write the component**

```tsx
// src/primitives/BrollClip.tsx
import React from "react";
import { Video, staticFile } from "remotion";
import { motion } from "../theme/tokens";

export type RampProfileId = keyof typeof motion.ramp;

export interface BrollClipProps {
  src: string;                     // filename under public/, or absolute URL
  startFrom?: number;              // Remotion frame offset into source; default 0
  objectFit?: "cover" | "contain"; // default cover
  rampProfile?: RampProfileId;     // applies per-segment playback rate
  style?: React.CSSProperties;
}

export const BrollClip: React.FC<BrollClipProps> = ({
  src,
  startFrom = 0,
  objectFit = "cover",
  rampProfile,
  style,
}) => {
  // Note: Remotion's <Video/> does not natively support variable playbackRate
  // across a clip — it takes one playbackRate for the whole clip. For multi-
  // phase speed ramps, the rig composes several <BrollClip/> instances end-to-
  // end with different playbackRate values. This component exposes a single
  // effective playbackRate; rigs that want the ramp curve build it in Sequence.
  const playbackRate =
    rampProfile != null ? motion.ramp[rampProfile].rip : 1;

  const resolved = src.startsWith("http") ? src : staticFile(src);

  return (
    <Video
      src={resolved}
      startFrom={startFrom}
      playbackRate={playbackRate}
      style={{ width: "100%", height: "100%", objectFit, ...style }}
    />
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/primitives/BrollClip.tsx
git commit -m "feat(primitives): BrollClip — Video wrapper w/ rampProfile playback rate"
```

---

### Task 22: useBeatMap hook + primitives barrel export

**Files:**
- Create: `/home/aialfred/remotion/src/primitives/useBeatMap.ts`
- Create: `/home/aialfred/remotion/src/primitives/index.ts`

- [ ] **Step 1: Write useBeatMap.ts**

```ts
// src/primitives/useBeatMap.ts
// Simple beat map utility. Rigs receive beat frames as props (produced by the
// Python provider side using librosa). This hook provides small helpers for
// reasoning about "am I on a beat right now?" and "what's the next beat?".
import { useCurrentFrame } from "remotion";

export type BeatMap = number[];  // sorted ascending frame indices

export function useBeatMap(beats: BeatMap) {
  const frame = useCurrentFrame();

  const isOnBeat = (windowFrames = 2): boolean =>
    beats.some((b) => Math.abs(b - frame) <= windowFrames);

  const nextBeat = (): number | null => {
    const n = beats.find((b) => b > frame);
    return n ?? null;
  };

  const lastBeat = (): number | null => {
    let prev: number | null = null;
    for (const b of beats) {
      if (b <= frame) prev = b;
      else break;
    }
    return prev;
  };

  return { frame, isOnBeat, nextBeat, lastBeat };
}
```

- [ ] **Step 2: Write primitives/index.ts barrel**

```ts
// src/primitives/index.ts
export * from "./BigWordCaption";
export * from "./KaraokeCaptionLine";
export * from "./LowerThirdBrandBar";
export * from "./EpisodeBadge";
export * from "./AudioWaveformStrip";
export * from "./BrandLockup";
export * from "./GradeOverlay";
export * from "./EndCard";
export * from "./BrollClip";
export * from "./useBeatMap";
```

- [ ] **Step 3: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/primitives/useBeatMap.ts src/primitives/index.ts
git commit -m "feat(primitives): useBeatMap hook + barrel export"
```

---

# PHASE D — Rigs (Tasks 23-29)

Each rig reads brand from `BrandProvider` wrapped at the composition root. All rig components require `brand: BrandId` prop at the outer level so Remotion's composition defaultProps can carry it.

---

### Task 23: GritDocRig

**Files:**
- Create: `/home/aialfred/remotion/src/rigs/GritDocRig.tsx`

- [ ] **Step 1: Write the rig**

```tsx
// src/rigs/GritDocRig.tsx
import React from "react";
import { AbsoluteFill, Sequence } from "remotion";
import { BrandProvider, getBrand, type BrandId } from "../theme";
import { BrollClip, BigWordCaption, BrandLockup, GradeOverlay } from "../primitives";

export interface GritDocClip {
  src: string;            // video file in public/
  durationFrames: number;
  rampProfile?: "holdRip" | "dropIn" | "rampOut";
}

export interface GritDocHeroWord {
  word: string;
  startFrame: number;
  endFrame: number;
}

export interface GritDocRigProps {
  brand: BrandId;
  clips: GritDocClip[];                 // rendered sequentially
  heroWords?: GritDocHeroWord[];        // optional mid-clip hero words
  gradePreset?: "teal-orange-crushed" | "bw-film" | "warm-studio" | "product-black";
}

export const GritDocRig: React.FC<GritDocRigProps> = ({
  brand,
  clips,
  heroWords = [],
  gradePreset = "teal-orange-crushed",
}) => {
  const theme = getBrand(brand);

  // Lay out clips sequentially
  let acc = 0;
  const segments = clips.map((c) => {
    const seg = { from: acc, durationInFrames: c.durationFrames, clip: c };
    acc += c.durationFrames;
    return seg;
  });

  return (
    <BrandProvider brand={theme}>
      <AbsoluteFill style={{ background: theme.colors.bgDark }}>
        {segments.map((s, i) => (
          <Sequence key={i} from={s.from} durationInFrames={s.durationInFrames}>
            <BrollClip src={s.clip.src} rampProfile={s.clip.rampProfile} />
          </Sequence>
        ))}
        <GradeOverlay preset={gradePreset} />
        {heroWords.map((hw, i) => (
          <BigWordCaption
            key={i}
            word={hw.word}
            startFrame={hw.startFrame}
            endFrame={hw.endFrame}
            variant="scaleOnBeat"
          />
        ))}
        <BrandLockup position="top-right" size={56} opacity={0.8} />
      </AbsoluteFill>
    </BrandProvider>
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/rigs/GritDocRig.tsx
git commit -m "feat(rigs): GritDocRig — B-roll montage + sparse captions"
```

---

### Task 24: KineticTypeRig

**Files:**
- Create: `/home/aialfred/remotion/src/rigs/KineticTypeRig.tsx`

- [ ] **Step 1: Write the rig**

```tsx
// src/rigs/KineticTypeRig.tsx
import React from "react";
import { AbsoluteFill } from "remotion";
import { BrandProvider, getBrand, type BrandId } from "../theme";
import type { GradeId } from "../theme";
import { BrollClip, BigWordCaption, KaraokeCaptionLine, BrandLockup, GradeOverlay } from "../primitives";
import type { KaraokeWord } from "../primitives";

export interface KineticTypeWordBeat {
  word: string;
  startFrame: number;
  endFrame: number;
  variant?: "single" | "stacked" | "scaleOnBeat";
}

export interface KineticTypeRigProps {
  brand: BrandId;
  bgClip: string;                  // video or image under public/
  wordBeats: KineticTypeWordBeat[];
  karaokeLine?: KaraokeWord[];     // optional lower-tier karaoke line
  gradePreset?: GradeId;
}

export const KineticTypeRig: React.FC<KineticTypeRigProps> = ({
  brand,
  bgClip,
  wordBeats,
  karaokeLine,
  gradePreset = "warm-studio",
}) => {
  const theme = getBrand(brand);

  return (
    <BrandProvider brand={theme}>
      <AbsoluteFill style={{ background: theme.colors.bgDark }}>
        <BrollClip src={bgClip} />
        <GradeOverlay preset={gradePreset} />
        <AbsoluteFill style={{ background: "rgba(0,0,0,0.35)" }} />
        {wordBeats.map((wb, i) => (
          <BigWordCaption
            key={i}
            word={wb.word}
            startFrame={wb.startFrame}
            endFrame={wb.endFrame}
            variant={wb.variant ?? "scaleOnBeat"}
            sizeRatio={0.42}
          />
        ))}
        {karaokeLine && karaokeLine.length > 0 && (
          <KaraokeCaptionLine words={karaokeLine} bottomPx={200} />
        )}
        <BrandLockup position="bottom-right" size={48} opacity={0.7} />
      </AbsoluteFill>
    </BrandProvider>
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/rigs/KineticTypeRig.tsx
git commit -m "feat(rigs): KineticTypeRig — text drives frame, slow b-roll underneath"
```

---

### Task 25: MagazineRig (★ most load-bearing)

**Files:**
- Create: `/home/aialfred/remotion/src/rigs/MagazineRig.tsx`

- [ ] **Step 1: Write the rig**

```tsx
// src/rigs/MagazineRig.tsx
import React from "react";
import { AbsoluteFill, Audio, staticFile } from "remotion";
import { BrandProvider, getBrand, type BrandId } from "../theme";
import type { GradeId } from "../theme";
import {
  BrollClip, KaraokeCaptionLine, EpisodeBadge, BrandLockup,
  AudioWaveformStrip, GradeOverlay,
} from "../primitives";
import type { KaraokeWord } from "../primitives";

export interface MagazineRigProps {
  brand: BrandId;
  episodeNumber: number;
  episodeTitle: string;
  clipSrc: string;                 // mp4 under public/
  audioSrc?: string;               // wav/mp3 under public/, optional
  captionPhrases: KaraokeWord[];   // karaoke-style
  hostName: string;
  guestName?: string;
  gradePreset?: GradeId;
}

export const MagazineRig: React.FC<MagazineRigProps> = ({
  brand,
  episodeNumber,
  episodeTitle,
  clipSrc,
  audioSrc,
  captionPhrases,
  hostName,
  guestName,
  gradePreset = "warm-studio",
}) => {
  const theme = getBrand(brand);

  return (
    <BrandProvider brand={theme}>
      <AbsoluteFill style={{ background: theme.colors.bgDark }}>
        {/* Video band — upper 65% */}
        <div style={{
          position: "absolute", top: 0, left: 0, right: 0, height: "65%",
          overflow: "hidden",
        }}>
          <BrollClip src={clipSrc} />
        </div>

        {/* Optional audio track mixed over */}
        {audioSrc && <Audio src={staticFile(audioSrc)} />}

        {/* Grade */}
        <GradeOverlay preset={gradePreset} />

        {/* Dark gradient footer blend */}
        <div style={{
          position: "absolute",
          top: "40%", left: 0, right: 0, bottom: 0,
          background: `linear-gradient(to bottom, transparent 0%, rgba(10,10,10,0.6) 25%, ${theme.colors.bgDark} 55%)`,
        }} />

        {/* Episode badge top-left */}
        <EpisodeBadge number={episodeNumber} variant="pill" position={{ top: 40, left: 44 }} />

        {/* Brand lockup top-right */}
        <BrandLockup position="top-right" size={60} showHandle={false} />

        {/* Karaoke caption in lower area */}
        <KaraokeCaptionLine words={captionPhrases} bottomPx={340} fontSize={64} />

        {/* Waveform + title + host credit footer */}
        <div style={{
          position: "absolute", left: 0, right: 0, bottom: 80,
          display: "flex", flexDirection: "column", alignItems: "center", gap: 18,
        }}>
          <AudioWaveformStrip barCount={22} color={theme.colors.accent} />
          <div style={{
            fontFamily: `${theme.fonts.headline}, sans-serif`,
            fontWeight: 800,
            color: theme.colors.fgLight,
            textTransform: "uppercase",
            fontSize: 22,
            letterSpacing: 2,
            textAlign: "center",
            padding: "0 44px",
          }}>{episodeTitle}</div>
          <div style={{
            display: "flex", gap: 8,
            fontFamily: `${theme.fonts.body}, sans-serif`,
            fontWeight: 600,
            color: theme.colors.fgLight,
            opacity: 0.75,
            fontSize: 14,
            letterSpacing: 2,
            textTransform: "uppercase",
          }}>
            <span>{hostName}</span>
            {guestName && <><span style={{ opacity: 0.4 }}>·</span><span>ft. {guestName}</span></>}
          </div>
        </div>

        {/* URL bottom-right */}
        <div style={{
          position: "absolute", bottom: 36, right: 44,
          fontFamily: `${theme.fonts.body}, sans-serif`,
          fontWeight: 600,
          fontSize: 14,
          color: theme.colors.fgLight,
          opacity: 0.6,
          letterSpacing: 1,
        }}>{theme.meta.url}</div>
      </AbsoluteFill>
    </BrandProvider>
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/rigs/MagazineRig.tsx
git commit -m "feat(rigs): MagazineRig — RuckTalk editorial podcast frame"
```

---

### Task 26: SpeedRampRig

**Files:**
- Create: `/home/aialfred/remotion/src/rigs/SpeedRampRig.tsx`

- [ ] **Step 1: Write the rig**

```tsx
// src/rigs/SpeedRampRig.tsx
import React from "react";
import { AbsoluteFill, Sequence } from "remotion";
import { BrandProvider, getBrand, type BrandId } from "../theme";
import type { GradeId } from "../theme";
import { motion } from "../theme/tokens";
import { BrollClip, BigWordCaption, GradeOverlay, EndCard } from "../primitives";
import type { RampProfileId } from "../primitives/BrollClip";

export interface SpeedRampRigProps {
  brand: BrandId;
  clipSrc: string;
  rampProfile?: RampProfileId;     // default "holdRip"
  heroWord: { word: string; atFrame: number; holdFrames?: number };
  gradePreset?: GradeId;
  endCardFrames?: number;           // trailing end card duration, default 45
}

export const SpeedRampRig: React.FC<SpeedRampRigProps> = ({
  brand,
  clipSrc,
  rampProfile = "holdRip",
  heroWord,
  gradePreset = "teal-orange-crushed",
  endCardFrames = 45,
}) => {
  const theme = getBrand(brand);
  const rampLen = 180; // 6s @ 30fps — total ramp phase duration

  return (
    <BrandProvider brand={theme}>
      <AbsoluteFill style={{ background: theme.colors.bgDark }}>
        <Sequence from={0} durationInFrames={rampLen}>
          <BrollClip src={clipSrc} rampProfile={rampProfile} />
          <GradeOverlay preset={gradePreset} />
          <BigWordCaption
            word={heroWord.word}
            startFrame={heroWord.atFrame}
            endFrame={heroWord.atFrame + (heroWord.holdFrames ?? motion.caption.hold)}
            variant="scaleOnBeat"
          />
        </Sequence>
        <Sequence from={rampLen} durationInFrames={endCardFrames}>
          <EndCard variant="logo-url" startFrame={0} endFrame={endCardFrames} />
        </Sequence>
      </AbsoluteFill>
    </BrandProvider>
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/rigs/SpeedRampRig.tsx
git commit -m "feat(rigs): SpeedRampRig — single-hero speed-ramped clip w/ end card"
```

---

### Task 27: BeatMontageRig

**Files:**
- Create: `/home/aialfred/remotion/src/rigs/BeatMontageRig.tsx`

- [ ] **Step 1: Write the rig**

```tsx
// src/rigs/BeatMontageRig.tsx
import React from "react";
import { AbsoluteFill, Sequence } from "remotion";
import { BrandProvider, getBrand, type BrandId } from "../theme";
import type { GradeId } from "../theme";
import { BrollClip, GradeOverlay, BrandLockup, EndCard } from "../primitives";

export interface BeatMontageClip {
  src: string;
  startFromSource?: number;  // frame offset INTO source video
  lengthFrames: number;      // how long this clip shows
}

export interface BeatMontageRigProps {
  brand: BrandId;
  clips: BeatMontageClip[];           // rendered sequentially on beats
  gradePreset?: GradeId;
  endCardFrames?: number;
}

export const BeatMontageRig: React.FC<BeatMontageRigProps> = ({
  brand,
  clips,
  gradePreset = "teal-orange-crushed",
  endCardFrames = 30,
}) => {
  const theme = getBrand(brand);

  let acc = 0;
  const segs = clips.map((c) => {
    const from = acc;
    acc += c.lengthFrames;
    return { from, clip: c };
  });
  const montageEnd = acc;

  return (
    <BrandProvider brand={theme}>
      <AbsoluteFill style={{ background: theme.colors.bgDark }}>
        {segs.map((s, i) => (
          <Sequence key={i} from={s.from} durationInFrames={s.clip.lengthFrames}>
            <BrollClip src={s.clip.src} startFrom={s.clip.startFromSource ?? 0} />
          </Sequence>
        ))}
        <GradeOverlay preset={gradePreset} />
        <BrandLockup position="bottom-right" size={44} opacity={0.6} />
        <Sequence from={montageEnd} durationInFrames={endCardFrames}>
          <EndCard variant="logo-url" startFrame={0} endFrame={endCardFrames} />
        </Sequence>
      </AbsoluteFill>
    </BrandProvider>
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/rigs/BeatMontageRig.tsx
git commit -m "feat(rigs): BeatMontageRig — rapid beat-cut montage"
```

---

### Task 28: VfxFlexRig

**Files:**
- Create: `/home/aialfred/remotion/src/rigs/VfxFlexRig.tsx`

- [ ] **Step 1: Write the rig**

```tsx
// src/rigs/VfxFlexRig.tsx
// Transition-heavy rig. Ships with simple crossfade transitions between
// clips; whip/morph/light-leak variants are future work.
import React from "react";
import { AbsoluteFill, Sequence, interpolate, useCurrentFrame } from "remotion";
import { BrandProvider, getBrand, type BrandId } from "../theme";
import type { GradeId } from "../theme";
import { BrollClip, BigWordCaption, GradeOverlay } from "../primitives";

export interface VfxFlexClip {
  src: string;
  lengthFrames: number;
}

export interface VfxFlexRigProps {
  brand: BrandId;
  clips: VfxFlexClip[];               // 2-4 clips
  heroWord: { word: string; atFrame: number; holdFrames?: number };
  transitionFrames?: number;          // default 12
  gradePreset?: GradeId;
}

const Crossfade: React.FC<{ durationInFrames: number; transitionFrames: number; children: React.ReactNode }> =
  ({ durationInFrames, transitionFrames, children }) => {
    const frame = useCurrentFrame();
    const inFade = interpolate(frame, [0, transitionFrames], [0, 1], { extrapolateRight: "clamp" });
    const outFade = interpolate(
      frame,
      [durationInFrames - transitionFrames, durationInFrames],
      [1, 0],
      { extrapolateLeft: "clamp" }
    );
    const opacity = Math.min(inFade, outFade);
    return <div style={{ opacity, width: "100%", height: "100%" }}>{children}</div>;
  };

export const VfxFlexRig: React.FC<VfxFlexRigProps> = ({
  brand,
  clips,
  heroWord,
  transitionFrames = 12,
  gradePreset = "teal-orange-crushed",
}) => {
  const theme = getBrand(brand);

  // Clips overlap by transitionFrames so crossfades blend them
  let acc = 0;
  const segs = clips.map((c, i) => {
    const from = i === 0 ? 0 : acc - transitionFrames;
    const seg = { from, clip: c };
    acc = from + c.lengthFrames;
    return seg;
  });

  return (
    <BrandProvider brand={theme}>
      <AbsoluteFill style={{ background: theme.colors.bgDark }}>
        {segs.map((s, i) => (
          <Sequence key={i} from={s.from} durationInFrames={s.clip.lengthFrames}>
            <Crossfade durationInFrames={s.clip.lengthFrames} transitionFrames={transitionFrames}>
              <BrollClip src={s.clip.src} />
            </Crossfade>
          </Sequence>
        ))}
        <GradeOverlay preset={gradePreset} />
        <BigWordCaption
          word={heroWord.word}
          startFrame={heroWord.atFrame}
          endFrame={heroWord.atFrame + (heroWord.holdFrames ?? 45)}
          variant="scaleOnBeat"
        />
      </AbsoluteFill>
    </BrandProvider>
  );
};
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/rigs/VfxFlexRig.tsx
git commit -m "feat(rigs): VfxFlexRig — crossfade-chained clips + hero word"
```

---

### Task 29: CinematicRig + rigs barrel export

**Files:**
- Create: `/home/aialfred/remotion/src/rigs/CinematicRig.tsx`
- Create: `/home/aialfred/remotion/src/rigs/index.ts`

- [ ] **Step 1: Write CinematicRig.tsx**

```tsx
// src/rigs/CinematicRig.tsx
import React from "react";
import { AbsoluteFill, Sequence } from "remotion";
import { BrandProvider, getBrand, type BrandId } from "../theme";
import type { GradeId } from "../theme";
import { BrollClip, GradeOverlay, EndCard } from "../primitives";

export interface CinematicScene {
  src: string;
  lengthFrames: number;
  rampProfile?: "holdRip" | "dropIn" | "rampOut";
}

export interface CinematicRigProps {
  brand: BrandId;
  scenes: CinematicScene[];
  gradePreset?: Extract<GradeId, "warm-studio" | "bw-film">;
  endCardFrames?: number;
}

export const CinematicRig: React.FC<CinematicRigProps> = ({
  brand,
  scenes,
  gradePreset = "warm-studio",
  endCardFrames = 45,
}) => {
  const theme = getBrand(brand);

  let acc = 0;
  const segs = scenes.map((s) => {
    const from = acc;
    acc += s.lengthFrames;
    return { from, scene: s };
  });
  const montageEnd = acc;

  return (
    <BrandProvider brand={theme}>
      <AbsoluteFill style={{ background: theme.colors.bgDark }}>
        {segs.map((s, i) => (
          <Sequence key={i} from={s.from} durationInFrames={s.scene.lengthFrames}>
            <BrollClip src={s.scene.src} rampProfile={s.scene.rampProfile} />
          </Sequence>
        ))}
        <GradeOverlay preset={gradePreset} letterbox letterboxOpacity={1} />
        <Sequence from={montageEnd} durationInFrames={endCardFrames}>
          <EndCard variant="logo-url" startFrame={0} endFrame={endCardFrames} />
        </Sequence>
      </AbsoluteFill>
    </BrandProvider>
  );
};
```

- [ ] **Step 2: Write rigs/index.ts**

```ts
// src/rigs/index.ts
export * from "./GritDocRig";
export * from "./KineticTypeRig";
export * from "./MagazineRig";
export * from "./SpeedRampRig";
export * from "./BeatMontageRig";
export * from "./VfxFlexRig";
export * from "./CinematicRig";
```

- [ ] **Step 3: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/rigs/CinematicRig.tsx src/rigs/index.ts
git commit -m "feat(rigs): CinematicRig + rigs barrel export"
```

---

# PHASE E — Engine + Integration (Tasks 30-34)

---

### Task 30: Zod schemas for all rig props

**Files:**
- Create: `/home/aialfred/remotion/src/engine/schemas.ts`
- Create: `/home/aialfred/remotion/tests/fixtures/.test-schemas.mjs` (inline run test)

- [ ] **Step 1: Write schemas.ts**

```ts
// src/engine/schemas.ts
// Runtime validation for rig props. autoProps() runs these before returning
// so the daily cron never renders invalid props.
import { z } from "zod";

export const BrandIdSchema = z.enum(["rucktalk", "loovacast", "grl"]);
export const GradeIdSchema = z.enum([
  "teal-orange-crushed", "bw-film", "warm-studio", "product-black",
]);
export const RampProfileSchema = z.enum(["holdRip", "dropIn", "rampOut"]);

const HeroWordSchema = z.object({
  word: z.string().min(1),
  startFrame: z.number().int().nonnegative(),
  endFrame: z.number().int().positive(),
});

const KaraokeWordSchema = z.object({
  text: z.string().min(1),
  startFrame: z.number().int().nonnegative(),
  endFrame: z.number().int().positive(),
});

export const GritDocRigSchema = z.object({
  brand: BrandIdSchema,
  clips: z.array(z.object({
    src: z.string().min(1),
    durationFrames: z.number().int().positive(),
    rampProfile: RampProfileSchema.optional(),
  })).min(1),
  heroWords: z.array(HeroWordSchema).optional(),
  gradePreset: GradeIdSchema.optional(),
});

export const KineticTypeRigSchema = z.object({
  brand: BrandIdSchema,
  bgClip: z.string().min(1),
  wordBeats: z.array(z.object({
    word: z.string().min(1),
    startFrame: z.number().int().nonnegative(),
    endFrame: z.number().int().positive(),
    variant: z.enum(["single", "stacked", "scaleOnBeat"]).optional(),
  })).min(1),
  karaokeLine: z.array(KaraokeWordSchema).optional(),
  gradePreset: GradeIdSchema.optional(),
});

export const MagazineRigSchema = z.object({
  brand: BrandIdSchema,
  episodeNumber: z.number().int().positive(),
  episodeTitle: z.string().min(1).max(200),
  clipSrc: z.string().min(1),
  audioSrc: z.string().optional(),
  captionPhrases: z.array(KaraokeWordSchema).min(1),
  hostName: z.string().min(1),
  guestName: z.string().optional(),
  gradePreset: GradeIdSchema.optional(),
});

export const SpeedRampRigSchema = z.object({
  brand: BrandIdSchema,
  clipSrc: z.string().min(1),
  rampProfile: RampProfileSchema.optional(),
  heroWord: z.object({
    word: z.string().min(1),
    atFrame: z.number().int().nonnegative(),
    holdFrames: z.number().int().positive().optional(),
  }),
  gradePreset: GradeIdSchema.optional(),
  endCardFrames: z.number().int().positive().optional(),
});

export const BeatMontageRigSchema = z.object({
  brand: BrandIdSchema,
  clips: z.array(z.object({
    src: z.string().min(1),
    startFromSource: z.number().int().nonnegative().optional(),
    lengthFrames: z.number().int().positive(),
  })).min(1),
  gradePreset: GradeIdSchema.optional(),
  endCardFrames: z.number().int().positive().optional(),
});

export const VfxFlexRigSchema = z.object({
  brand: BrandIdSchema,
  clips: z.array(z.object({
    src: z.string().min(1),
    lengthFrames: z.number().int().positive(),
  })).min(2).max(4),
  heroWord: z.object({
    word: z.string().min(1),
    atFrame: z.number().int().nonnegative(),
    holdFrames: z.number().int().positive().optional(),
  }),
  transitionFrames: z.number().int().positive().optional(),
  gradePreset: GradeIdSchema.optional(),
});

export const CinematicRigSchema = z.object({
  brand: BrandIdSchema,
  scenes: z.array(z.object({
    src: z.string().min(1),
    lengthFrames: z.number().int().positive(),
    rampProfile: RampProfileSchema.optional(),
  })).min(1),
  gradePreset: z.enum(["warm-studio", "bw-film"]).optional(),
  endCardFrames: z.number().int().positive().optional(),
});

export const RigName = z.enum([
  "GritDocRig", "KineticTypeRig", "MagazineRig", "SpeedRampRig",
  "BeatMontageRig", "VfxFlexRig", "CinematicRig",
]);
export type RigNameT = z.infer<typeof RigName>;

export const rigSchemas = {
  GritDocRig: GritDocRigSchema,
  KineticTypeRig: KineticTypeRigSchema,
  MagazineRig: MagazineRigSchema,
  SpeedRampRig: SpeedRampRigSchema,
  BeatMontageRig: BeatMontageRigSchema,
  VfxFlexRig: VfxFlexRigSchema,
  CinematicRig: CinematicRigSchema,
} as const;
```

- [ ] **Step 2: Write schema smoke test (node ESM script)**

`/home/aialfred/remotion/tests/fixtures/.test-schemas.mjs`:

```javascript
// Compiles and validates a minimum valid fixture for each rig schema.
// Run with: node --experimental-strip-types tests/fixtures/.test-schemas.mjs
import { rigSchemas } from "../../src/engine/schemas.ts";

const cases = {
  GritDocRig: { brand: "rucktalk", clips: [{ src: "a.mp4", durationFrames: 60 }] },
  KineticTypeRig: { brand: "rucktalk", bgClip: "bg.mp4", wordBeats: [{ word: "HI", startFrame: 0, endFrame: 30 }] },
  MagazineRig: { brand: "rucktalk", episodeNumber: 1, episodeTitle: "X", clipSrc: "c.mp4",
                  captionPhrases: [{ text: "HI", startFrame: 0, endFrame: 30 }], hostName: "Mike" },
  SpeedRampRig: { brand: "rucktalk", clipSrc: "c.mp4", heroWord: { word: "GO", atFrame: 30 } },
  BeatMontageRig: { brand: "rucktalk", clips: [{ src: "a.mp4", lengthFrames: 15 }] },
  VfxFlexRig: { brand: "rucktalk",
                  clips: [{ src: "a.mp4", lengthFrames: 45 }, { src: "b.mp4", lengthFrames: 45 }],
                  heroWord: { word: "X", atFrame: 30 } },
  CinematicRig: { brand: "rucktalk", scenes: [{ src: "s.mp4", lengthFrames: 90 }] },
};

let fail = 0;
for (const [name, fixture] of Object.entries(cases)) {
  const result = rigSchemas[name].safeParse(fixture);
  if (!result.success) {
    console.error(`FAIL ${name}:`, result.error.flatten());
    fail++;
  } else {
    console.log(`OK   ${name}`);
  }
}
process.exit(fail === 0 ? 0 : 1);
```

- [ ] **Step 3: Run the schema test**

```bash
cd /home/aialfred/remotion
npx tsx tests/fixtures/.test-schemas.mjs 2>&1 | tail -20
# If tsx is not installed: npm install --save-dev tsx
```

Expected: 7 lines of `OK <RigName>`, exit code 0.

- [ ] **Step 4: Commit**

```bash
cd /home/aialfred/remotion
# install tsx if not present
npm install --save-dev tsx 2>&1 | tail -3
git add src/engine/schemas.ts tests/fixtures/.test-schemas.mjs package.json package-lock.json
git commit -m "feat(engine): zod schemas for all 7 rigs + validation smoke test"
```

---

### Task 31: heroProps loader (hero briefs → typed rig props)

**Files:**
- Create: `/home/aialfred/remotion/src/engine/heroProps.ts`

- [ ] **Step 1: Write heroProps.ts**

```ts
// src/engine/heroProps.ts
// Loads a hero brief JSON file, validates against the rig schema, returns
// typed props ready to pass to `npx remotion render`.
import fs from "node:fs";
import path from "node:path";
import { rigSchemas, RigName, type RigNameT } from "./schemas";

export interface LoadedHeroBrief {
  rig: RigNameT;
  props: unknown;  // validated against rigSchemas[rig]
}

export function loadHeroBrief(briefPath: string): LoadedHeroBrief {
  const abs = path.resolve(briefPath);
  if (!fs.existsSync(abs)) throw new Error(`Brief not found: ${abs}`);
  const raw = JSON.parse(fs.readFileSync(abs, "utf-8"));

  const rigParse = RigName.safeParse(raw.rig);
  if (!rigParse.success) {
    throw new Error(`Invalid rig name in brief: ${JSON.stringify(raw.rig)}`);
  }
  const rig = rigParse.data;

  const schema = rigSchemas[rig];
  const validated = schema.safeParse(raw.props);
  if (!validated.success) {
    throw new Error(
      `Invalid props for ${rig}:\n${JSON.stringify(validated.error.flatten(), null, 2)}`
    );
  }
  return { rig, props: validated.data };
}
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/engine/heroProps.ts
git commit -m "feat(engine): heroProps loader — validates briefs against zod schemas"
```

---

### Task 32: autoProps builder (daily engine → rig + props)

**Files:**
- Create: `/home/aialfred/remotion/src/engine/autoProps.ts`

- [ ] **Step 1: Write autoProps.ts**

```ts
// src/engine/autoProps.ts
// Deterministic prop builder for the daily engine. Called with an AutoBrief,
// returns a { rig, props } tuple ready for `npx remotion render`.
//
// Only emits rigs 1-3 (MagazineRig, GritDocRig, KineticTypeRig) — the
// auto-safe tier. Rotation picks based on brief.rotation with guardrails.
import type { z } from "zod";
import {
  GritDocRigSchema, KineticTypeRigSchema, MagazineRigSchema, RigName, type RigNameT,
} from "./schemas";

export type TtsVoice = { provider: "kokoro" | "qwen3"; voice: string };

export interface EpisodeData {
  episodeNumber: number;
  episodeTitle: string;
  clipSrc: string;
  audioSrc?: string;
  captionPhrases: { text: string; startFrame: number; endFrame: number }[];
  hostName: string;
  guestName?: string;
}

export interface AutoBrief {
  brand: "rucktalk" | "loovacast" | "grl";
  date: string;              // ISO
  rotation: number;          // seed for which rig gets picked
  episode?: EpisodeData;
  bgClip?: string;           // for KineticTypeRig
  wordBeats?: { word: string; startFrame: number; endFrame: number }[];
  clips?: { src: string; durationFrames: number }[];  // for GritDocRig
  tts?: TtsVoice;
}

export type AutoOutput =
  | { rig: "MagazineRig"; props: z.infer<typeof MagazineRigSchema> }
  | { rig: "GritDocRig"; props: z.infer<typeof GritDocRigSchema> }
  | { rig: "KineticTypeRig"; props: z.infer<typeof KineticTypeRigSchema> };

const AUTO_RIG_ROTATION: RigNameT[] = ["MagazineRig", "GritDocRig", "KineticTypeRig"];

export function autoProps(brief: AutoBrief): AutoOutput {
  const pick = AUTO_RIG_ROTATION[Math.abs(brief.rotation) % AUTO_RIG_ROTATION.length];

  // Guardrail: MagazineRig requires real episode data. Fall back to KineticType.
  const effective: RigNameT =
    pick === "MagazineRig" && !brief.episode ? "KineticTypeRig" : pick;

  switch (effective) {
    case "MagazineRig": {
      const ep = brief.episode!;
      const props = MagazineRigSchema.parse({
        brand: brief.brand,
        episodeNumber: ep.episodeNumber,
        episodeTitle: ep.episodeTitle,
        clipSrc: ep.clipSrc,
        audioSrc: ep.audioSrc,
        captionPhrases: ep.captionPhrases,
        hostName: ep.hostName,
        guestName: ep.guestName,
      });
      return { rig: "MagazineRig", props };
    }
    case "GritDocRig": {
      if (!brief.clips || brief.clips.length === 0) {
        throw new Error("GritDocRig requires at least 1 clip in AutoBrief.clips");
      }
      const props = GritDocRigSchema.parse({
        brand: brief.brand,
        clips: brief.clips,
      });
      return { rig: "GritDocRig", props };
    }
    case "KineticTypeRig": {
      if (!brief.bgClip) throw new Error("KineticTypeRig requires AutoBrief.bgClip");
      if (!brief.wordBeats || brief.wordBeats.length === 0) {
        throw new Error("KineticTypeRig requires AutoBrief.wordBeats");
      }
      const props = KineticTypeRigSchema.parse({
        brand: brief.brand,
        bgClip: brief.bgClip,
        wordBeats: brief.wordBeats,
      });
      return { rig: "KineticTypeRig", props };
    }
    default:
      throw new Error(`autoProps: unexpected effective rig: ${effective}`);
  }
}
```

- [ ] **Step 2: Typecheck + commit**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit
git add src/engine/autoProps.ts
git commit -m "feat(engine): autoProps — deterministic rig+props for daily cron"
```

---

### Task 33: Root.tsx — add 7 new compositions

**Files:**
- Modify: `/home/aialfred/remotion/src/Root.tsx`

- [ ] **Step 1: Replace Root.tsx content**

```tsx
// src/Root.tsx
import { Composition } from "remotion";

// Deprecated — kept to avoid breaking existing cron until migration plan ships.
import { LoovaCastPromo } from "./templates/_deprecated/LoovaCastPromo";
import { RuckTalkPromo } from "./templates/_deprecated/RuckTalkPromo";
import { RuckTalkShort } from "./templates/_deprecated/RuckTalkShort";
import { RuckTalkClip } from "./templates/_deprecated/RuckTalkClip";
import { RuckTalkGrit } from "./templates/_deprecated/RuckTalkGrit";
import { BrandPromo } from "./templates/_deprecated/BrandPromo";

// New library rigs
import {
  GritDocRig, KineticTypeRig, MagazineRig, SpeedRampRig,
  BeatMontageRig, VfxFlexRig, CinematicRig,
} from "./rigs";

const PORTRAIT = { width: 1080, height: 1920, fps: 30 };

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/* ============ New Library ============ */}
      <Composition
        id="GritDocRig"
        component={GritDocRig}
        durationInFrames={450}
        {...PORTRAIT}
        defaultProps={{
          brand: "rucktalk" as const,
          clips: [{ src: "bg-video.mp4", durationFrames: 450 }],
          heroWords: [],
          gradePreset: "teal-orange-crushed" as const,
        }}
      />
      <Composition
        id="KineticTypeRig"
        component={KineticTypeRig}
        durationInFrames={300}
        {...PORTRAIT}
        defaultProps={{
          brand: "rucktalk" as const,
          bgClip: "bg-video.mp4",
          wordBeats: [
            { word: "TACTICAL", startFrame: 15, endFrame: 60, variant: "scaleOnBeat" as const },
            { word: "IDEAS",    startFrame: 65, endFrame: 110, variant: "scaleOnBeat" as const },
          ],
          gradePreset: "warm-studio" as const,
        }}
      />
      <Composition
        id="MagazineRig"
        component={MagazineRig}
        durationInFrames={1800}
        {...PORTRAIT}
        defaultProps={{
          brand: "rucktalk" as const,
          episodeNumber: 100,
          episodeTitle: "Reinvent Or Get Left Behind",
          clipSrc: "yt-clip.mp4",
          captionPhrases: [
            { text: "COMFORT", startFrame: 30, endFrame: 90 },
            { text: "IS", startFrame: 95, endFrame: 120 },
            { text: "THE ENEMY", startFrame: 125, endFrame: 200 },
          ],
          hostName: "MIKE JOHNSON",
          gradePreset: "warm-studio" as const,
        }}
      />
      <Composition
        id="SpeedRampRig"
        component={SpeedRampRig}
        durationInFrames={225}
        {...PORTRAIT}
        defaultProps={{
          brand: "rucktalk" as const,
          clipSrc: "bg-video.mp4",
          rampProfile: "holdRip" as const,
          heroWord: { word: "IMPACT", atFrame: 90, holdFrames: 60 },
          gradePreset: "teal-orange-crushed" as const,
        }}
      />
      <Composition
        id="BeatMontageRig"
        component={BeatMontageRig}
        durationInFrames={300}
        {...PORTRAIT}
        defaultProps={{
          brand: "rucktalk" as const,
          clips: Array.from({ length: 10 }, () => ({ src: "bg-video.mp4", lengthFrames: 27 })),
          gradePreset: "teal-orange-crushed" as const,
        }}
      />
      <Composition
        id="VfxFlexRig"
        component={VfxFlexRig}
        durationInFrames={300}
        {...PORTRAIT}
        defaultProps={{
          brand: "rucktalk" as const,
          clips: [
            { src: "bg-video.mp4", lengthFrames: 120 },
            { src: "bg-video.mp4", lengthFrames: 120 },
          ],
          heroWord: { word: "GROUND RUSH", atFrame: 200, holdFrames: 50 },
          gradePreset: "teal-orange-crushed" as const,
        }}
      />
      <Composition
        id="CinematicRig"
        component={CinematicRig}
        durationInFrames={450}
        {...PORTRAIT}
        defaultProps={{
          brand: "rucktalk" as const,
          scenes: [
            { src: "bg-video.mp4", lengthFrames: 150 },
            { src: "bg-video.mp4", lengthFrames: 150 },
          ],
          gradePreset: "warm-studio" as const,
        }}
      />

      {/* ============ Deprecated (kept until migration plan ships) ============ */}
      <Composition id="LoovaCastPromo" component={LoovaCastPromo} durationInFrames={300} fps={30} width={1080} height={1080}
        defaultProps={{ headline: "Your Voice. Your Station. Your Way.", subtext: "Launch your internet radio station in minutes", bgImage: "", accentColor: "#f97316" }} />
      <Composition id="LoovaCastPromoPortrait" component={LoovaCastPromo} durationInFrames={300} fps={30} width={1080} height={1920}
        defaultProps={{ headline: "Your Voice. Your Station. Your Way.", subtext: "Launch your internet radio station in minutes", bgImage: "", accentColor: "#f97316" }} />
      <Composition id="RuckTalkPromo" component={RuckTalkPromo} durationInFrames={300} fps={30} width={1080} height={1080}
        defaultProps={{ headline: "Success isn't clean. Show up anyway.", subtext: "#RuckTalk", bgImage: "" }} />
      <Composition id="RuckTalkShort" component={RuckTalkShort} durationInFrames={900} fps={30} width={1080} height={1920}
        defaultProps={{ images: [], captionWords: [
          { text: "Success", startFrame: 30, endFrame: 45 }, { text: "isn't", startFrame: 46, endFrame: 55 },
          { text: "clean.", startFrame: 56, endFrame: 70 }, { text: "Show", startFrame: 75, endFrame: 85 },
          { text: "up", startFrame: 86, endFrame: 95 }, { text: "anyway.", startFrame: 96, endFrame: 115 },
        ], durationPerImage: 90 }} />
      <Composition id="RuckTalkClip" component={RuckTalkClip} durationInFrames={1800} fps={30} width={1080} height={1920}
        defaultProps={{ videoSrc: "", episodeNumber: 1, episodeTitle: "Reinvent Or Get Left Behind",
          contextLine: "Ford got comfortable. The competition reinvented.", hostName: "MIKE JOHNSON", guestName: "",
          captionPhrases: [
            { text: "REINVENT OR GET LEFT BEHIND.", startFrame: 30, endFrame: 90 },
            { text: "COMFORT IS THE ENEMY.", startFrame: 100, endFrame: 160 },
            { text: "ADAPT. OVERCOME. DOMINATE.", startFrame: 170, endFrame: 240 },
          ] }} />
      <Composition id="RuckTalkGrit" component={RuckTalkGrit} durationInFrames={1110} fps={30} width={1080} height={1920}
        defaultProps={{ voiceover: "grit-vo.wav", scenes: [] }} />
      <Composition id="BrandPromo" component={BrandPromo} durationInFrames={300} fps={30} width={1080} height={1080}
        defaultProps={{ brandName: "Ground Rush Labs", headline: "Building the future", subtext: "Innovation. Automation. Results.",
          bgImage: "", accentColor: "#f97316", logoUrl: "" }} />
    </>
  );
};
```

- [ ] **Step 2: Verify Remotion studio boots**

```bash
cd /home/aialfred/remotion
npx tsc --noEmit 2>&1 | tail -10
# Optional human check: npx remotion studio — should show 7 new + 8 old compositions.
```

- [ ] **Step 3: Commit**

```bash
cd /home/aialfred/remotion
git add src/Root.tsx
git commit -m "feat(remotion): Root.tsx registers 7 new rigs alongside deprecated"
```

---

### Task 34: Hero CLI (npm run hero)

**Files:**
- Create: `/home/aialfred/remotion/scripts/hero.mjs`
- Create: `/home/aialfred/remotion/briefs/.gitkeep` (directory placeholder)

- [ ] **Step 1: Write hero.mjs**

```javascript
#!/usr/bin/env node
// npm run hero <brief.json> [--out=path.mp4] [--low]
// Validates the brief via engine/heroProps, then runs `npx remotion render`.
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REMOTION_ROOT = path.resolve(__dirname, "..");

const args = process.argv.slice(2);
if (args.length === 0 || args.includes("--help")) {
  console.log("Usage: npm run hero <brief.json> [-- --out=path.mp4] [-- --low]");
  process.exit(0);
}

const briefArg = args.find(a => !a.startsWith("--"));
const outArg = args.find(a => a.startsWith("--out="))?.slice(6);
const low = args.includes("--low");

if (!briefArg) { console.error("Missing brief path"); process.exit(1); }
const briefPath = path.resolve(briefArg);
if (!existsSync(briefPath)) { console.error(`Brief not found: ${briefPath}`); process.exit(1); }

// Validate the brief by shelling to tsx against heroProps
const validateScript = `
import { loadHeroBrief } from "./src/engine/heroProps.ts";
const b = loadHeroBrief(${JSON.stringify(briefPath)});
console.log(JSON.stringify(b));
`;
const tmpValidator = path.join(REMOTION_ROOT, "scripts", ".hero-validate.mjs");
writeFileSync(tmpValidator, validateScript);

let validated;
try {
  const out = execFileSync("npx", ["tsx", tmpValidator], {
    cwd: REMOTION_ROOT,
    encoding: "utf-8",
    stdio: ["ignore", "pipe", "inherit"],
  });
  validated = JSON.parse(out);
} catch (e) {
  console.error("Brief validation failed.");
  process.exit(1);
}

// Write props JSON that remotion render can read
const propsPath = path.join(REMOTION_ROOT, `.hero-props-${Date.now()}.json`);
writeFileSync(propsPath, JSON.stringify(validated.props));

const outPath = outArg ?? path.join(REMOTION_ROOT, `hero-${validated.rig}-${Date.now()}.mp4`);
const renderArgs = [
  "remotion", "render",
  "src/index.ts", validated.rig, outPath,
  `--props=${propsPath}`,
];
if (low) renderArgs.push("--scale=0.5");

console.log(`Rendering ${validated.rig} → ${outPath}...`);
execFileSync("npx", renderArgs, { cwd: REMOTION_ROOT, stdio: "inherit" });
console.log(`Done. ${outPath}`);
```

- [ ] **Step 2: Create briefs dir**

```bash
cd /home/aialfred/remotion
mkdir -p briefs
touch briefs/.gitkeep
```

- [ ] **Step 3: Smoke test by rendering a sample brief**

Create sample brief `/home/aialfred/remotion/briefs/sample-magazine.json`:

```json
{
  "rig": "MagazineRig",
  "props": {
    "brand": "rucktalk",
    "episodeNumber": 100,
    "episodeTitle": "Reinvent Or Get Left Behind",
    "clipSrc": "yt-clip.mp4",
    "captionPhrases": [
      { "text": "COMFORT", "startFrame": 30, "endFrame": 90 },
      { "text": "IS THE ENEMY", "startFrame": 100, "endFrame": 180 }
    ],
    "hostName": "MIKE JOHNSON"
  }
}
```

Then:

```bash
cd /home/aialfred/remotion
npm run hero briefs/sample-magazine.json -- --low 2>&1 | tail -5
```

Expected: a file `hero-MagazineRig-<timestamp>.mp4` exists and is non-zero-byte.

- [ ] **Step 4: Commit**

```bash
cd /home/aialfred/remotion
git add scripts/hero.mjs briefs/ .gitignore
# Append .hero-props-*.json and hero-*.mp4 to .gitignore
cat >> .gitignore <<'EOF'
.hero-props-*.json
hero-*.mp4
scripts/.hero-validate.mjs
EOF
git add .gitignore
git commit -m "feat(scripts): npm run hero — validate brief + remotion render"
```

---

# PHASE F — Testing Gates (Tasks 35-38)

---

### Task 35: Smoke test — render every rig from fixture

**Files:**
- Create: `/home/aialfred/remotion/tests/fixtures/{grit-doc,kinetic-type,magazine,speed-ramp,beat-montage,vfx-flex,cinematic}.json`
- Create: `/home/aialfred/remotion/tests/smoke.sh`

- [ ] **Step 1: Write the 7 fixtures**

`/home/aialfred/remotion/tests/fixtures/magazine.json`:

```json
{
  "rig": "MagazineRig",
  "props": {
    "brand": "rucktalk",
    "episodeNumber": 1,
    "episodeTitle": "Smoke Test Episode",
    "clipSrc": "yt-clip.mp4",
    "captionPhrases": [{ "text": "TEST", "startFrame": 15, "endFrame": 60 }],
    "hostName": "MIKE JOHNSON"
  }
}
```

`tests/fixtures/grit-doc.json`:

```json
{
  "rig": "GritDocRig",
  "props": {
    "brand": "rucktalk",
    "clips": [
      { "src": "bg-video.mp4", "durationFrames": 120 },
      { "src": "bg-video.mp4", "durationFrames": 120 }
    ],
    "heroWords": [{ "word": "GRIT", "startFrame": 60, "endFrame": 120 }]
  }
}
```

`tests/fixtures/kinetic-type.json`:

```json
{
  "rig": "KineticTypeRig",
  "props": {
    "brand": "rucktalk",
    "bgClip": "bg-video.mp4",
    "wordBeats": [
      { "word": "WORDS", "startFrame": 10, "endFrame": 60, "variant": "scaleOnBeat" },
      { "word": "HIT", "startFrame": 70, "endFrame": 120, "variant": "scaleOnBeat" }
    ]
  }
}
```

`tests/fixtures/speed-ramp.json`:

```json
{
  "rig": "SpeedRampRig",
  "props": {
    "brand": "rucktalk",
    "clipSrc": "bg-video.mp4",
    "heroWord": { "word": "RIP", "atFrame": 90, "holdFrames": 45 }
  }
}
```

`tests/fixtures/beat-montage.json`:

```json
{
  "rig": "BeatMontageRig",
  "props": {
    "brand": "rucktalk",
    "clips": [
      { "src": "bg-video.mp4", "lengthFrames": 15 },
      { "src": "bg-video.mp4", "lengthFrames": 15 },
      { "src": "bg-video.mp4", "lengthFrames": 15 }
    ]
  }
}
```

`tests/fixtures/vfx-flex.json`:

```json
{
  "rig": "VfxFlexRig",
  "props": {
    "brand": "rucktalk",
    "clips": [
      { "src": "bg-video.mp4", "lengthFrames": 120 },
      { "src": "bg-video.mp4", "lengthFrames": 120 }
    ],
    "heroWord": { "word": "FLEX", "atFrame": 150, "holdFrames": 45 }
  }
}
```

`tests/fixtures/cinematic.json`:

```json
{
  "rig": "CinematicRig",
  "props": {
    "brand": "rucktalk",
    "scenes": [
      { "src": "bg-video.mp4", "lengthFrames": 120 },
      { "src": "bg-video.mp4", "lengthFrames": 120 }
    ]
  }
}
```

- [ ] **Step 2: Write smoke.sh**

```bash
#!/usr/bin/env bash
# tests/smoke.sh — renders every rig at --scale=0.5 from its fixture.
# Requires: npm, ffprobe, public/bg-video.mp4 + yt-clip.mp4 already present.
set -euo pipefail

cd "$(dirname "$0")/.."

FIXTURES=(magazine grit-doc kinetic-type speed-ramp beat-montage vfx-flex cinematic)
OUT_DIR="tests/_smoke_output"
mkdir -p "$OUT_DIR"

FAIL=0
for f in "${FIXTURES[@]}"; do
  out="$OUT_DIR/$f.mp4"
  echo "→ $f → $out"
  if npm run --silent hero -- "tests/fixtures/$f.json" --out="$out" --low > "$OUT_DIR/$f.log" 2>&1; then
    size=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out")
    if [ "$size" -lt 10000 ]; then
      echo "  FAIL: output too small ($size bytes)"
      FAIL=$((FAIL + 1))
    else
      dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out" 2>/dev/null || echo "unknown")
      echo "  OK   size=${size} duration=${dur}s"
    fi
  else
    echo "  FAIL: render errored (see $OUT_DIR/$f.log)"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "SMOKE: all $((${#FIXTURES[@]})) rigs rendered OK"
else
  echo "SMOKE: $FAIL of ${#FIXTURES[@]} rigs FAILED"
  exit 1
fi
```

- [ ] **Step 3: Make executable and run**

```bash
cd /home/aialfred/remotion
chmod +x tests/smoke.sh
npm run smoke 2>&1 | tail -30
```

Expected: 7 lines `OK` and final `SMOKE: all 7 rigs rendered OK`. Rendering all 7 at --scale=0.5 takes ~5-10 minutes total on the server's GPU.

- [ ] **Step 4: Gitignore smoke output + commit**

```bash
cd /home/aialfred/remotion
cat >> .gitignore <<'EOF'
tests/_smoke_output/
EOF
git add tests/fixtures/ tests/smoke.sh .gitignore
git commit -m "test(remotion): per-rig fixtures + smoke.sh renders all 7 at scale 0.5"
```

---

### Task 36: Brand-swap test — render MagazineRig in rucktalk + loovacast, verify color

**Files:**
- Create: `/home/aialfred/remotion/tests/brand-swap.sh`
- Create: `/home/aialfred/remotion/tests/fixtures/magazine-loovacast.json`

- [ ] **Step 1: Write loovacast fixture**

`/home/aialfred/remotion/tests/fixtures/magazine-loovacast.json`:

```json
{
  "rig": "MagazineRig",
  "props": {
    "brand": "loovacast",
    "episodeNumber": 1,
    "episodeTitle": "Brand Swap Test",
    "clipSrc": "yt-clip.mp4",
    "captionPhrases": [{ "text": "PURPLE", "startFrame": 15, "endFrame": 60 }],
    "hostName": "LOOVACAST HOST"
  }
}
```

- [ ] **Step 2: Write brand-swap.sh**

```bash
#!/usr/bin/env bash
# tests/brand-swap.sh — render MagazineRig for rucktalk + loovacast,
# extract a frame, sample pixel near the top-left episode badge, verify
# brand-specific primary color appears.
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="tests/_swap_output"; mkdir -p "$OUT"
# Render both
npm run --silent hero -- tests/fixtures/magazine.json           --out="$OUT/rucktalk.mp4"  --low > "$OUT/rt.log" 2>&1
npm run --silent hero -- tests/fixtures/magazine-loovacast.json --out="$OUT/loovacast.mp4" --low > "$OUT/lc.log" 2>&1

# Extract frame at t=1.5s from each
ffmpeg -y -loglevel error -ss 1.5 -i "$OUT/rucktalk.mp4"  -vframes 1 "$OUT/rt.png"
ffmpeg -y -loglevel error -ss 1.5 -i "$OUT/loovacast.mp4" -vframes 1 "$OUT/lc.png"

# Sample pixel at approx x=120, y=55 (top-left where episode badge sits)
# Reduced coordinates because we render at --scale=0.5:
# full frame is 1080x1920, scaled → 540x960. Badge is around (60,40)*0.5 = approx (60, 40).
SAMPLE_X=80
SAMPLE_Y=50

sample_hex() {
  local img="$1"
  # Use ImageMagick if available, else fall back to python PIL
  if command -v convert >/dev/null 2>&1; then
    convert "$img" -format "%[pixel:p{${SAMPLE_X},${SAMPLE_Y}}]" info: | tr -d '()' | awk -F',' '{printf "%02X%02X%02X", $1, $2, $3}'
  else
    python3 - "$img" "$SAMPLE_X" "$SAMPLE_Y" <<'PY'
import sys
from PIL import Image
img = Image.open(sys.argv[1]).convert("RGB")
r, g, b = img.getpixel((int(sys.argv[2]), int(sys.argv[3])))
print(f"{r:02X}{g:02X}{b:02X}")
PY
  fi
}

RT_PIXEL=$(sample_hex "$OUT/rt.png")
LC_PIXEL=$(sample_hex "$OUT/lc.png")

echo "rucktalk  badge pixel: #$RT_PIXEL  (expect orange-ish, high R low B)"
echo "loovacast badge pixel: #$LC_PIXEL  (expect purple-ish, high R+B low G)"

# Very loose sanity: R channel of rucktalk pixel > B channel
R_RT=$(( 16#${RT_PIXEL:0:2} )); B_RT=$(( 16#${RT_PIXEL:4:2} ))
R_LC=$(( 16#${LC_PIXEL:0:2} )); G_LC=$(( 16#${LC_PIXEL:2:2} )); B_LC=$(( 16#${LC_PIXEL:4:2} ))

FAIL=0
if [ "$R_RT" -le "$B_RT" ]; then
  echo "FAIL: rucktalk expected R > B (orange), got R=$R_RT B=$B_RT"
  FAIL=1
fi
if [ "$B_LC" -le "$G_LC" ]; then
  echo "FAIL: loovacast expected B > G (purple), got G=$G_LC B=$B_LC"
  FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
  echo "BRAND-SWAP: OK"
else
  exit 1
fi
```

- [ ] **Step 3: Make executable and run**

```bash
cd /home/aialfred/remotion
chmod +x tests/brand-swap.sh
# Ensure a loovacast-logo.png placeholder exists (copy rucktalk logo as fallback)
cp -n public/rucktalk-logo.png public/loovacast-logo.png 2>/dev/null || true
cp -n public/rucktalk-logo.png public/grl-logo.png 2>/dev/null || true
npm run test:brand-swap 2>&1 | tail -10
```

Expected: BRAND-SWAP: OK.

- [ ] **Step 4: Gitignore + commit**

```bash
cd /home/aialfred/remotion
cat >> .gitignore <<'EOF'
tests/_swap_output/
EOF
git add tests/fixtures/magazine-loovacast.json tests/brand-swap.sh .gitignore public/loovacast-logo.png public/grl-logo.png
git commit -m "test(remotion): brand-swap test — MagazineRig rucktalk vs loovacast pixel check"
```

---

### Task 37: Visual-diff opt-in (goldens)

**Files:**
- Create: `/home/aialfred/remotion/scripts/visual-diff.mjs`

- [ ] **Step 1: Write visual-diff.mjs**

```javascript
#!/usr/bin/env node
// Visual regression: renders each rig fixture, extracts 3 keyframes,
// compares against tests/golden/<rig>/<brand>/frame-XX.png. Threshold 2%.
// Usage:
//   npm run test:visual            # compare, fail if drift > threshold
//   npm run test:visual:approve    # regenerate goldens from current output
import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const APPROVE = process.argv.includes("--approve");

const CASES = [
  { rig: "MagazineRig", brand: "rucktalk", fixture: "tests/fixtures/magazine.json" },
  { rig: "MagazineRig", brand: "loovacast", fixture: "tests/fixtures/magazine-loovacast.json" },
  { rig: "GritDocRig", brand: "rucktalk", fixture: "tests/fixtures/grit-doc.json" },
  { rig: "KineticTypeRig", brand: "rucktalk", fixture: "tests/fixtures/kinetic-type.json" },
];

const FRAME_TIMES = [0.5, 1.5, 3.0]; // seconds — 3 keyframes per case

const OUT = path.join(ROOT, "tests/_visual_output");
fs.mkdirSync(OUT, { recursive: true });

async function compareImages(golden, current) {
  // Compare file sizes as a cheap proxy; proper pixel diff requires ImageMagick.
  if (!fs.existsSync(golden)) return { missing: true, diff: 1 };
  try {
    const res = execSync(
      `compare -metric AE -fuzz 5% "${golden}" "${current}" null: 2>&1`,
      { encoding: "utf-8", shell: "/bin/bash" }
    );
    return { missing: false, diff: parseInt(res.trim(), 10) || 0 };
  } catch (e) {
    // ImageMagick compare returns non-zero on difference; parse stderr
    const n = parseInt((e.stderr || e.stdout || "0").toString().trim(), 10);
    return { missing: false, diff: isNaN(n) ? 999999 : n };
  }
}

let fail = 0;
for (const c of CASES) {
  const mp4 = path.join(OUT, `${c.rig}-${c.brand}.mp4`);
  execSync(`npm run --silent hero -- ${c.fixture} --out="${mp4}" --low`,
    { cwd: ROOT, stdio: "inherit" });

  for (let i = 0; i < FRAME_TIMES.length; i++) {
    const t = FRAME_TIMES[i];
    const curPng = path.join(OUT, `${c.rig}-${c.brand}-${i}.png`);
    execSync(`ffmpeg -y -loglevel error -ss ${t} -i "${mp4}" -vframes 1 "${curPng}"`);

    const goldenDir = path.join(ROOT, `tests/golden/${c.rig}/${c.brand}`);
    fs.mkdirSync(goldenDir, { recursive: true });
    const goldenPng = path.join(goldenDir, `frame-${String(i).padStart(2, "0")}.png`);

    if (APPROVE) {
      fs.copyFileSync(curPng, goldenPng);
      console.log(`APPROVED ${c.rig}/${c.brand}/frame-${i}`);
      continue;
    }

    const { missing, diff } = await compareImages(goldenPng, curPng);
    // 540x960 @ scale 0.5 = 518400 pixels; 2% threshold = 10368
    const threshold = 10368;
    if (missing) {
      console.log(`MISSING  ${c.rig}/${c.brand}/frame-${i} (run with --approve to create)`);
      fail++;
    } else if (diff > threshold) {
      console.log(`FAIL     ${c.rig}/${c.brand}/frame-${i}  diff=${diff}px > ${threshold}`);
      fail++;
    } else {
      console.log(`OK       ${c.rig}/${c.brand}/frame-${i}  diff=${diff}px`);
    }
  }
}

if (APPROVE) {
  console.log("\nGolden images regenerated. Commit tests/golden/ to lock in.");
  process.exit(0);
}
console.log(`\nVISUAL: ${fail === 0 ? "PASS" : `FAIL (${fail} frames)`}`);
process.exit(fail === 0 ? 0 : 1);
```

- [ ] **Step 2: Ensure ImageMagick present, then create initial goldens**

```bash
cd /home/aialfred/remotion
which compare || echo "WARN: ImageMagick not found — install via: sudo apt-get install imagemagick"
npm run test:visual:approve 2>&1 | tail -20
```

Expected: initial 12 goldens (4 cases × 3 frames) written under `tests/golden/`.

- [ ] **Step 3: Verify baseline comparison passes**

```bash
cd /home/aialfred/remotion
npm run test:visual 2>&1 | tail -20
```

Expected: all 12 frames OK, VISUAL: PASS.

- [ ] **Step 4: Commit goldens + script**

```bash
cd /home/aialfred/remotion
cat >> .gitignore <<'EOF'
tests/_visual_output/
EOF
git add scripts/visual-diff.mjs tests/golden/ .gitignore
git commit -m "test(remotion): opt-in visual-diff against golden frames"
```

---

### Task 38: Plan completion — run full test gate + tag

**Files:** (no new files; final verification)

- [ ] **Step 1: Run full verification in Remotion repo**

```bash
cd /home/aialfred/remotion
npm run typecheck 2>&1 | tail -3
npm run smoke 2>&1 | tail -10
npm run test:brand-swap 2>&1 | tail -5
npm run test:visual 2>&1 | tail -15
```

Expected: `typecheck` clean, `smoke` all 7 rigs OK, brand-swap OK, visual all 12 frames OK.

- [ ] **Step 2: Run Python provider tests**

```bash
cd /home/aialfred/alfred
pytest tests/providers/ -v 2>&1 | tail -20
```

Expected: all provider tests PASS (including live Kokoro + Qwen3 integration).

- [ ] **Step 3: Tag both repos**

```bash
cd /home/aialfred/remotion
git tag -a remotion-library-v1.0 -m "Remotion template library Phase 1 complete — 8 primitives, 7 rigs, 3 brand themes, all tests passing"

cd /home/aialfred/alfred
git tag -a providers-v1.0 -m "Python TTS/image/video provider abstraction — Kokoro+Qwen3 live, Higgsfield stubbed"
```

- [ ] **Step 4: Announce to Mike via email**

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, "/home/aialfred/alfred")
from integrations.email.client import EmailClient
body = """Phase 1 of the Remotion template library is complete, sir.

Built:
- 8 atomic primitives (BigWordCaption, KaraokeCaptionLine, LowerThirdBrandBar,
  EpisodeBadge, AudioWaveformStrip, BrandLockup, GradeOverlay, EndCard)
- 7 rigs (GritDoc, KineticType, Magazine, SpeedRamp, BeatMontage, VfxFlex, Cinematic)
- 3 brand themes (RuckTalk, LoovaCast, GRL)
- Python provider abstraction (Kokoro + Qwen3 live; ComfyUI local/cloud + Higgsfield stub)
- Zod prop validation, smoke tests, brand-swap test, opt-in visual diffs

All existing templates untouched — daily cron keeps working.

Render any rig: npm run hero briefs/<file>.json
Verify: npm run smoke && npm run test:brand-swap

Next step on your word: migration plan (cutover episode pipeline and daily social
to the new library, then delete old templates)."""
EmailClient().send_email(
    account="alfred-gw",
    to="mjohnson@groundrushinc.com",
    subject="Remotion Library Phase 1 — complete",
    body=body,
)
PY
```

---

## Self-Review

**Spec coverage check:**
- §1 Directory + theme → Tasks 1-6 ✓
- §2 Primitives (8) → Tasks 13-21 ✓
- §3 Rigs (7) → Tasks 23-29 ✓
- §4 Input contract (AutoBrief, HeroBrief, voice/provider) → Tasks 30-32, 34 ✓
- §5 Motion + grade tokens → Task 3 ✓
- §6 Migration + render pipeline → Partially — Root.tsx done (Task 33), hero CLI done (34), cutover of daily/episode pipelines DEFERRED to follow-up plan (stated in intro)
- §7 Testing → Tasks 35-38 (smoke, brand-swap, visual-diff) ✓
- Provider abstraction (Kokoro + Qwen3 + ComfyUI + Higgsfield stub) → Tasks 7-12 ✓
- Voice rotation pools + MJ reserved → Tasks 4, 12 ✓

**Placeholder scan:** No TODO/TBD. Every task shows the actual code. Two narrative notes are documented caveats, not placeholders:
- Task 19 `GradeOverlay` — CSS-based approximation; documented that a true LUT pipeline is follow-up
- Task 21 `BrollClip` — single playbackRate per clip; rigs compose multiple clips for multi-phase ramps

**Type consistency spot-check:**
- `BrandId` defined once (Task 5), referenced consistently in Tasks 6, 23-29, 30, 32
- `KaraokeWord` type exported from `KaraokeCaptionLine`, consumed in `MagazineRig` and `KineticTypeRig`
- `RampProfileId` exported from `BrollClip`, consumed in `SpeedRampRig` and fixture schemas
- `GradeId` exported from `tokens.ts`, referenced in all rigs + `GradeOverlay`
- `AutoBrief.tts` uses the same `{ provider, voice }` shape as `theme/providers.ts` `TtsVoice`
- `LoadedHeroBrief.rig` is `RigNameT`, used by hero.mjs to select the right composition id

Plan verified against spec. Ready for execution.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-17-remotion-template-library.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Good for a 38-task plan because it prevents my context from drowning.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach?
