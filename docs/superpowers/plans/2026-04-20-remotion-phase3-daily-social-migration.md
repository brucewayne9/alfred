# Remotion Phase 3 — Daily Social Migration (ffmpeg → Remotion Rigs)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `scripts/rucktalk_daily_social.py` from raw ffmpeg audio+video composition to Remotion rig rendering via the Phase 1 `autoProps()` engine. Daily shorts gain brand furniture, typography, and caption treatment that match the MagazineRig quality bar.

**Architecture:** Add a thin Node CLI (`scripts/auto-render.mjs`) to the Remotion repo that takes an `AutoBrief` JSON, calls `autoProps()` to resolve rig+props, and invokes `npx remotion render`. On the Python side, add a brief-builder module that converts each daily social mode (monologue / conversation / clip) into an `AutoBrief`. Daily social orchestrator gets a `DAILY_SOCIAL_ENGINE` env flag for safe rollout — defaults to legacy ffmpeg, flips to remotion after human approval.

**Tech Stack:** Python 3.11 (pytest), Node 22 + Remotion 4.0.438 (existing), tsx (existing dev dep), zod (existing from Phase 1).

**Source spec:** `/home/aialfred/alfred/docs/superpowers/specs/2026-04-17-remotion-template-library-design.md` Section 4 (Input Contract) + Section 6 Phase 3.

**Scope boundary:** Only `rucktalk_daily_social.py` and the new `scripts/auto-render.mjs` bridge. Do not modify the Remotion library components, the rigs, or `autoProps.ts` itself — they are v1.0-stable from Phase 1. Do not modify the episode pipeline — Phase 2 is stable on MagazineRig. Phase 4 (delete `_deprecated/`) comes after this plan.

**Repos:**
- Alfred `/home/aialfred/alfred/` — brief builder, daily social refactor, tests
- Remotion `/home/aialfred/remotion/` — new `scripts/auto-render.mjs` CLI

---

## Prerequisites Check

Before Task 1, verify:

- [ ] **P1:** Phase 2 stable on main. `cd /home/aialfred/alfred && git tag -l rucktalk-phase2-stable` — should exist after 2 clean episode cycles. If not yet, Phase 3 can still start (tasks are independent), but delay the flip (Task 9) until Phase 2 is tagged stable.
- [ ] **P2:** `autoProps` + schemas live on Remotion master. Verify: `cd /home/aialfred/remotion && test -f src/engine/autoProps.ts && test -f src/engine/schemas.ts && echo OK`.
- [ ] **P3:** All three auto-tier rigs (MagazineRig, GritDocRig, KineticTypeRig) render from fixtures cleanly. Verify: `cd /home/aialfred/remotion && npm run smoke 2>&1 | tail -5`.

---

## File Structure

**Create:**
- `/home/aialfred/remotion/scripts/auto-render.mjs` — Node CLI that reads `AutoBrief` JSON, calls `autoProps()`, invokes `npx remotion render`
- `/home/aialfred/alfred/scripts/daily_social_briefs.py` — Python module that produces an `AutoBrief`-shaped dict per daily social mode
- `/home/aialfred/alfred/tests/scripts/test_daily_social_briefs.py` — unit tests for the brief builders
- `/home/aialfred/alfred/tests/scripts/test_daily_social_engine_flag.py` — smoke tests around the new `DAILY_SOCIAL_ENGINE` flag routing

**Modify:**
- `/home/aialfred/alfred/scripts/rucktalk_daily_social.py` — add `DAILY_SOCIAL_ENGINE` constant, route `_produce_monologue_video()` and `_produce_conversation_video()` to either legacy ffmpeg (current) or new `render_via_remotion()` helper that calls `auto-render.mjs` with a built brief
- `/home/aialfred/remotion/package.json` — add `"auto-render"` npm script pointing at the new mjs

**Do not touch:**
- Any file under `/home/aialfred/remotion/src/` — library is locked at v1.0
- `scripts/rucktalk_episode_pipeline.py` — Phase 2 stable
- `scripts/rucktalk_rig_props.py` — episode-pipeline-specific (not reused here)

---

## Task 1: Add `auto-render.mjs` CLI to Remotion repo

**Files:**
- Create: `/home/aialfred/remotion/scripts/auto-render.mjs`
- Modify: `/home/aialfred/remotion/package.json` — add `"auto-render"` script

- [ ] **Step 1: Write the CLI**

Create `/home/aialfred/remotion/scripts/auto-render.mjs`:

```javascript
#!/usr/bin/env node
// Usage:
//   npm run auto-render -- <brief.json> [--out=path.mp4] [--low]
//
// Reads an AutoBrief JSON file, runs it through engine/autoProps()
// (pure TS, validated via Zod schemas), then invokes remotion render.
// Output: a single mp4 at --out (or timestamped default).
import { execFileSync, execSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

const args = process.argv.slice(2);
if (!args.length || args.includes("--help")) {
  console.log("Usage: npm run auto-render -- <brief.json> [--out=path.mp4] [--low]");
  process.exit(args.includes("--help") ? 0 : 1);
}

const briefArg = args.find(a => !a.startsWith("--"));
const outArg = args.find(a => a.startsWith("--out="))?.slice(6);
const low = args.includes("--low");
if (!briefArg) { console.error("Missing brief path"); process.exit(1); }

const briefPath = path.resolve(briefArg);
if (!existsSync(briefPath)) { console.error(`Brief not found: ${briefPath}`); process.exit(1); }

// Resolve rig+props by shelling to tsx against autoProps.
const resolveScript = `
import { readFileSync } from "node:fs";
import { autoProps } from "./src/engine/autoProps.ts";
const brief = JSON.parse(readFileSync(${JSON.stringify(briefPath)}, "utf-8"));
const result = autoProps(brief);
console.log(JSON.stringify(result));
`;
const tmpResolver = path.join(ROOT, "scripts", ".auto-render-resolve.mjs");
writeFileSync(tmpResolver, resolveScript);

let resolved;
try {
  const out = execFileSync("npx", ["tsx", tmpResolver], {
    cwd: ROOT,
    encoding: "utf-8",
    stdio: ["ignore", "pipe", "inherit"],
  });
  resolved = JSON.parse(out);
} catch (e) {
  console.error("AutoProps resolution failed.");
  process.exit(1);
}

// Write props JSON for remotion render subprocess
const propsPath = path.join(ROOT, `.auto-props-${Date.now()}.json`);
writeFileSync(propsPath, JSON.stringify(resolved.props));

const outPath = outArg ?? path.join(ROOT, `auto-${resolved.rig}-${Date.now()}.mp4`);
const renderArgs = [
  "remotion", "render",
  "src/index.ts", resolved.rig, outPath,
  `--props=${propsPath}`,
];
if (low) renderArgs.push("--scale=0.5");

console.log(`Rendering ${resolved.rig} → ${outPath}...`);
execFileSync("npx", renderArgs, { cwd: ROOT, stdio: "inherit" });
console.log(`RESOLVED_RIG=${resolved.rig}`);
console.log(`OUTPUT=${outPath}`);
```

- [ ] **Step 2: Add the npm script**

Edit `/home/aialfred/remotion/package.json` `"scripts"` block — add one line:

```json
"auto-render": "node scripts/auto-render.mjs",
```

Keep all existing scripts. Verify the JSON is valid after edit.

- [ ] **Step 3: Smoke test against a synthetic AutoBrief**

Write a quick test brief. Since MagazineRig requires episode data and KineticTypeRig is simpler, use KineticType:

```bash
cd /home/aialfred/remotion
cat > /tmp/test_brief_kinetic.json <<'EOF'
{
  "brand": "rucktalk",
  "date": "2026-04-20",
  "rotation": 2,
  "bgClip": "bg-video.mp4",
  "wordBeats": [
    {"word": "SHOW UP", "startFrame": 10, "endFrame": 60, "variant": "scaleOnBeat"},
    {"word": "ANYWAY", "startFrame": 70, "endFrame": 140, "variant": "scaleOnBeat"}
  ]
}
EOF
npm run auto-render -- /tmp/test_brief_kinetic.json --low 2>&1 | tail -10
ls -l auto-KineticTypeRig-*.mp4 2>/dev/null | tail -1
```

Expected: exits 0, prints `RESOLVED_RIG=KineticTypeRig`, `OUTPUT=…`, and an mp4 file exists.

- [ ] **Step 4: Gitignore the throwaway files and commit**

```bash
cd /home/aialfred/remotion
cat >> .gitignore <<'EOF'
.auto-props-*.json
auto-*.mp4
scripts/.auto-render-resolve.mjs
EOF
git add scripts/auto-render.mjs package.json package-lock.json .gitignore
git commit -m "feat(scripts): auto-render.mjs CLI — validate AutoBrief + render

Bridges Python daily social engine to Remotion. Reads an AutoBrief JSON,
resolves rig+props via engine/autoProps.ts, invokes remotion render.
Prints RESOLVED_RIG and OUTPUT on stdout for caller to parse."
```

---

## Task 2: Brief builder module — `daily_social_briefs.py`

**Files:**
- Create: `/home/aialfred/alfred/scripts/daily_social_briefs.py`
- Create: `/home/aialfred/alfred/tests/scripts/test_daily_social_briefs.py`

- [ ] **Step 1: Write the failing tests**

Create `/home/aialfred/alfred/tests/scripts/test_daily_social_briefs.py`:

```python
"""Tests for AutoBrief builders — Phase 3 daily social migration."""
from scripts.daily_social_briefs import (
    build_monologue_brief,
    build_conversation_brief,
    derive_word_beats_from_script,
)


def test_derive_word_beats_basic():
    script = "Show up anyway. Comfort is the enemy."
    beats = derive_word_beats_from_script(script, audio_duration_s=6.0, fps=30)
    # 2 sentences → 2 beat groups minimum
    assert len(beats) >= 2
    for b in beats:
        assert "word" in b and "startFrame" in b and "endFrame" in b
        assert b["endFrame"] > b["startFrame"]
        assert b["startFrame"] >= 0
        assert b["endFrame"] <= 6 * 30
        assert b.get("variant") in {"single", "stacked", "scaleOnBeat"}


def test_derive_word_beats_preserves_order():
    beats = derive_word_beats_from_script(
        "First. Second. Third.", audio_duration_s=6.0, fps=30
    )
    # ending frames should be monotonically non-decreasing
    ends = [b["endFrame"] for b in beats]
    assert ends == sorted(ends)


def test_build_monologue_brief_shape():
    brief = build_monologue_brief(
        date="2026-04-20",
        rotation=2,
        script="Show up anyway. Comfort is the enemy.",
        bg_clip_public_name="daily_mono_20260420.mp4",
        audio_duration_s=6.0,
    )
    assert brief["brand"] == "rucktalk"
    assert brief["date"] == "2026-04-20"
    assert brief["rotation"] == 2
    assert brief["bgClip"] == "daily_mono_20260420.mp4"
    assert len(brief["wordBeats"]) >= 2


def test_build_conversation_brief_shape():
    brief = build_conversation_brief(
        date="2026-04-20",
        rotation=1,
        bg_clips=[
            {"src": "bg_a.mp4", "durationFrames": 240},
            {"src": "bg_b.mp4", "durationFrames": 240},
        ],
    )
    assert brief["brand"] == "rucktalk"
    # Must align with autoProps expectations for GritDocRig — clips list required
    assert len(brief["clips"]) == 2
    assert all("src" in c and "durationFrames" in c for c in brief["clips"])
    # rotation value must land on GritDocRig (index 1 of ["MagazineRig","GritDocRig","KineticTypeRig"])
    assert brief["rotation"] % 3 == 1
```

- [ ] **Step 2: Run tests to verify FAIL**

```bash
cd /home/aialfred/alfred
python3 -m pytest tests/scripts/test_daily_social_briefs.py -v 2>&1 | tail -8
```

Expected: `ModuleNotFoundError: No module named 'scripts.daily_social_briefs'`.

- [ ] **Step 3: Implement the module**

Create `/home/aialfred/alfred/scripts/daily_social_briefs.py`:

```python
"""AutoBrief builders for the daily social engine (Phase 3 migration).

Converts the Python-side daily content state (LLM-generated script, Kokoro
audio duration, ComfyUI Cloud video) into the AutoBrief shape consumed by
the Remotion engine/autoProps.ts.

One builder per daily social mode:
  - build_monologue_brief   -> resolves to KineticTypeRig via autoProps rotation
  - build_conversation_brief -> resolves to GritDocRig   via autoProps rotation

Rotation is controlled by the caller so autoProps picks the expected rig.
autoProps cycles through ["MagazineRig", "GritDocRig", "KineticTypeRig"];
the caller must pass rotation values that land on the intended rig.
"""
from __future__ import annotations

import re
from typing import TypedDict


class CaptionPhrase(TypedDict):
    word: str
    startFrame: int
    endFrame: int
    variant: str  # "single" | "stacked" | "scaleOnBeat"


class MonologueAutoBrief(TypedDict):
    brand: str
    date: str
    rotation: int
    bgClip: str
    wordBeats: list[CaptionPhrase]


class ConversationClip(TypedDict):
    src: str
    durationFrames: int


class ConversationAutoBrief(TypedDict):
    brand: str
    date: str
    rotation: int
    clips: list[ConversationClip]


# autoProps.ts cycles through these three at index `rotation % 3`.
# MagazineRig is skipped for daily content (no episode data available),
# so valid rotation values are those that land on GritDocRig (1) or
# KineticTypeRig (2).
_ROTATION_FOR_KINETIC_TYPE = 2
_ROTATION_FOR_GRIT_DOC = 1


def derive_word_beats_from_script(
    script: str,
    audio_duration_s: float,
    fps: int = 30,
) -> list[CaptionPhrase]:
    """Break a TTS script into timed caption phrases.

    Sentences are the unit of typographic emphasis. Time is allocated
    proportionally to each sentence's character count (approximation of
    speaking time). Result is a list of phrases with non-overlapping
    frame ranges, guaranteed to fit within [0, audio_duration_s*fps].
    """
    # Split on sentence-ending punctuation, drop empties
    raw = re.split(r"(?<=[.!?])\s+", script.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    if not sentences:
        return []

    total_chars = sum(len(s) for s in sentences) or 1
    total_frames = int(audio_duration_s * fps)

    beats: list[CaptionPhrase] = []
    cursor = 0
    for i, s in enumerate(sentences):
        share = len(s) / total_chars
        span = max(1, int(total_frames * share))
        end = cursor + span if i < len(sentences) - 1 else total_frames
        # Upper-case short punchy phrase for hero caption display
        clean = re.sub(r"[^A-Za-z0-9\s'-]", "", s).upper().strip()
        variant = "scaleOnBeat" if len(clean) <= 20 else "stacked"
        beats.append({
            "word": clean,
            "startFrame": cursor,
            "endFrame": end,
            "variant": variant,
        })
        cursor = end

    return beats


def build_monologue_brief(
    *,
    date: str,
    rotation: int,
    script: str,
    bg_clip_public_name: str,
    audio_duration_s: float,
    fps: int = 30,
) -> MonologueAutoBrief:
    """Build an AutoBrief that resolves to KineticTypeRig.

    Caller is responsible for choosing a `rotation` that lands on
    KineticTypeRig (rotation % 3 == 2). `bg_clip_public_name` is the
    filename inside Remotion's public/ dir (the mp4 must be copied there
    before render).
    """
    beats = derive_word_beats_from_script(script, audio_duration_s, fps)
    if not beats:
        raise ValueError("script produced zero word beats")
    return {
        "brand": "rucktalk",
        "date": date,
        "rotation": rotation,
        "bgClip": bg_clip_public_name,
        "wordBeats": beats,
    }


def build_conversation_brief(
    *,
    date: str,
    rotation: int,
    bg_clips: list[ConversationClip],
) -> ConversationAutoBrief:
    """Build an AutoBrief that resolves to GritDocRig (no captions, just B-roll montage).

    Caller must pass rotation such that rotation % 3 == 1 (GritDocRig).
    """
    if not bg_clips:
        raise ValueError("bg_clips must not be empty")
    return {
        "brand": "rucktalk",
        "date": date,
        "rotation": rotation,
        "clips": bg_clips,
    }


def pick_rotation_for_kinetic_type() -> int:
    """Return a rotation index that makes autoProps pick KineticTypeRig."""
    return _ROTATION_FOR_KINETIC_TYPE


def pick_rotation_for_grit_doc() -> int:
    """Return a rotation index that makes autoProps pick GritDocRig."""
    return _ROTATION_FOR_GRIT_DOC
```

- [ ] **Step 4: Run tests to verify PASS**

```bash
cd /home/aialfred/alfred
python3 -m pytest tests/scripts/test_daily_social_briefs.py -v 2>&1 | tail -10
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add scripts/daily_social_briefs.py tests/scripts/test_daily_social_briefs.py
git commit -m "feat(daily-social): AutoBrief builders for monologue + conversation

New scripts/daily_social_briefs.py exposes:
  - build_monologue_brief   -> rotation 2 -> KineticTypeRig (Kokoro TTS + bg clip)
  - build_conversation_brief -> rotation 1 -> GritDocRig   (NotebookLM audio + bg clip montage)
  - derive_word_beats_from_script  — proportional sentence timing
  - pick_rotation_* helpers

4 unit tests cover shape, ordering, variant selection."
```

---

## Task 3: Add `DAILY_SOCIAL_ENGINE` flag + `render_via_remotion` helper

**Files:**
- Modify: `/home/aialfred/alfred/scripts/rucktalk_daily_social.py`

- [ ] **Step 1: Add env-var flag + imports near top of module**

Open `scripts/rucktalk_daily_social.py`. After the existing imports (around line 42), add:

```python
import os
import shutil
import subprocess

from scripts.daily_social_briefs import (
    build_monologue_brief,
    build_conversation_brief,
    pick_rotation_for_kinetic_type,
    pick_rotation_for_grit_doc,
)

# Phase 3 migration flag. "legacy" = current ffmpeg direct composition.
# "remotion" = new path via scripts/auto-render.mjs.
# Default stays legacy during rollout; Task 8 flips it to remotion after Mike approves.
DAILY_SOCIAL_ENGINE = os.environ.get("DAILY_SOCIAL_ENGINE", "legacy")

REMOTION_DIR = "/home/aialfred/remotion"
REMOTION_PUBLIC = Path(REMOTION_DIR) / "public"
NPX_PATH = "/home/aialfred/.nvm/versions/node/v22.22.0/bin/npx"
NODE_PATH = "/home/aialfred/.nvm/versions/node/v22.22.0/bin/node"
```

(`os`, `shutil`, `subprocess` may already be imported — don't duplicate. `Path` is already imported via `pathlib` at line 26.)

- [ ] **Step 2: Add the render_via_remotion helper function**

Append just before the existing `_produce_monologue_video` function (around line 173):

```python
def _render_via_remotion(brief: dict, output_path: Path) -> bool:
    """Invoke scripts/auto-render.mjs with an AutoBrief and capture output.

    The brief's bgClip / clips.src fields must already be filenames that
    exist in Remotion's public/ dir. Returns True on render success.
    """
    brief_file = WORK_DIR / f"brief_{output_path.stem}.json"
    brief_file.write_text(json.dumps(brief))
    try:
        cmd = [
            NPX_PATH, "--prefix", REMOTION_DIR,
            "auto-render", "--",
            str(brief_file),
            f"--out={output_path}",
        ]
        result = subprocess.run(
            cmd, cwd=REMOTION_DIR,
            capture_output=True, text=True, timeout=900,
            env={**os.environ, "PATH": f"/home/aialfred/.nvm/versions/node/v22.22.0/bin:{os.environ.get('PATH','')}"},
        )
        brief_file.unlink(missing_ok=True)
        if result.returncode != 0:
            logger.warning("Remotion auto-render failed: %s", result.stderr[-400:])
            return False
        # Parse "RESOLVED_RIG=..." from stdout for logging
        rig = next((l.split("=", 1)[1] for l in result.stdout.splitlines()
                    if l.startswith("RESOLVED_RIG=")), "unknown")
        logger.info("Rendered via Remotion %s: %s (%.1f MB)",
                    rig, output_path.name, output_path.stat().st_size / 1024 / 1024)
        return True
    except subprocess.TimeoutExpired:
        logger.error("Remotion auto-render timed out: %s", output_path.name)
        return False
    except Exception as exc:
        logger.error("Remotion auto-render error: %s", exc)
        return False


def _copy_to_remotion_public(src: Path, target_name: str) -> str:
    """Copy a local asset to Remotion's public/ dir so staticFile() can serve it.

    Returns the target_name (what the Remotion brief references).
    """
    REMOTION_PUBLIC.mkdir(parents=True, exist_ok=True)
    dest = REMOTION_PUBLIC / target_name
    shutil.copy2(str(src), str(dest))
    return target_name
```

- [ ] **Step 3: Syntax check**

```bash
cd /home/aialfred/alfred
python3 -c "import scripts.rucktalk_daily_social" 2>&1 | tail -3
```

Expected: no output (clean import). If `ModuleNotFoundError`, check your imports.

- [ ] **Step 4: Commit**

```bash
cd /home/aialfred/alfred
git add scripts/rucktalk_daily_social.py
git commit -m "feat(daily-social): add DAILY_SOCIAL_ENGINE flag + Remotion render helper

New env var DAILY_SOCIAL_ENGINE (default 'legacy') selects between the
current ffmpeg flow and the new Remotion-rendered flow. Adds
_render_via_remotion() that invokes scripts/auto-render.mjs with an
AutoBrief, and _copy_to_remotion_public() utility for asset staging.

No behavior change at default — Task 4/5 wire the monologue/conversation
producers to use the new path when EPISODE_RIG='remotion'."
```

---

## Task 4: Route monologue producer through flag

**Files:**
- Modify: `/home/aialfred/alfred/scripts/rucktalk_daily_social.py` — `_produce_monologue_video()`

- [ ] **Step 1: Read the current function**

Locate `def _produce_monologue_video(content: dict, mode: str, run_id: str)` (line ~173 before Task 3's additions). It runs Kokoro TTS, generates ComfyUI Cloud video, composes them with ffmpeg, writes to `WORK_DIR / f"monologue_{run_id}.mp4"`.

- [ ] **Step 2: Wrap it with the flag**

Rename the existing function to `_produce_monologue_video_legacy` (internal only). Then add a new entry-point function with the original name that dispatches:

```python
def _produce_monologue_video(content: dict, mode: str, run_id: str) -> str | None:
    """Dispatcher — routes to legacy ffmpeg path or new Remotion path based on DAILY_SOCIAL_ENGINE."""
    if DAILY_SOCIAL_ENGINE == "remotion":
        return _produce_monologue_video_remotion(content, mode, run_id)
    return _produce_monologue_video_legacy(content, mode, run_id)
```

- [ ] **Step 3: Implement the Remotion variant**

Add after `_produce_monologue_video_legacy`:

```python
def _produce_monologue_video_remotion(content: dict, mode: str, run_id: str) -> str | None:
    """Format A via Remotion KineticTypeRig.

    Same inputs as legacy (Kokoro TTS audio + ComfyUI Cloud video) but
    composed through the Remotion rig instead of raw ffmpeg. Captions
    appear on-screen in sync with the voiceover.
    """
    script = content.get("script") or content.get("narration") or ""
    if not script:
        logger.error("Monologue content has no script/narration text — cannot derive captions.")
        return None

    # 1. Run Kokoro TTS to get the voiceover mp3
    narration_path = WORK_DIR / f"monologue_{run_id}_narration.mp3"
    run_tts(script, narration_path)
    if not narration_path.exists():
        logger.error("Kokoro TTS failed to produce narration.")
        return None

    # Discover the audio's duration via ffprobe
    try:
        dur_result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(narration_path)],
            capture_output=True, text=True, timeout=30,
        )
        audio_duration_s = float(dur_result.stdout.strip())
    except Exception as exc:
        logger.error("Could not probe narration duration: %s", exc)
        return None

    # 2. Generate a single ComfyUI Cloud video of matching-or-longer duration
    bg_video_path = WORK_DIR / f"monologue_{run_id}_bg.mp4"
    topic_prompt = content.get("image_prompt") or content.get("topic") or "rucking motivation"
    cloud_ok = run_comfyui_video_cloud(topic_prompt, bg_video_path,
                                        duration_s=max(8.0, audio_duration_s + 1.0))
    if not cloud_ok or not bg_video_path.exists():
        logger.warning("ComfyUI Cloud bg video failed — aborting remotion monologue.")
        return None

    # 3. Copy bg + audio to Remotion public/ so staticFile() serves them
    bg_public_name = _copy_to_remotion_public(bg_video_path, f"daily_{run_id}_bg.mp4")

    # 4. Build the AutoBrief
    brief = build_monologue_brief(
        date=datetime.now(EST).date().isoformat(),
        rotation=pick_rotation_for_kinetic_type(),
        script=script,
        bg_clip_public_name=bg_public_name,
        audio_duration_s=audio_duration_s,
    )

    # 5. Render
    output = WORK_DIR / f"monologue_{run_id}.mp4"
    ok = _render_via_remotion(brief, output)
    # Best-effort cleanup of staged asset (not critical if it fails)
    (REMOTION_PUBLIC / bg_public_name).unlink(missing_ok=True)

    if not ok:
        return None

    # NOTE: Remotion output does NOT include the Kokoro audio yet — KineticTypeRig
    # currently renders silent with bg video only. Mux the audio in as a final step.
    muxed = WORK_DIR / f"monologue_{run_id}_final.mp4"
    mux_cmd = [
        "ffmpeg", "-y",
        "-i", str(output),
        "-i", str(narration_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(muxed),
    ]
    mux_result = subprocess.run(mux_cmd, capture_output=True, text=True, timeout=120)
    if mux_result.returncode != 0:
        logger.warning("ffmpeg mux failed: %s", mux_result.stderr[-300:])
        return str(output)  # fall back to silent render rather than nothing
    return str(muxed)
```

- [ ] **Step 4: Syntax check**

```bash
cd /home/aialfred/alfred
python3 -c "import scripts.rucktalk_daily_social" 2>&1 | tail -3
```

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add scripts/rucktalk_daily_social.py
git commit -m "feat(daily-social): route monologue through DAILY_SOCIAL_ENGINE flag

New _produce_monologue_video dispatches to:
  - _produce_monologue_video_legacy  (ffmpeg direct, current behavior)
  - _produce_monologue_video_remotion (KineticTypeRig via auto-render.mjs)

The new path: Kokoro TTS -> ComfyUI Cloud bg video -> KineticTypeRig render
-> ffmpeg mux for audio. Default stays legacy; flag flips in Task 8."
```

---

## Task 5: Route conversation producer through flag

**Files:**
- Modify: `/home/aialfred/alfred/scripts/rucktalk_daily_social.py` — `_produce_conversation_video()`

- [ ] **Step 1: Rename + wrap**

Same pattern as Task 4. Rename the existing `_produce_conversation_video` to `_produce_conversation_video_legacy`. Add a dispatcher:

```python
def _produce_conversation_video(content: dict, mode: str, run_id: str) -> str | None:
    if DAILY_SOCIAL_ENGINE == "remotion":
        return _produce_conversation_video_remotion(content, mode, run_id)
    return _produce_conversation_video_legacy(content, mode, run_id)
```

- [ ] **Step 2: Implement the Remotion variant**

Append:

```python
def _produce_conversation_video_remotion(content: dict, mode: str, run_id: str) -> str | None:
    """Format B via Remotion GritDocRig.

    NotebookLM 2-host podcast audio over a B-roll montage assembled from
    ComfyUI Cloud video segments. Falls back to monologue-remotion if
    NotebookLM fails.
    """
    topic = content.get("topic") or "RuckTalk"
    logger.info("Generating NotebookLM conversation for Remotion path: %s", topic)

    # 1. NotebookLM audio (reuses the legacy path's NotebookLM generation)
    audio_path = WORK_DIR / f"conversation_{run_id}.mp3"
    try:
        # NotebookLM client code is in the legacy function; extract to a shared helper
        # if you prefer. For now the simplest bridge is to call the legacy function
        # up to the audio step — copy that block here or refactor it into a helper.
        # Minimum viable: rely on legacy function's conversation-prep logic via a short
        # sub-call OR re-implement the audio generation step inline.
        import asyncio
        from integrations.notebooklm.client import NotebookLMClient  # adjust if different
        async def _gen():
            async with NotebookLMClient() as client:
                nb = await client.notebooks.create(name=f"RuckTalk daily {run_id}")
                await client.sources.add_text(nb.id, f"Topic: {topic}\n\nScript: {content.get('script','')}")
                await client.artifacts.generate_audio(nb.id,
                    instructions="Two hosts, energetic, under 3 minutes.")
                audio_bytes = await client.artifacts.download_audio(nb.id)
                audio_path.write_bytes(audio_bytes)
        asyncio.run(_gen())
    except Exception as exc:
        logger.warning("NotebookLM unavailable (%s) — falling back to monologue-remotion.", exc)
        return _produce_monologue_video_remotion(content, mode, run_id)

    if not audio_path.exists():
        return _produce_monologue_video_remotion(content, mode, run_id)

    # Probe audio duration
    try:
        dur_result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True, timeout=30,
        )
        audio_duration_s = float(dur_result.stdout.strip())
    except Exception:
        return _produce_monologue_video_remotion(content, mode, run_id)

    # 2. Generate 2-3 ComfyUI Cloud video segments to montage
    seg_duration = max(4.0, audio_duration_s / 3.0)
    seg_frames = int(seg_duration * 30)
    bg_clips = []
    prompts = [
        content.get("image_prompt") or "rucking outdoors",
        f"{content.get('topic','rucking')} atmosphere",
        "mountain rucking golden hour",
    ]
    for i, prompt in enumerate(prompts):
        seg_path = WORK_DIR / f"conversation_{run_id}_seg{i}.mp4"
        if run_comfyui_video_cloud(prompt, seg_path, duration_s=seg_duration):
            name = _copy_to_remotion_public(seg_path, f"daily_{run_id}_seg{i}.mp4")
            bg_clips.append({"src": name, "durationFrames": seg_frames})
    if not bg_clips:
        logger.warning("No bg segments generated — aborting conversation-remotion.")
        return None

    # 3. Build brief + render
    brief = build_conversation_brief(
        date=datetime.now(EST).date().isoformat(),
        rotation=pick_rotation_for_grit_doc(),
        bg_clips=bg_clips,
    )
    silent_out = WORK_DIR / f"conversation_{run_id}.mp4"
    ok = _render_via_remotion(brief, silent_out)
    # Cleanup staged segments
    for c in bg_clips:
        (REMOTION_PUBLIC / c["src"]).unlink(missing_ok=True)
    if not ok:
        return None

    # 4. Mux NotebookLM audio
    muxed = WORK_DIR / f"conversation_{run_id}_final.mp4"
    mux_cmd = [
        "ffmpeg", "-y",
        "-i", str(silent_out),
        "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(muxed),
    ]
    mux_result = subprocess.run(mux_cmd, capture_output=True, text=True, timeout=120)
    if mux_result.returncode != 0:
        return str(silent_out)
    return str(muxed)
```

- [ ] **Step 3: Syntax check + commit**

```bash
cd /home/aialfred/alfred
python3 -c "import scripts.rucktalk_daily_social" 2>&1 | tail -3
git add scripts/rucktalk_daily_social.py
git commit -m "feat(daily-social): route conversation through DAILY_SOCIAL_ENGINE flag

New _produce_conversation_video dispatches to legacy or remotion. The
remotion variant: NotebookLM audio -> 3x ComfyUI Cloud bg segments ->
GritDocRig montage -> ffmpeg mux. Falls back to monologue-remotion if
NotebookLM unavailable."
```

---

## Task 6: Flag-smoke test

**Files:**
- Create: `/home/aialfred/alfred/tests/scripts/test_daily_social_engine_flag.py`

- [ ] **Step 1: Write the test**

Create the file:

```python
"""Smoke test for the DAILY_SOCIAL_ENGINE dispatcher — verifies the flag
routes calls to the correct producer without invoking the heavy
TTS/ComfyUI/Remotion pipelines."""
from unittest.mock import patch
import scripts.rucktalk_daily_social as ds


def test_monologue_dispatches_legacy_by_default(monkeypatch):
    monkeypatch.setattr(ds, "DAILY_SOCIAL_ENGINE", "legacy")
    with patch.object(ds, "_produce_monologue_video_legacy", return_value="/tmp/legacy.mp4") as legacy, \
         patch.object(ds, "_produce_monologue_video_remotion", return_value="/tmp/remotion.mp4") as remo:
        result = ds._produce_monologue_video({"script": "x"}, "pillar", "r1")
        assert result == "/tmp/legacy.mp4"
        assert legacy.called
        assert not remo.called


def test_monologue_dispatches_remotion_when_flagged(monkeypatch):
    monkeypatch.setattr(ds, "DAILY_SOCIAL_ENGINE", "remotion")
    with patch.object(ds, "_produce_monologue_video_legacy", return_value="/tmp/legacy.mp4") as legacy, \
         patch.object(ds, "_produce_monologue_video_remotion", return_value="/tmp/remotion.mp4") as remo:
        result = ds._produce_monologue_video({"script": "x"}, "pillar", "r1")
        assert result == "/tmp/remotion.mp4"
        assert not legacy.called
        assert remo.called


def test_conversation_dispatches_legacy_by_default(monkeypatch):
    monkeypatch.setattr(ds, "DAILY_SOCIAL_ENGINE", "legacy")
    with patch.object(ds, "_produce_conversation_video_legacy", return_value="/tmp/c_legacy.mp4") as legacy, \
         patch.object(ds, "_produce_conversation_video_remotion", return_value="/tmp/c_remotion.mp4") as remo:
        result = ds._produce_conversation_video({"topic": "x"}, "pillar", "r1")
        assert result == "/tmp/c_legacy.mp4"
        assert legacy.called
        assert not remo.called


def test_conversation_dispatches_remotion_when_flagged(monkeypatch):
    monkeypatch.setattr(ds, "DAILY_SOCIAL_ENGINE", "remotion")
    with patch.object(ds, "_produce_conversation_video_legacy", return_value="/tmp/c_legacy.mp4") as legacy, \
         patch.object(ds, "_produce_conversation_video_remotion", return_value="/tmp/c_remotion.mp4") as remo:
        result = ds._produce_conversation_video({"topic": "x"}, "pillar", "r1")
        assert result == "/tmp/c_remotion.mp4"
        assert not legacy.called
        assert remo.called
```

- [ ] **Step 2: Run tests**

```bash
cd /home/aialfred/alfred
python3 -m pytest tests/scripts/test_daily_social_engine_flag.py -v 2>&1 | tail -10
```

Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
cd /home/aialfred/alfred
git add tests/scripts/test_daily_social_engine_flag.py
git commit -m "test(daily-social): verify DAILY_SOCIAL_ENGINE flag routes correctly"
```

---

## Task 7: Real side-by-side — email Mike for approval

**Files:** (no new files — produces artifacts)

- [ ] **Step 1: Run one full legacy render**

```bash
cd /home/aialfred/alfred
# Use the existing legacy path — just invoke the main engine once with the current content
DAILY_SOCIAL_ENGINE=legacy python3 scripts/rucktalk_daily_social.py --dry-run=false --force-format=monologue 2>&1 | tail -10
# Locate the output mp4 — it will be in WORK_DIR named monologue_*.mp4
ls -t /home/aialfred/rucktalk_pipeline/*.mp4 | head -2
```

If the daily engine doesn't support `--force-format`, invoke `_produce_monologue_video` directly via a short Python snippet using whatever content dict the LLM last produced.

- [ ] **Step 2: Run one full remotion render of the same content**

```bash
DAILY_SOCIAL_ENGINE=remotion python3 scripts/rucktalk_daily_social.py --dry-run=false --force-format=monologue 2>&1 | tail -10
ls -t /home/aialfred/rucktalk_pipeline/*.mp4 | head -2
```

- [ ] **Step 3: Email Mike both files**

```bash
python3 <<'PY'
import os, smtplib, glob
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("/home/aialfred/alfred/config/.env")

pw = os.environ["EMAIL_PASS_ALFRED_GW"]
# EDIT: set these to the real paths of the two renders you just produced.
old = Path("/home/aialfred/rucktalk_pipeline/monologue_LEGACY.mp4")
new = Path("/home/aialfred/rucktalk_pipeline/monologue_REMOTION.mp4")

msg = MIMEMultipart()
msg["From"] = "Alfred <alfred@groundrushinc.com>"
msg["To"]   = "mjohnson@groundrushinc.com"
msg["Subject"] = "Remotion Phase 3 — daily social rig cutover gate"
msg.attach(MIMEText("""<html><body style="font-family:-apple-system,sans-serif;line-height:1.55;max-width:640px;padding:20px;color:#111">
<p>Sir,</p>
<p>Phase 3 cutover gate. Same content, two renders:</p>
<ul>
  <li><b>monologue_LEGACY.mp4</b> — current ffmpeg direct composition (what your daily cron posts today).</li>
  <li><b>monologue_REMOTION.mp4</b> — new KineticTypeRig from Phase 1 library with on-screen kinetic captions.</li>
</ul>
<p>Reply: <b>Approve</b> / <b>Revise</b> (with notes) / <b>Keep old</b>.</p>
<p>Rollback after approval: <code>DAILY_SOCIAL_ENGINE=legacy</code> on the cron shell.</p>
<p>— Alfred</p></body></html>""", "html"))

for f in (old, new):
    with open(f, "rb") as fh:
        part = MIMEBase("video", "mp4")
        part.set_payload(fh.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment", filename=f.name)
    msg.attach(part)

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
    s.login("alfred@groundrushinc.com", pw)
    s.send_message(msg)
print("sent")
PY
```

- [ ] **Step 4: STOP — wait for Mike's reply**

Human approval gate. Do not proceed to Task 8 until Mike replies.

- **Approve** → Task 8.
- **Revise** → collect notes; loop back to Task 4 (for monologue) or Task 5 (for conversation) to adjust prop builders or caption timing, re-render, re-email.
- **Keep old** → Phase 3 is parked. No flip.

No commit for Task 7.

---

## Task 8: Flip the default to remotion

**Files:**
- Modify: `/home/aialfred/alfred/scripts/rucktalk_daily_social.py`

Do only after Task 7 returns **Approve**.

- [ ] **Step 1: Flip the default**

Locate:

```python
DAILY_SOCIAL_ENGINE = os.environ.get("DAILY_SOCIAL_ENGINE", "legacy")
```

Replace with:

```python
DAILY_SOCIAL_ENGINE = os.environ.get("DAILY_SOCIAL_ENGINE", "remotion")
```

- [ ] **Step 2: Verify**

```bash
cd /home/aialfred/alfred
python3 -c "from scripts.rucktalk_daily_social import DAILY_SOCIAL_ENGINE; print(DAILY_SOCIAL_ENGINE)"
```

Expected: `remotion`.

- [ ] **Step 3: Full test suite**

```bash
python3 -m pytest tests/scripts/ tests/providers/ -v 2>&1 | tail -15
```

Expected: all green (rig_props, compare, daily_social_briefs, daily_social_engine_flag, providers).

- [ ] **Step 4: Commit**

```bash
git add scripts/rucktalk_daily_social.py
git commit -m "feat(daily-social): flip DAILY_SOCIAL_ENGINE default to remotion

Phase 3 cutover approved by Mike via email gate. Tomorrow's 7 AM ET
daily social run produces its video through KineticTypeRig (monologue)
or GritDocRig (conversation) instead of ffmpeg direct composition.

Rollback: DAILY_SOCIAL_ENGINE=legacy on the cron shell."
```

---

## Task 9: First-run watch

**Files:** (no new files — monitoring)

- [ ] **Step 1: Wait for the next 7 AM ET run**

Or force one immediately if Mike wants to verify sooner:

```bash
cd /home/aialfred/alfred
python3 scripts/rucktalk_daily_social.py --dry-run=false 2>&1 | tail -15
```

- [ ] **Step 2: Verify the posted clip**

Check Postiz queue or Telegram notification for the scheduled post. Pull the mp4 URL it references; confirm it was rendered via Remotion (log line should contain `"Rendered via Remotion"`).

- [ ] **Step 3: Telegram Mike if it worked, email if not**

Success case — just let the normal daily-engine Telegram notification fire.
Failure case — immediate email:

```bash
cd /home/aialfred/alfred
python3 <<'PY'
import sys; sys.path.insert(0, ".")
from integrations.email.client import EmailClient
EmailClient().send_email(
    account="alfred-gw", to="mjohnson@groundrushinc.com",
    subject="[ACTION] Phase 3 first run failed — reverting to legacy",
    body="Sir, the first DAILY_SOCIAL_ENGINE=remotion run failed. Reverting "
         "the cron shell to DAILY_SOCIAL_ENGINE=legacy. Logs: "
         "/home/aialfred/rucktalk_pipeline/rucktalk.log. — Alfred",
)
PY
```

If it fails, set the rollback: edit crontab to add `DAILY_SOCIAL_ENGINE=legacy` before the daily social command.

No commit.

---

## Task 10: Three-day stability + tag

**Files:** (no new files — monitoring)

- [ ] **Step 1: Monitor 3 consecutive daily runs**

After each 7 AM ET run:
- Log contains `"Rendered via Remotion"`
- Postiz scheduling succeeded
- No Telegram error notifications

- [ ] **Step 2: Tag stable**

```bash
cd /home/aialfred/alfred
git tag -a rucktalk-phase3-stable -m "Phase 3 stable — daily social on Remotion, 3 clean days"
```

- [ ] **Step 3: Notify Mike**

```bash
cd /home/aialfred/alfred
python3 <<'PY'
import sys; sys.path.insert(0, ".")
from integrations.email.client import EmailClient
EmailClient().send_email(
    account="alfred-gw", to="mjohnson@groundrushinc.com",
    subject="Phase 3 stable — daily social on Remotion",
    body="Sir, three consecutive daily social runs have produced their video "
         "through the Remotion library (monologue -> KineticTypeRig, conversation "
         "-> GritDocRig). No errors, no rollbacks. Tagged rucktalk-phase3-stable.\n\n"
         "Phase 4 next: delete the _deprecated templates in the Remotion repo + "
         "the RuckTalkClip prop builder in Alfred. On your word.\n\n— Alfred",
)
PY
```

---

## Self-Review

**Spec coverage (Section 6 Phase 3):**
- "Migrate `rucktalk_daily_social.py` from its current template mix → `GritDocRig` / `KineticTypeRig` / `MagazineRig` via `autoProps()`" → Tasks 1-5 cover the migration; Tasks 7-8 the cutover. MagazineRig isn't used by daily social (requires episode data); the plan makes that explicit.
- Data sources table (Section 4): Kokoro TTS → KineticTypeRig (Task 4 remotion variant); NotebookLM → GritDocRig (Task 5); ComfyUI Cloud video for bg (both tasks).
- Approval gate before flip → Task 7.
- One-env-var rollback → Task 3 (`DAILY_SOCIAL_ENGINE`), Task 9 (use it if first run fails).

**Placeholder scan:** clean. Task 7 step 3 has a variable path placeholder (`monologue_LEGACY.mp4` vs real path) which is flagged as needing the actual filename from the side-by-side run. Task 5's NotebookLM client import path is flagged with "adjust if different" — it's best-effort; implementer may need to align with the actual integration layer.

**Type consistency:**
- `CaptionPhrase` defined once in `daily_social_briefs.py`, used by both builders.
- `MonologueAutoBrief` / `ConversationAutoBrief` fields match what `autoProps.ts` expects for KineticTypeRig / GritDocRig respectively (verified against Phase 1 Zod schemas in `src/engine/schemas.ts`).
- `DAILY_SOCIAL_ENGINE` is the single source of truth for rig selection, read at call time in both dispatchers.
- `REMOTION_DIR`, `REMOTION_PUBLIC`, `NPX_PATH` defined once at module top.

**Known shortcuts / future work (not in this plan):**
- KineticTypeRig composition doesn't embed audio — this plan muxes via ffmpeg. Long-term, the rig should accept an `audioSrc` prop like MagazineRig. Tracked as follow-on.
- `_produce_conversation_video_remotion` inlines a NotebookLM call path; the legacy function's NotebookLM logic should ideally be extracted to a shared helper. Tracked.
- Beat-synced cuts are not implemented for GritDocRig (clips are even-length). Hero tier can author beat-aware briefs; auto tier uses uniform cuts.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-04-20-remotion-phase3-daily-social-migration.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review, works well for the 10-task size.
2. **Inline Execution** — I execute in this session with checkpoints.

Which, sir?
