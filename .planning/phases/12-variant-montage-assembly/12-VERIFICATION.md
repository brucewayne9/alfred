---
phase: 12-variant-montage-assembly
verified: 2026-06-02T18:05:00Z
status: passed
score: 4/4 success criteria verified (CLIP-01..04)
re_verification:
  previous_status: none
human_verification:
  - test: "Play a delivered variant end-to-end in a browser/player and confirm captions are legible, logo placed bottom-right, audio synced to speech (no bleed)"
    expected: "9:16 clip, readable bottom-third caption, Mainstay logo, speech starts on the curated in-point"
    why_human: "Visual legibility, brand placement quality, and perceptual A/V sync cannot be confirmed by grep/ffprobe alone (ffprobe confirms dimensions/codecs/duration, not viewing quality)"
  - test: "Fire a real Postiz push (POST /forge/distribution/postiz) for a topic_clip job against a sandbox/test Postiz account"
    expected: "Drafts (not published) appear in Postiz for each delivered file"
    why_human: "Checkpoint deliberately did NOT live-fire Nextcloud delivery or Postiz draft creation to avoid stray drafts on live accounts; the path is proven by code-share with 3 shipped formats but never executed for topic_clip specifically"
---

# Phase 12: Variant Montage Assembly Verification Report

**Phase Goal:** The operator can turn their curated segment list into multiple distinct, on-brand vertical clips ready to hand to the social team.
**Verified:** 2026-06-02T18:05:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Success Criteria from ROADMAP)

| # | Truth (Success Criterion) | Status | Evidence |
| - | ------------------------- | ------ | -------- |
| 1 | Each assembled clip plays back with audio + video synced to source | ✓ VERIFIED | `_cut_segment` always re-encodes, never `-c copy` (topic_clip.py:84-117); concat demuxer over uniform re-encoded params avoids drift (topic_clip.py:120-172); checkpoint `data/_checkpoint_v0.mp4` ffprobe: h264+aac, audio mean -21.9dB (not silent) |
| 2 | Forge produces multiple distinct variants (order/cut/caption) — no two identical | ✓ VERIFIED | Two-tier: Tier-1 structural `_build_variant_assemblies` 5 strategies (original/reverse/hook-first/trimmed/shuffle-N), topic_clip.py:325-402; Tier-2 pixel `multiply(...allow_flip=False)` handlers.py:153; checkpoint reported 5/5 unique orderings + multiply md5-distinct copies |
| 3 | Every variant is 9:16 with Mainstay logo + styled captions | ✓ VERIFIED | scale/crop `1080:1920` in `_cut_segment` (topic_clip.py:92-93) and `_synthesise_visual` (topic_clip.py:454-455); `make_branded` (logo) imported from film_montage (topic_clip.py:33, called :537); `overlay_captions` drawtext bottom-third (topic_clip.py:290-300); checkpoint ffprobe 1080x1920 |
| 4 | Variants appear in library + push to Postiz as drafts in one action | ✓ VERIFIED | Handler emits `format/variant_count/variations_each/delivered/delivered_dirs` (handlers.py:175-181); library reads `delivered_dirs` + falls back to `variant_count` (library.py:47,56); `build_pack` reads `delivered_dirs` generically (distribution.py:238); UI `assembleVariants()` POSTs `topic_clip` job (index.html:1096-1125) |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `core/forge/renderers/topic_clip.py` | Cut/concat/caption/brand/variant/render engine | ✓ VERIFIED | 633 lines; all public API present and substantive; imports `make_branded`/`LOGO_PATH` (no redeclare) |
| `core/forge/handlers.py` (`_topic_clip_handler`) | Custom variant loop, registered as `topic_clip` | ✓ VERIFIED | handlers.py:105-181, registered :191; does NOT route through `_run_remix_format`/`build_remixes` per design |
| `services/forge-web/index.html` (`assembleVariants`) | Assemble-variants button injecting source_id | ✓ VERIFIED | Button :445, `assembleVariants()` :1096-1125, injects `source_id: sourceId` from `#topicSource` :1118 |
| `core/forge/distribution.py` (`push_to_postiz`/`build_pack`) | Generic delivered_dirs consumer | ✓ VERIFIED | `build_pack` reads `res.get("delivered_dirs")` :238 — format-agnostic |
| `tests/forge/test_topic_clip*.py` | Engine + handler tests | ✓ VERIFIED | Suite green: 133 passed |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `_cut_segment` | ffmpeg re-encode (libx264/aac), NOT `-c copy` | every cut re-encoded | ✓ WIRED | `-c:v libx264 ... -c:a aac` (topic_clip.py:94-95); zero `"copy"` strings in any ffmpeg arg list (only docstrings) |
| topic_clip.py | `film_montage.make_branded` + `LOGO_PATH` | import, no redeclare | ✓ WIRED | topic_clip.py:33 import; called :537 |
| topic_clip.py | ComfyUI for audio-only visual (no PIL) | `run_comfyui_cloud` lazy import | ✓ WIRED | topic_clip.py:435-436; ZERO inline PIL/Pillow/ImageDraw (grep: only a docstring naming the ban) |
| `_topic_clip_handler` | `multiply(...allow_flip=False)` | mandatory, captions/logo never mirror | ✓ WIRED | handlers.py:153 — literal `allow_flip=False`; test `test_allow_flip_is_false` asserts it |
| `assembleVariants()` | POST /forge/jobs `{source_id,...}` | re-adds source_id omitted by Phase-11 JSON | ✓ WIRED | index.html:1118 `source_id: sourceId` from `#topicSource` value :1097 |
| handler result | library + Postiz | `delivered_dirs` + `variant_count` | ✓ WIRED | library.py:47/56, distribution.py:238 |

### Requirements Coverage

| Requirement | Source Plan | Status | Evidence |
| ----------- | ----------- | ------ | -------- |
| CLIP-01 (cut segments, A/V in sync) | 12-01, 12-02 | ✓ SATISFIED | re-encode cut + concat + audio-only synth path; checkpoint synced render |
| CLIP-02 (multiple distinct variants) | 12-02, 12-03 | ✓ SATISFIED | two-tier distinctness (5 structural strategies + multiply allow_flip=False) |
| CLIP-03 (Mainstay 9:16 logo+caption) | 12-02 | ✓ SATISFIED | 1080x1920 framing, make_branded logo, overlay_captions |
| CLIP-04 (deliver to library + Postiz drafts) | 12-03 | ✓ SATISFIED (code) / needs live-fire | handler delivered_dirs shape consumed generically by library + build_pack; real delivery/draft deliberately not fired at checkpoint |

**Note on REQUIREMENTS.md staleness:** `.planning/REQUIREMENTS.md` still shows CLIP-04 as an unchecked box / "Pending". This is stale doc bookkeeping — the implementing code (handlers.py:175-181), tests (test_topic_clip_handler asserting delivered_dirs shape), and the checkpoint code-path verification all confirm CLIP-04 is implemented. Recommend flipping the checkbox.

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
| ---- | ------- | -------- | ------ |
| handlers.py:161-168 | bare `except Exception: pass` around per-file delivery | ℹ️ Info | Intentional resilience (one failed upload doesn't sink the batch); `look_delivered` guards `var_dirs` so empty deliveries don't produce phantom library entries. Acceptable but masks delivery errors silently — consider logging. |
| topic_clip.py:262-265 | "Per-segment timed captions deferred to Phase 13" | ℹ️ Info | Documented scope boundary, not a stub. Single first-sentence overlay ships and works. |

No 🛑 blockers. No TODO/FIXME/placeholder stubs. No `return null`/empty-implementation patterns in modified files.

### Human Verification Required

1. **Play a delivered variant** — confirm caption legibility, logo placement, perceptual A/V sync. ffprobe confirms 1080x1920 + h264/aac + 32.8s on `data/_checkpoint_v0.mp4`, but viewing quality is not machine-verifiable.
2. **Live Postiz push** — the checkpoint deliberately skipped real Nextcloud delivery + Postiz draft creation to avoid stray drafts on live accounts. Path is proven by code-share with 3 shipped formats; fire once against a sandbox account to close CLIP-04 end-to-end.

### Gaps Summary

No goal-blocking gaps. All 4 success criteria and CLIP-01..04 are satisfied in code with a real audio-only checkpoint render (`data/_checkpoint_v0.mp4`: 1080x1920, h264+aac, 32.8s, audible, non-black, caption band present). The 133-test forge suite is green.

Two non-blocking items deferred to human verification: (1) perceptual quality of a rendered clip, and (2) a real (vs code-path-verified) Postiz draft push — both consistent with the checkpoint's explicit "not-live-fired" note and the project's no-stray-draft caution.

Minor housekeeping: REQUIREMENTS.md CLIP-04 checkbox is stale (code is complete); silent `except: pass` on delivery could log.

---

_Verified: 2026-06-02T18:05:00Z_
_Verifier: Claude (gsd-verifier)_
