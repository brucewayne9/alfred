# Forge Studio — Milestone Arc

**Date:** 2026-06-12
**Status:** Arc approved (shape + keystone architecture decision). Per-milestone specs to follow.
**Owner:** Mike Johnson / Alfred

---

## Vision

One screen. Drop a source — a RuckTalk episode, or any long-form video — and Forge returns a
**batch** of intelligently-clipped, montage-cut, transition-polished, caption-styled vertical
clips, reviewed and queued to schedule. The manual "drop files in a folder" workflow is retired.

RuckTalk is the **first/proving show**. The same engine becomes a sellable product for other
podcasts and creators (the productize milestone, M5).

## Keystone architecture decision (APPROVED 2026-06-12)

**Consolidate into Forge.** RuckTalk's existing clip intelligence — `rucktalk_thesis_montage.py`,
the 5-angle thesis montage standard (hook_twist / cliff_notes / contrarian / personal_story /
advice), the daily social engine — is **ported into Forge's engine as reusable "show profiles,"**
not called as external scripts.

- **One engine, one codebase.** No drift between RuckTalk scripts and Forge.
- **A "show profile"** = a named config bundle (angle set, caption style, montage cut standard,
  branding, target platforms). RuckTalk becomes the first profile; new creators get their own.
- This is what makes the product **sellable** (M5) and what truly kills the folder dance (M3).
- Rejected alternatives: orchestrate RT scripts as-is (two codebases, not productizable);
  hybrid (defers the hard merge — declined in favor of doing it right once).

## Milestones

| # | Milestone | Delivers | Depends on |
|---|---|---|---|
| **M1** | **Transitions** | Curated transition menu (Phase 1) + beat-sync auto-edit (Phase 2). Polishes every montage. | none — ship first |
| **M2** | **Batch clipping** | One source → N intelligent clips automatically, with angle variety. | M1 helps, not required |
| **M3** | **RuckTalk ingest unified** | Pick/drop an episode *inside Forge*; engine produces the batch. Folder cron retired. Show-profile system born here. | M2 + engine consolidation |
| **M4** | **Review + schedule queue** | "Ready-to-go" board of generated clips → approve → batch-schedule via Postiz, per platform. | M2 |
| **M5** | **Productize** | Multi-show / multi-creator profiles, onboarding, the business. | M1–M4 |

**Recommended order:** M1 → M2 → M3 → M4 → M5.
Polish + business hook land immediately (M1); batch by M2; folder dies by M3; "schedule a whole
episode's worth and walk away" by M4; sellable product by M5.

## What already exists (build on, don't rebuild)

- **Forge engine:** intelligent single-clip extraction — Whisper → topic/semantic search → 9:16
  reframe → speech-synced karaoke captions (`core/forge/`: `topic_clip.py`, `ingest.py`,
  `search.py`, `clips.py`, `caption_styles.py`).
- **Transitions foundation:** `film_montage.py` `_concat_xfade` already crossfade-dissolves via
  ffmpeg `xfade` (toggleable). ffmpeg `xfade` ships ~50 transition types — M1 is largely *exposing*
  and curating existing capability + adding beat-sync.
- **RuckTalk intelligence:** `rucktalk_thesis_montage.py` (5 angles), episode pipeline, daily
  social engine — the logic M3 ports into Forge as the RuckTalk show profile.
- **Distribution:** Postiz already wired into Forge (`core/forge/postiz_client.py`,
  `distribution.py`) — M4's scheduling builds on it.
- **Audio analysis:** ducking/bed analysis already runs — M1 Phase 2 beat-sync extends it
  (tempo + onset detection) rather than adding a new subsystem.

## M1 — Transitions (next to be specced)

**Phase 1 — Curated menu.** Parameterize `_concat_xfade(transition=type)`, bring `multi_montage.py`
(currently hard cuts) onto the same path, expose a tasteful ~9-option picker with "Auto" default:
Auto · Cut · Dissolve (`fade`) · Whip (`slideleft/right`) · Wipe (`wipeleft/right`) · Zoom
(`zoomin`) · Flash (`fadewhite`) · Blur (`smoothleft`) · Glitch (`pixelize`). One style per montage.

**Phase 2 — Beat-sync.** Detect tempo + beat onsets (librosa/aubio) on the music bed; snap cut
points and transitions to the nearest beat; "Auto-edit to the beat" toggle; optionally vary
transition by music energy. This is the Instagram-Edits feel.

**Guardrail:** restraint. Default Cut + Dissolve; flashy transitions opt-in; bias toward one style
per montage (per-cut variety reads as amateur). Keep transition durations 0.15–0.3s. `xfade`
re-encodes (already accepted in `film_montage`).

## Deferred (resolved per-milestone, not now)

- M2: how angle variety is chosen for a batch (fixed profile angles vs. content-adaptive).
- M3: episode source UX (upload vs. pick-from-library vs. point-at-drop-folder), profile schema.
- M4: per-platform scheduling rules, approval granularity (per-clip vs. per-batch).
- M5: tenancy model, pricing, onboarding — full scope at milestone time.

## Next step

Brainstorm **M1 (Transitions)** through the normal design flow → its own spec → implementation plan.
