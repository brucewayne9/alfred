# Mainstay Forge — Phase 08: Viral Scoring ("Auto-Clips")

**Status:** SCOPED — ready to build
**Date:** 2026-06-12
**Builder:** Alfred (105)
**Project Manager:** Lucius (111) — timelines, check-ins with Jordan, team update emails, tracks to done
**Branch:** `feat/mainstay-forge`
**Judge model:** `kimi-k2.6:cloud` via local Ollama (`localhost:11434`) — **$0 marginal cost, no GPU load**

---

## 1. The gap this closes

Forge today finds clips by **topic** (semantic search) or by **operator hand-pick**. It cannot yet say
*"here are the 20 moments that will pop, ranked, with a score and a reason."* That ranked-grid behavior is
the one thing Opus Clip does that we don't — the **Virality Score (0–100)**.

This phase adds an automated **clip-worthiness scorer** and a new **Auto-Clips** tab that surfaces a ranked
grid of scored moments, each one click away from the existing render pipeline.

## 2. Design principles

- **Reuse everything already built.** Ingest, Whisper word-timings, ChromaDB windows, the 37-style caption
  engine, the topic_clip renderer, and the distribution pack all stay exactly as-is. This phase only adds a
  **scoring layer** in front of the render step and a **grid UI** to surface it.
- **No paid API.** The judge is `kimi-k2.6:cloud`, already present in 105's Ollama and billed on the existing
  flat Ollama Cloud subscription. Same call pattern as `core/forge/search.py` uses for `bge-m3:latest`.
- **GPU stays lean.** Cloud inference for scoring keeps the 3090 free for Whisper + Qwen3 TTS, which must
  run locally. We do NOT stand up a local scoring LLM.
- **Pluggable judge (one config flag).** Default `FORGE_JUDGE_MODEL=kimi-k2.6:cloud`. Swappable to any other
  Ollama model (`glm-5.1:cloud`, a local 7–14B, etc.) without code changes, so we can A/B without lock-in.
- **The data flywheel is already half-wired.** `core/forge/intel.py` already pulls real
  views/likes/comments/shares per posted clip. The moment we score *before* posting, we automatically build a
  `predicted_score → actual_engagement` dataset with zero manual labeling. That dataset is the eventual moat.

## 3. Architecture (where it plugs in)

```
ingest → whisper (word-level timings)   [EXISTING]
              │
              ▼
   [NEW] scorer.py  ── reads full timestamped transcript
              │        ── one Ollama call to kimi-k2.6:cloud
              │        ── returns ranked moments (JSON)
              │        ── snaps in/out to word boundaries (reuse existing word data)
              ▼
   [NEW] clip_candidates table  (predicted scores stored)
              │
              ▼
   [NEW] Auto-Clips UI tab  ── ranked grid, score badges, hook line, reason
              │ (one click)
              ▼
   topic_clip renderer → caption engine → distribution pack   [EXISTING]
              │
              ▼
   intel.py engagement  ── joins back to clip_candidates → calibration   [EXISTING + small join]
```

## 4. The judge call

- **Endpoint:** `http://localhost:11434/api/chat` (same Ollama the embeddings already use).
- **Model:** `kimi-k2.6:cloud` (config: `FORGE_JUDGE_MODEL`).
- **Input:** full transcript for one source, with per-segment `start_s`/`end_s` and word timings already in
  `transcript_segments`. A 45-min interview ≈ ~10K tokens — a single call, no chunking needed for typical
  lengths. (Add windowed batching only if a source exceeds the model's context.)
- **Output (strict JSON, validated):** array of moments —
  ```json
  {
    "start_s": 412.3, "end_s": 447.8,
    "score": 87,
    "hook": "the exact line that opens the clip",
    "emotion": "vulnerable | hot-take | origin-story | quotable | funny | aspirational",
    "reason": "one sentence: why this pops",
    "caption": "suggested social caption"
  }
  ```
- **Robustness:** Kimi is historically shakier than Claude on strict JSON. Mitigate with:
  (a) `format: json` in the Ollama request, (b) a strict schema + reject-and-retry on parse failure
  (max 2 retries), (c) timestamp-snap to nearest word boundary using existing word data so model drift on
  decimals can't produce a bad cut.

## 5. The rubric (music/artist-tuned — NOT generic)

The judge is prompted as a **viral short-form editor for a music artist's social team**. It scores each
candidate moment on:

1. **Hook strength (first 3s)** — does the opening line stop the scroll?
2. **Emotional intensity** — vulnerable admissions, origin-story beats, raw moments (Rod Wave's lane).
3. **Quotability** — a line a fan would screenshot or stitch.
4. **Story completeness** — setup → payoff inside the clip; no dangling context.
5. **Standalone clarity** — makes sense with zero prior context.
6. **Controversy / hot-take** — opinion that drives comments (without being a liability).

Tuned for what pops on *artist interviews*, which a generic talking-head model gets wrong. The rubric lives in
a single prompt template so it's tunable monthly against real numbers (Phase 2).

## 6. New components

| Component | File | Notes |
|---|---|---|
| Scorer | `core/forge/scorer.py` | NEW. Ollama call, JSON parse + retry, word-boundary snap, returns ranked candidates. |
| Storage | `core/forge/jobs.py` (schema) | NEW table `clip_candidates` (`source_id`, `start_s`, `end_s`, `score`, `hook`, `emotion`, `reason`, `caption`, `judge_model`, `created_at`, `rendered`, `posted`). |
| Job handler | `core/forge/handlers.py` | NEW `score_source` job type → runs scorer, writes candidates. Auto-enqueue after transcription completes. |
| API | `core/api/forge.py` | NEW `POST /forge/sources/{id}/score`, `GET /forge/sources/{id}/candidates?sort=score`. |
| UI | `services/forge-web/` | NEW **Auto-Clips** tab: ranked grid, score badge, hook + reason, emotion chip, one-click → topic_clip render. |
| Calibration | `core/forge/intel.py` | SMALL add: join `posted` candidates → actual engagement; calibration report endpoint. |

## 7. Phases

### Phase 1 — LLM-judge scorer (MVP) — *ship first*
- `scorer.py` + `kimi-k2.6:cloud` call + JSON validation/retry + word-snap.
- `clip_candidates` table + `score_source` job (auto-fires after transcribe).
- Auto-Clips API endpoints.
- Auto-Clips UI tab: ranked grid → one-click into existing render.
- **Deliverable:** ingest a long Rod Wave interview → get an Opus-Clip-style ranked grid of scored moments,
  each renderable into a captioned 9:16 clip in one click. Feels like Opus Clip to Jordan, costs $0/video.

### Phase 2 — Close the loop (instrumentation)
- Log every rendered/posted candidate with its predicted score.
- `intel.py` joins predicted score → actual engagement.
- Calibration view: is the score predictive? Tune the rubric against real Rod Wave numbers.

### Phase 3 — Trained ranker (only once data justifies — likely post-tour)
- After a few hundred posted clips, train a lightweight ranker on features
  (window embedding + LLM sub-scores + hook/audio energy) to predict engagement; blend with the Kimi judge.
- This is the proprietary moat — trained on Mainstay's *own* audience, uncopyable by generic tools.

## 8. Explicitly out of scope (this phase)
- Face-tracking reframe (separate gap; still center-crop). Worth its own phase later.
- Visual/audio-energy scoring (Phase 3 territory).
- Auto-posting (stays draft handoff by design).

## 9. Open items for the builder
- Confirm `kimi-k2.6:cloud` honors Ollama `format: json` cleanly; if not, fall back to strict prompt + parse-retry.
- Decide candidate count default (start: top 20 per source, operator-adjustable).
- Confirm context headroom for the longest expected sources; add windowed batching only if needed.

## 10. Cost summary
- **Per video:** $0 marginal (Kimi runs on the existing Ollama Cloud subscription).
- **GPU:** untouched (cloud judge; local 3090 stays on Whisper + TTS).
- **No Claude API. No new keys. No new infra.**
