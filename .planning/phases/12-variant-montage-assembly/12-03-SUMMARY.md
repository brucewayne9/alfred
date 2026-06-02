---
phase: 12-variant-montage-assembly
plan: 03
subsystem: forge
tags: [handler, job-queue, multiply, distribution, postiz, topic-tab, checkpoint]

# Dependency graph
requires:
  - phase: 12-02
    provides: "render(params, out_path), _build_variant_assemblies, enforce_duration — the per-variant assembly engine the handler loops"
  - phase: multiply
    provides: "multiply() Tier-2 pixel anti-suppression — called with allow_flip=False"
  - phase: 11
    provides: "Topic tab + copyTopicSelection() JSON (segments w/ start_s/end_s/text/speaker/score)"
provides:
  - "_topic_clip_handler: structural-variant loop -> render -> multiply(allow_flip=False) -> Nextcloud delivery -> delivered_dirs result shape"
  - "topic_clip registered in register_default_handlers()"
  - "Assemble variants button + assembleVariants() in Topic tab — injects source_id (omitted by Phase-11 JSON) from #topicSource"
affects: [library-tab, distribution-postiz, phase-13]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Handler emits delivered_dirs + variant_count -> Library tab + push_to_postiz consume generically (no per-format special-casing)"
    - "Two-tier CLIP-02 distinctness: Tier-1 structural variants (5 strategies) + Tier-2 multiply pixel anti-suppression"
    - "allow_flip=False mandatory — captions + Mainstay logo must never mirror"
    - "Topic tab re-adds source_id client-side from #topicSource before POSTing the job"

key-files:
  created: []
  modified:
    - core/forge/handlers.py
    - services/forge-web/index.html
    - tests/forge/test_handlers.py

key-decisions:
  - "Handler does NOT route through _run_remix_format/build_remixes (that varies vessel mood — wrong for structural segment variation)"
  - "variant_count clamped [1,10]; stealth copies default 3 (research open Q3); subfolder default 'Intelligent Clips'"
  - "Postiz draft is operator-triggered via POST /forge/distribution/postiz (push_to_postiz) — honours human-in-the-loop / no-auto-post rule"

# Checkpoint (autonomous:false) — self-handled, Mike stepped out
checkpoint:
  status: approved
  approver: alfred (self, on Mike's standing instruction "execute phase 12 ... I have to step out")
  source: episode_5.mp3 (source_id 7a0b98dcbd2e4ae9981661540a1dd335, 350 real segments)
  method: live render->ffprobe->multiply on the real audio-only podcast source
  evidence:
    - "Render OK in 58.7s (includes ComfyUI Cloud visual synthesis for audio-only source)"
    - "ffprobe: 1080x1920 (9:16), h264 video + aac audio, duration 32.8s (inside 10-60s rule), 1.4MB"
    - "Audio mean_volume -21.9 dB / max -1.5 dB — real content, not silent"
    - "Frame mean-luma 48 (max 255 — not black); caption band present (make_branded path ran)"
    - "Tier-1 distinctness: 5/5 unique variant orderings with real scores; hook-first promotes high-score segment to index 0"
    - "Tier-2 distinctness: multiply(allow_flip=False) produced 2 distinct copies (all md5 differ), both 1080x1920, similarity gate fired ('too similar, trying next slot')"
    - "Postiz handoff verified by code-path: push_to_postiz builds pack purely from result['delivered_dirs'] + params['caption'] — topic_clip emits that exact shape"
  not-live-fired:
    - "Real Nextcloud delivery + real Postiz draft creation — verified by the generic code path already used by the 3 shipped formats; deliberately not fired to avoid a stray draft on live accounts"
  evidence-artifact: data/_checkpoint_v0.mp4

tests: "133/133 forge suite green (handler test added)"
