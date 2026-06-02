# Phase 12: Variant Montage Assembly — Research

**Researched:** 2026-06-02
**Domain:** ffmpeg segment concatenation, audio-only-to-vertical pipeline, multiply.py extension, Forge job/handler/delivery pattern
**Confidence:** HIGH — all findings verified directly from source code; no stale training data relied on for any architectural claim

---

## Summary

Phase 12 sits squarely inside the existing Forge infrastructure. The job/handler/delivery/distribution/Postiz pipeline is fully built and proven across three formats (leak_graphic, kinetic_lyric, film_montage). Phase 12 adds a fourth handler — `topic_clip` — that receives a `source_id` and the Phase-11 Copy-selection JSON, cuts segments from the on-disk source file using ffmpeg, and routes output through the existing delivery + distribution chain.

The two genuine technical problems to solve are: (1) audio-only sources (e.g. `episode_5.mp3`) have no video track — a 9:16 vertical must be synthesised (kinetic caption overlay on black/waveform/generated background); and (2) multiply.py currently differentiates variants by *pixel-level visual transforms on a master render*, not by *segment reorder/recut/recaption* — CLIP-02 requires a meaningful structural variation strategy, which means producing multiple distinct cut assemblies before multiply.py does its stealth-uniqueness pass on each.

**Primary recommendation:** Build a new `core/forge/renderers/topic_clip.py` renderer that (a) cuts and concatenates transcript segments directly from the ingested source file with full audio, (b) overlays ffmpeg `drawtext` captions derived from the segment `text` field, (c) brands with the Mainstay logo via the existing `make_branded` helper in `film_montage.py`, and (d) generates multiple variation assemblies before handing each to `multiply.py`. Wire it through a new `topic_clip` handler registered in `handlers.py` following the `_run_remix_format` pattern.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLIP-01 | Forge cuts the matched segments out of the source into clips with audio and video in sync | `source.file_path` in sources table is the on-disk path; `audio.clip_audio` + ffmpeg `filter_complex concat` handles both video and audio tracks; audio-only path requires synthesised visual (see Audio-Only Vertical below) |
| CLIP-02 | Forge generates multiple distinct variants from the matched segments (different segment order, cut points, and captions) | multiply.py differentiates a *single master* at the pixel level — it does NOT reorder/recut; CLIP-02 requires generating multiple distinct *assembly masters* first, then multiply.py does anti-suppression uniqueness per master |
| CLIP-03 | Each variant is branded in the Mainstay style (logo + caption) as a 9:16 vertical | `make_branded()` in `film_montage.py` already handles logo-overlay + audio mux at 1080x1920; `drawtext` filter for captions; `LOGO_PATH` already points to `/home/aialfred/remotion/public/mainstay-logo.png` |
| CLIP-04 | Variants deliver to the library and can be handed to Postiz as drafts | `delivery.deliver()` + `distribution.push_to_postiz()` already exist and are proven; `_run_remix_format` in `handlers.py` is the exact pattern to follow |
</phase_requirements>

---

## Hand-off Contract from Phase 11

The `copyTopicSelection()` function in `index.html` (line 1075–1091) produces this exact JSON shape:

```json
[
  {
    "start_s": 142.3,
    "end_s": 178.9,
    "text": "you really can't sleep when your body is in recovery mode...",
    "speaker": "A",
    "score": 0.79
  }
]
```

Fields confirmed from source:
- `start_s` — `s.trimIn` (operator-adjusted in-point, float seconds)
- `end_s` — `s.trimOut` (operator-adjusted out-point, float seconds)
- `text` — full segment transcript text
- `speaker` — "A" or "B" or null
- `score` — raw cosine similarity score (0.0–1.0)

The source file is NOT in this JSON. The handler must receive `source_id` separately so it can look up `source.file_path` via `ingest.get_source(source_id)`.

---

## What Already Exists (Reuse, Do Not Rebuild)

### Core infrastructure — zero changes needed

| File | What it provides | How Phase 12 uses it |
|------|-----------------|----------------------|
| `core/forge/jobs.py` | `enqueue()`, `register_handler()`, `worker_loop()`, `check_cancel()` | New `topic_clip` job type drops straight in |
| `core/forge/handlers.py` | `register_default_handlers()`, `_run_remix_format()` pattern | `_topic_clip_handler()` follows exact same shape as `_film_montage_handler()` |
| `core/forge/delivery.py` | `deliver(local_path, subfolder, filename)` | Upload each assembled variant to Nextcloud |
| `core/forge/distribution.py` | `push_to_postiz()`, `build_pack()`, `assign_posts()`, `build_caption()` | Postiz draft flow is unchanged |
| `core/forge/library.py` | `list_done_jobs()`, `list_dir_files()`, `read_file()` | Library tab shows `topic_clip` jobs alongside other formats |
| `core/forge/audio.py` | `duration_seconds()`, `clip_audio()`, `assert_audible()` | Duration checks, audio cutting, audibility guard |
| `core/forge/multiply.py` | `multiply(master, count, out_dir, allow_flip=False)` | Anti-suppression uniqueness pass on each assembled variant master |
| `core/forge/ingest.py` | `get_source(source_id)`, `get_segments(source_id)` | Resolve `file_path` and load full transcript for caption timing |
| `core/forge/db.py` | `sources`, `transcript_segments` tables | No schema changes needed |
| `core/api/forge.py` | `POST /forge/jobs`, `GET /forge/distribution/postiz` | No new endpoints needed for CLIP-01..04 |
| `services/forge-web/index.html` | Topic tab already has source_id + Copy-selection JSON | Needs a new "Assemble variants" button and variant count input |

### make_branded() — reuse directly

`film_montage.py:make_branded(body, hook, caption, out_path)` already:
- Overlays logo at `W-w-40:H-h-60` bottom-right (150px scaled)
- Muxes audio from a separate `hook` file
- Falls back gracefully if logo is absent
- Outputs libx264 + aac 192k + yuv420p

For `topic_clip`, the "hook" audio is the concatenated segment audio extracted from the source — same API shape.

### multiply.py — use as-is, with allow_flip=False

`multiply(master, count, out_dir, allow_flip=False)` produces N pixel-distinct variants of a rendered master. Pass `allow_flip=False` because caption text would be mirrored otherwise. Already used this way for leak_graphic.

The 34-test suite covers this path. Do not touch multiply.py.

---

## What Phase 12 Must Build

### New file: `core/forge/renderers/topic_clip.py`

This is the only new core module. Everything else is wiring.

Key functions:
1. `cut_segment(src_path, start_s, end_s, out_path, has_video)` — ffmpeg cut of a single segment with audio; handles video and audio-only sources differently
2. `concat_segments(seg_paths, out_path, has_video)` — concatenate non-contiguous segments into one file
3. `overlay_captions(body_path, segments, out_path)` — drawtext filter for speaker-attributed caption text
4. `assemble_variant(source_path, segments, variant_params, out_path, work_dir)` — full pipeline for one variant master
5. `render(params, out_path)` — top-level renderer following the same signature as `film_montage.render(params, out_path)`

### Registration in `handlers.py`

Add `_topic_clip_handler()` following the exact `_run_remix_format` pattern, registered as `"topic_clip"`.

### UI additions in `index.html`

Add to the Topic tab:
- Variant count input (default 3, max 10, min 1)
- Caption style picker (optional — can default to bottom-third text)
- "Assemble variants" button that posts `{job_type: "topic_clip", params: {source_id, segments: [...], variant_count, caption, subfolder}}`

---

## Technical Risk 1 — Audio/Video Sync (CLIP-01)

### Source types on disk

From `ingest.py` and `db.py`: `sources.file_path` is the local path set during ingest. Confirmed from Phase 10 testing that `episode_5.mp3` is an audio-only source (`.mp3` extension). Video sources are `.mp4`.

**Detection:** `ffprobe -select_streams v:0` returns empty stdout for audio-only files. The `_has_audio()` helper in `multiply.py` already does this pattern for audio — mirror it for video.

### Cutting a single segment

For both video and audio-only, the ffmpeg cut pattern is:

```bash
# Seek BEFORE -i for keyframe-accurate fast seek, then re-encode the cut for
# sample-accurate trim. Using -ss before -i then -ss 0 after is the reliable
# 2-pass pattern for non-contiguous segments.
ffmpeg -y -v error \
  -ss {start_s} -i {src} \
  -t {duration} \
  -c:v libx264 -preset veryfast -crf 23 \
  -c:a aac -b:a 192k \
  {out}.mp4
```

For audio-only sources: add `-vn` and output `.mp3` (or `.aac`) for the audio track. The visual track is synthesised separately (see Audio-Only Vertical below).

### Concatenating non-contiguous segments — filter_complex vs demux concat

**Critical finding:** The ffmpeg `concat` demuxer (the `file '{path}'` concat.txt approach used in `film_montage.py`) works correctly ONLY if all input files have identical codec parameters. For segments cut from the same source with re-encode (`-c:v libx264 -preset veryfast -crf 23`), they will be uniform — the film_montage approach (concat.txt + `-f concat`) is safe and already proven in this codebase.

The `filter_complex concat` alternative (`[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1`) is more robust for mixed inputs but scales poorly with segment count (filter string grows with N). **Recommendation:** Re-encode each cut segment to uniform params, then use the concat demuxer (same as `film_montage.py` lines 162–171). This avoids A/V drift because the re-encode locks timestamps.

**Standing duration constraint:** Before concatenating, sum `(end_s - start_s)` for all kept segments. If total < 10s, raise an informative error. If total > 60s, trim the last segment to fit the 60s cap.

### Audio-Only Vertical (the hardest CLIP-01 sub-problem)

When the source is audio-only (e.g. `episode_5.mp3`), there is no video track to cut. The 9:16 vertical needs a synthesised visual. Three options, in order of complexity:

| Option | Approach | Cost | Quality |
|--------|----------|------|---------|
| A — Black background + drawtext captions | Pure ffmpeg: `color=black:s=1080x1920` + `drawtext` for the transcript text | Zero extra deps | Functional, minimal |
| B — Static ComfyUI background + captions | Generate one topic-keyed still image via `comfyui_gen.py`, use it as the video layer, overlay drawtext | 1 ComfyUI call per variant | On-brand |
| C — Waveform visualiser | `avectorscope` or `showwaves` filter | Pure ffmpeg | Distinctive but not Mainstay-branded |

**Recommendation:** Option B for audio-only sources. Use `comfyui_gen.py generate "{topic_from_caption}"` (project rule: never inline PIL). The generated image is a static background scaled to 1080x1920. This is consistent with how `film_montage.py` generates vessel clips via `clips.generate_clip()`. The `caption` param already present in the job params provides the topic seed.

ffmpeg command for audio-only assembly:
```bash
ffmpeg -y -v error \
  -loop 1 -i {background_image} \
  -i {concatenated_audio} \
  -c:v libx264 -preset veryfast -crf 23 \
  -c:a aac -b:a 192k \
  -vf "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,format=yuv420p" \
  -shortest \
  {body.mp4}
```

Then pipe through `make_branded(body, hook_audio, caption, out_path)` as-is.

---

## Technical Risk 2 — Variant Distinctness (CLIP-02)

### What multiply.py currently does

`multiply.py` takes ONE master video and produces N pixel-distinct copies via zoom/crop/flip/rotate/speed transforms. It does NOT change segment order, cut points, or caption text. It is a visual anti-suppression layer — all N copies tell the same story, just slightly different pixels.

### What CLIP-02 requires

"Different segment order, cut points, and captions" — these are STRUCTURAL variations, not pixel tweaks. multiply.py cannot produce these. The solution is a two-tier system:

**Tier 1 — Structural variation (new, per-variant assembly):**
Produce `variant_count` distinct assemblies before touching multiply.py:

| Variant | Segment strategy |
|---------|-----------------|
| V0 — Original order | Segments in the order the operator curated (Phase-11 JSON order) |
| V1 — Reverse order | Segments concatenated last-to-first |
| V2 — Hook-first | Best-scoring segment moved to position 0 |
| V3 — Trimmed cut points | Each segment trimmed by +/- 1–2s at random (within bounds) |
| V4+ — Random shuffles | If variant_count > 4, additional random permutations seeded by variant index |

Each structural variant produces one master `.mp4`.

**Tier 2 — Pixel uniqueness (existing multiply.py):**
Each master is fed through `multiply(master, stealth_copies, work_dir, allow_flip=False)` to produce platform-ready uniqueness copies. The `stealth_copies` count can be small (3–5) since structural variation is the primary differentiator.

Caption text also varies per structural variant: V0 uses the segment `text` verbatim, V1–V4 can shorten to the key sentence or use a different line of the same segment.

### Duration enforcement across variants

```python
def enforce_duration(segments, min_s=10.0, max_s=60.0):
    total = sum(s['end_s'] - s['start_s'] for s in segments)
    if total < min_s:
        raise ValueError(f"total segment duration {total:.1f}s < {min_s}s minimum")
    if total > max_s:
        # trim last segment back
        over = total - max_s
        segments = list(segments)
        last = dict(segments[-1])
        last['end_s'] = max(last['start_s'] + 1.0, last['end_s'] - over)
        segments[-1] = last
    return segments
```

---

## Technical Risk 3 — Captions/Branding (CLIP-03)

### make_branded() — what it already does

From `film_montage.py:80–109`: overlays logo (150px, bottom-right, `W-w-40:H-h-60`), muxes audio. **The existing function skips caption drawtext with the comment: "apostrophes/quotes break ffmpeg drawtext and would risk the whole render."**

This is the correct conservative choice for the existing use case (caption from user input, arbitrary text). For Phase 12, the transcript `text` field can contain apostrophes and special chars.

**Recommendation:** Build a separate caption-overlay step BEFORE `make_branded()`:

```python
def overlay_captions(body_path, caption_text, out_path, font_size=52):
    """Overlay bottom-third caption. Escape apostrophes for ffmpeg drawtext."""
    safe = caption_text.replace("'", "’").replace(":", "\\:").replace("\\", "\\\\")
    # Wrap to ~40 chars per line using textwrap, then join with '\n'
    # drawtext uses \n for line breaks when fix_bounds=1
    ...
```

The safe approach is to sanitise the text before drawtext: replace `'` with curly apostrophe `'` (U+2019), escape `:` and `\`. Keep lines <= 40 chars to fit 1080px at 52px font.

Font to use: the project uses Hanken Grotesk for UI. For ffmpeg drawtext, embed the font file path. Check `/usr/share/fonts/` or use the system default (`fontfile` param). If font resolution is fragile across environments, use `font=Helvetica` as a safe fallback.

**Caption positioning for interview clips:** Bottom-third, white text with black stroke/drop-shadow is the standard vertical clip pattern. ffmpeg drawtext params:
```
drawtext=fontsize=52:fontcolor=white:bordercolor=black:borderw=3:
  x=(w-text_w)/2:y=h-text_h-120:
  text='{safe_text}'
```

### 9:16 aspect enforcement

From `film_montage._cut_segment()`: the existing scale filter is `scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,format=yuv420p`. Use the identical filter for CLIP-03. Works for widescreen 16:9 source video (letterbox → pillarbox → crop).

### Logo path

`LOGO_PATH = Path("/home/aialfred/remotion/public/mainstay-logo.png")` — hardcoded in `film_montage.py`. Phase 12 should import this constant rather than redeclare it.

---

## Technical Risk 4 — Delivery + Postiz Handoff (CLIP-04)

### Delivery path

`delivery.deliver(local_path, subfolder, filename)` uploads to `Content/Mainstay-RodWave/{subfolder}`. Proposed subfolder: `"Intelligent Clips/{job_timestamp}"` or `"Intelligent Clips/{source_name}"` — consistent with the existing `Viral Music Verticals/Film Montage` pattern.

### Library integration

`library.list_done_jobs()` already reads `result.delivered_dirs` and `result.format` to build library cards. The `topic_clip` handler result must emit `{"format": "topic_clip", "delivered_dirs": [...], "delivered": N, "variant_count": N}`. The `variations_each` key in `list_done_jobs()` falls back to `res.get("variant_count")` — so naming it `variant_count` (not `variations_each`) in the result dict covers the library display.

### Postiz draft flow

`distribution.push_to_postiz(job_id)` calls `build_pack(job_id)`, which reads `job.result.delivered_dirs`, lists Nextcloud files, and calls `postiz.py create-draft`. This is fully format-agnostic — it works for any job type that emits `delivered_dirs`. Phase 12 gets Postiz for free.

**Auto-post constraint:** `push_to_postiz` creates DRAFTS only — `postiz.py create-draft` is the call, not `publish`. The UI "Push to Postiz (drafts)" button is the human trigger. This constraint is already built into the distribution module.

**Caption for Postiz:** `distribution.build_caption(hook, platform)` appends `#RodWave #DontLookDown #fyp` etc. The `hook` param should be the first segment's `text` field (truncated to ~120 chars) for the topic_clip format.

---

## Architecture Patterns

### Recommended new module structure

```
core/forge/renderers/
├── film_montage.py    # existing — unchanged
├── kinetic_lyric.py   # existing — unchanged
├── leak_graphic.py    # existing — unchanged
└── topic_clip.py      # NEW — Phase 12
```

### topic_clip.py internal design

```
render(params, out_path)
  └── _resolve_source_file(source_id)           # ingest.get_source() → file_path
  └── _detect_has_video(file_path)              # ffprobe video stream check
  └── _build_variant_assemblies(segments, n)   # structural variation strategies
      └── enforce_duration(segments)
      └── _cut_segment(src, start_s, end_s, out)
      └── _concat_segments(seg_paths, out)
  └── _synthesise_visual(audio_path, caption)   # audio-only: comfyui_gen.py bg
  └── overlay_captions(body, caption_text, out)
  └── make_branded(body, audio, caption, out)   # import from film_montage
  └── multiply(master, stealth_n, out_dir,      # anti-suppression layer
               allow_flip=False)
  └── delivery.deliver(variant, subfolder)
```

### Handler registration pattern (handlers.py)

```python
def _topic_clip_handler(params: dict) -> dict:
    from core.forge.renderers.topic_clip import render
    return _run_remix_format(render, params, fmt="topic_clip",
                             default_subfolder="Intelligent Clips")
```

**Wait — important gap:** `_run_remix_format` calls `build_remixes(params, remix_count)` which varies the `vessel_prompt`. For `topic_clip`, the visual variation is structural (segment order) not vessel mood. The `_run_remix_format` wrapper is close but calls `remix.build_remixes()` before the renderer. The renderer itself must handle structural variation internally, while `_run_remix_format` handles multiply + delivery. This means the `render()` function for `topic_clip` should accept variant_index or similar in params if called through `_run_remix_format`, OR the handler bypasses `_run_remix_format` and implements the variant loop itself. Recommend the latter for clarity: write a custom `_topic_clip_handler` that iterates variant assemblies directly, calling `multiply()` and `delivery.deliver()` per variant. This avoids awkwardly repurposing the remix system.

---

## Common Pitfalls

### Pitfall 1: Seeking accuracy on long-form sources

**What goes wrong:** `ffmpeg -ss {start} -i file.mp3` with copy codec (`-c:a copy`) uses keyframe seek — the actual start may be up to 1–2 seconds before the requested timestamp. For speech clips this produces a few words of bleed from the previous segment.

**Prevention:** Always re-encode when cutting segments: use `-c:a aac -b:a 192k` (not `-c:a copy`). For video: `-c:v libx264 -preset veryfast`. Re-encode ensures sample-accurate cuts. Confirmed from `film_montage._cut_segment()` — it never uses copy codec.

### Pitfall 2: Concat demuxer requires uniform params

**What goes wrong:** concat.txt approach fails silently or produces corrupt output if input segments have different frame rates, sample rates, or pixel formats.

**Prevention:** All cut segments are re-encoded to the same params (libx264, 30fps, yuv420p, aac 192k). Check that `_cut_segment` always specifies `-pix_fmt yuv420p -fps_fps 30` for video segments. Audio-only segments: normalize to `-ar 44100 -ac 2`.

### Pitfall 3: drawtext apostrophe crash

**What goes wrong:** Interview transcript text is full of contractions (`it's`, `I've`, `can't`). Unescaped single quotes in ffmpeg `drawtext=text='...'` cause the filter graph to fail.

**Prevention:** Sanitise before drawtext: `text.replace("'", "’")`. Also escape `:` → `\:` and `\` → `\\`. Build a `_safe_drawtext(text)` helper and test it with a fixture containing apostrophes.

### Pitfall 4: source_id not in the Copy-selection JSON

**What goes wrong:** The Phase-11 hand-off JSON (`copyTopicSelection()`) does NOT include `source_id`. If the UI sends only the segments array, the handler has no path to the source file.

**Prevention:** The "Assemble variants" button must POST `{source_id: currentTopicSourceId, segments: [...], variant_count: N, caption: str}`. The `currentTopicSourceId` is the value of `#topicSource` at submit time. Document this in the UI JS explicitly.

### Pitfall 5: multiply.py hflip on text-bearing clips

**What goes wrong:** If `allow_flip=True` (the default), multiply.py applies `hflip` to half the variants, mirroring any text overlays (captions, logos).

**Prevention:** Always call `multiply(master, n, out_dir, allow_flip=False)` for topic_clip. This is the same pattern already used for `leak_graphic`.

### Pitfall 6: Duration guard before concat, not after

**What goes wrong:** Enforcing the 60s cap AFTER cutting all segments wastes time (the last cut is thrown away or re-cut).

**Prevention:** Calculate `total_s = sum(seg['end_s'] - seg['start_s'] for seg in segments)` before any ffmpeg calls. Adjust the last segment's `end_s` down to fit the cap. Then cut.

### Pitfall 7: Audio-only ComfyUI call blocks the worker

**What goes wrong:** `comfyui_gen.py generate` can take 30–90 seconds. If called synchronously inside the renderer, the worker_loop is blocked from processing other jobs (it uses `loop.run_in_executor` which runs the handler in a thread, so other async work proceeds, but a second forge job won't start until this one finishes).

**Impact:** This is the existing model for all renderers (film_montage also calls ComfyUI). Not a new problem. Acceptable for Phase 12. Document for Phase 13 flow if ComfyUI is expected to be heavily used.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Anti-suppression pixel uniqueness | Custom ffmpeg filter loop | `multiply.py` — 34 tests, proven |
| Nextcloud upload | Direct WebDAV calls | `delivery.deliver()` |
| Postiz draft creation | Direct Postiz API | `distribution.push_to_postiz()` |
| Library indexing | Custom job listing | `library.list_done_jobs()` + existing `result.variant_count` key |
| Job queue + async worker | asyncio task management | `jobs.enqueue()` + `worker_loop()` |
| Logo overlay branding | New ffmpeg overlay | `film_montage.make_branded()` — import directly |
| Image generation for audio-only bg | Inline PIL/Pillow | `comfyui_gen.py generate "prompt"` (project rule, non-negotiable) |

---

## Recommended Plan / Wave Breakdown

Based on CLIP-01..04 dependencies and the test-first pattern established in Phases 10–11:

### Wave 1 — Segment cutting + concat engine (CLIP-01 core)
- `core/forge/renderers/topic_clip.py`: `_detect_has_video`, `_cut_segment`, `_concat_segments`, `enforce_duration`
- Tests: synthetic segments cut from a test mp3/mp4 source, concat output duration check, duration enforcement edge cases
- No branding yet, no handler registration

### Wave 2 — Structural variant assembly + caption overlay (CLIP-01 complete + CLIP-02 + CLIP-03)
- `assemble_variant()` with the 5 structural strategies
- `overlay_captions()` with drawtext + `_safe_drawtext()` sanitiser
- `render()` full pipeline for video sources: cut → concat → captions → `make_branded()`
- Audio-only path: `_synthesise_visual()` using `comfyui_gen.py`
- Tests: variant order strategies produce different segment sequences; caption text with apostrophes renders without crash

### Wave 3 — Handler registration + delivery + Postiz (CLIP-04)
- `_topic_clip_handler()` in `handlers.py` + `register_default_handlers()` call
- Result dict shape: `{"format": "topic_clip", "delivered_dirs": [...], "delivered": N, "variant_count": N}`
- UI: "Assemble variants" button on Topic tab + variant count input
- Tests: handler enqueue → job runs → delivered_dirs populated; `build_pack` picks up delivered files; Postiz push returns skipped (no postiz_id in test) without error

---

## State of the Art

| Old Approach | Current Approach | Notes |
|--------------|-----------------|-------|
| ffmpeg `filter_complex concat` | ffmpeg concat demuxer (concat.txt) | concat demuxer is simpler at scale; filter_complex grows O(N); existing codebase uses concat demuxer in film_montage.py |
| Separate audio + video pipelines | Unified re-encode at cut time | film_montage pattern; re-encoding guarantees uniform params for concat |
| Separate multiply invocation per format | `_run_remix_format` wrapper in handlers.py | Phase 12 should replicate the handler structure but implement variant loop directly, not via `build_remixes()` which was designed for vessel-mood variation |

---

## Open Questions

1. **Caption font availability**
   - What we know: ffmpeg drawtext needs a font file path or name; Hanken Grotesk is loaded via Google Fonts in the browser only
   - What's unclear: is Hanken Grotesk installed as a system font on the 105 server?
   - Recommendation: `fc-list | grep -i hanken` at plan time; if absent, fall back to `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf` and document substitution

2. **Caption strategy for multi-segment clips**
   - What we know: concatenating 3 segments produces a 30–50s clip; showing the full text of each segment simultaneously is too much
   - What's unclear: should captions track segment boundaries (show segment N text while segment N plays), or show just a single hook sentence?
   - Recommendation: default to showing the first sentence of each segment while it plays; timestamps from the word-level `words` JSON already stored in `transcript_segments` can drive timed drawtext if desired (Phase 13 stretch)

3. **Stealth multiply count per structural variant**
   - What we know: existing formats use 18 `variations` (stealth copies per look) for anti-suppression; for topic clips the structural variation is the primary differentiator
   - What's unclear: how many stealth copies per structural variant to generate (fewer = smaller batch, more = better anti-suppression)
   - Recommendation: default to 3 stealth copies per structural variant (`variations=3`, overridable), total delivered = `variant_count * 3`

---

## Sources

### Primary (HIGH confidence — direct source code inspection)
- `/home/aialfred/alfred/core/forge/renderers/film_montage.py` — make_branded, _cut_segment, concat pattern, LOGO_PATH
- `/home/aialfred/alfred/core/forge/multiply.py` — full multiply() API, allow_flip param, dHash uniqueness
- `/home/aialfred/alfred/core/forge/clips.py` — resolve_source, fetch_source, _has_audio pattern (no; this is in multiply.py)
- `/home/aialfred/alfred/core/forge/delivery.py` — deliver() signature and Nextcloud path
- `/home/aialfred/alfred/core/forge/distribution.py` — push_to_postiz(), build_pack(), build_caption(), PLATFORMS list
- `/home/aialfred/alfred/core/forge/handlers.py` — _run_remix_format, all handler registrations
- `/home/aialfred/alfred/core/forge/jobs.py` — enqueue, register_handler, worker_loop, check_cancel
- `/home/aialfred/alfred/core/forge/ingest.py` — get_source(), get_segments(), source file_path resolution
- `/home/aialfred/alfred/core/forge/db.py` — sources and transcript_segments schema
- `/home/aialfred/alfred/core/forge/library.py` — list_done_jobs(), delivered_dirs/variant_count key
- `/home/aialfred/alfred/core/api/forge.py` — POST /forge/jobs endpoint, distribution endpoints
- `/home/aialfred/alfred/services/forge-web/index.html` — copyTopicSelection() exact JSON shape (line 1075–1091), topic tab source_id picker, pushPostiz() button

### Primary (HIGH confidence — Phase 11 hand-off)
- `.planning/phases/11-topic-targeted-segment-retrieval/11-03-SUMMARY.md` — Copy-selection JSON shape confirmed: `[{start_s, end_s, text, speaker, score}]`

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all components verified from source code
- Architecture: HIGH — reuse pattern follows established handlers.py / _run_remix_format pattern
- Audio/video sync: HIGH — ffmpeg patterns confirmed from existing film_montage.py
- CLIP-02 variation strategy: MEDIUM — structural variation approach is sound but specific reorder strategies are recommendations; calibrate variant count with the team
- Pitfalls: HIGH — all verified from either existing code comments, git history, or direct pattern analysis

**Research date:** 2026-06-02
**Valid until:** Stable architecture — valid until film_montage.py or multiply.py change (neither is on the roadmap for Phase 12)
