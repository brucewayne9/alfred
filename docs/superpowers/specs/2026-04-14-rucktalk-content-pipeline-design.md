# RuckTalk Content Pipeline — Design Spec

## Overview

A unified content automation system for the RuckTalk brand. Three engines working together: an event-driven episode pipeline triggered by NextCloud file drops, a daily social engine that produces one high-quality post per morning, and a daily blog engine that publishes SEO-optimized articles to the RuckTalk WordPress site.

## Brand Context

**RuckTalk** — "Tactical Living for the Modern Entrepreneur"

**Core Pillars**: Entrepreneurship, hustle, motivation, health, family

**Sub-pillars**: Culture, lifestyle, opinion, sports commentary, hot takes

**Voice**: Direct, no-BS, like a mentor in the trenches. Raw, honest, motivating. No fluff, no corporate speak. A dad who runs businesses, stays fit, keeps family first — writing for others doing the same.

**Content Strategy**: Existing `rucktalk_strategy.json` defines 7 weighted pillars, 4 target personas, SEO requirements, and image style guidelines. This spec builds on that foundation.

---

## Engine 1: Episode Pipeline (Event-Driven)

### Trigger

A cron job runs every 10 minutes, polling the NextCloud folder `/RuckTalk/Episodes` via WebDAV for new MP4 files not listed in `/home/aialfred/rucktalk_pipeline/processed_files.json`.

### Pipeline Sequence

```
MP4 detected in NextCloud
    |
    v
1. Download MP4 via WebDAV to /home/aialfred/rucktalk_pipeline/incoming/
    |
    v
2. Extract audio (ffmpeg -> MP3)
    |
    v
3. Transcribe (Whisper, CPU, with timestamps)
    |
    v
4. AI Analysis of transcript:
    - Generate episode title
    - Generate description / show notes
    - Extract 5-7 best moments (timestamp ranges + summary)
    - Identify key quotes for social clips
    - Generate SEO keywords
    - Determine primary pillar category
    |
    v
5. Generate cover image (ComfyUI, cinematic style, no text)
    |
    v
6. Publish audio episode to WordPress:
    - Upload MP3 to media library
    - Upload cover image as featured image
    - Create episode post (custom post type or "Episodes" category)
    - Include: title, description, embedded audio player, show notes
    |
    v
7. Upload full video to YouTube:
    - Via youtube.py integration (Oracle/117)
    - Title, description, tags from step 4
    - Thumbnail from cover image
    |
    v
8. Create video page on WordPress:
    - YouTube embed (responsive iframe)
    - Episode title, description
    - Styled consistently — builds into a channel-like archive page
    |
    v
9. Generate episode blog post:
    - Auto-blogger transforms transcript into 1200+ word SEO article
    - 5+ ComfyUI images (cinematic, no text)
    - Featured image
    - Published to WordPress blog section
    |
    v
10. Smart clip generation:
    - AI identifies 5-7 highlight moments from transcript
    - Can stitch non-contiguous segments that form a coherent thought
    - ffmpeg cuts video at Whisper timestamp boundaries
    - Captions generated from transcript timestamps
    - Caption styling: bold, centered, word-by-word highlight
    - Remotion adds branded intro/outro bumpers
    - Rendered in portrait (1080x1920) for Shorts/Reels/TikTok
    - Rendered in landscape (1920x1080) for YouTube/LinkedIn
    |
    v
11. Queue social content:
    - Day 1: Full episode announcement (link to YouTube + website)
    - Days 2-7: One clip per day, scheduled via Postiz
    - Clips take priority over evergreen content in daily social engine
    |
    v
12. Mark processed + notify:
    - Add filename to processed_files.json
    - Telegram notification to Mike with summary of everything published
```

### Smart Clipping Detail

The intelligent clipping system works as follows:

1. Whisper produces a word-level timestamped transcript
2. AI (via Ollama/cloud LLM) reads the full transcript and identifies the strongest moments — complete thoughts, quotable takes, emotional peaks
3. Each "moment" is defined as one or more timestamp ranges. If a thought begins at 4:12 and its punchline is at 18:45 with filler in between, both ranges are captured and stitched together
4. ffmpeg extracts and concatenates the video segments with crossfade transitions
5. Captions are generated from the corresponding transcript segments, synchronized to the stitched video
6. Remotion overlays: branded lower third, intro bumper (1-2 seconds), outro with CTA
7. Output formats: portrait (Shorts/Reels/TikTok), square (IG Feed), landscape (YouTube/LinkedIn)

### File Structure

```
/home/aialfred/rucktalk_pipeline/
    processed_files.json          # tracks what's been handled
    pipeline.log                  # execution log
    incoming/                     # downloaded MP4s (cleaned after processing)
    audio/                        # extracted MP3s
    transcripts/                  # Whisper output (JSON with timestamps)
    clips/                        # generated video clips
    metadata/                     # AI-generated titles, descriptions, keywords (JSON per episode)
    images/                       # ComfyUI cover images
```

---

## Engine 2: Daily Social Engine (Cron, Every Morning)

### Schedule

One post per day, every morning. Exact time configured in cron (aligned with existing Postiz schedule — currently 2AM UTC / 10PM ET for overnight scheduling, goes live in the morning).

### Content Selection Logic

```
1. Check clip queue (from recent episodes)
   |
   ├── Clip available? -> Post the clip with caption
   |
   └── No clip? -> Generate original content
         |
         ├── ~40% chance: Current events mode
         |     - Search trending topics (via search.py / SearXNG)
         |     - Find angle that connects to a RuckTalk pillar
         |     - Write take in RuckTalk voice
         |
         └── ~60% chance: Evergreen mode
               - Pick pillar (weighted rotation from strategy)
               - Generate timeless content on that pillar
               - Motivational, tactical, or opinion-based
```

### Post Production (Non-Episode Days)

Every daily post is a **video**, not a static image. Production flow:

1. AI writes a 60-90 second script tied to the topic, in RuckTalk voice
2. Kokoro TTS (port 8880, voice: bm_daniel) narrates the script
3. ComfyUI generates 4-6 cinematic images matching the narration beats (no text in images)
4. video_render.py or Remotion stitches: images + narration + transitions + captions
5. Output in portrait (1080x1920) for Shorts/Reels/TikTok
6. Caption styling: bold, word-by-word highlight, synced to narration

### Post Production (Episode Clip Days)

When a clip from an episode is in the queue:

1. Clip is already rendered (portrait + landscape) from Episode Pipeline step 10
2. AI writes a caption in RuckTalk voice that teases the full episode
3. Include link to full episode (YouTube + website)

### Distribution

All posts go to: Facebook, Instagram, YouTube (Shorts/Community), LinkedIn, TikTok — via Postiz scheduling using existing integration IDs.

### Platforms

- Instagram: `cmmm0ck4m000iqudf4zkc2huz`
- Facebook: `cmmm0d0e8000kqudfvkjm6hzi`
- YouTube: `cmmm1r9n0000mqudfhdc436va`
- LinkedIn: `cmnd9rvnx003bqtnvd9n6z7c6`
- TikTok: via tiktok.py (draft mode)

---

## Engine 3: Daily Blog Engine (Cron, Weekdays)

### What It Does

Revives the existing `auto_blogger.py --site rucktalk --auto` pipeline that was previously running well.

### Spec

- Runs Monday through Friday
- 1200+ word SEO-optimized articles
- 5+ ComfyUI-generated images (cinematic, no text, no stock feel)
- Featured image generated per post
- Pillar rotation per `rucktalk_strategy.json` weights:
  - Entrepreneur (25%), Fitness (15%), Family (15%), Mindset (15%), Nutrition (10%), Faith (10%), Tactical (10%)
- Content mix: ~60% timeless pillar content, ~40% current-event-driven (topic ties back to a pillar)
- Published directly to WordPress (rucktalk site on server 100)
- Social promo scheduled via Postiz for the following day

### Existing Infrastructure

- Script: `~/.openclaw/workspace/scripts/integrations/auto_blogger.py`
- Strategy: `~/.openclaw/workspace/scripts/integrations/rucktalk_strategy.json`
- History: `~/.openclaw/workspace/scripts/integrations/rucktalk_history.json`
- WordPress: rucktalk site on server 100, managed via `wordpress.py`

---

## Engine Coordination

### Priority Rules

1. **Episode drop day**: Episode announcement becomes the daily social post. Blog engine still runs independently.
2. **Days after episode**: Queued episode clips take priority over generated content for daily social.
3. **No episode content available**: Daily social engine generates original narrated video content.
4. **Blog engine**: Always independent. Runs on its own weekday schedule regardless of episode activity.

### State Management

```
/home/aialfred/rucktalk_pipeline/
    processed_files.json     # Episode Pipeline: which MP4s have been processed
    clip_queue.json          # Episode Pipeline -> Social Engine: queued clips with metadata
    social_history.json      # Social Engine: what's been posted, pillar rotation tracking
```

Existing state files:
- `rucktalk_history.json` — Blog Engine post history (already exists)
- `rucktalk_strategy.json` — Content strategy config (already exists)

---

## Technical Stack

| Component | Tool | Location |
|-----------|------|----------|
| NextCloud polling | `integrations/nextcloud/client.py` (WebDAV) | Server 105 |
| Audio extraction | `ffmpeg` | Server 105 |
| Transcription | `whisper` (CPU) | Server 105 |
| AI analysis | Ollama (gemma4/minimax/kimi) via local endpoint | Server 105:11434 |
| Cover images | `comfyui_gen.py` (FLUX.1, --lowvram) | Server 105:8188 |
| TTS narration | Kokoro TTS (bm_daniel voice) | Server 105:8880 |
| Video rendering | `video_render.py` (ffmpeg/melt) | Server 105 |
| Animated video | `remotion_render.py` (RuckTalkPromo template) | Server 105 |
| YouTube upload | `youtube.py` | Oracle/Server 117 |
| WordPress publishing | `wordpress.py` (WP-CLI via SSH) | Server 100 |
| Blog generation | `auto_blogger.py --site rucktalk` | Oracle/Server 117 |
| Social scheduling | Postiz API / `postiz.py` | Server 117 |
| Trending topics | `search.py` (SearXNG) | Oracle/Server 117 |
| Notifications | Telegram (chat 7582976864) | Server 105 |

---

## Cron Schedule

| Job | Schedule | Description |
|-----|----------|-------------|
| Episode watcher | `*/10 * * * *` | Poll NextCloud for new MP4s |
| Daily social post | `0 11 * * *` (7AM ET) | Generate and schedule morning post |
| Daily blog post | `15 11 * * 1-5` (7:15AM ET) | Generate and publish weekday blog (existing cron) |

---

## New Code Required

1. **`scripts/rucktalk_episode_pipeline.py`** — Main episode pipeline orchestrator. Handles: NextCloud polling, download, audio extraction, transcription, AI metadata generation, WordPress publishing (audio + video pages), YouTube upload delegation, smart clip generation, social queue management, notifications.

2. **`scripts/rucktalk_daily_social.py`** — Daily social content engine. Handles: clip queue checking, trending topic research, evergreen content generation, TTS narration, ComfyUI image generation, video assembly, Postiz scheduling.

3. **Remotion template updates** — New `RuckTalkShort` template for narrated shorts (portrait, word-by-word captions, branded bumpers). Extends existing `RuckTalkPromo.tsx`.

4. **WordPress episode infrastructure** — Episode post type or category setup on the rucktalk WordPress site. Video archive page template with YouTube embeds.

---

## What Already Exists (Reuse)

- `auto_blogger.py` + `rucktalk_strategy.json` — Blog engine, ready to re-enable
- `comfyui_gen.py` — Image generation with RuckTalk style suffix
- `video_render.py` — Social video rendering with TTS integration
- `remotion_render.py` — RuckTalkPromo template
- `youtube.py` — YouTube upload via Oracle
- `wordpress.py` — WordPress management (16 sites including rucktalk)
- `postiz.py` — Social scheduling with all RuckTalk integrations connected
- `search.py` — SearXNG for trending topics
- `integrations/nextcloud/client.py` — WebDAV file operations
- `/home/aialfred/rucktalk_pipeline/` — Pipeline directory with state tracking

---

## GPU Considerations

ComfyUI and Kokoro TTS share the RTX 4070 (12GB). The pipeline should:

- Check GPU availability before generating images or TTS
- Queue image generation requests (not parallel)
- Stop Ollama if needed to free VRAM for ComfyUI (Ollama auto-restarts)
- Use `--lowvram` mode for ComfyUI (already configured)

---

## Error Handling

- If any step in the episode pipeline fails, log the error, notify Mike via Telegram with what failed, and do NOT mark the file as processed (so it retries next cycle)
- If daily social generation fails (GPU busy, API down), fall back to a text-only post with a strong quote from the strategy pillars
- Blog engine failures handled by existing auto_blogger retry logic

---

## Success Criteria

- Dropping an MP4 into NextCloud results in: YouTube upload, WordPress audio page, WordPress video page, blog post, 5-7 queued social clips — all within ~30 minutes
- One polished video post appears on all social platforms every morning
- One SEO blog post publishes every weekday
- Mike's only manual step is recording and dropping the MP4 into NextCloud
