# RuckTalk Content Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified content automation system where Mike drops an MP4 into NextCloud and everything else — YouTube upload, WordPress audio/video pages, blog post, smart clips, daily social videos, daily blog articles — happens automatically.

**Architecture:** Three engines sharing state via JSON files in `/home/aialfred/rucktalk_pipeline/`. Engine 1 (episode pipeline) watches NextCloud and orchestrates the full episode flow. Engine 2 (daily social) generates one narrated video post per morning. Engine 3 (daily blog) revives the existing auto_blogger for weekday SEO articles. An existing 762-line `rucktalk_pipeline.py` on Oracle handles basic episode processing already — we'll build the upgraded version on Server 105 (Alfred) where all the GPU tools live.

**Tech Stack:** Python 3, ffmpeg, Whisper (CPU), Kokoro TTS, ComfyUI (FLUX.1), Remotion, WordPress REST API via `wordpress.py`, YouTube via `youtube.py`, Postiz API, SearXNG via `search.py`, NextCloud WebDAV, Ollama LLMs

**Spec:** `docs/superpowers/specs/2026-04-14-rucktalk-content-pipeline-design.md`

---

## File Structure Overview

### New Files
| File | Responsibility |
|------|---------------|
| `scripts/rucktalk_episode_pipeline.py` | Episode pipeline orchestrator — NextCloud watcher, download, audio extraction, transcription, AI metadata, WordPress publishing, YouTube delegation, smart clip generation, social queue, notifications |
| `scripts/rucktalk_daily_social.py` | Daily social engine — clip queue, trending topics, evergreen content, TTS narration, ComfyUI images, video assembly, Postiz scheduling |
| `scripts/rucktalk_common.py` | Shared constants, state management, notification helpers, LLM calling, GPU helpers used by both scripts |
| `/home/aialfred/remotion/src/templates/RuckTalkShort.tsx` | Remotion template for narrated shorts — portrait 1080x1920, word-by-word captions, branded bumpers |
| `/home/aialfred/remotion/src/components/CaptionOverlay.tsx` | Word-by-word caption animation component for shorts |

### Modified Files
| File | Change |
|------|--------|
| `/home/aialfred/remotion/src/Root.tsx` | Register new RuckTalkShort composition |
| `/home/aialfred/rucktalk_pipeline/processed_files.json` | Add `next_episode_number` counter |

### Existing Files (used as-is, not modified)
| File | Purpose |
|------|---------|
| `~/.openclaw/workspace/scripts/integrations/comfyui_gen.py` | Image generation |
| `~/.openclaw/workspace/scripts/integrations/youtube.py` | YouTube upload |
| `~/.openclaw/workspace/scripts/integrations/wordpress.py` | WordPress management |
| `~/.openclaw/workspace/scripts/integrations/postiz.py` | Social scheduling |
| `~/.openclaw/workspace/scripts/integrations/auto_blogger.py` | Blog engine |
| `~/.openclaw/workspace/scripts/integrations/search.py` | Trending topics |
| `~/.openclaw/workspace/scripts/integrations/video_render.py` | Video rendering + TTS |
| `integrations/nextcloud/client.py` | NextCloud WebDAV client |

---

## Task 1: Shared Module — `rucktalk_common.py`

**Files:**
- Create: `scripts/rucktalk_common.py`

This module holds all constants, state management, notification helpers, LLM integration, and GPU management shared between the episode pipeline and daily social engine.

- [ ] **Step 1: Create the shared constants and config**

```python
#!/usr/bin/env python3
"""RuckTalk Content Pipeline — Shared utilities.

Constants, state management, notifications, LLM calls, and GPU helpers
used by both rucktalk_episode_pipeline.py and rucktalk_daily_social.py.
"""

import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytz

# ── Paths ────────────────────────────────────────────────────────────────────
WORK_DIR = Path("/home/aialfred/rucktalk_pipeline")
STATE_FILE = WORK_DIR / "processed_files.json"
CLIP_QUEUE_FILE = WORK_DIR / "clip_queue.json"
SOCIAL_HISTORY_FILE = WORK_DIR / "social_history.json"
INCOMING_DIR = WORK_DIR / "incoming"
AUDIO_DIR = WORK_DIR / "audio"
TRANSCRIPTS_DIR = WORK_DIR / "transcripts"
CLIPS_DIR = WORK_DIR / "clips"
METADATA_DIR = WORK_DIR / "metadata"
IMAGES_DIR = WORK_DIR / "images"

SCRIPTS = Path("/home/aialfred/.openclaw/workspace/scripts/integrations")
STATIC_MEDIA = Path("/home/aialfred/alfred/static/media")
PUBLIC_MEDIA_URL = "https://aialfred.groundrushcloud.com/media"

# ── NextCloud ────────────────────────────────────────────────────────────────
NC_EPISODE_FOLDER = "/RuckTalk/Episodes"
NC_PROCESSED_FOLDER = "/RuckTalk/Episodes/Processed"

# ── Telegram ─────────────────────────────────────────────────────────────────
TG_TARGET = "7582976864"
OPENCLAW_BIN = "/home/aialfred/.nvm/versions/node/v22.22.0/bin/openclaw"

# ── Timezone ─────────────────────────────────────────────────────────────────
EST = pytz.timezone("America/New_York")

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_URL = "http://75.43.156.105:11434/v1/chat/completions"
LLM_MODELS = [
    "gemma4:31b-cloud",
    "gpt-oss:120b-cloud",
    "minimax-m2:cloud",
    "kimi-k2.5:cloud",
    "deepseek-v3.2:cloud",
]
LLM_RETRIES_PER_MODEL = 2
LLM_RETRY_DELAY = 30
LLM_TIMEOUT = 300

# ── Postiz Integration IDs ───────────────────────────────────────────────────
POSTIZ_IDS = {
    "instagram": "cmmm0ck4m000iqudf4zkc2huz",
    "facebook": "cmmm0d0e8000kqudfvkjm6hzi",
    "youtube": "cmmm1r9n0000mqudfhdc436va",
    "linkedin": "cmnd9rvnx003bqtnvd9n6z7c6",
}

# ── Image Style ──────────────────────────────────────────────────────────────
IMAGE_STYLE_SUFFIX = (
    "cinematic photorealism, bold vibrant energy, warm natural tones, "
    "aspirational lifestyle, real-world authenticity, no text, no watermark, "
    "no logos, no stock photo feel"
)

# ── Brand ────────────────────────────────────────────────────────────────────
BRAND_VOICE = (
    "Direct, no-BS, like a mentor in the trenches. Raw, honest, motivating. "
    "No fluff, no corporate speak. A dad who runs businesses, stays fit, "
    "keeps family first — writing for others doing the same."
)

PILLARS = {
    "entrepreneur": {"name": "Entrepreneurship & Business", "weight": 0.25},
    "fitness": {"name": "Fitness & Discipline", "weight": 0.15},
    "family": {"name": "Family & Fatherhood", "weight": 0.15},
    "mindset": {"name": "Mindset & Grit", "weight": 0.15},
    "nutrition": {"name": "Nutrition & Fuel", "weight": 0.10},
    "faith": {"name": "Faith & Purpose", "weight": 0.10},
    "tactical": {"name": "Tactical Life Hacks", "weight": 0.10},
}

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE = WORK_DIR / "pipeline.log"

logger = logging.getLogger("rucktalk")


def setup_logging():
    """Configure logging to both file and stdout."""
    WORK_DIR.mkdir(exist_ok=True)
    handler_file = logging.FileHandler(LOG_FILE)
    handler_stdout = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler_file.setFormatter(formatter)
    handler_stdout.setFormatter(formatter)
    logger.addHandler(handler_file)
    logger.addHandler(handler_stdout)
    logger.setLevel(logging.INFO)
```

- [ ] **Step 2: Add state management functions**

Append to `scripts/rucktalk_common.py`:

```python
# ── State Management ─────────────────────────────────────────────────────────

def ensure_dirs():
    """Create all pipeline directories."""
    for d in [WORK_DIR, INCOMING_DIR, AUDIO_DIR, TRANSCRIPTS_DIR,
              CLIPS_DIR, METADATA_DIR, IMAGES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    """Load episode processing state."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"processed": [], "next_episode_number": 4}


def save_state(state: dict):
    """Save episode processing state."""
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_clip_queue() -> dict:
    """Load the clip queue (episode clips waiting for daily social)."""
    if CLIP_QUEUE_FILE.exists():
        return json.loads(CLIP_QUEUE_FILE.read_text())
    return {"clips": []}


def save_clip_queue(queue: dict):
    """Save the clip queue."""
    CLIP_QUEUE_FILE.write_text(json.dumps(queue, indent=2))


def load_social_history() -> dict:
    """Load social posting history."""
    if SOCIAL_HISTORY_FILE.exists():
        return json.loads(SOCIAL_HISTORY_FILE.read_text())
    return {"posts": [], "pillar_counts": {}, "last_mode": None}


def save_social_history(history: dict):
    """Save social posting history."""
    SOCIAL_HISTORY_FILE.write_text(json.dumps(history, indent=2))
```

- [ ] **Step 3: Add notification and LLM helpers**

Append to `scripts/rucktalk_common.py`:

```python
# ── Notifications ────────────────────────────────────────────────────────────

def notify_telegram(msg: str):
    """Send a Telegram notification to Mike."""
    try:
        subprocess.run(
            [OPENCLAW_BIN, "message", "send", "--channel", "telegram",
             "--target", TG_TARGET, "--message", msg],
            capture_output=True, timeout=30,
        )
    except Exception as e:
        logger.error(f"Telegram notification failed: {e}")


# ── LLM ──────────────────────────────────────────────────────────────────────

def llm_call(prompt: str, temperature: float = 0.7, max_tokens: int = 8192) -> str | None:
    """Call the LLM with model fallback chain. Returns response text or None."""
    for model in LLM_MODELS:
        for attempt in range(LLM_RETRIES_PER_MODEL):
            try:
                data = json.dumps({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }).encode()
                req = urllib.request.Request(
                    LLM_URL, data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
                    result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"LLM {model} attempt {attempt + 1} failed: {e}")
                if attempt < LLM_RETRIES_PER_MODEL - 1:
                    time.sleep(LLM_RETRY_DELAY)
    logger.error("All LLM models exhausted")
    return None


def llm_json(prompt: str, temperature: float = 0.5) -> dict | None:
    """Call LLM expecting JSON response. Strips markdown fences and parses."""
    raw = llm_call(prompt, temperature=temperature)
    if not raw:
        return None
    # Strip markdown code fences
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        logger.error(f"LLM returned invalid JSON: {raw[:200]}")
        return None


# ── GPU Helpers ──────────────────────────────────────────────────────────────

def run_comfyui(prompt: str, width: int = 1024, height: int = 1024) -> str | None:
    """Generate an image via ComfyUI. Returns local file path or None."""
    full_prompt = f"{prompt}, {IMAGE_STYLE_SUFFIX}"
    try:
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "comfyui_gen.py"), "generate",
             full_prompt, "--width", str(width), "--height", str(height),
             "--model", "flux"],
            capture_output=True, text=True, timeout=180,
            cwd=str(SCRIPTS),
        )
        for line in r.stdout.split("\n"):
            if "http" in line and ("groundrushcloud" in line or "/output/" in line):
                url = line.strip().split()[-1]
                if url.startswith("http"):
                    return _download_url(url)
    except Exception as e:
        logger.error(f"ComfyUI error: {e}")
    return None


def run_tts(text: str, output_path: str | Path) -> bool:
    """Generate TTS audio via Kokoro. Returns True on success."""
    try:
        payload = json.dumps({
            "model": "kokoro",
            "input": text,
            "voice": "bm_daniel",
            "response_format": "mp3",
        }).encode()
        req = urllib.request.Request(
            "http://75.43.156.105:8880/v1/audio/speech",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            Path(output_path).write_bytes(resp.read())
        return True
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return False


def _download_url(url: str) -> str | None:
    """Download a URL to a temp file in IMAGES_DIR. Returns local path."""
    try:
        import uuid
        ext = Path(url.split("?")[0]).suffix or ".png"
        local = IMAGES_DIR / f"{uuid.uuid4().hex[:12]}{ext}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            local.write_bytes(resp.read())
        return str(local)
    except Exception as e:
        logger.error(f"Download failed {url}: {e}")
        return None


def run_script(script_name: str, *args, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run an Oracle integration script by name."""
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script_name)] + list(args),
        capture_output=True, text=True, timeout=timeout,
        cwd=str(SCRIPTS),
    )
```

- [ ] **Step 4: Verify the module loads cleanly**

Run:
```bash
cd /home/aialfred/alfred && python3 -c "import scripts.rucktalk_common as rc; rc.setup_logging(); print('OK')"
```

Expected: `OK` with no import errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/rucktalk_common.py
git commit -m "feat(rucktalk): add shared utilities module for content pipeline"
```

---

## Task 2: Update Pipeline State with Episode Numbering

**Files:**
- Modify: `/home/aialfred/rucktalk_pipeline/processed_files.json`

- [ ] **Step 1: Update the state file to include episode counter**

```python
import json
from pathlib import Path

state_file = Path("/home/aialfred/rucktalk_pipeline/processed_files.json")
state = json.loads(state_file.read_text())
state["next_episode_number"] = 4  # 3 episodes already processed
state_file.write_text(json.dumps(state, indent=2))
print(json.dumps(state, indent=2))
```

Run: `python3 -c "<above code>"`

Expected output:
```json
{
  "processed": [
    "Relentless.mp4",
    "The Enemy Within.mp4",
    "Perspective Shift.mp4"
  ],
  "next_episode_number": 4
}
```

- [ ] **Step 2: Create initial clip queue and social history files**

```bash
echo '{"clips": []}' > /home/aialfred/rucktalk_pipeline/clip_queue.json
echo '{"posts": [], "pillar_counts": {}, "last_mode": null}' > /home/aialfred/rucktalk_pipeline/social_history.json
```

- [ ] **Step 3: Create pipeline subdirectories**

```bash
mkdir -p /home/aialfred/rucktalk_pipeline/{incoming,audio,transcripts,clips,metadata,images}
```

- [ ] **Step 4: Commit**

```bash
git add -f /home/aialfred/rucktalk_pipeline/processed_files.json
git commit -m "feat(rucktalk): add episode numbering counter to pipeline state"
```

Note: `clip_queue.json`, `social_history.json`, and subdirectories are runtime state — no need to commit.

---

## Task 3: Episode Pipeline — NextCloud Watcher & Download

**Files:**
- Create: `scripts/rucktalk_episode_pipeline.py`

- [ ] **Step 1: Create the episode pipeline script with NextCloud watcher**

```python
#!/usr/bin/env python3
"""RuckTalk Episode Pipeline — Full automation from NextCloud drop to everywhere.

Watches NextCloud /RuckTalk/Episodes for new MP4 files and runs the full pipeline:
  1. Download MP4
  2. Extract audio (ffmpeg → MP3)
  3. Transcribe (Whisper with timestamps)
  4. AI analysis — title, description, moments, keywords
  5. Generate cover image (ComfyUI)
  6. Publish audio episode to WordPress
  7. Upload full video to YouTube
  8. Create video page on WordPress (YouTube embed)
  9. Generate episode blog post
  10. Smart clip generation with captions
  11. Queue social content
  12. Notify Mike

Usage:
  python3 rucktalk_episode_pipeline.py              # Poll NextCloud for new episodes
  python3 rucktalk_episode_pipeline.py --reprocess "filename.mp4"  # Force reprocess
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rucktalk_common import (
    NC_EPISODE_FOLDER, NC_PROCESSED_FOLDER,
    WORK_DIR, INCOMING_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, CLIPS_DIR,
    METADATA_DIR, IMAGES_DIR, STATIC_MEDIA, PUBLIC_MEDIA_URL,
    SCRIPTS, POSTIZ_IDS,
    logger, setup_logging, ensure_dirs,
    load_state, save_state, load_clip_queue, save_clip_queue,
    notify_telegram, llm_call, llm_json,
    run_comfyui, run_tts, run_script,
    BRAND_VOICE, IMAGE_STYLE_SUFFIX, PILLARS,
)

from integrations.nextcloud.client import list_files, download_file, create_folder


# ── NextCloud ────────────────────────────────────────────────────────────────

def check_for_new_episodes() -> list[str]:
    """Poll NextCloud for new MP4 files not yet processed."""
    state = load_state()
    processed = set(state.get("processed", []))

    try:
        files = list_files(NC_EPISODE_FOLDER, depth=1)
    except Exception as e:
        logger.error(f"NextCloud list failed: {e}")
        return []

    video_exts = {".mp4", ".mov", ".mkv", ".avi"}
    pending = []
    for f in files:
        name = f["name"]
        ext = Path(name).suffix.lower()
        if ext in video_exts and name not in processed and "bumper" not in name.lower():
            pending.append(name)

    return pending


def download_episode(filename: str) -> Path | None:
    """Download an episode MP4 from NextCloud to incoming dir."""
    remote_path = f"{NC_EPISODE_FOLDER}/{filename}"
    local_path = INCOMING_DIR / filename
    logger.info(f"  Downloading {filename}...")
    try:
        content = download_file(remote_path)
        local_path.write_bytes(content)
        size_mb = local_path.stat().st_size / 1e6
        logger.info(f"  Downloaded: {size_mb:.1f} MB")
        return local_path
    except Exception as e:
        logger.error(f"  Download failed: {e}")
        return None


# ── Audio Extraction ─────────────────────────────────────────────────────────

def extract_audio(video_path: Path, episode_slug: str) -> Path | None:
    """Extract audio from video as MP3 using ffmpeg."""
    audio_path = AUDIO_DIR / f"{episode_slug}.mp3"
    logger.info("  Extracting audio...")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn",
         "-acodec", "libmp3lame", "-ab", "192k", "-ar", "44100",
         str(audio_path)],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode == 0 and audio_path.exists():
        size_mb = audio_path.stat().st_size / 1e6
        logger.info(f"  Audio extracted: {size_mb:.1f} MB")
        return audio_path
    logger.error(f"  Audio extraction failed: {r.stderr[-300:]}")
    return None


# ── Transcription ────────────────────────────────────────────────────────────

def transcribe(audio_path: Path, episode_slug: str) -> dict | None:
    """Transcribe audio with Whisper, producing JSON with timestamps."""
    logger.info("  Transcribing with Whisper (this may take several minutes)...")
    output_dir = TRANSCRIPTS_DIR
    try:
        r = subprocess.run(
            ["whisper", str(audio_path),
             "--model", "base",
             "--language", "en",
             "--output_format", "json",
             "--word_timestamps", "True",
             "--output_dir", str(output_dir)],
            capture_output=True, text=True, timeout=1800,
        )
        # Whisper names output after the input file stem
        json_file = output_dir / f"{audio_path.stem}.json"
        if json_file.exists():
            transcript = json.loads(json_file.read_text())
            # Also save a clean text version
            full_text = " ".join(seg["text"].strip() for seg in transcript.get("segments", []))
            text_file = output_dir / f"{episode_slug}.txt"
            text_file.write_text(full_text)
            logger.info(f"  Transcription complete: {len(full_text)} chars, {len(transcript.get('segments', []))} segments")
            return transcript
    except subprocess.TimeoutExpired:
        logger.error("  Whisper timed out (30 min limit)")
    except Exception as e:
        logger.error(f"  Transcription error: {e}")
    return None


def get_full_text(transcript: dict) -> str:
    """Extract plain text from Whisper JSON transcript."""
    return " ".join(seg["text"].strip() for seg in transcript.get("segments", []))


# ── AI Metadata ──────────────────────────────────────────────────────────────

def generate_episode_metadata(transcript_text: str, episode_number: int) -> dict | None:
    """Generate episode title, description, moments, keywords from transcript."""
    logger.info("  Generating episode metadata via AI...")
    prompt = f"""You are analyzing a podcast episode transcript for RuckTalk — a podcast about
entrepreneurship, discipline, fitness, family, and faith. The voice is direct, no-BS, raw,
and motivating.

This is Episode {episode_number}.

Transcript (first 6000 chars):
{transcript_text[:6000]}

Generate the following as JSON:
{{
  "title": "A compelling, punchy episode title (NOT just 'Episode {episode_number}' — give it a real name, 3-6 words, captures the core theme)",
  "description": "2-3 sentence episode description for YouTube and the website",
  "show_notes": "Bullet-point show notes in HTML (<ul><li>...) covering 5-8 key topics discussed",
  "moments": [
    {{
      "summary": "What this moment is about (1 sentence)",
      "quote": "The most powerful line from this moment (exact words from transcript)",
      "start_time": 0.0,
      "end_time": 30.0,
      "pillar": "entrepreneur|fitness|family|mindset|nutrition|faith|tactical"
    }}
  ],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "primary_pillar": "entrepreneur|fitness|family|mindset|nutrition|faith|tactical"
}}

For "moments": identify the 5-7 strongest moments. Each should be a self-contained thought
(30-90 seconds) that would make a compelling social media clip. The timestamps should
reference the approximate location in the transcript.

Return ONLY valid JSON."""

    result = llm_json(prompt)
    if result:
        logger.info(f"  Episode title: \"{result.get('title', 'Unknown')}\"")
        logger.info(f"  Found {len(result.get('moments', []))} clip moments")
    return result


# ── Cover Image ──────────────────────────────────────────────────────────────

def generate_cover_image(title: str, episode_number: int) -> str | None:
    """Generate an episode cover image via ComfyUI."""
    logger.info("  Generating cover image...")
    prompt = (
        f"cinematic portrait of a determined man, dramatic side lighting, "
        f"dark moody background with warm accent tones, podcast episode artwork, "
        f"masculine energy, grit and determination, professional photography"
    )
    return run_comfyui(prompt, width=1344, height=768)


# ── WordPress Publishing ─────────────────────────────────────────────────────

def upload_media_to_wp(local_path: str) -> dict | None:
    """Upload a file to RuckTalk WordPress media library."""
    r = run_script("wordpress.py", "upload-media", "rucktalk", local_path)
    try:
        return json.loads(r.stdout)
    except Exception:
        logger.error(f"WP media upload failed: {r.stdout[:200]} {r.stderr[:200]}")
        return None


def publish_audio_episode(episode_number: int, title: str, description: str,
                          show_notes: str, audio_url: str, cover_media_id: int | None) -> str | None:
    """Publish the audio episode to WordPress."""
    logger.info("  Publishing audio episode to WordPress...")
    ep_title = f"Episode {episode_number}: {title}"

    content = f"""<div class="rucktalk-episode">
<h3>Listen to this episode:</h3>
<audio controls style="width:100%;max-width:700px;">
<source src="{audio_url}" type="audio/mpeg">
Your browser does not support the audio element.
</audio>
<hr>
<h3>About This Episode</h3>
<p>{description}</p>
<h3>Show Notes</h3>
{show_notes}
<p><strong>Subscribe and follow RuckTalk for new episodes.</strong></p>
</div>"""

    args = ["create-post", "rucktalk", "--title", ep_title,
            "--content", content, "--status", "publish"]
    if cover_media_id:
        args.extend(["--featured-media", str(cover_media_id)])

    r = run_script("wordpress.py", *args)
    try:
        result = json.loads(r.stdout)
        url = result.get("link", result.get("url", ""))
        logger.info(f"  Audio episode published: {url}")
        return url
    except Exception:
        logger.error(f"WP audio publish failed: {r.stdout[:200]}")
        return None


def publish_video_page(episode_number: int, title: str, description: str,
                       youtube_id: str, cover_media_id: int | None) -> str | None:
    """Create a video page on WordPress with YouTube embed."""
    logger.info("  Publishing video page to WordPress...")
    ep_title = f"Episode {episode_number}: {title}"

    content = f"""<div class="rucktalk-video-episode">
<div class="video-container" style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;">
<iframe src="https://www.youtube.com/embed/{youtube_id}"
  style="position:absolute;top:0;left:0;width:100%;height:100%;"
  frameborder="0" allow="accelerometer;autoplay;clipboard-write;encrypted-media;gyroscope;picture-in-picture"
  allowfullscreen></iframe>
</div>
<h3>About This Episode</h3>
<p>{description}</p>
</div>"""

    args = ["create-post", "rucktalk", "--title", f"Watch: {ep_title}",
            "--content", content, "--status", "publish"]
    if cover_media_id:
        args.extend(["--featured-media", str(cover_media_id)])

    r = run_script("wordpress.py", *args)
    try:
        result = json.loads(r.stdout)
        url = result.get("link", result.get("url", ""))
        logger.info(f"  Video page published: {url}")
        return url
    except Exception:
        logger.error(f"WP video page failed: {r.stdout[:200]}")
        return None


# ── YouTube Upload ───────────────────────────────────────────────────────────

def upload_to_youtube(video_path: Path, episode_number: int, title: str,
                      description: str, keywords: list[str]) -> str | None:
    """Upload the full episode to YouTube. Returns video ID or None."""
    logger.info("  Uploading to YouTube...")
    yt_title = f"Episode {episode_number}: {title} | RuckTalk"
    yt_desc = (
        f"{description}\n\n"
        f"Welcome to RuckTalk — the podcast for men who lead with discipline, "
        f"show up for their families, and never stop growing.\n\n"
        f"Subscribe for new episodes.\n\n"
        f"#{' #'.join(keywords[:8])}\n"
        f"#RuckTalk #Podcast #Discipline #Entrepreneur"
    )
    r = run_script("youtube.py", "upload", str(video_path), yt_title, yt_desc, "public",
                   timeout=600)
    logger.info(f"  YouTube output: {r.stdout[:300]}")
    if r.returncode != 0:
        logger.error(f"  YouTube error: {r.stderr[:300]}")
        return None

    # Parse video ID from output
    import re
    for line in r.stdout.split("\n"):
        m = re.search(r'[A-Za-z0-9_-]{11}', line)
        if m:
            vid_id = m.group(0)
            logger.info(f"  YouTube video ID: {vid_id}")
            return vid_id
    return None


# ── Blog Post from Transcript ────────────────────────────────────────────────

def generate_episode_blog(episode_number: int, title: str, transcript_text: str,
                          description: str, cover_image_path: str | None) -> str | None:
    """Generate and publish a full blog post from the episode transcript."""
    logger.info("  Generating episode blog post...")

    prompt = f"""Write a blog post for RuckTalk.com based on this podcast episode transcript.

Brand voice: {BRAND_VOICE}

Episode {episode_number}: {title}
Description: {description}

Transcript (first 5000 chars):
{transcript_text[:5000]}

Requirements:
- SEO-friendly title (incorporate the episode name but optimize for search)
- Minimum 1200 words
- 5+ H2 section headers
- Written in the RuckTalk voice — direct, raw, motivating
- Include key takeaways and direct quotes from the episode
- End with a CTA to listen to the full episode
- HTML format (use <h2>, <p>, <blockquote>, <ul> tags)

Return JSON:
{{
  "blog_title": "Episode {episode_number}: {title} — SEO subtitle",
  "meta_description": "Under 155 chars",
  "content": "<h2>...</h2><p>...</p>...",
  "image_prompts": ["prompt for image 1", "prompt for image 2", "prompt for image 3", "prompt for image 4"]
}}"""

    blog = llm_json(prompt)
    if not blog:
        logger.error("  Blog generation failed")
        return None

    # Generate inline images
    content = blog.get("content", "")
    image_prompts = blog.get("image_prompts", [])
    for i, img_prompt in enumerate(image_prompts[:4]):
        img_path = run_comfyui(img_prompt, width=1024, height=1024)
        if img_path:
            media = upload_media_to_wp(img_path)
            if media and media.get("source_url"):
                img_html = (
                    f'<figure class="wp-block-image"><img src="{media["source_url"]}" '
                    f'alt="{img_prompt[:80]}"/></figure>'
                )
                # Insert after the i-th H2
                h2_positions = [j for j, c in enumerate(content.split("</h2>")) if j < len(content.split("</h2>")) - 1]
                if i < len(h2_positions):
                    parts = content.split("</h2>", i + 1)
                    if len(parts) > i:
                        parts[i] += f"</h2>\n{img_html}\n"
                        content = "".join(parts) if i == 0 else "</h2>".join(parts[:i]) + "".join(parts[i:])
        time.sleep(3)  # GPU cooldown between image gens

    # Upload cover as featured image
    featured_id = None
    if cover_image_path:
        media = upload_media_to_wp(cover_image_path)
        if media:
            featured_id = media.get("id")

    # Publish
    blog_title = blog.get("blog_title", f"Episode {episode_number}: {title}")
    args = ["create-post", "rucktalk", "--title", blog_title,
            "--content", content, "--status", "publish"]
    if featured_id:
        args.extend(["--featured-media", str(featured_id)])

    r = run_script("wordpress.py", *args)
    try:
        result = json.loads(r.stdout)
        url = result.get("link", result.get("url", ""))
        logger.info(f"  Blog published: {url}")
        return url
    except Exception:
        logger.error(f"  Blog publish failed: {r.stdout[:200]}")
        return None


# ── Smart Clip Generation ────────────────────────────────────────────────────

def generate_smart_clips(video_path: Path, transcript: dict, moments: list[dict],
                         episode_number: int, title: str, episode_slug: str) -> list[dict]:
    """Generate smart clips from identified moments with captions."""
    logger.info(f"  Generating {len(moments)} smart clips...")
    clips = []
    segments = transcript.get("segments", [])

    for i, moment in enumerate(moments):
        start = moment.get("start_time", 0)
        end = moment.get("end_time", start + 60)
        duration = min(end - start, 90)  # Cap at 90 seconds
        if duration < 15:
            duration = 30  # Minimum 15 seconds

        clip_name = f"ep{episode_number}_clip{i + 1}_{episode_slug}"

        # Cut portrait clip (1080x1920)
        portrait_path = CLIPS_DIR / f"{clip_name}_portrait.mp4"
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
             "-i", str(video_path),
             "-vf", f"scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-c:a", "aac", "-b:a", "128k",
             str(portrait_path)],
            capture_output=True, text=True, timeout=120,
        )

        if r.returncode != 0:
            logger.warning(f"  Clip {i + 1} portrait failed: {r.stderr[-200:]}")
            continue

        # Cut landscape clip (1920x1080)
        landscape_path = CLIPS_DIR / f"{clip_name}_landscape.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
             "-i", str(video_path),
             "-vf", f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-c:a", "aac", "-b:a", "128k",
             str(landscape_path)],
            capture_output=True, text=True, timeout=120,
        )

        # Generate caption SRT from transcript segments that fall in this time range
        srt_path = CLIPS_DIR / f"{clip_name}.srt"
        srt_lines = []
        srt_idx = 1
        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            if seg_start >= start and seg_end <= start + duration:
                adj_start = seg_start - start
                adj_end = seg_end - start
                srt_lines.append(f"{srt_idx}")
                srt_lines.append(f"{_srt_time(adj_start)} --> {_srt_time(adj_end)}")
                srt_lines.append(seg.get("text", "").strip())
                srt_lines.append("")
                srt_idx += 1
        srt_path.write_text("\n".join(srt_lines))

        # Copy portrait clip to static for web access
        web_clip_name = f"rucktalk_{clip_name}_portrait.mp4"
        web_path = STATIC_MEDIA / web_clip_name
        shutil.copy2(str(portrait_path), str(web_path))

        clip_info = {
            "index": i + 1,
            "episode_number": episode_number,
            "episode_title": title,
            "summary": moment.get("summary", ""),
            "quote": moment.get("quote", ""),
            "pillar": moment.get("pillar", "mindset"),
            "portrait_path": str(portrait_path),
            "landscape_path": str(landscape_path),
            "srt_path": str(srt_path),
            "web_url": f"{PUBLIC_MEDIA_URL}/{web_clip_name}",
            "duration": duration,
            "posted": False,
        }
        clips.append(clip_info)
        logger.info(f"  Clip {i + 1}: {moment.get('summary', '')[:60]}")

    logger.info(f"  Generated {len(clips)} clips")
    return clips


def _srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── Main Pipeline ────────────────────────────────────────────────────────────

def process_episode(filename: str) -> bool:
    """Run the full pipeline for a single episode."""
    state = load_state()
    episode_number = state.get("next_episode_number", 4)
    raw_title = Path(filename).stem
    episode_slug = raw_title.lower().replace(" ", "_").replace("-", "_")
    episode_slug = "".join(c for c in episode_slug if c.isalnum() or c == "_")

    logger.info(f"=== Processing Episode {episode_number}: {filename} ===")
    notify_telegram(f"Starting RuckTalk pipeline for: {filename}")

    try:
        # 1. Download
        video_path = download_episode(filename)
        if not video_path:
            raise RuntimeError("Download failed")

        # 2. Extract audio
        audio_path = extract_audio(video_path, episode_slug)
        if not audio_path:
            raise RuntimeError("Audio extraction failed")

        # Copy audio to static for web access
        web_audio_name = f"rucktalk_ep{episode_number}_{episode_slug}.mp3"
        web_audio = STATIC_MEDIA / web_audio_name
        shutil.copy2(str(audio_path), str(web_audio))
        audio_url = f"{PUBLIC_MEDIA_URL}/{web_audio_name}"

        # 3. Transcribe
        transcript = transcribe(audio_path, episode_slug)
        if not transcript:
            raise RuntimeError("Transcription failed")
        transcript_text = get_full_text(transcript)

        # 4. AI metadata
        metadata = generate_episode_metadata(transcript_text, episode_number)
        if not metadata:
            # Fallback: use filename as title
            metadata = {
                "title": raw_title.replace("-", " ").replace("_", " ").title(),
                "description": f"Episode {episode_number} of RuckTalk.",
                "show_notes": "<ul><li>New episode</li></ul>",
                "moments": [],
                "keywords": ["rucktalk", "podcast", "entrepreneur"],
                "primary_pillar": "mindset",
            }

        ep_title = metadata["title"]
        ep_full = f"Episode {episode_number}: {ep_title}"
        logger.info(f"  Title: {ep_full}")

        # Save metadata
        meta_file = METADATA_DIR / f"ep{episode_number}_{episode_slug}.json"
        meta_file.write_text(json.dumps({
            "episode_number": episode_number,
            "filename": filename,
            "title": ep_title,
            "full_title": ep_full,
            **metadata,
        }, indent=2))

        # 5. Cover image
        cover_path = generate_cover_image(ep_title, episode_number)

        # 6. Upload cover to WordPress
        cover_media_id = None
        if cover_path:
            media = upload_media_to_wp(cover_path)
            if media:
                cover_media_id = media.get("id")

        # 7. Publish audio episode to WordPress
        audio_wp_url = publish_audio_episode(
            episode_number, ep_title,
            metadata.get("description", ""),
            metadata.get("show_notes", ""),
            audio_url, cover_media_id,
        )

        # 8. Upload to YouTube
        youtube_id = upload_to_youtube(
            video_path, episode_number, ep_title,
            metadata.get("description", ""),
            metadata.get("keywords", []),
        )

        # 9. Create video page on WordPress
        video_wp_url = None
        if youtube_id:
            video_wp_url = publish_video_page(
                episode_number, ep_title,
                metadata.get("description", ""),
                youtube_id, cover_media_id,
            )

        # 10. Generate episode blog post
        blog_url = generate_episode_blog(
            episode_number, ep_title, transcript_text,
            metadata.get("description", ""), cover_path,
        )

        # 11. Smart clip generation
        moments = metadata.get("moments", [])
        clips = []
        if moments:
            clips = generate_smart_clips(
                video_path, transcript, moments,
                episode_number, ep_title, episode_slug,
            )
            # Add clips to queue for daily social engine
            queue = load_clip_queue()
            queue["clips"].extend(clips)
            save_clip_queue(queue)

        # 12. Mark processed and increment counter
        state["processed"].append(filename)
        state["next_episode_number"] = episode_number + 1
        save_state(state)

        # Clean up incoming video (keep audio and transcripts)
        video_path.unlink(missing_ok=True)

        # 13. Notify Mike
        yt_line = f"YouTube: https://youtu.be/{youtube_id}" if youtube_id else "YouTube: upload failed"
        audio_line = f"Audio: {audio_wp_url}" if audio_wp_url else "Audio: publish failed"
        video_line = f"Video page: {video_wp_url}" if video_wp_url else "Video page: skipped"
        blog_line = f"Blog: {blog_url}" if blog_url else "Blog: failed"
        clip_line = f"Clips: {len(clips)} queued for daily social"

        notify_telegram(
            f"RuckTalk pipeline complete: {ep_full}\n\n"
            f"{yt_line}\n{audio_line}\n{video_line}\n{blog_line}\n{clip_line}\n\n"
            f"Everything is live, sir."
        )

        logger.info(f"=== Episode {episode_number} complete ===")
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Pipeline failed for {filename}: {e}")
        notify_telegram(f"RuckTalk pipeline FAILED for {filename}: {e}")
        return False


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RuckTalk Episode Pipeline")
    parser.add_argument("--reprocess", help="Force reprocess a specific file")
    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    if args.reprocess:
        logger.info(f"Force reprocessing: {args.reprocess}")
        process_episode(args.reprocess)
        return

    logger.info("RuckTalk Episode Pipeline — checking for new episodes...")
    pending = check_for_new_episodes()

    if not pending:
        logger.info("Nothing new.")
        return

    logger.info(f"{len(pending)} new episode(s): {pending}")
    notify_telegram(f"RuckTalk pipeline starting — {len(pending)} episode(s) queued...")

    for ep in pending:
        process_episode(ep)
        time.sleep(5)

    logger.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses without errors**

Run:
```bash
cd /home/aialfred/alfred && python3 -c "import ast; ast.parse(open('scripts/rucktalk_episode_pipeline.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/rucktalk_episode_pipeline.py
git commit -m "feat(rucktalk): add episode pipeline — NextCloud to YouTube/WordPress/blog/clips"
```

---

## Task 4: Daily Social Engine — `rucktalk_daily_social.py`

**Files:**
- Create: `scripts/rucktalk_daily_social.py`

- [ ] **Step 1: Create the daily social engine**

```python
#!/usr/bin/env python3
"""RuckTalk Daily Social Engine — One powerful post every morning.

Priority order:
  1. Episode clip from queue (if available)
  2. Current-event-driven content (~40% of the time)
  3. Evergreen pillar content (~60% of the time)

Every post is a video — narrated shorts on non-clip days, episode clips on clip days.

Usage:
  python3 rucktalk_daily_social.py              # Generate and schedule today's post
  python3 rucktalk_daily_social.py --dry-run     # Generate but don't post
  python3 rucktalk_daily_social.py --mode trend   # Force trending mode
  python3 rucktalk_daily_social.py --mode evergreen  # Force evergreen mode
"""

import argparse
import json
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rucktalk_common import (
    WORK_DIR, CLIPS_DIR, IMAGES_DIR, STATIC_MEDIA, PUBLIC_MEDIA_URL,
    SCRIPTS, POSTIZ_IDS, EST,
    logger, setup_logging, ensure_dirs,
    load_clip_queue, save_clip_queue,
    load_social_history, save_social_history,
    notify_telegram, llm_call, llm_json,
    run_comfyui, run_tts, run_script,
    BRAND_VOICE, IMAGE_STYLE_SUFFIX, PILLARS,
)


# ── Content Selection ────────────────────────────────────────────────────────

def select_content_mode(history: dict) -> str:
    """Decide: clip, trend, or evergreen."""
    # Check clip queue first
    queue = load_clip_queue()
    unposted = [c for c in queue.get("clips", []) if not c.get("posted", False)]
    if unposted:
        return "clip"

    # 40% trend, 60% evergreen
    return "trend" if random.random() < 0.4 else "evergreen"


def select_pillar(history: dict) -> tuple[str, str]:
    """Pick a pillar using weighted rotation, avoiding recent repeats."""
    pillar_counts = history.get("pillar_counts", {})
    recent_posts = history.get("posts", [])[-5:]
    recent_pillars = {p.get("pillar") for p in recent_posts}

    # Build weighted choices excluding recently used pillars
    candidates = []
    for pid, info in PILLARS.items():
        if pid not in recent_pillars:
            candidates.append((pid, info["weight"]))

    if not candidates:
        # All pillars used recently, reset
        candidates = [(pid, info["weight"]) for pid, info in PILLARS.items()]

    # Weighted random selection
    total = sum(w for _, w in candidates)
    r = random.random() * total
    cumulative = 0
    for pid, weight in candidates:
        cumulative += weight
        if r <= cumulative:
            return pid, PILLARS[pid]["name"]

    return candidates[0][0], PILLARS[candidates[0][0]]["name"]


# ── Clip Mode ────────────────────────────────────────────────────────────────

def post_episode_clip(dry_run: bool = False) -> dict | None:
    """Post the next episode clip from the queue."""
    queue = load_clip_queue()
    unposted = [c for c in queue.get("clips", []) if not c.get("posted", False)]
    if not unposted:
        return None

    clip = unposted[0]
    logger.info(f"  Posting episode clip: Ep {clip['episode_number']} clip {clip['index']}")

    # Generate caption
    caption = llm_call(
        f"""Write a social media caption for a podcast clip from RuckTalk.

Brand voice: {BRAND_VOICE}

Episode {clip['episode_number']}: {clip['episode_title']}
Clip summary: {clip['summary']}
Key quote: {clip['quote']}

Write a 2-3 line caption that:
- Opens with a hook that stops the scroll
- Teases what the clip is about without giving it all away
- Ends with a CTA to watch the full episode
- Include 4-5 relevant hashtags on the last line
- NO escape characters, plain text only, use real line breaks

Keep it under 200 words.""",
        temperature=0.7,
    )

    if not caption:
        caption = (
            f"From Episode {clip['episode_number']}: {clip['episode_title']}\n\n"
            f"{clip.get('quote', 'New episode out now.')}\n\n"
            f"Full episode on YouTube — link in bio.\n\n"
            f"#RuckTalk #Podcast #Discipline #Entrepreneur #Mindset"
        )

    if dry_run:
        logger.info(f"  [DRY RUN] Would post clip: {caption[:100]}...")
        return {"mode": "clip", "caption": caption, "clip": clip}

    # Schedule via Postiz
    schedule_dt = _next_morning_utc()
    video_url = clip.get("web_url", "")
    success = schedule_to_postiz(caption, schedule_dt, video_url=video_url)

    if success:
        # Mark clip as posted
        for c in queue["clips"]:
            if c["index"] == clip["index"] and c["episode_number"] == clip["episode_number"]:
                c["posted"] = True
        save_clip_queue(queue)
        logger.info("  Clip scheduled successfully")

    return {"mode": "clip", "caption": caption, "clip": clip, "scheduled": success}


# ── Trending Mode ────────────────────────────────────────────────────────────

def generate_trending_content(dry_run: bool = False) -> dict | None:
    """Find a trending topic and create a RuckTalk take on it."""
    logger.info("  Mode: Trending content")

    # Search for trending topics
    pillar_id, pillar_name = select_pillar(load_social_history())
    search_queries = [
        f"trending {pillar_name.lower()} news today",
        "trending entrepreneurship motivation news",
        "viral fitness discipline story today",
    ]

    trending_results = []
    for query in search_queries[:2]:
        r = run_script("search.py", "query", query)
        try:
            results = json.loads(r.stdout)
            if isinstance(results, list):
                trending_results.extend(results[:3])
        except Exception:
            pass

    # Have LLM pick the best topic and write a script
    trending_context = "\n".join(
        f"- {r.get('title', '')}: {r.get('content', '')[:100]}"
        for r in trending_results[:8]
    ) if trending_results else "No specific trending stories found. Generate a timely, relevant take."

    result = llm_json(f"""You are writing a 60-90 second narrated video script for RuckTalk's social media.

Brand voice: {BRAND_VOICE}
Pillar focus: {pillar_name}

Trending topics today:
{trending_context}

Pick the most relevant trending topic and write a RuckTalk take on it. Connect it back
to the {pillar_name} pillar.

Return JSON:
{{
  "topic": "The topic you chose",
  "script": "The full narration script (60-90 seconds when spoken, ~150-220 words). Written in first person, direct to camera style. No stage directions.",
  "caption": "Social media caption (under 200 words, 2-3 lines, hook + value + CTA, 4-5 hashtags on last line, plain text with real line breaks)",
  "image_prompts": ["prompt for visual 1", "prompt for visual 2", "prompt for visual 3", "prompt for visual 4"],
  "pillar": "{pillar_id}"
}}""")

    if not result:
        logger.warning("  Trending content generation failed, falling back to evergreen")
        return generate_evergreen_content(dry_run)

    return _produce_narrated_video(result, "trend", dry_run)


# ── Evergreen Mode ───────────────────────────────────────────────────────────

def generate_evergreen_content(dry_run: bool = False) -> dict | None:
    """Generate timeless content on a RuckTalk pillar."""
    logger.info("  Mode: Evergreen content")

    pillar_id, pillar_name = select_pillar(load_social_history())
    logger.info(f"  Pillar: {pillar_name}")

    result = llm_json(f"""You are writing a 60-90 second narrated video script for RuckTalk's social media.

Brand voice: {BRAND_VOICE}
Pillar: {pillar_name}

Write a timeless, powerful piece of content about {pillar_name}. Something that could be
posted any day and still hit hard. Not tied to any current event. Think: motivational truth
bomb, tactical life advice, a story that teaches a lesson.

Return JSON:
{{
  "topic": "The topic",
  "script": "The full narration script (60-90 seconds when spoken, ~150-220 words). Written in first person, direct to camera style. No stage directions.",
  "caption": "Social media caption (under 200 words, 2-3 lines, hook + value + CTA, 4-5 hashtags on last line, plain text with real line breaks)",
  "image_prompts": ["prompt for visual 1", "prompt for visual 2", "prompt for visual 3", "prompt for visual 4"],
  "pillar": "{pillar_id}"
}}""")

    if not result:
        logger.error("  Evergreen content generation failed")
        return None

    return _produce_narrated_video(result, "evergreen", dry_run)


# ── Video Production ─────────────────────────────────────────────────────────

def _produce_narrated_video(content: dict, mode: str, dry_run: bool = False) -> dict | None:
    """Produce a narrated video from AI-generated content."""
    topic = content.get("topic", "RuckTalk")
    script = content.get("script", "")
    caption = content.get("caption", "")
    image_prompts = content.get("image_prompts", [])
    pillar = content.get("pillar", "mindset")

    logger.info(f"  Topic: {topic}")
    logger.info(f"  Script: {len(script)} chars")

    # 1. Generate TTS narration
    narration_path = WORK_DIR / f"narration_{uuid.uuid4().hex[:8]}.mp3"
    logger.info("  Generating narration (Kokoro TTS)...")
    if not run_tts(script, narration_path):
        logger.error("  TTS failed")
        return None

    # 2. Generate images
    logger.info("  Generating visuals (ComfyUI)...")
    image_paths = []
    for i, prompt in enumerate(image_prompts[:4]):
        img = run_comfyui(prompt, width=1080, height=1920)
        if img:
            image_paths.append(img)
            logger.info(f"  Image {i + 1} generated")
        time.sleep(3)  # GPU cooldown

    if not image_paths:
        logger.error("  No images generated")
        return None

    # 3. Build video using video_render.py slideshow with audio
    logger.info("  Assembling video...")
    video_name = f"rucktalk_social_{uuid.uuid4().hex[:12]}.mp4"
    video_path = STATIC_MEDIA / video_name

    # Use video_render.py slideshow: images + narration audio
    args = ["slideshow"] + image_paths + [
        "--audio", str(narration_path),
        "--transition", "fade",
        "--ratio", "portrait",
    ]
    r = run_script("video_render.py", *args, timeout=300)

    # Parse output for the video path
    output_path = None
    try:
        result = json.loads(r.stdout)
        output_path = result.get("output", result.get("path"))
    except Exception:
        # Try to find path in stdout
        for line in r.stdout.split("\n"):
            if "/static/media/" in line or STATIC_MEDIA.name in line:
                potential = line.strip().split()[-1]
                if Path(potential).exists():
                    output_path = potential

    if not output_path or not Path(output_path).exists():
        logger.error(f"  Video assembly failed: {r.stderr[:300]}")
        return None

    # Move to our expected location if needed
    if str(output_path) != str(video_path):
        shutil.copy2(output_path, str(video_path))

    video_url = f"{PUBLIC_MEDIA_URL}/{video_name}"
    logger.info(f"  Video ready: {video_url}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would schedule: {caption[:100]}...")
        # Clean up narration
        narration_path.unlink(missing_ok=True)
        return {"mode": mode, "topic": topic, "caption": caption, "video_url": video_url, "pillar": pillar}

    # 4. Schedule via Postiz
    schedule_dt = _next_morning_utc()
    success = schedule_to_postiz(caption, schedule_dt, video_url=video_url)

    # 5. Record in history
    history = load_social_history()
    history["posts"].append({
        "date": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "topic": topic,
        "pillar": pillar,
        "video_url": video_url,
        "scheduled": success,
    })
    history["pillar_counts"][pillar] = history["pillar_counts"].get(pillar, 0) + 1
    history["last_mode"] = mode
    save_social_history(history)

    # Clean up narration temp file
    narration_path.unlink(missing_ok=True)

    return {"mode": mode, "topic": topic, "caption": caption, "video_url": video_url,
            "pillar": pillar, "scheduled": success}


# ── Postiz Scheduling ────────────────────────────────────────────────────────

def _next_morning_utc() -> str:
    """Get tomorrow morning 11:00 UTC (7 AM ET) as ISO string."""
    now = datetime.now(timezone.utc)
    tomorrow = now.date() + timedelta(days=1)
    return f"{tomorrow}T11:00:00"


def schedule_to_postiz(caption: str, schedule_dt: str, video_url: str = "",
                       image_url: str = "") -> bool:
    """Schedule a post to all RuckTalk platforms via Postiz."""
    logger.info(f"  Scheduling to Postiz for {schedule_dt}...")

    # Upload media first if we have a video or image
    media_id = ""
    media_to_upload = video_url or image_url
    if media_to_upload:
        sys.path.insert(0, str(SCRIPTS))
        try:
            import postiz as _postiz
            media = _postiz.upload_media(media_to_upload)
            media_id = media.get("id", "")
            if not media_id:
                logger.warning(f"  Postiz media upload failed: {media}")
        except Exception as e:
            logger.warning(f"  Postiz media upload error: {e}")

    # Build posts for each platform
    all_ids = ",".join(POSTIZ_IDS.values())
    args = ["create-post", caption, schedule_dt, all_ids]
    if media_to_upload:
        args.append(media_to_upload)

    r = run_script("postiz.py", *args)
    if r.returncode == 0:
        logger.info("  Postiz scheduling successful")
        return True
    else:
        logger.error(f"  Postiz error: {r.stderr[:200]}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RuckTalk Daily Social Engine")
    parser.add_argument("--dry-run", action="store_true", help="Generate but don't post")
    parser.add_argument("--mode", choices=["trend", "evergreen", "clip"],
                        help="Force a specific content mode")
    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    logger.info("RuckTalk Daily Social — generating today's post...")

    # Select mode
    history = load_social_history()
    if args.mode:
        mode = args.mode
    else:
        mode = select_content_mode(history)

    logger.info(f"  Content mode: {mode}")

    # Generate and post
    if mode == "clip":
        result = post_episode_clip(dry_run=args.dry_run)
        if not result:
            logger.info("  No clips available, switching to generated content")
            mode = "trend" if random.random() < 0.4 else "evergreen"

    if mode == "trend":
        result = generate_trending_content(dry_run=args.dry_run)
    elif mode == "evergreen":
        result = generate_evergreen_content(dry_run=args.dry_run)

    if result:
        logger.info(f"  Done: {result.get('mode', mode)} — {result.get('topic', 'N/A')}")
        if not args.dry_run:
            notify_telegram(
                f"RuckTalk daily social posted.\n"
                f"Mode: {result.get('mode', mode)}\n"
                f"Topic: {result.get('topic', 'N/A')}\n"
                f"Scheduled for tomorrow morning."
            )
    else:
        logger.error("  Failed to generate content")
        notify_telegram("RuckTalk daily social FAILED — check logs.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add missing import**

At the top of the file, after `from pathlib import Path`, add:

```python
import shutil
```

- [ ] **Step 3: Verify syntax**

Run:
```bash
cd /home/aialfred/alfred && python3 -c "import ast; ast.parse(open('scripts/rucktalk_daily_social.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/rucktalk_daily_social.py
git commit -m "feat(rucktalk): add daily social engine — narrated video posts every morning"
```

---

## Task 5: Remotion Template — RuckTalkShort

**Files:**
- Create: `/home/aialfred/remotion/src/components/CaptionOverlay.tsx`
- Create: `/home/aialfred/remotion/src/templates/RuckTalkShort.tsx`
- Modify: `/home/aialfred/remotion/src/Root.tsx`

- [ ] **Step 1: Create the CaptionOverlay component**

```tsx
// /home/aialfred/remotion/src/components/CaptionOverlay.tsx
import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";

interface CaptionWord {
  text: string;
  startFrame: number;
  endFrame: number;
}

interface CaptionOverlayProps {
  words: CaptionWord[];
}

export const CaptionOverlay: React.FC<CaptionOverlayProps> = ({ words }) => {
  const frame = useCurrentFrame();

  // Group words into lines of ~5 words
  const linesOfWords: CaptionWord[][] = [];
  for (let i = 0; i < words.length; i += 5) {
    linesOfWords.push(words.slice(i, i + 5));
  }

  // Find active line
  const activeLine = linesOfWords.find((line) =>
    line.some((w) => frame >= w.startFrame && frame <= w.endFrame)
  );

  if (!activeLine) return null;

  return (
    <div
      style={{
        position: "absolute",
        bottom: 200,
        left: 0,
        right: 0,
        display: "flex",
        justifyContent: "center",
        flexWrap: "wrap",
        gap: 8,
        padding: "0 40px",
      }}
    >
      {activeLine.map((word, i) => {
        const isActive = frame >= word.startFrame && frame <= word.endFrame;
        const opacity = interpolate(
          frame,
          [word.startFrame - 2, word.startFrame],
          [0.5, 1],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
        );

        return (
          <span
            key={i}
            style={{
              fontSize: 52,
              fontWeight: "900",
              fontFamily: "Montserrat, sans-serif",
              color: isActive ? "#ffffff" : "rgba(255,255,255,0.5)",
              textShadow: isActive
                ? "0 0 20px rgba(220,38,38,0.8), 0 4px 20px rgba(0,0,0,0.9)"
                : "0 4px 20px rgba(0,0,0,0.9)",
              textTransform: "uppercase",
              opacity,
              transition: "color 0.1s",
            }}
          >
            {word.text}
          </span>
        );
      })}
    </div>
  );
};
```

- [ ] **Step 2: Create the RuckTalkShort template**

```tsx
// /home/aialfred/remotion/src/templates/RuckTalkShort.tsx
import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  Img,
  interpolate,
  Sequence,
} from "remotion";
import { GradientOverlay } from "../components/GradientOverlay";
import { CaptionOverlay } from "../components/CaptionOverlay";

interface RuckTalkShortProps {
  images: string[];
  captionWords: Array<{ text: string; startFrame: number; endFrame: number }>;
  durationPerImage: number;
}

export const RuckTalkShort: React.FC<RuckTalkShortProps> = ({
  images,
  captionWords,
  durationPerImage = 90,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <AbsoluteFill style={{ backgroundColor: "#0a0a0a" }}>
      {/* Image slideshow with crossfade */}
      {images.map((img, i) => {
        const startFrame = i * durationPerImage;
        const fadeIn = interpolate(
          frame,
          [startFrame, startFrame + 15],
          [0, 1],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
        );
        const zoom = interpolate(
          frame,
          [startFrame, startFrame + durationPerImage],
          [1.0, 1.15],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
        );

        return (
          <Sequence key={i} from={startFrame} durationInFrames={durationPerImage + 15}>
            <AbsoluteFill style={{ opacity: fadeIn }}>
              <Img
                src={img}
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  transform: `scale(${zoom})`,
                  filter: "contrast(1.15) saturate(0.85)",
                }}
              />
            </AbsoluteFill>
          </Sequence>
        );
      })}

      {/* Dark gradient overlay */}
      <GradientOverlay color="#0a0a0a" opacity={0.5} />

      {/* RuckTalk branding — top left */}
      <div
        style={{
          position: "absolute",
          top: 60,
          left: 40,
          fontSize: 28,
          fontWeight: "900",
          fontFamily: "Montserrat, sans-serif",
          color: "rgba(255,255,255,0.6)",
          letterSpacing: 6,
          textTransform: "uppercase",
        }}
      >
        RUCKTALK
      </div>

      {/* Accent line */}
      <div
        style={{
          position: "absolute",
          top: 100,
          left: 40,
          width: 50,
          height: 4,
          backgroundColor: "#dc2626",
        }}
      />

      {/* Caption overlay */}
      <CaptionOverlay words={captionWords} />
    </AbsoluteFill>
  );
};
```

- [ ] **Step 3: Register the composition in Root.tsx**

Read the current Root.tsx and add the RuckTalkShort composition. Add after the existing RuckTalkPromo composition:

```tsx
import { RuckTalkShort } from "./templates/RuckTalkShort";

// Add this Composition inside the fragment in Root:
<Composition
  id="RuckTalkShort"
  component={RuckTalkShort}
  durationInFrames={900}
  fps={30}
  width={1080}
  height={1920}
  defaultProps={{
    images: [],
    captionWords: [],
    durationPerImage: 90,
  }}
/>
```

- [ ] **Step 4: Verify Remotion builds**

```bash
cd /home/aialfred/remotion && npx remotion compositions 2>&1 | head -20
```

Expected: Should list RuckTalkShort among the compositions.

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/remotion && git add src/components/CaptionOverlay.tsx src/templates/RuckTalkShort.tsx src/Root.tsx
git commit -m "feat(remotion): add RuckTalkShort template with word-by-word captions"
```

---

## Task 6: NextCloud Folder Setup & Ensure Connectivity

**Files:** None (infrastructure setup)

- [ ] **Step 1: Verify NextCloud connectivity**

```bash
cd /home/aialfred/alfred && python3 -c "
from integrations.nextcloud.client import is_connected, list_files
print('Connected:', is_connected())
"
```

Expected: `Connected: True`

- [ ] **Step 2: Create the NextCloud folder structure**

```bash
cd /home/aialfred/alfred && python3 -c "
from integrations.nextcloud.client import create_folder, list_files
import json

# Create folders
for folder in ['/RuckTalk', '/RuckTalk/Episodes', '/RuckTalk/Episodes/Processed']:
    try:
        create_folder(folder)
        print(f'Created: {folder}')
    except Exception as e:
        print(f'Exists or error: {folder} — {e}')

# Verify
files = list_files('/RuckTalk', depth=1)
print(json.dumps(files, indent=2))
"
```

Expected: Folder structure created, listing shows Episodes subfolder.

- [ ] **Step 3: Check if the old NextCloud path has content to migrate**

The existing pipeline watches `Content/RuckTalk/Episodes/Raw/`. Check if there's content there:

```bash
cd /home/aialfred/alfred && python3 -c "
from integrations.nextcloud.client import list_files
import json
try:
    files = list_files('/Content/RuckTalk/Episodes/Raw', depth=1)
    print(json.dumps(files, indent=2))
except Exception as e:
    print(f'Old path not found or error: {e}')
"
```

If files exist at the old path, note them but do NOT move — Mike may still use that path. The new pipeline watches `/RuckTalk/Episodes`.

---

## Task 7: Cron Job Setup

**Files:** System crontab

- [ ] **Step 1: Add cron jobs for all three engines**

```bash
# Show current crontab for reference
crontab -l
```

Add these cron entries (confirm with Mike before installing since this affects shared state — T1 per action tiers):

```bash
# RuckTalk Episode Pipeline — check NextCloud every 10 minutes
*/10 * * * * cd /home/aialfred/alfred && /home/aialfred/.pyenv/shims/python3 scripts/rucktalk_episode_pipeline.py >> /home/aialfred/rucktalk_pipeline/pipeline.log 2>&1

# RuckTalk Daily Social — one post every morning at 7 AM ET (11 UTC)
0 11 * * * cd /home/aialfred/alfred && /home/aialfred/.pyenv/shims/python3 scripts/rucktalk_daily_social.py >> /home/aialfred/rucktalk_pipeline/social.log 2>&1

# RuckTalk Daily Blog — weekdays at 7:15 AM ET (11:15 UTC), via auto_blogger
15 11 * * 1-5 cd /home/aialfred/.openclaw/workspace/scripts/integrations && /home/aialfred/.pyenv/shims/python3 auto_blogger.py --site rucktalk --auto --publish >> /home/aialfred/rucktalk_pipeline/blog.log 2>&1
```

- [ ] **Step 2: Disable the old pipeline cron**

The existing cron runs `rucktalk_pipeline.py` from Oracle's scripts every 30 minutes on weekdays. Comment it out to avoid conflicts:

```bash
# Comment out the old cron (find the line with rucktalk_pipeline.py and prefix with #)
crontab -l | sed 's|^\(.*rucktalk_pipeline\.py.*\)$|# OLD: \1|' | crontab -
```

- [ ] **Step 3: Verify cron is installed**

```bash
crontab -l | grep rucktalk
```

Expected: Three new cron entries visible, old one commented out.

- [ ] **Step 4: Commit a record of the cron configuration**

No git commit needed — cron is system config, not code. But log it:

```bash
echo "Cron jobs installed $(date)" >> /home/aialfred/rucktalk_pipeline/pipeline.log
```

---

## Task 8: End-to-End Smoke Test

**Files:** None (testing)

- [ ] **Step 1: Test the common module**

```bash
cd /home/aialfred/alfred && python3 -c "
from scripts.rucktalk_common import *
setup_logging()
ensure_dirs()
state = load_state()
print('State:', json.dumps(state))
print('Next episode:', state.get('next_episode_number'))
print('Dirs created:', all(d.exists() for d in [INCOMING_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, CLIPS_DIR, METADATA_DIR, IMAGES_DIR]))
"
```

Expected: State shows next_episode_number=4, all directories exist.

- [ ] **Step 2: Test the daily social engine in dry-run mode**

```bash
cd /home/aialfred/alfred && python3 scripts/rucktalk_daily_social.py --dry-run --mode evergreen
```

Expected: Script generates content (LLM call, TTS, ComfyUI images, video assembly) but does not post. Check logs for each step completing.

- [ ] **Step 3: Test NextCloud episode detection**

```bash
cd /home/aialfred/alfred && python3 -c "
from scripts.rucktalk_episode_pipeline import check_for_new_episodes
pending = check_for_new_episodes()
print(f'Pending episodes: {pending}')
"
```

Expected: Empty list (or any unprocessed files in the new NextCloud path).

- [ ] **Step 4: Test episode pipeline with a small test file (optional)**

If Mike has a test MP4 to drop into `/RuckTalk/Episodes`, run:

```bash
cd /home/aialfred/alfred && python3 scripts/rucktalk_episode_pipeline.py
```

Watch the logs for each pipeline step. If no test file available, skip to deployment.

- [ ] **Step 5: Verify Telegram notifications work**

```bash
cd /home/aialfred/alfred && python3 -c "
from scripts.rucktalk_common import notify_telegram
notify_telegram('RuckTalk pipeline smoke test — all systems go, sir.')
"
```

Expected: Mike receives the Telegram message.

---

## Task 9: Final Commit & Cleanup

**Files:** All new files

- [ ] **Step 1: Final commit with all pipeline files**

```bash
cd /home/aialfred/alfred
git add scripts/rucktalk_common.py scripts/rucktalk_episode_pipeline.py scripts/rucktalk_daily_social.py
git status
git commit -m "feat(rucktalk): complete content pipeline — episode automation, daily social, daily blog

Three-engine system:
- Episode pipeline: NextCloud watch -> audio/transcribe -> YouTube -> WordPress -> blog -> clips
- Daily social: narrated video posts every morning (clips/trending/evergreen)
- Daily blog: auto_blogger revival for weekday SEO articles

Episode numbering starts at Episode 4."
```

- [ ] **Step 2: Verify all files are committed**

```bash
git log --oneline -5
```

Expected: Commits for common module, episode pipeline, daily social, Remotion template.
