"""
rucktalk_common.py — Shared constants, state management, notification helpers,
LLM calling, and GPU management for the RuckTalk content pipeline.

Used by:
  - rucktalk_episode_pipeline.py  (Engine 1 & 2)
  - rucktalk_daily_social.py      (Engine 3)
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytz
import requests

# ─────────────────────────────────────────────
# Constants & Config
# ─────────────────────────────────────────────

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

NC_EPISODE_FOLDER = "/RuckTalk/Episodes"
NC_PROCESSED_FOLDER = "/RuckTalk/Episodes/Processed"

TG_TARGET = "7582976864"
OPENCLAW_BIN = "/home/aialfred/.nvm/versions/node/v22.22.0/bin/openclaw"

EST = pytz.timezone("America/New_York")

# LLM config
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

# Postiz integration IDs
POSTIZ_IDS = {
    "instagram": "cmmm0ck4m000iqudf4zkc2huz",
    "facebook": "cmmm0d0e8000kqudfvkjm6hzi",
    "youtube": "cmmm1r9n0000mqudfhdc436va",
    "linkedin": "cmnd9rvnx003bqtnvd9n6z7c6",
}

# ComfyUI image style suffix
IMAGE_STYLE_SUFFIX = (
    "cinematic photorealism, 8k, dramatic lighting, shallow depth of field, "
    "film grain, anamorphic lens flare, color graded, professional photography"
)

# Brand voice description
BRAND_VOICE = (
    "RuckTalk is a rucking podcast hosted by Mike Johnson. The tone is casual, "
    "knowledgeable, motivational, and community-driven. Think: a buddy who knows "
    "the science but keeps it real. Speaks to weekend warriors and serious ruckers "
    "alike. No fluff, no gatekeeping. Action-oriented."
)

# Content pillars with weights (higher = more frequent)
PILLARS = {
    "training_tips": 3,
    "gear_reviews": 2,
    "nutrition": 2,
    "community_stories": 2,
    "event_coverage": 1,
    "science_and_research": 1,
    "mindset_and_motivation": 2,
}

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logger = logging.getLogger("rucktalk")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging to both file and stdout."""
    log_file = WORK_DIR / "rucktalk.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(str(log_file))
    fh.setFormatter(fmt)
    fh.setLevel(level)

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)

    logger.setLevel(level)
    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(sh)


# ─────────────────────────────────────────────
# State Management
# ─────────────────────────────────────────────

_DEFAULT_STATE = {"processed": [], "next_episode_number": 4}


def ensure_dirs() -> None:
    """Create all pipeline working directories."""
    for d in (INCOMING_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, CLIPS_DIR, METADATA_DIR, IMAGES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    """Load pipeline state from processed_files.json."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load state file, using defaults: %s", exc)
    return dict(_DEFAULT_STATE)


def save_state(state: dict) -> None:
    """Persist pipeline state to processed_files.json."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_clip_queue() -> list:
    """Load the clip queue (list of clip dicts)."""
    if CLIP_QUEUE_FILE.exists():
        try:
            return json.loads(CLIP_QUEUE_FILE.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load clip queue, returning empty: %s", exc)
    return []


def save_clip_queue(queue: list) -> None:
    """Persist the clip queue."""
    CLIP_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLIP_QUEUE_FILE.write_text(json.dumps(queue, indent=2))


def load_social_history() -> dict:
    """Load social posting history."""
    if SOCIAL_HISTORY_FILE.exists():
        try:
            return json.loads(SOCIAL_HISTORY_FILE.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load social history, returning empty: %s", exc)
    return {}


def save_social_history(history: dict) -> None:
    """Persist social posting history."""
    SOCIAL_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SOCIAL_HISTORY_FILE.write_text(json.dumps(history, indent=2))


# ─────────────────────────────────────────────
# Notification
# ─────────────────────────────────────────────


def notify_telegram(msg: str) -> None:
    """Send a notification message to Mike via Telegram (openclaw CLI)."""
    try:
        subprocess.run(
            [OPENCLAW_BIN, "message", "send", TG_TARGET, msg],
            capture_output=True,
            text=True,
            timeout=60,
        )
        logger.info("Telegram notification sent.")
    except Exception as exc:
        logger.error("Failed to send Telegram notification: %s", exc)


# ─────────────────────────────────────────────
# LLM Helpers
# ─────────────────────────────────────────────


def llm_call(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> str | None:
    """
    Call the LLM with model fallback chain.
    Tries each model up to LLM_RETRIES_PER_MODEL times before moving on.
    Returns the response text or None on total failure.
    """
    for model in LLM_MODELS:
        for attempt in range(1, LLM_RETRIES_PER_MODEL + 1):
            try:
                logger.debug("LLM call: model=%s attempt=%d", model, attempt)
                resp = requests.post(
                    LLM_URL,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=LLM_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                if text and text.strip():
                    logger.info("LLM response from %s (%d chars)", model, len(text))
                    return text.strip()
                logger.warning("Empty LLM response from %s, retrying...", model)
            except Exception as exc:
                logger.warning(
                    "LLM call failed (model=%s attempt=%d): %s", model, attempt, exc
                )
            if attempt < LLM_RETRIES_PER_MODEL:
                time.sleep(LLM_RETRY_DELAY)

    logger.error("All LLM models exhausted — no response.")
    return None


def llm_json(prompt: str, temperature: float = 0.5) -> dict | None:
    """
    Call LLM expecting a JSON response.
    Strips markdown fences and parses JSON.
    Returns dict or None on failure.
    """
    raw = llm_call(prompt, temperature=temperature)
    if not raw:
        return None
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM JSON: %s\nRaw: %s", exc, raw[:500])
        return None


# ─────────────────────────────────────────────
# GPU Helpers
# ─────────────────────────────────────────────


COMFYUI_CLOUD_API_KEY = "comfyui-22acca191efd4a61b4b2f79886ab60015f5fe6d0859b84f6337e728cef9986b3"


def run_comfyui(prompt: str, width: int = 1024, height: int = 1024) -> str | None:
    """
    Generate an image via LOCAL ComfyUI (GPU on 105).
    Used for daily social images.
    Returns the local file path on success, or None on failure.
    """
    return _run_comfyui_impl(prompt, width, height, use_cloud=False)


def run_comfyui_cloud(prompt: str, width: int = 1024, height: int = 1024) -> str | None:
    """
    Generate an image via ComfyUI Cloud.
    Used for blog images and episode cover art — offloads from local GPU.
    Returns the local file path on success, or None on failure.
    """
    result = _run_comfyui_impl(prompt, width, height, use_cloud=True)
    if result is None:
        # Fallback to local if cloud fails
        logger.warning("ComfyUI Cloud failed, falling back to local GPU.")
        return _run_comfyui_impl(prompt, width, height, use_cloud=False)
    return result


def _run_comfyui_impl(prompt: str, width: int, height: int, use_cloud: bool) -> str | None:
    """Internal: generate image via local or cloud ComfyUI."""
    env = os.environ.copy()
    if use_cloud:
        env["COMFYUI_CLOUD_API_KEY"] = COMFYUI_CLOUD_API_KEY
        mode_label = "cloud"
    else:
        env.pop("COMFYUI_CLOUD_API_KEY", None)
        mode_label = "local"

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "comfyui_gen.py"),
                "generate",
                prompt,
                "--width",
                str(width),
                "--height",
                str(height),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        if result.returncode != 0:
            logger.error("ComfyUI failed: %s", result.stderr[:500])
            return None

        # Parse output — comfyui_gen.py outputs JSON with image_url and filename
        output = result.stdout.strip()

        # Try to parse JSON output
        try:
            import json as _json
            data = _json.loads(output)
            filename = data.get("filename", "")
            if filename:
                # Check all possible locations
                for check_path in [
                    f"/home/aialfred/ComfyUI/output/{filename}",  # local mode
                    f"/home/aialfred/alfred/static/media/{filename}",  # media dir
                    f"/tmp/{filename}",  # cloud mode downloads here
                ]:
                    if os.path.isfile(check_path):
                        logger.info("ComfyUI (%s) generated: %s", mode_label, check_path)
                        return check_path
        except (ValueError, KeyError):
            pass

        # Fallback: look for local file paths in output
        for line in reversed(output.splitlines()):
            line = line.strip()
            if line and (line.endswith(".png") or line.endswith(".jpg") or line.endswith(".webp")):
                if os.path.isfile(line):
                    logger.info("ComfyUI generated: %s", line)
                    return line
            # Check if it's a filename (not a path) — look in ComfyUI output dir
            if line and not line.startswith("/") and (line.endswith(".png") or line.endswith(".jpg")):
                comfyui_path = f"/home/aialfred/ComfyUI/output/{line}"
                if os.path.isfile(comfyui_path):
                    logger.info("ComfyUI generated: %s", comfyui_path)
                    return comfyui_path

        # Fallback: look for any path-like string
        for line in reversed(output.splitlines()):
            line = line.strip()
            if line.startswith("/") and os.path.isfile(line):
                logger.info("ComfyUI generated: %s", line)
                return line

        # Last resort: find the most recent file in ComfyUI output dir
        comfyui_out = Path("/home/aialfred/ComfyUI/output")
        if comfyui_out.exists():
            recent = sorted(comfyui_out.glob("alfred_flux_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            if recent:
                logger.info("ComfyUI generated (by recency): %s", recent[0])
                return str(recent[0])

        logger.error("Could not find output file in ComfyUI output:\n%s", output[:500])
        return None
    except Exception as exc:
        logger.error("ComfyUI error: %s", exc)
        return None


def run_comfyui_video_cloud(prompt: str, duration: int = 10) -> str | None:
    """
    Generate a short AI video via ComfyUI Cloud (LTX-2 model).
    Portrait orientation (9:16) for Shorts/Reels/TikTok.
    Returns local file path of the MP4, or None on failure.
    """
    env = os.environ.copy()
    env["COMFYUI_CLOUD_API_KEY"] = COMFYUI_CLOUD_API_KEY

    # LTX-2: 9:16 portrait, duration in frames (~16fps)
    frames = duration * 16

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "comfyui_gen.py"),
                "video",
                prompt,
                "--model", "ltx",
                "--width", "768",
                "--height", "1344",
                "--length", str(frames),
            ],
            capture_output=True,
            text=True,
            timeout=1200,  # 20 min max for video
            env=env,
        )

        output = result.stdout.strip()
        logger.info("ComfyUI Cloud video output: %s", output[:300])

        if result.returncode != 0:
            logger.error("ComfyUI Cloud video failed: %s", result.stderr[:500])
            return None

        # Parse JSON output for filename
        try:
            data = json.loads(output)
            filename = data.get("filename", "")
            if filename:
                for check_path in [
                    f"/home/aialfred/ComfyUI/output/{filename}",
                    f"/home/aialfred/ComfyUI/output/video/{filename}",
                    f"/home/aialfred/alfred/static/drafts/{filename}",
                    f"/tmp/{filename}",
                ]:
                    if os.path.isfile(check_path):
                        logger.info("ComfyUI Cloud video generated: %s", check_path)
                        return check_path
                # Check video subdirectories
                video_dir = Path("/home/aialfred/ComfyUI/output/video")
                if video_dir.exists():
                    for f in video_dir.rglob(f"*{filename}*"):
                        if f.is_file():
                            logger.info("ComfyUI Cloud video found: %s", f)
                            return str(f)
            # Check for video_url in output
            video_url = data.get("video_url", "")
            if video_url and video_url.startswith("http"):
                # Download to local
                local = WORK_DIR / f"cloud_video_{uuid.uuid4().hex[:8]}.mp4"
                urllib.request.urlretrieve(video_url, str(local))
                logger.info("ComfyUI Cloud video downloaded: %s", local)
                return str(local)
        except (ValueError, KeyError):
            pass

        # Fallback: find most recent video in output dirs
        for search_dir in [Path("/home/aialfred/ComfyUI/output/video"),
                           Path("/home/aialfred/ComfyUI/output"),
                           Path("/home/aialfred/alfred/static/drafts")]:
            if search_dir.exists():
                recent = sorted(search_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
                if recent and (time.time() - recent[0].stat().st_mtime) < 300:  # within 5 min
                    logger.info("ComfyUI Cloud video (by recency): %s", recent[0])
                    return str(recent[0])

        logger.error("Could not find video output: %s", output[:500])
        return None
    except Exception as exc:
        logger.error("ComfyUI Cloud video error: %s", exc)
        return None


def run_tts(text: str, output_path: str) -> bool:
    """
    Generate TTS audio via Kokoro at port 8880.
    Returns True on success, False on failure.
    """
    try:
        resp = requests.post(
            "http://127.0.0.1:8880/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": text,
                "voice": "am_michael",
                "response_format": "mp3",
            },
            timeout=120,
        )
        resp.raise_for_status()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(resp.content)
        logger.info("TTS generated: %s (%d bytes)", output_path, len(resp.content))
        return True
    except Exception as exc:
        logger.error("TTS failed: %s", exc)
        return False


def _download_url(url: str) -> str | None:
    """
    Download a URL to IMAGES_DIR.
    Returns the local file path or None on failure.
    """
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        # Determine extension from content-type or URL
        content_type = resp.headers.get("content-type", "")
        if "png" in content_type or url.endswith(".png"):
            ext = ".png"
        elif "jpeg" in content_type or "jpg" in content_type or url.endswith(".jpg"):
            ext = ".jpg"
        elif "webp" in content_type or url.endswith(".webp"):
            ext = ".webp"
        else:
            ext = ".png"

        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}{ext}"
        local_path = IMAGES_DIR / filename
        local_path.write_bytes(resp.content)
        logger.info("Downloaded %s -> %s (%d bytes)", url[:80], local_path, len(resp.content))
        return str(local_path)
    except Exception as exc:
        logger.error("Download failed for %s: %s", url[:80], exc)
        return None


def run_script(script_name: str, *args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """
    Run an Oracle integration script from SCRIPTS directory.
    Returns the CompletedProcess result.
    """
    script_path = SCRIPTS / script_name
    cmd = [sys.executable, str(script_path)] + list(args)
    logger.info("Running script: %s %s", script_name, " ".join(args))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            logger.warning(
                "Script %s exited %d: %s",
                script_name,
                result.returncode,
                result.stderr[:500],
            )
        else:
            logger.info("Script %s completed successfully.", script_name)
        return result
    except subprocess.TimeoutExpired:
        logger.error("Script %s timed out after %ds", script_name, timeout)
        raise
    except Exception as exc:
        logger.error("Script %s failed: %s", script_name, exc)
        raise
