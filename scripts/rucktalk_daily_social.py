#!/usr/bin/env python3
"""
rucktalk_daily_social.py — Daily Social Content Engine for RuckTalk.

Runs daily via cron at 7 AM ET. Generates and schedules one social media
post across all RuckTalk platforms (Instagram, Facebook, YouTube, LinkedIn).

Content priority:
  1. Episode clip from queue (if available)
  2. Current-event-driven content (~40% when no clips)
  3. Evergreen pillar content (~60% when no clips)

Every post is a VIDEO:
  - Clip days: pre-rendered episode clip
  - Non-clip days: LLM writes 60-90s script → Kokoro TTS → ComfyUI images → slideshow video
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rucktalk_common import (
    WORK_DIR, CLIPS_DIR, IMAGES_DIR, STATIC_MEDIA, PUBLIC_MEDIA_URL,
    SCRIPTS, POSTIZ_IDS, EST,
    logger, setup_logging, ensure_dirs,
    load_clip_queue, save_clip_queue,
    load_social_history, save_social_history,
    notify_telegram, llm_call, llm_json,
    run_comfyui, run_comfyui_video_cloud, run_tts, run_script,
    BRAND_VOICE, IMAGE_STYLE_SUFFIX, PILLARS,
)

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


# ─────────────────────────────────────────────
# Content Selection
# ─────────────────────────────────────────────


def select_content_mode(history: dict) -> str:
    """
    Decide today's content type.

    Priority:
      1. "clip" if unposted clips exist in queue
      2. "trend" (40%) or "evergreen" (60%) random
    """
    queue = load_clip_queue()
    unposted = [c for c in queue.get("clips", []) if not c.get("posted")]
    if unposted:
        logger.info("Clip queue has %d unposted clips — selecting clip mode.", len(unposted))
        return "clip"

    roll = random.random()
    mode = "trend" if roll < 0.4 else "evergreen"
    logger.info("No clips available. Random roll %.2f → %s mode.", roll, mode)
    return mode


def select_pillar(history: dict) -> tuple[str, str]:
    """
    Weighted random pillar selection, avoiding the last 5 recently used pillars.

    Returns (pillar_id, pillar_display_name).
    """
    recent_posts = history.get("posts", [])
    recent_pillars = []
    for post in recent_posts[-5:]:
        p = post.get("pillar")
        if p:
            recent_pillars.append(p)

    # Build weighted pool excluding recent pillars
    pool = {k: v for k, v in PILLARS.items() if k not in recent_pillars}

    # If all pillars were recently used, fall back to full set
    if not pool:
        logger.info("All pillars recently used — resetting exclusion list.")
        pool = dict(PILLARS)

    # Weighted random selection
    pillar_ids = list(pool.keys())
    weights = [pool[k] for k in pillar_ids]
    chosen = random.choices(pillar_ids, weights=weights, k=1)[0]

    display_name = chosen.replace("_", " ").title()
    logger.info("Selected pillar: %s (%s)", chosen, display_name)
    return chosen, display_name


# ─────────────────────────────────────────────
# Scheduling Helpers
# ─────────────────────────────────────────────


def _next_morning_utc() -> str:
    """
    Returns tomorrow at 11:00 UTC (7 AM ET) as 'YYYY-MM-DDT11:00:00'.
    """
    now = datetime.utcnow()
    tomorrow = now + timedelta(days=1)
    schedule_dt = tomorrow.replace(hour=11, minute=0, second=0, microsecond=0)
    return schedule_dt.strftime("%Y-%m-%dT%H:%M:%S")


def schedule_to_postiz(
    caption: str,
    schedule_dt: str,
    video_url: str | None = None,
    image_url: str | None = None,
) -> bool:
    """
    Schedule a post to all RuckTalk platforms via postiz.py.

    Args:
        caption: Post text/caption.
        schedule_dt: ISO datetime string for scheduling.
        video_url: Public URL for video media.
        image_url: Public URL for image media (fallback if no video).

    Returns True on success, False on failure.
    """
    all_ids = ",".join(POSTIZ_IDS.values())
    media_url = video_url or image_url or ""

    args = [caption, schedule_dt, all_ids]
    if media_url:
        args.append(media_url)

    try:
        result = run_script("postiz.py", "create-post", *args)
        if result.returncode == 0:
            logger.info("Postiz scheduling succeeded for %s", schedule_dt)
            return True
        else:
            logger.error("Postiz scheduling failed: %s", result.stderr[:500])
            return False
    except Exception as exc:
        logger.error("Postiz scheduling error: %s", exc)
        return False


# ─────────────────────────────────────────────
# Video Production
# ─────────────────────────────────────────────


def _select_format(history: dict) -> str:
    """Pick between 'monologue' (Kokoro+ComfyUI) and 'conversation' (NotebookLM).

    Rotates formats so they alternate, giving variety.
    """
    recent_formats = [p.get("format", "monologue") for p in history.get("posts", [])[-5:]]
    last_format = recent_formats[-1] if recent_formats else "conversation"

    # Alternate, with slight bias toward monologue (it's more reliable)
    if last_format == "conversation":
        return "monologue"
    else:
        return "conversation"


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


def _produce_monologue_video(content: dict, mode: str, run_id: str) -> str | None:
    """Dispatcher — routes to legacy ffmpeg path or new Remotion path based on DAILY_SOCIAL_ENGINE."""
    if DAILY_SOCIAL_ENGINE == "remotion":
        return _produce_monologue_video_remotion(content, mode, run_id)
    return _produce_monologue_video_legacy(content, mode, run_id)


def _produce_monologue_video_legacy(content: dict, mode: str, run_id: str) -> str | None:
    """Format A — 'The Monologue': Kokoro TTS narration over ComfyUI Cloud AI video.

    Returns local video file path, or None on failure.
    """
    narration_script = content["narration_script"]
    video_prompt = content.get("video_prompt", content.get("image_prompts", [""])[0])

    # Step 1: Generate TTS narration
    narration_path = WORK_DIR / f"narration_{run_id}.mp3"
    logger.info("Generating TTS narration (%d chars)...", len(narration_script))
    tts_ok = run_tts(narration_script, str(narration_path))
    if not tts_ok:
        logger.error("TTS generation failed.")
        return None

    # Step 2: Generate AI video via ComfyUI Cloud (LTX-2, portrait)
    full_prompt = f"{video_prompt}, {IMAGE_STYLE_SUFFIX}"
    logger.info("Generating AI video via ComfyUI Cloud...")
    video_path = run_comfyui_video_cloud(full_prompt, duration=10)

    if not video_path:
        logger.warning("Cloud video failed — falling back to local image slideshow.")
        # Fallback: generate one image and loop it with narration
        image_prompts = content.get("image_prompts", [video_prompt])
        img_path = run_comfyui(f"{image_prompts[0]}, {IMAGE_STYLE_SUFFIX}", width=1080, height=1920)
        if not img_path:
            logger.error("Fallback image generation also failed.")
            return None

        # Build simple video from single image + narration
        output = WORK_DIR / f"monologue_{run_id}.mp4"
        import subprocess as _sp
        r = _sp.run([
            "ffmpeg", "-y",
            "-loop", "1", "-i", img_path,
            "-i", str(narration_path),
            "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            "-shortest", str(output),
        ], capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and output.exists():
            logger.info("Fallback video built: %s", output)
            return str(output)
        logger.error("Fallback ffmpeg failed: %s", r.stderr[-300:])
        return None

    # Step 3: Composite narration over AI video
    output = WORK_DIR / f"monologue_{run_id}.mp4"
    import subprocess as _sp
    r = _sp.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", str(narration_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-map", "0:v", "-map", "1:a",
        "-shortest", str(output),
    ], capture_output=True, text=True, timeout=120)

    if r.returncode == 0 and output.exists():
        logger.info("Monologue video assembled: %s", output)
        return str(output)

    logger.error("ffmpeg composite failed: %s", r.stderr[-300:])
    return None


def _produce_monologue_video_remotion(content: dict, mode: str, run_id: str) -> str | None:
    """Format A via Remotion KineticTypeRig.

    Same inputs as legacy (Kokoro TTS audio + ComfyUI Cloud video) but
    composed through the Remotion rig instead of raw ffmpeg. Captions
    appear on-screen in sync with the voiceover.
    """
    script = content.get("script") or content.get("narration_script") or ""
    if not script:
        logger.error("Monologue content has no script/narration text — cannot derive captions.")
        return None

    # 1. Run Kokoro TTS to get the voiceover mp3
    narration_path = WORK_DIR / f"monologue_{run_id}_narration.mp3"
    tts_ok = run_tts(script, str(narration_path))
    if not tts_ok or not narration_path.exists():
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
    cloud_ok = run_comfyui_video_cloud(topic_prompt, str(bg_video_path),
                                        duration_s=max(8.0, audio_duration_s + 1.0))
    if not cloud_ok or not bg_video_path.exists():
        logger.warning("ComfyUI Cloud bg video failed — aborting remotion monologue.")
        return None

    # 3. Copy bg to Remotion public/ so staticFile() serves it
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


def _produce_conversation_video_legacy(content: dict, mode: str, run_id: str) -> str | None:
    """Format B — 'The Conversation': NotebookLM 2-person podcast over ComfyUI Cloud video.

    Returns local video file path, or None on failure.
    Falls back to monologue format if NotebookLM fails.
    """
    topic = content.get("topic", "RuckTalk")
    source_text = content.get("narration_script", "")
    video_prompt = content.get("video_prompt", content.get("image_prompts", [""])[0])

    # Step 1: Try NotebookLM podcast generation
    logger.info("Generating NotebookLM conversation for: %s", topic)
    audio_path = WORK_DIR / f"conversation_{run_id}.mp3"

    try:
        import subprocess as _sp

        # Build source text for NotebookLM — use the narration script + topic context
        nlm_source = (
            f"RuckTalk — Tactical Living for the Modern Entrepreneur\n\n"
            f"Topic: {topic}\n\n"
            f"{source_text}\n\n"
            f"This content is about: {content.get('pillar', 'entrepreneurship and discipline')}. "
            f"The tone should be direct, motivating, no-BS — like two guys who train together "
            f"talking about real life."
        )

        # Write source to temp file
        source_file = WORK_DIR / f"nlm_source_{run_id}.txt"
        source_file.write_text(nlm_source)

        # Call notebooklm-py to generate audio
        # Flow: create notebook → add source → generate audio → download
        nlm_script = f'''
import asyncio
from notebooklm import NotebookLMClient

async def main():
    async with NotebookLMClient() as client:
        nb = await client.notebooks.create("RuckTalk: {topic[:50]}")
        await client.sources.add_text(nb.id, open("{source_file}").read(), wait=True)
        await client.artifacts.generate_audio(nb.id, instructions="Make it engaging, conversational, two hosts discussing this topic with energy and authenticity. Keep it under 3 minutes.")

        # Poll for completion
        for _ in range(60):
            artifact = await client.artifacts.get(nb.id)
            if hasattr(artifact, "status") and artifact.status == "completed":
                break
            await asyncio.sleep(10)

        audio = await client.artifacts.download(artifact.id)
        with open("{audio_path}", "wb") as f:
            f.write(audio)

        # Clean up notebook
        await client.notebooks.delete(nb.id)

asyncio.run(main())
'''
        nlm_result = _sp.run(
            [sys.executable, "-c", nlm_script],
            capture_output=True, text=True, timeout=600,
        )

        if nlm_result.returncode != 0 or not audio_path.exists():
            logger.warning("NotebookLM failed: %s", nlm_result.stderr[:300])
            raise RuntimeError("NotebookLM generation failed")

        logger.info("NotebookLM audio generated: %s (%d bytes)", audio_path, audio_path.stat().st_size)

    except Exception as exc:
        logger.warning("NotebookLM unavailable (%s) — falling back to monologue format.", exc)
        return _produce_monologue_video(content, mode, run_id)

    # Step 2: Generate AI video via ComfyUI Cloud
    full_prompt = f"{video_prompt}, {IMAGE_STYLE_SUFFIX}"
    logger.info("Generating background video via ComfyUI Cloud...")
    video_path = run_comfyui_video_cloud(full_prompt, duration=10)

    if not video_path:
        # Fallback: generate a single image and loop it
        img_path = run_comfyui(f"{video_prompt}, {IMAGE_STYLE_SUFFIX}", width=1080, height=1920)
        if img_path:
            video_path = WORK_DIR / f"bg_{run_id}.mp4"
            import subprocess as _sp2
            _sp2.run([
                "ffmpeg", "-y", "-loop", "1", "-i", img_path,
                "-vf", "scale=1080:1920,zoompan=z='min(zoom+0.0003,1.1)':d=500:s=1080x1920",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-t", "180",  # 3 min max
                str(video_path),
            ], capture_output=True, timeout=60)
            video_path = str(video_path)

    # Step 3: Composite NotebookLM audio over video
    output = WORK_DIR / f"conversation_{run_id}_final.mp4"
    import subprocess as _sp
    if video_path and Path(video_path).exists():
        r = _sp.run([
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            "-map", "0:v", "-map", "1:a",
            "-shortest", str(output),
        ], capture_output=True, text=True, timeout=300)
    else:
        # Audio only — create a simple black background video with audio
        r = _sp.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=300",
            "-i", str(audio_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(output),
        ], capture_output=True, text=True, timeout=300)

    if r.returncode == 0 and output.exists():
        logger.info("Conversation video assembled: %s", output)
        return str(output)

    logger.error("Conversation video assembly failed: %s", r.stderr[-300:])
    # Last resort fallback to monologue
    return _produce_monologue_video(content, mode, run_id)


def _produce_conversation_video(content: dict, mode: str, run_id: str) -> str | None:
    """Dispatcher — routes to legacy ffmpeg path or new Remotion path based on DAILY_SOCIAL_ENGINE."""
    if DAILY_SOCIAL_ENGINE == "remotion":
        return _produce_conversation_video_remotion(content, mode, run_id)
    return _produce_conversation_video_legacy(content, mode, run_id)


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
        import asyncio
        from notebooklm import NotebookLMClient

        async def _gen():
            async with NotebookLMClient() as client:
                nb = await client.notebooks.create(name=f"RuckTalk daily {run_id}")
                await client.sources.add_text(nb.id, f"Topic: {topic}\n\nScript: {content.get('narration_script','')}")
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
        content.get("image_prompt") or content.get("video_prompt") or "rucking outdoors",
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


def _produce_video(content: dict, mode: str, dry_run: bool = False) -> dict | None:
    """
    Produce a social video using the best available format.

    Rotates between:
      - Format A: 'monologue' (Kokoro TTS + ComfyUI Cloud video)
      - Format B: 'conversation' (NotebookLM + ComfyUI Cloud video)

    Returns result dict on success, None on failure.
    """
    run_id = uuid.uuid4().hex[:8]
    caption = content["caption"]
    history = load_social_history()

    # Pick format
    fmt = _select_format(history)
    logger.info("Selected format: %s", fmt)

    # Produce the video
    if fmt == "conversation":
        video_path = _produce_conversation_video(content, mode, run_id)
    else:
        video_path = _produce_monologue_video(content, mode, run_id)

    if not video_path or not Path(video_path).exists():
        logger.error("Video production failed for format: %s", fmt)
        notify_telegram(f"RuckTalk daily social: {fmt} video failed. No post today.")
        return None

    # ── Copy to static media ──
    rucktalk_media = STATIC_MEDIA / "rucktalk"
    rucktalk_media.mkdir(parents=True, exist_ok=True)
    video_filename = f"rucktalk_{mode}_{run_id}.mp4"
    static_path = rucktalk_media / video_filename
    shutil.copy2(video_path, str(static_path))
    video_url = f"{PUBLIC_MEDIA_URL}/rucktalk/{video_filename}"
    logger.info("Video copied to static: %s", static_path)

    # ── Step 5: Schedule to Postiz ──
    schedule_dt = _next_morning_utc()
    posted = False
    if not dry_run:
        posted = schedule_to_postiz(caption, schedule_dt, video_url=video_url)
    else:
        logger.info("[DRY RUN] Would schedule to Postiz at %s", schedule_dt)

    # ── Step 6: Record in social history ──
    history = load_social_history()
    if "posts" not in history:
        history["posts"] = []

    record = {
        "id": run_id,
        "mode": mode,
        "format": fmt,
        "date": datetime.now(EST).strftime("%Y-%m-%d"),
        "caption": caption[:500],
        "video_url": video_url,
        "schedule_dt": schedule_dt,
        "posted": posted,
        "pillar": content.get("pillar"),
        "topic": content.get("topic"),
    }
    history["posts"].append(record)
    save_social_history(history)

    return record


# ─────────────────────────────────────────────
# Content Generators
# ─────────────────────────────────────────────


def post_episode_clip(dry_run: bool = False) -> dict | None:
    """
    Post the next unposted episode clip from the queue.

    Pops the next clip, generates a caption via LLM, schedules via Postiz,
    and marks the clip as posted.
    """
    queue = load_clip_queue()
    unposted = [c for c in queue.get("clips", []) if not c.get("posted")]
    if not unposted:
        logger.warning("No unposted clips in queue.")
        return None

    clip = unposted[0]
    clip_path = clip.get("portrait_path") or clip.get("path", "")
    clip_title = clip.get("label") or clip.get("title", "RuckTalk Episode Clip")
    episode = clip.get("episode_number") or clip.get("episode", "")

    logger.info("Processing clip: %s (episode: %s)", clip_title, episode)

    # Generate caption via LLM
    caption_prompt = f"""You are the social media voice for RuckTalk, a rucking podcast.
{BRAND_VOICE}

Write a short, punchy social media caption for this episode clip:
Title: {clip_title}
Episode: {episode}
Description: {clip.get('description', 'A clip from the latest RuckTalk episode.')}

Rules:
- Start with a strong hook (question or bold statement)
- Tease the full episode
- Include a clear CTA (watch/listen to full episode)
- Add 5-8 relevant hashtags at the end
- Keep it under 280 characters (before hashtags)
- NO escape characters, use natural line breaks
- Write in plain text only
"""
    caption = llm_call(caption_prompt, temperature=0.8)
    if not caption:
        logger.error("Failed to generate caption for clip.")
        notify_telegram("📝 RuckTalk daily social: caption generation failed for clip.")
        return None

    # Determine media URL
    # If clip has a public URL, use it; otherwise copy to static
    video_url = clip.get("web_url") or clip.get("url")
    if not video_url and clip_path and Path(clip_path).exists():
        STATIC_MEDIA.mkdir(parents=True, exist_ok=True)
        clip_filename = f"rucktalk_clip_{uuid.uuid4().hex[:8]}{Path(clip_path).suffix}"
        static_path = STATIC_MEDIA / clip_filename
        shutil.copy2(clip_path, static_path)
        video_url = f"{PUBLIC_MEDIA_URL}/{clip_filename}"
        logger.info("Clip copied to static: %s", static_path)

    # Schedule to Postiz
    schedule_dt = _next_morning_utc()
    posted = False
    if not dry_run:
        if video_url:
            posted = schedule_to_postiz(caption, schedule_dt, video_url=video_url)
        else:
            logger.warning("No video URL available for clip — skipping Postiz.")
    else:
        logger.info("[DRY RUN] Would schedule clip to Postiz at %s", schedule_dt)

    # Mark clip as posted in queue
    for c in queue.get("clips", []):
        if c is clip or (c.get("path") == clip.get("path") and c.get("title") == clip.get("title")):
            c["posted"] = True
            c["posted_date"] = datetime.now(EST).strftime("%Y-%m-%d")
            break
    save_clip_queue(queue)

    # Record in social history
    history = load_social_history()
    if "posts" not in history:
        history["posts"] = []

    record = {
        "id": uuid.uuid4().hex[:8],
        "mode": "clip",
        "date": datetime.now(EST).strftime("%Y-%m-%d"),
        "caption": caption[:500],
        "video_url": video_url,
        "schedule_dt": schedule_dt,
        "posted": posted,
        "episode": episode,
        "clip_title": clip_title,
    }
    history["posts"].append(record)
    save_social_history(history)

    return record


def generate_trending_content(dry_run: bool = False) -> dict | None:
    """
    Generate a trending-topic narrated video post.

    Searches for trending rucking/fitness topics, picks the best one,
    writes a 60-90 second narration script, and produces a video.
    """
    logger.info("Generating trending content...")

    # Search for trending topics
    search_terms = [
        "rucking fitness trending news",
        "ruck march training 2026",
        "weighted hiking trend",
        "rucking benefits new research",
    ]
    search_results = []
    for term in search_terms:
        try:
            result = run_script("search.py", "query", term)
            if result.returncode == 0 and result.stdout.strip():
                search_results.append(result.stdout.strip())
        except Exception as exc:
            logger.warning("Search failed for '%s': %s", term, exc)

    search_context = "\n\n".join(search_results[:3]) if search_results else "No search results available."

    # LLM picks topic and writes content
    content_prompt = f"""You are the content strategist for RuckTalk, a rucking podcast.
{BRAND_VOICE}

Based on these recent search results about rucking and fitness trends:
---
{search_context[:3000]}
---

Create a trending social media video post. Return valid JSON with these exact keys:
{{
  "topic": "short topic title",
  "narration_script": "60-90 second narration script. Conversational, informative, engaging. Written to be spoken aloud. No stage directions.",
  "caption": "social media caption with hook + insight + CTA + 5-8 hashtags. Plain text, natural line breaks only.",
  "image_prompts": [
    "prompt 1 for a dramatic portrait image related to the topic",
    "prompt 2 for a different angle on the topic",
    "prompt 3 showing action/movement related to the topic",
    "prompt 4 showing results/community related to the topic"
  ]
}}

Rules for narration_script:
- 60-90 seconds when read aloud (~150-220 words)
- Open with a hook that grabs attention
- Reference the trending angle
- End with a call to action
- Conversational tone, not scripted

Rules for image_prompts:
- Each prompt should describe a photorealistic scene
- Portrait orientation (1080x1920)
- Related to rucking, fitness, outdoor training
- No text or logos in the image
"""

    content = llm_json(content_prompt)
    if not content:
        logger.error("Failed to generate trending content from LLM.")
        notify_telegram("📝 RuckTalk daily social: LLM failed to generate trending content.")
        return None

    # Validate required keys
    required_keys = ["narration_script", "caption", "image_prompts"]
    for key in required_keys:
        if key not in content:
            logger.error("LLM response missing key: %s", key)
            notify_telegram(f"📝 RuckTalk daily social: LLM response missing '{key}'.")
            return None

    if not isinstance(content["image_prompts"], list) or len(content["image_prompts"]) < 2:
        logger.error("LLM returned insufficient image prompts.")
        return None

    content["topic"] = content.get("topic", "Trending Rucking Topic")

    return _produce_video(content, mode="trend", dry_run=dry_run)


def generate_evergreen_content(dry_run: bool = False) -> dict | None:
    """
    Generate an evergreen pillar-based narrated video post.

    Picks a content pillar, writes timeless content with narration script,
    and produces a video.
    """
    history = load_social_history()
    pillar_id, pillar_name = select_pillar(history)
    logger.info("Generating evergreen content for pillar: %s", pillar_name)

    content_prompt = f"""You are the content strategist for RuckTalk, a rucking podcast.
{BRAND_VOICE}

Create an EVERGREEN social media video post about: {pillar_name}

This content should be timeless — valuable today and in 6 months.

Return valid JSON with these exact keys:
{{
  "narration_script": "60-90 second narration script. Conversational, informative, engaging. Written to be spoken aloud. No stage directions.",
  "caption": "social media caption with hook + value + CTA + 5-8 hashtags. Plain text, natural line breaks only.",
  "image_prompts": [
    "prompt 1 for a dramatic portrait image about {pillar_name.lower()}",
    "prompt 2 for a different angle on {pillar_name.lower()}",
    "prompt 3 showing action/movement related to {pillar_name.lower()}",
    "prompt 4 showing results/community related to {pillar_name.lower()}"
  ]
}}

Rules for narration_script:
- 60-90 seconds when read aloud (~150-220 words)
- Open with a hook or bold statement
- Provide actionable value
- End with encouragement and call to action
- Conversational tone, think podcast host talking to a friend

Rules for image_prompts:
- Each prompt should describe a photorealistic scene
- Portrait orientation (1080x1920)
- Related to rucking and {pillar_name.lower()}
- No text or logos in the image

Pillar context for {pillar_name}:
- training_tips: Practical rucking workouts, form tips, programming advice
- gear_reviews: Rucksacks, plates, boots, hydration — honest takes
- nutrition: Fueling for rucks, recovery meals, hydration strategies
- community_stories: Real ruckers, their journeys, transformation stories
- event_coverage: GORUCK events, ruck marches, community meetups
- science_and_research: Studies on loaded carries, benefits, injury prevention
- mindset_and_motivation: Mental toughness, discipline, the ruck mindset
"""

    content = llm_json(content_prompt)
    if not content:
        logger.error("Failed to generate evergreen content from LLM.")
        notify_telegram(f"📝 RuckTalk daily social: LLM failed for pillar '{pillar_name}'.")
        return None

    # Validate required keys
    required_keys = ["narration_script", "caption", "image_prompts"]
    for key in required_keys:
        if key not in content:
            logger.error("LLM response missing key: %s", key)
            notify_telegram(f"📝 RuckTalk daily social: LLM response missing '{key}'.")
            return None

    if not isinstance(content["image_prompts"], list) or len(content["image_prompts"]) < 2:
        logger.error("LLM returned insufficient image prompts.")
        return None

    content["pillar"] = pillar_id

    return _produce_video(content, mode="evergreen", dry_run=dry_run)


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RuckTalk Daily Social Engine — generate and schedule one post per day."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate content but don't schedule to Postiz.",
    )
    parser.add_argument(
        "--mode",
        choices=["clip", "trend", "evergreen"],
        default=None,
        help="Force a specific content mode (default: auto-select).",
    )
    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    logger.info("=" * 60)
    logger.info("RuckTalk Daily Social Engine — starting")
    logger.info("Dry run: %s", args.dry_run)

    history = load_social_history()

    # Select content mode
    if args.mode:
        mode = args.mode
        logger.info("Mode forced via CLI: %s", mode)
    else:
        mode = select_content_mode(history)

    logger.info("Content mode: %s", mode)

    # Generate and post
    result = None
    try:
        if mode == "clip":
            result = post_episode_clip(dry_run=args.dry_run)
        elif mode == "trend":
            result = generate_trending_content(dry_run=args.dry_run)
        elif mode == "evergreen":
            result = generate_evergreen_content(dry_run=args.dry_run)
        else:
            logger.error("Unknown mode: %s", mode)
    except Exception as exc:
        logger.exception("Unhandled error in %s mode: %s", mode, exc)
        notify_telegram(f"💥 RuckTalk daily social crashed in {mode} mode: {exc}")
        sys.exit(1)

    # Report outcome
    if result:
        dry_tag = " [DRY RUN]" if args.dry_run else ""
        msg = (
            f"📢 RuckTalk Daily Social{dry_tag}\n"
            f"Mode: {mode}\n"
            f"Scheduled: {result.get('schedule_dt', 'N/A')}\n"
            f"Caption: {result.get('caption', '')[:120]}..."
        )
        logger.info("Post generated successfully: %s", result.get("id"))
        if not args.dry_run:
            notify_telegram(msg)
    else:
        logger.warning("No content generated for mode: %s", mode)
        if not args.dry_run:
            notify_telegram(f"⚠️ RuckTalk daily social: no content generated ({mode} mode).")

    logger.info("RuckTalk Daily Social Engine — done")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
