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
import random
import shutil
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

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
    unposted = [c for c in queue if not c.get("posted")]
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


def _produce_narrated_video(content: dict, mode: str, dry_run: bool = False) -> dict | None:
    """
    Produce a narrated slideshow video from LLM-generated content.

    Args:
        content: dict with keys: narration_script, caption, image_prompts (list of 4),
                 pillar (optional), topic (optional)
        mode: "trend" or "evergreen"
        dry_run: If True, skip posting/scheduling.

    Returns result dict on success, None on failure.
    """
    run_id = uuid.uuid4().hex[:8]
    narration_script = content["narration_script"]
    caption = content["caption"]
    image_prompts = content["image_prompts"]

    # ── Step 1: Generate TTS narration ──
    narration_path = WORK_DIR / f"narration_{run_id}.mp3"
    logger.info("Generating TTS narration (%d chars)...", len(narration_script))
    tts_ok = run_tts(narration_script, str(narration_path))
    if not tts_ok:
        logger.error("TTS generation failed — aborting video production.")
        notify_telegram("🎙️ RuckTalk daily social: TTS failed. No post today.")
        return None

    # ── Step 2: Generate 4 ComfyUI images (portrait 1080x1920) ──
    image_paths = []
    for i, prompt in enumerate(image_prompts[:4]):
        full_prompt = f"{prompt}, {IMAGE_STYLE_SUFFIX}"
        logger.info("Generating image %d/4: %.60s...", i + 1, prompt)
        img_path = run_comfyui(full_prompt, width=1080, height=1920)
        if img_path:
            image_paths.append(img_path)
        else:
            logger.warning("Image %d/4 failed — continuing with available images.", i + 1)

        # 3-second cooldown between GPU calls (except after last)
        if i < 3:
            time.sleep(3)

    if len(image_paths) < 2:
        logger.error("Only %d images generated — need at least 2. Aborting.", len(image_paths))
        notify_telegram(
            f"🖼️ RuckTalk daily social: only {len(image_paths)}/4 images generated. "
            "GPU may be struggling. No post today."
        )
        return None

    logger.info("Generated %d/4 images successfully.", len(image_paths))

    # ── Step 3: Assemble video via video_render.py ──
    logger.info("Assembling slideshow video...")
    try:
        render_args = [
            "slideshow",
            *image_paths,
            "--audio", str(narration_path),
            "--transition", "fade",
            "--ratio", "portrait",
        ]
        result = run_script("video_render.py", *render_args, timeout=600)

        if result.returncode != 0:
            logger.error("Video render failed: %s", result.stderr[:500])
            notify_telegram("🎬 RuckTalk daily social: video render failed. No post today.")
            return None

        # Parse output path from JSON result
        video_path = None
        try:
            render_output = json.loads(result.stdout.strip())
            video_path = render_output.get("output") or render_output.get("path")
        except (json.JSONDecodeError, AttributeError):
            # Fallback: look for file path in stdout
            for line in reversed(result.stdout.strip().splitlines()):
                line = line.strip()
                if line.startswith("/") and Path(line).exists():
                    video_path = line
                    break

        if not video_path or not Path(video_path).exists():
            logger.error("Could not determine video output path from render.")
            notify_telegram("🎬 RuckTalk daily social: video render output not found.")
            return None

        logger.info("Video assembled: %s", video_path)

    except Exception as exc:
        logger.error("Video render error: %s", exc)
        notify_telegram(f"🎬 RuckTalk daily social: video render exception: {exc}")
        return None

    # ── Step 4: Copy to static media ──
    STATIC_MEDIA.mkdir(parents=True, exist_ok=True)
    video_filename = f"rucktalk_{mode}_{run_id}.mp4"
    static_path = STATIC_MEDIA / video_filename
    shutil.copy2(video_path, static_path)
    video_url = f"{PUBLIC_MEDIA_URL}/{video_filename}"
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
    unposted = [c for c in queue if not c.get("posted")]
    if not unposted:
        logger.warning("No unposted clips in queue.")
        return None

    clip = unposted[0]
    clip_path = clip.get("path", "")
    clip_title = clip.get("title", "RuckTalk Episode Clip")
    episode = clip.get("episode", "")

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
    video_url = clip.get("url")
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
    for c in queue:
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

    return _produce_narrated_video(content, mode="trend", dry_run=dry_run)


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

    return _produce_narrated_video(content, mode="evergreen", dry_run=dry_run)


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
