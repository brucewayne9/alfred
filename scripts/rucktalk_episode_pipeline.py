#!/usr/bin/env python3
"""
rucktalk_episode_pipeline.py — RuckTalk Episode Pipeline Orchestrator

Polls NextCloud for new MP4 files in /RuckTalk/Episodes and runs a full
automation pipeline: download → transcribe → AI analysis → cover art →
WordPress publish → YouTube upload → blog post → smart clips → social queue.

Usage:
    python3 rucktalk_episode_pipeline.py                          # Poll NextCloud
    python3 rucktalk_episode_pipeline.py --reprocess "file.mp4"   # Force reprocess
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rucktalk_common import (
    NC_EPISODE_FOLDER,
    NC_PROCESSED_FOLDER,
    WORK_DIR,
    INCOMING_DIR,
    AUDIO_DIR,
    TRANSCRIPTS_DIR,
    CLIPS_DIR,
    METADATA_DIR,
    IMAGES_DIR,
    STATIC_MEDIA,
    PUBLIC_MEDIA_URL,
    SCRIPTS,
    POSTIZ_IDS,
    logger,
    setup_logging,
    ensure_dirs,
    load_state,
    save_state,
    load_clip_queue,
    save_clip_queue,
    notify_telegram,
    llm_call,
    llm_json,
    run_comfyui,
    run_tts,
    run_script,
    BRAND_VOICE,
    IMAGE_STYLE_SUFFIX,
    PILLARS,
)
from integrations.nextcloud.client import list_files, download_file, create_folder


# ─────────────────────────────────────────────
# Step 1: Check NextCloud for new episodes
# ─────────────────────────────────────────────


def check_for_new_episodes(force_file: str | None = None) -> list[dict]:
    """
    Poll NextCloud /RuckTalk/Episodes for unprocessed MP4 files.
    If force_file is set, return that file regardless of processed state.
    Returns list of file info dicts.
    """
    state = load_state()
    processed = set(state.get("processed", []))

    if force_file:
        logger.info("Force reprocessing: %s", force_file)
        return [{"name": force_file, "path": f"{NC_EPISODE_FOLDER}/{force_file}"}]

    try:
        files = list_files(NC_EPISODE_FOLDER, depth=1)
    except Exception as exc:
        logger.error("Failed to list NextCloud episodes: %s", exc)
        return []

    new_files = []
    for f in files:
        if f.get("is_folder"):
            continue
        name = f.get("name", "")
        if not name.lower().endswith(".mp4"):
            continue
        if name in processed:
            continue
        new_files.append(f)

    if new_files:
        logger.info("Found %d new episode(s): %s", len(new_files), [f["name"] for f in new_files])
    else:
        logger.debug("No new episodes found.")

    return new_files


# ─────────────────────────────────────────────
# Step 2: Download MP4 from NextCloud
# ─────────────────────────────────────────────


def download_episode(nc_path: str, filename: str) -> Path:
    """Download MP4 from NextCloud to INCOMING_DIR. Returns local path."""
    local_path = INCOMING_DIR / filename
    if local_path.exists():
        logger.info("File already downloaded: %s", local_path)
        return local_path

    logger.info("Downloading %s from NextCloud...", filename)
    data = download_file(nc_path)
    local_path.write_bytes(data)
    logger.info("Downloaded %s (%.1f MB)", filename, len(data) / 1024 / 1024)
    return local_path


# ─────────────────────────────────────────────
# Step 3: Extract audio with ffmpeg
# ─────────────────────────────────────────────


def extract_audio(video_path: Path, episode_slug: str) -> Path:
    """Extract MP3 audio from MP4 using ffmpeg."""
    audio_path = AUDIO_DIR / f"{episode_slug}.mp3"
    if audio_path.exists():
        logger.info("Audio already extracted: %s", audio_path)
        return audio_path

    logger.info("Extracting audio from %s...", video_path.name)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "libmp3lame", "-q:a", "2",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[:500]}")

    logger.info("Audio extracted: %s (%.1f MB)", audio_path.name, audio_path.stat().st_size / 1024 / 1024)
    return audio_path


# ─────────────────────────────────────────────
# Step 4: Transcribe with Whisper
# ─────────────────────────────────────────────


def transcribe_audio(audio_path: Path, episode_slug: str) -> dict:
    """Transcribe audio with Whisper. Returns parsed JSON with segments."""
    transcript_path = TRANSCRIPTS_DIR / f"{episode_slug}.json"
    if transcript_path.exists():
        logger.info("Transcript already exists: %s", transcript_path)
        return json.loads(transcript_path.read_text())

    logger.info("Transcribing %s with Whisper...", audio_path.name)
    result = subprocess.run(
        [
            "whisper", str(audio_path),
            "--model", "base",
            "--language", "en",
            "--output_format", "json",
            "--word_timestamps", "True",
            "--output_dir", str(TRANSCRIPTS_DIR),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Whisper transcription failed: {result.stderr[:500]}")

    # Whisper outputs as <filename_without_ext>.json
    whisper_output = TRANSCRIPTS_DIR / f"{audio_path.stem}.json"
    if whisper_output.exists() and whisper_output != transcript_path:
        shutil.move(str(whisper_output), str(transcript_path))

    if not transcript_path.exists():
        raise RuntimeError(f"Whisper output not found at {transcript_path}")

    transcript = json.loads(transcript_path.read_text())
    seg_count = len(transcript.get("segments", []))
    logger.info("Transcription complete: %d segments", seg_count)
    return transcript


def get_full_text(transcript: dict) -> str:
    """Extract full text from Whisper transcript."""
    return transcript.get("text", "").strip() or " ".join(
        seg.get("text", "").strip() for seg in transcript.get("segments", [])
    )


# ─────────────────────────────────────────────
# Step 5: AI Analysis
# ─────────────────────────────────────────────


def analyze_episode(transcript: dict, episode_number: int) -> dict | None:
    """
    Use LLM to generate episode title, description, show notes,
    clip moments, and keywords from the transcript.
    """
    full_text = get_full_text(transcript)
    # Truncate to ~12k chars to stay within LLM context
    text_sample = full_text[:12000]

    prompt = f"""You are a podcast production assistant for RuckTalk, a rucking podcast.
Brand voice: {BRAND_VOICE}

Below is the transcript of Episode {episode_number}. Analyze it and return a JSON object with these exact keys:

1. "title" — A catchy, concise episode title (no "Episode X:" prefix, just the title itself). 5-8 words max.
2. "description" — YouTube/podcast description, 2-3 paragraphs. Include key topics discussed.
3. "show_notes" — Bullet-point show notes in markdown. Include timestamps (MM:SS) for major topics.
4. "clip_moments" — Array of 5-7 objects, each with:
   - "label": Short name for the clip (3-5 words)
   - "start_time": Start timestamp in seconds (float)
   - "end_time": End timestamp in seconds (float), clips should be 30-90 seconds long
   - "caption": Social media caption for this clip (1-2 sentences, engaging, no hashtags)
   - "pillar": One of {list(PILLARS.keys())}
5. "keywords" — Array of 8-12 SEO keywords/tags
6. "blog_outline" — Array of 3-5 section headers for a blog post derived from this episode

Return ONLY valid JSON, no markdown fences, no explanation.

TRANSCRIPT:
{text_sample}
"""

    logger.info("Running AI analysis for Episode %d...", episode_number)
    analysis = llm_json(prompt)
    if not analysis:
        logger.error("AI analysis failed — no response from LLM.")
        return None

    # Validate required keys
    required = ["title", "description", "clip_moments", "keywords"]
    missing = [k for k in required if k not in analysis]
    if missing:
        logger.error("AI analysis missing keys: %s", missing)
        return None

    logger.info(
        "AI analysis complete: title=%r, %d clips, %d keywords",
        analysis.get("title"),
        len(analysis.get("clip_moments", [])),
        len(analysis.get("keywords", [])),
    )
    return analysis


# ─────────────────────────────────────────────
# Step 6: Generate cover image
# ─────────────────────────────────────────────


def generate_cover_image(title: str, episode_number: int) -> str | None:
    """Generate episode cover art via ComfyUI. Returns local path or None."""
    prompt = (
        f"Podcast cover art for a rucking podcast episode titled '{title}'. "
        f"Episode {episode_number}. Rugged outdoor scene with rucksack, "
        f"bold energetic composition. {IMAGE_STYLE_SUFFIX}"
    )
    logger.info("Generating cover image for Episode %d...", episode_number)
    result = run_comfyui(prompt, width=1400, height=1400)
    if result:
        # Copy to images dir with a stable name
        dest = IMAGES_DIR / f"episode_{episode_number}_cover.png"
        shutil.copy2(result, str(dest))
        logger.info("Cover image saved: %s", dest)
        return str(dest)
    logger.warning("Cover image generation failed, proceeding without.")
    return None


# ─────────────────────────────────────────────
# Step 7: Publish audio to WordPress
# ─────────────────────────────────────────────


def publish_to_wordpress(
    audio_path: Path,
    title: str,
    episode_number: int,
    description: str,
    show_notes: str,
    cover_image_path: str | None,
) -> dict | None:
    """
    Upload audio + cover to WordPress and create a podcast post.
    Returns dict with 'post_link', 'post_id', 'audio_url'.
    """
    full_title = f"Episode {episode_number}: {title}"

    # Upload audio file
    logger.info("Uploading audio to WordPress...")
    audio_result = run_script(
        "wordpress.py", "upload-media", "rucktalk", str(audio_path)
    )
    if audio_result.returncode != 0:
        logger.error("WordPress audio upload failed: %s", audio_result.stderr[:500])
        return None

    try:
        audio_media = json.loads(audio_result.stdout)
    except (json.JSONDecodeError, ValueError):
        logger.error("Could not parse WordPress audio upload response: %s", audio_result.stdout[:500])
        return None

    audio_url = audio_media.get("source_url", "")

    # Upload cover image if available
    featured_media_id = None
    if cover_image_path and os.path.isfile(cover_image_path):
        logger.info("Uploading cover image to WordPress...")
        img_result = run_script(
            "wordpress.py", "upload-media", "rucktalk", cover_image_path
        )
        if img_result.returncode == 0:
            try:
                img_media = json.loads(img_result.stdout)
                featured_media_id = str(img_media.get("id", ""))
            except (json.JSONDecodeError, ValueError):
                logger.warning("Could not parse cover image upload response.")

    # Build post content with audio player
    content = f"""<div class="podcast-episode">
<h2>{full_title}</h2>

<audio controls preload="metadata">
  <source src="{audio_url}" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

<div class="episode-description">
{description}
</div>

<div class="show-notes">
<h3>Show Notes</h3>
{show_notes}
</div>
</div>"""

    # Create post
    logger.info("Creating WordPress podcast post...")
    post_args = [
        "wordpress.py", "create-post", "rucktalk",
        "--title", full_title,
        "--content", content,
        "--status", "publish",
    ]
    if featured_media_id:
        post_args.extend(["--featured-media", featured_media_id])

    post_result = run_script(*post_args)
    if post_result.returncode != 0:
        logger.error("WordPress post creation failed: %s", post_result.stderr[:500])
        return None

    try:
        post_data = json.loads(post_result.stdout)
    except (json.JSONDecodeError, ValueError):
        logger.error("Could not parse WordPress post response: %s", post_result.stdout[:500])
        return None

    result = {
        "post_link": post_data.get("link", ""),
        "post_id": post_data.get("id"),
        "audio_url": audio_url,
    }
    logger.info("WordPress podcast post published: %s", result["post_link"])
    return result


# ─────────────────────────────────────────────
# Step 8: Upload to YouTube
# ─────────────────────────────────────────────


def upload_to_youtube(
    video_path: Path,
    title: str,
    episode_number: int,
    description: str,
    keywords: list[str],
) -> str | None:
    """Upload full video to YouTube. Returns video ID or None."""
    full_title = f"Episode {episode_number}: {title}"
    yt_desc = f"{description}\n\n#rucking #rucktalk #podcast"
    if keywords:
        yt_desc += " " + " ".join(f"#{kw.replace(' ', '')}" for kw in keywords[:8])

    logger.info("Uploading to YouTube: %s", full_title)
    result = run_script(
        "youtube.py", "upload",
        str(video_path), full_title, yt_desc, "public",
        timeout=600,
    )
    if result.returncode != 0:
        logger.error("YouTube upload failed: %s", result.stderr[:500])
        return None

    # Parse video ID from output — look for YouTube video ID pattern
    output = result.stdout.strip()
    # Try to find a video ID (11-char alphanumeric string)
    match = re.search(r"(?:video[_ ]?id|id)[:\s]*([A-Za-z0-9_-]{11})", output, re.IGNORECASE)
    if match:
        video_id = match.group(1)
    else:
        # Fallback: look for any 11-char ID-like string on its own line
        for line in reversed(output.splitlines()):
            line = line.strip()
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", line):
                video_id = line
                break
        else:
            # Last resort: try JSON
            try:
                yt_data = json.loads(output)
                video_id = yt_data.get("id") or yt_data.get("video_id")
            except (json.JSONDecodeError, ValueError):
                logger.error("Could not parse YouTube video ID from output: %s", output[:300])
                return None

    logger.info("YouTube upload complete: https://youtu.be/%s", video_id)
    return video_id


# ─────────────────────────────────────────────
# Step 9: Create video page on WordPress
# ─────────────────────────────────────────────


def create_video_page(
    title: str,
    episode_number: int,
    youtube_id: str,
    description: str,
    show_notes: str,
    featured_media_id: str | None = None,
) -> str | None:
    """Create a WordPress page with the YouTube embed. Returns page link."""
    full_title = f"Episode {episode_number}: {title} (Video)"
    yt_url = f"https://www.youtube.com/watch?v={youtube_id}"

    content = f"""<div class="episode-video">
<div class="video-embed" style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;">
  <iframe src="https://www.youtube.com/embed/{youtube_id}"
    style="position:absolute;top:0;left:0;width:100%;height:100%;"
    frameborder="0" allowfullscreen></iframe>
</div>

<div class="episode-description">
{description}
</div>

<div class="show-notes">
<h3>Show Notes</h3>
{show_notes}
</div>

<p><a href="{yt_url}" target="_blank">Watch on YouTube</a></p>
</div>"""

    logger.info("Creating WordPress video page...")
    post_args = [
        "wordpress.py", "create-post", "rucktalk",
        "--title", full_title,
        "--content", content,
        "--status", "publish",
    ]

    result = run_script(*post_args)
    if result.returncode != 0:
        logger.error("WordPress video page failed: %s", result.stderr[:500])
        return None

    try:
        post_data = json.loads(result.stdout)
        link = post_data.get("link", "")
        logger.info("WordPress video page published: %s", link)
        return link
    except (json.JSONDecodeError, ValueError):
        logger.warning("Could not parse video page response.")
        return None


# ─────────────────────────────────────────────
# Step 9b: Generate blog post from transcript
# ─────────────────────────────────────────────


def generate_blog_post(
    transcript: dict,
    title: str,
    episode_number: int,
    analysis: dict,
    wp_podcast_link: str | None,
    youtube_id: str | None,
    cover_image_path: str | None,
) -> str | None:
    """Generate and publish a blog post from the episode transcript. Returns post link."""
    full_text = get_full_text(transcript)
    text_sample = full_text[:10000]
    blog_outline = analysis.get("blog_outline", ["Introduction", "Key Takeaways", "Conclusion"])

    prompt = f"""You are a blog writer for RuckTalk, a rucking podcast.
Brand voice: {BRAND_VOICE}

Write a blog post based on Episode {episode_number}: "{title}".
Use these section headers: {json.dumps(blog_outline)}

The post should:
- Be 800-1200 words
- Use conversational but informative tone
- Include practical takeaways
- Reference specific points from the episode
- Use HTML formatting (h2, h3, p, ul/li, strong, em)
- Do NOT include the title as h1 (WordPress handles that)

TRANSCRIPT EXCERPT:
{text_sample}

Return ONLY the HTML content of the blog post, no markdown fences.
"""

    logger.info("Generating blog post for Episode %d...", episode_number)
    blog_html = llm_call(prompt, temperature=0.7)
    if not blog_html:
        logger.error("Blog post generation failed.")
        return None

    # Clean up any markdown fences
    blog_html = re.sub(r"^```(?:html)?\s*\n?", "", blog_html.strip())
    blog_html = re.sub(r"\n?```\s*$", "", blog_html)

    # Add episode links at the top
    links = []
    if wp_podcast_link:
        links.append(f'<a href="{wp_podcast_link}">Listen to the full episode</a>')
    if youtube_id:
        links.append(f'<a href="https://youtu.be/{youtube_id}">Watch on YouTube</a>')

    if links:
        blog_html = f'<p><strong>Episode {episode_number}:</strong> {" | ".join(links)}</p>\n\n{blog_html}'

    full_title = f"Episode {episode_number}: {title} — Blog"

    post_args = [
        "wordpress.py", "create-post", "rucktalk",
        "--title", full_title,
        "--content", blog_html,
        "--status", "publish",
    ]

    result = run_script(*post_args)
    if result.returncode != 0:
        logger.error("Blog post creation failed: %s", result.stderr[:500])
        return None

    try:
        post_data = json.loads(result.stdout)
        link = post_data.get("link", "")
        logger.info("Blog post published: %s", link)
        return link
    except (json.JSONDecodeError, ValueError):
        logger.warning("Could not parse blog post response.")
        return None


# ─────────────────────────────────────────────
# Step 10: Smart clip generation
# ─────────────────────────────────────────────


def generate_srt_for_clip(
    transcript: dict, start_time: float, end_time: float
) -> str:
    """Generate SRT subtitle content from transcript segments within a time range."""
    segments = transcript.get("segments", [])
    srt_lines = []
    counter = 1

    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        seg_text = seg.get("text", "").strip()

        # Include segments that overlap with clip range
        if seg_end < start_time or seg_start > end_time:
            continue
        if not seg_text:
            continue

        # Adjust timestamps relative to clip start
        rel_start = max(0, seg_start - start_time)
        rel_end = min(end_time - start_time, seg_end - start_time)

        srt_lines.append(str(counter))
        srt_lines.append(
            f"{_format_srt_time(rel_start)} --> {_format_srt_time(rel_end)}"
        )
        srt_lines.append(seg_text)
        srt_lines.append("")
        counter += 1

    return "\n".join(srt_lines)


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_clips(
    video_path: Path,
    transcript: dict,
    clip_moments: list[dict],
    episode_number: int,
    episode_slug: str,
) -> list[dict]:
    """
    Cut video clips at identified moments in both portrait and landscape.
    Returns list of clip metadata dicts for the social queue.
    """
    clips_generated = []
    episode_clips_dir = CLIPS_DIR / f"episode_{episode_number}"
    episode_clips_dir.mkdir(parents=True, exist_ok=True)

    for i, moment in enumerate(clip_moments, 1):
        label = moment.get("label", f"clip_{i}")
        start = moment.get("start_time", 0)
        end = moment.get("end_time", start + 60)
        caption = moment.get("caption", "")
        pillar = moment.get("pillar", "training_tips")
        duration = end - start

        clip_slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        logger.info("Generating clip %d/%d: %s (%.0fs-%.0fs)", i, len(clip_moments), label, start, end)

        # Generate SRT subtitles for this clip
        srt_content = generate_srt_for_clip(transcript, start, end)
        srt_path = episode_clips_dir / f"{clip_slug}.srt"
        srt_path.write_text(srt_content)

        # Landscape clip (1920x1080)
        landscape_path = episode_clips_dir / f"{clip_slug}_landscape.mp4"
        landscape_ok = _cut_clip(
            video_path, start, duration, landscape_path, srt_path,
            scale="1920:1080", crop=None,
        )

        # Portrait clip (1080x1920) — center-crop from landscape
        portrait_path = episode_clips_dir / f"{clip_slug}_portrait.mp4"
        portrait_ok = _cut_clip(
            video_path, start, duration, portrait_path, srt_path,
            scale=None, crop="1080:1920",
        )

        if not landscape_ok and not portrait_ok:
            logger.warning("Clip %d (%s) failed to generate.", i, label)
            continue

        # Copy portrait clips to STATIC_MEDIA for web access
        web_filename = f"rucktalk_ep{episode_number}_{clip_slug}_portrait.mp4"
        web_path = STATIC_MEDIA / "rucktalk" / web_filename
        web_path.parent.mkdir(parents=True, exist_ok=True)

        if portrait_ok and portrait_path.exists():
            shutil.copy2(str(portrait_path), str(web_path))

        clip_meta = {
            "episode_number": episode_number,
            "episode_slug": episode_slug,
            "clip_index": i,
            "label": label,
            "caption": caption,
            "pillar": pillar,
            "start_time": start,
            "end_time": end,
            "duration": duration,
            "landscape_path": str(landscape_path) if landscape_ok else None,
            "portrait_path": str(portrait_path) if portrait_ok else None,
            "srt_path": str(srt_path),
            "web_url": f"{PUBLIC_MEDIA_URL}/rucktalk/{web_filename}" if portrait_ok else None,
            "posted": False,
            "created_at": datetime.now().isoformat(),
        }
        clips_generated.append(clip_meta)

    logger.info("Generated %d/%d clips successfully.", len(clips_generated), len(clip_moments))
    return clips_generated


def _cut_clip(
    video_path: Path,
    start: float,
    duration: float,
    output_path: Path,
    srt_path: Path,
    scale: str | None,
    crop: str | None,
) -> bool:
    """Cut a single clip with ffmpeg. Returns True on success."""
    if output_path.exists():
        return True

    # Build filter chain
    filters = []
    if crop:
        # For portrait: scale to ensure height, then crop to 1080x1920
        filters.append(f"scale=-2:1920")
        filters.append(f"crop=1080:1920")
    elif scale:
        filters.append(f"scale={scale}:force_original_aspect_ratio=decrease")
        filters.append(f"pad={scale}:(ow-iw)/2:(oh-ih)/2")

    # Burn in subtitles if SRT exists and has content
    if srt_path.exists() and srt_path.stat().st_size > 10:
        # Escape path for ffmpeg subtitles filter
        escaped_srt = str(srt_path).replace("'", "'\\''").replace(":", "\\:")
        filters.append(f"subtitles='{escaped_srt}'")

    vf = ",".join(filters) if filters else None

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
    ]
    if vf:
        cmd.extend(["-vf", vf])

    cmd.append(str(output_path))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning("ffmpeg clip cut failed for %s: %s", output_path.name, result.stderr[:300])
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out cutting clip: %s", output_path.name)
        return False
    except Exception as exc:
        logger.error("ffmpeg error cutting clip %s: %s", output_path.name, exc)
        return False


# ─────────────────────────────────────────────
# Step 11: Queue clips for daily social engine
# ─────────────────────────────────────────────


def queue_clips_for_social(clips: list[dict]) -> int:
    """Add generated clips to the social posting queue. Returns count added."""
    queue = load_clip_queue()
    added = 0
    for clip in clips:
        if clip.get("portrait_path") or clip.get("landscape_path"):
            queue.append(clip)
            added += 1
    save_clip_queue(queue)
    logger.info("Added %d clips to social queue (total queue: %d).", added, len(queue))
    return added


# ─────────────────────────────────────────────
# Step 12: Move processed file on NextCloud
# ─────────────────────────────────────────────


def move_to_processed(filename: str) -> None:
    """Move the processed MP4 to the Processed folder on NextCloud."""
    src = f"{NC_EPISODE_FOLDER}/{filename}"
    dst = f"{NC_PROCESSED_FOLDER}/{filename}"
    try:
        # Ensure Processed folder exists
        try:
            create_folder(NC_PROCESSED_FOLDER)
        except Exception:
            pass  # Already exists

        from integrations.nextcloud.client import move_file
        move_file(src, dst)
        logger.info("Moved %s to Processed folder on NextCloud.", filename)
    except Exception as exc:
        logger.warning("Could not move %s to Processed: %s (non-fatal)", filename, exc)


# ─────────────────────────────────────────────
# Main Pipeline Orchestrator
# ─────────────────────────────────────────────


def process_episode(file_info: dict) -> bool:
    """
    Run the full pipeline for a single episode file.
    Returns True on success, False on failure.
    """
    filename = file_info["name"]
    nc_path = file_info.get("path", f"{NC_EPISODE_FOLDER}/{filename}")
    state = load_state()
    episode_number = state.get("next_episode_number", 4)
    episode_slug = f"episode_{episode_number}"

    logger.info("=" * 60)
    logger.info("PROCESSING EPISODE %d: %s", episode_number, filename)
    logger.info("=" * 60)

    results = {
        "filename": filename,
        "episode_number": episode_number,
        "title": None,
        "wp_podcast_link": None,
        "youtube_id": None,
        "youtube_url": None,
        "wp_video_link": None,
        "wp_blog_link": None,
        "clips_generated": 0,
        "clips_queued": 0,
    }

    try:
        # Step 1: Download
        video_path = download_episode(nc_path, filename)

        # Step 2: Extract audio
        audio_path = extract_audio(video_path, episode_slug)

        # Step 3: Transcribe
        transcript = transcribe_audio(audio_path, episode_slug)

        # Step 4: AI analysis
        analysis = analyze_episode(transcript, episode_number)
        if not analysis:
            raise RuntimeError("AI analysis failed — cannot proceed without metadata.")

        title = analysis["title"]
        description = analysis.get("description", "")
        show_notes = analysis.get("show_notes", "")
        clip_moments = analysis.get("clip_moments", [])
        keywords = analysis.get("keywords", [])
        results["title"] = title

        # Save analysis metadata
        meta_path = METADATA_DIR / f"{episode_slug}_analysis.json"
        meta_path.write_text(json.dumps(analysis, indent=2))
        logger.info("Episode %d: \"%s\"", episode_number, title)

        # Step 5: Generate cover image
        cover_image_path = generate_cover_image(title, episode_number)

        # Step 6: Publish to WordPress (podcast post)
        wp_result = publish_to_wordpress(
            audio_path, title, episode_number, description, show_notes, cover_image_path
        )
        if wp_result:
            results["wp_podcast_link"] = wp_result["post_link"]

        # Step 7: Upload to YouTube
        youtube_id = upload_to_youtube(video_path, title, episode_number, description, keywords)
        if youtube_id:
            results["youtube_id"] = youtube_id
            results["youtube_url"] = f"https://youtu.be/{youtube_id}"

        # Step 8: Create video page on WordPress
        if youtube_id:
            video_link = create_video_page(
                title, episode_number, youtube_id, description, show_notes
            )
            results["wp_video_link"] = video_link

        # Step 9: Generate blog post
        blog_link = generate_blog_post(
            transcript, title, episode_number, analysis,
            results.get("wp_podcast_link"), youtube_id, cover_image_path,
        )
        results["wp_blog_link"] = blog_link

        # Step 10: Generate clips
        if clip_moments:
            clips = generate_clips(
                video_path, transcript, clip_moments, episode_number, episode_slug
            )
            results["clips_generated"] = len(clips)

            # Step 11: Queue clips for social
            if clips:
                queued = queue_clips_for_social(clips)
                results["clips_queued"] = queued

        # Mark as processed
        state = load_state()
        if filename not in state.get("processed", []):
            state.setdefault("processed", []).append(filename)
        state["next_episode_number"] = episode_number + 1
        save_state(state)

        # Move file on NextCloud
        move_to_processed(filename)

        # Save episode results
        results_path = METADATA_DIR / f"{episode_slug}_results.json"
        results_path.write_text(json.dumps(results, indent=2))

        # Notify Mike
        _send_success_notification(results)

        logger.info("Episode %d pipeline COMPLETE.", episode_number)
        return True

    except Exception as exc:
        logger.error("Pipeline FAILED for %s: %s", filename, exc)
        logger.error(traceback.format_exc())
        _send_failure_notification(filename, episode_number, exc)
        return False


def _send_success_notification(results: dict) -> None:
    """Send Telegram notification on successful episode processing."""
    ep = results["episode_number"]
    title = results.get("title", "Unknown")
    lines = [
        f"RuckTalk Episode {ep}: \"{title}\" — Pipeline Complete",
        "",
    ]
    if results.get("wp_podcast_link"):
        lines.append(f"Podcast: {results['wp_podcast_link']}")
    if results.get("youtube_url"):
        lines.append(f"YouTube: {results['youtube_url']}")
    if results.get("wp_video_link"):
        lines.append(f"Video page: {results['wp_video_link']}")
    if results.get("wp_blog_link"):
        lines.append(f"Blog: {results['wp_blog_link']}")
    if results.get("clips_queued"):
        lines.append(f"Clips queued for social: {results['clips_queued']}")

    notify_telegram("\n".join(lines))


def _send_failure_notification(filename: str, episode_number: int, exc: Exception) -> None:
    """Send Telegram notification on pipeline failure."""
    notify_telegram(
        f"RuckTalk Pipeline FAILED\n"
        f"File: {filename}\n"
        f"Episode: {episode_number}\n"
        f"Error: {str(exc)[:200]}"
    )


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="RuckTalk Episode Pipeline")
    parser.add_argument(
        "--reprocess",
        type=str,
        default=None,
        help="Force reprocess a specific MP4 filename",
    )
    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    logger.info("RuckTalk Episode Pipeline starting...")

    new_episodes = check_for_new_episodes(force_file=args.reprocess)
    if not new_episodes:
        logger.info("No new episodes to process.")
        return

    for file_info in new_episodes:
        success = process_episode(file_info)
        if not success:
            logger.error("Failed to process: %s — continuing with next.", file_info["name"])

    logger.info("Pipeline run complete.")


if __name__ == "__main__":
    main()
