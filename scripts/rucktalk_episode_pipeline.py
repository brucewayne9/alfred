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
    run_comfyui_cloud,
    run_tts,
    run_script,
    BRAND_VOICE,
    IMAGE_STYLE_SUFFIX,
    PILLARS,
)
from integrations.nextcloud.client import list_files, download_file, create_folder
from scripts.rucktalk_rig_props import (
    build_rucktalkclip_props,
    build_magazinerig_props,
)

# Phase 2 migration flag. Default is the deprecated rig during cutover;
# Task 5 flips it to "MagazineRig". Set EPISODE_RIG env var to override at runtime.
EPISODE_RIG = os.environ.get("EPISODE_RIG", "MagazineRig")


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
    """Generate branded episode cover art — ComfyUI background + text overlay.

    Matches the RuckTalk cover style:
    - Cinematic AI background (ComfyUI Cloud)
    - "RUCK TALK" top left
    - "EPISODE N" badge top right (orange)
    - "NEW EPISODE" label in orange
    - Episode title in huge bold text, key word in orange
    - Dark overlay for readability
    """
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance

    # Step 1: Generate background image via ComfyUI Cloud
    prompt = (
        f"cinematic dramatic scene related to '{title}', "
        f"dark moody atmosphere, single person, masculine energy, "
        f"studio or outdoor setting, professional photography, "
        f"{IMAGE_STYLE_SUFFIX}"
    )
    logger.info("Generating cover image for Episode %d...", episode_number)
    bg_path = run_comfyui_cloud(prompt, width=1400, height=1400)

    if not bg_path:
        logger.warning("ComfyUI Cloud failed for cover — trying local.")
        bg_path = run_comfyui(prompt, width=1400, height=1400)

    if not bg_path:
        logger.warning("Cover image generation failed entirely.")
        return None

    # Step 2: Composite branded text overlay
    try:
        bg = Image.open(bg_path).resize((1400, 1400))
        bg = ImageEnhance.Brightness(bg).enhance(0.45)

        draw = ImageDraw.Draw(bg)

        # Find a bold font
        font_path = None
        for fp in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]:
            if Path(fp).exists():
                font_path = fp
                break

        if not font_path:
            logger.warning("No bold font found — using default.")
            font_path = None

        def _font(size):
            return ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()

        # "RUCK TALK" top left
        draw.text((60, 50), "RUCK TALK", fill="white", font=_font(48))

        # "EPISODE N" badge top right
        ep_text = f"EPISODE {episode_number}"
        ep_bbox = draw.textbbox((0, 0), ep_text, font=_font(32))
        ep_w = ep_bbox[2] - ep_bbox[0] + 30
        ep_h = ep_bbox[3] - ep_bbox[1] + 20
        ep_x = 1400 - ep_w - 60
        draw.rounded_rectangle([ep_x, 50, ep_x + ep_w, 50 + ep_h], radius=10, fill="#f97316")
        draw.text((ep_x + 15, 58), ep_text, fill="white", font=_font(32))

        # "NEW EPISODE" label
        draw.text((60, 950), "NEW EPISODE", fill="#f97316", font=_font(36))

        # Episode title — split into lines, last significant word in orange
        words = title.upper().split()
        lines = []
        current_line = []
        max_chars = 18  # rough chars per line at this font size

        for word in words:
            if sum(len(w) for w in current_line) + len(current_line) + len(word) > max_chars:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        if current_line:
            lines.append(" ".join(current_line))

        # Render title lines — last line in orange
        y = 1010
        title_font = _font(110)
        for i, line in enumerate(lines[:4]):
            if i == len(lines) - 1:
                draw.text((60, y), line, fill="#f97316", font=title_font)
            else:
                draw.text((60, y), line, fill="white", font=title_font)
            y += 125

        # Save
        dest = IMAGES_DIR / f"episode_{episode_number}_cover.png"
        bg.save(str(dest), quality=95)
        logger.info("Branded cover image saved: %s", dest)
        return str(dest)

    except Exception as exc:
        logger.error("Cover image branding failed: %s", exc)
        # Fall back to raw ComfyUI image
        dest = IMAGES_DIR / f"episode_{episode_number}_cover.png"
        shutil.copy2(bg_path, str(dest))
        return str(dest)


# ─────────────────────────────────────────────
# Step 7: Publish audio to WordPress
# ─────────────────────────────────────────────


SSH_100 = "ssh -i /home/aialfred/.ssh/alfred_100 -o ConnectTimeout=10 -o StrictHostKeyChecking=no brucewayne9@75.43.156.100"
WP_CLI = "docker exec rt-wordpress wp"


def _wp_cli(cmd: str, timeout: int = 30) -> tuple[bool, str]:
    """Run a WP-CLI command on the RuckTalk WordPress server. Returns (success, output)."""
    full_cmd = f'{SSH_100} "{WP_CLI} {cmd} --allow-root"'
    try:
        r = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        output = r.stdout.strip()
        if r.returncode != 0:
            output = r.stderr.strip() or output
        return r.returncode == 0, output
    except Exception as exc:
        return False, str(exc)


def publish_to_wordpress(
    audio_path: Path,
    title: str,
    episode_number: int,
    description: str,
    show_notes: str,
    cover_image_path: str | None,
) -> dict | None:
    """
    Upload audio + cover to WordPress and create a proper Sonaar podcast episode.
    Uses the 'podcast' custom post type with correct meta fields so the episode
    appears on the show page at /show/ruck-talk/.
    Returns dict with 'post_link', 'post_id', 'audio_url'.
    """
    full_title = f"Episode {episode_number}: {title}"

    # Upload audio file
    logger.info("Uploading audio to WordPress...")
    audio_result = run_script("wordpress.py", "upload-media", "rucktalk", str(audio_path))
    if audio_result.returncode != 0:
        logger.error("WordPress audio upload failed: %s", audio_result.stderr[:500])
        return None

    try:
        audio_media = json.loads(audio_result.stdout)
    except (json.JSONDecodeError, ValueError):
        logger.error("Could not parse audio upload response: %s", audio_result.stdout[:500])
        return None

    audio_url = audio_media.get("source_url", "")
    audio_media_id = audio_media.get("id", "")

    # Upload cover image
    featured_media_id = None
    if cover_image_path and os.path.isfile(cover_image_path):
        logger.info("Uploading cover image to WordPress...")
        img_result = run_script("wordpress.py", "upload-media", "rucktalk", cover_image_path)
        if img_result.returncode == 0:
            try:
                img_media = json.loads(img_result.stdout)
                featured_media_id = str(img_media.get("id", ""))
            except (json.JSONDecodeError, ValueError):
                logger.warning("Could not parse cover image upload response.")

    # Build post content
    content = (
        f"<p>{description}</p>"
        f"<h3>Show Notes</h3>"
        f"{show_notes}"
        f"<p><strong>Subscribe to RuckTalk for new episodes. No fluff, no excuses.</strong></p>"
    )
    # Escape single quotes for shell
    safe_title = full_title.replace("'", "'\\''")
    safe_content = content.replace("'", "'\\''")

    # Create as podcast post type via WP-CLI
    logger.info("Creating podcast episode on WordPress...")
    ok, output = _wp_cli(
        f"post create --post_type=podcast --post_title='{safe_title}' "
        f"--post_status=publish --post_content='{safe_content}' --porcelain",
        timeout=30,
    )

    if not ok:
        logger.error("WordPress podcast post creation failed: %s", output)
        return None

    post_id = output.strip()
    logger.info("Created podcast post ID: %s", post_id)

    # Set Sonaar podcast meta fields
    meta_cmds = [
        f"post meta update {post_id} FileOrStreamPodCast mp3",
        f"post meta update {post_id} track_mp3_podcast {audio_media_id}",
        f"post meta update {post_id} podcast_player_position above",
        f"post meta update {post_id} podcast_itunes_episode_number {episode_number}",
        f"post meta update {post_id} podcast_itunes_episode_title '{title.replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}'",
        f"post meta update {post_id} podcast_explicit_episode 0",
        f"post meta update {post_id} no_track_skip 0",
    ]
    if featured_media_id:
        meta_cmds.append(f"post meta update {post_id} _thumbnail_id {featured_media_id}")

    # Assign to podcast category 184
    meta_cmds.append(f"post term add {post_id} podcast-category 184")

    for cmd in meta_cmds:
        ok, out = _wp_cli(cmd)
        if not ok:
            logger.warning("Meta command failed: %s → %s", cmd, out)

    # Get the post URL
    ok, post_url = _wp_cli(f"post get {post_id} --field=url")
    if not ok:
        post_url = f"https://rucktalk.com/podcast/{post_id}/"

    result = {
        "post_link": post_url,
        "post_id": post_id,
        "audio_url": audio_url,
    }
    logger.info("Podcast episode published: %s", post_url)
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
    analysis: dict | None = None,
) -> list[dict]:
    """
    Cut video clips at identified moments in both portrait and landscape,
    then render branded versions through the RuckTalkClip Remotion template.
    Returns list of clip metadata dicts for the social queue.
    """
    analysis = analysis or {}
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

        # Render branded version through Remotion RuckTalkClip template
        branded_path = episode_clips_dir / f"{clip_slug}_branded.mp4"
        branded_ok = _render_branded_clip(
            raw_clip_path=portrait_path if portrait_ok else landscape_path,
            output_path=branded_path,
            episode_number=episode_number,
            episode_title=analysis.get("title", ""),
            context_line=caption,
            host_name="MIKE JOHNSON",
            guest_name=analysis.get("guest_name"),
            transcript=transcript,
            clip_start=start,
            clip_end=end,
            duration_frames=int(duration * 30),
        ) if (portrait_ok or landscape_ok) else False

        # Use branded version for social, fall back to raw portrait
        social_clip = branded_path if branded_ok else portrait_path
        web_filename = f"rucktalk_ep{episode_number}_{clip_slug}_branded.mp4"
        web_path = STATIC_MEDIA / "rucktalk" / web_filename
        web_path.parent.mkdir(parents=True, exist_ok=True)

        if social_clip and Path(social_clip).exists():
            shutil.copy2(str(social_clip), str(web_path))

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
            "branded_path": str(branded_path) if branded_ok else None,
            "srt_path": str(srt_path),
            "web_url": f"{PUBLIC_MEDIA_URL}/rucktalk/{web_filename}" if (branded_ok or portrait_ok) else None,
            "posted": False,
            "created_at": datetime.now().isoformat(),
        }
        clips_generated.append(clip_meta)

    logger.info("Generated %d/%d clips successfully.", len(clips_generated), len(clip_moments))
    return clips_generated


def _render_branded_clip(
    raw_clip_path: Path,
    output_path: Path,
    episode_number: int,
    episode_title: str,
    context_line: str,
    host_name: str,
    guest_name: str | None,
    transcript: dict,
    clip_start: float,
    clip_end: float,
    duration_frames: int,
) -> bool:
    """Render a clip through the Remotion RuckTalkClip template with branded overlays."""
    if output_path.exists():
        return True

    # Build caption phrases from transcript segments in this time range
    segments = transcript.get("segments", [])
    phrases = []
    fps = 30
    phrase_words = []
    phrase_start = None

    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        if seg_start >= clip_start and seg_end <= clip_end:
            adj_start = seg_start - clip_start
            adj_end = seg_end - clip_start
            text = seg.get("text", "").strip().upper()
            if not text:
                continue

            if phrase_start is None:
                phrase_start = adj_start

            phrase_words.append(text)

            # Group into phrases of ~4-6 words
            if len(phrase_words) >= 4 or adj_end - phrase_start > 4.0:
                phrases.append({
                    "text": " ".join(phrase_words),
                    "startFrame": int(phrase_start * fps),
                    "endFrame": int(adj_end * fps),
                })
                phrase_words = []
                phrase_start = None

    # Flush remaining words
    if phrase_words and phrase_start is not None:
        phrases.append({
            "text": " ".join(phrase_words),
            "startFrame": int(phrase_start * fps),
            "endFrame": duration_frames,
        })

    if not phrases:
        # Fallback: single phrase from context
        phrases = [{"text": context_line.upper(), "startFrame": 30, "endFrame": duration_frames - 30}]

    # Copy raw clip to Remotion's public/ dir so staticFile() can serve it
    remotion_dir = "/home/aialfred/remotion"
    public_dir = Path(remotion_dir) / "public"
    public_dir.mkdir(exist_ok=True)

    clip_filename = f"clip_{output_path.stem}.mp4"
    public_clip = public_dir / clip_filename
    shutil.copy2(str(raw_clip_path), str(public_clip))
    logger.info("Copied clip to Remotion public: %s", clip_filename)

    # Build props + pick composition id based on EPISODE_RIG flag
    if EPISODE_RIG == "MagazineRig":
        props = build_magazinerig_props(
            clip_filename=clip_filename,
            episode_number=episode_number,
            episode_title=episode_title,
            host_name=host_name,
            guest_name=guest_name,
            caption_phrases=phrases,
        )
        composition_id = "MagazineRig"
    else:
        props = build_rucktalkclip_props(
            clip_filename=clip_filename,
            episode_number=episode_number,
            episode_title=episode_title,
            context_line=context_line,
            host_name=host_name,
            guest_name=guest_name,
            caption_phrases=phrases,
        )
        composition_id = "RuckTalkClip"

    # Write props to a temp file to avoid shell escaping issues with JSON
    props_file = Path(remotion_dir) / f"props_{output_path.stem}.json"
    props_file.write_text(json.dumps(props))

    logger.info("Rendering clip via %s (EPISODE_RIG=%s)", composition_id, EPISODE_RIG)

    npx = "/home/aialfred/.nvm/versions/node/v22.22.0/bin/npx"
    cmd = [
        npx, "remotion", "render",
        "src/index.ts", composition_id,
        f"--props={str(props_file)}",
        f"--frames=0-{min(duration_frames, 1800)}",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd, cwd=remotion_dir,
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "PATH": f"/home/aialfred/.nvm/versions/node/v22.22.0/bin:{os.environ.get('PATH', '')}"},
        )

        # Clean up temp files
        public_clip.unlink(missing_ok=True)
        props_file.unlink(missing_ok=True)

        if result.returncode != 0:
            logger.warning("Remotion render failed for %s: %s", output_path.name, result.stderr[-300:])
            return False
        logger.info("Branded clip rendered: %s", output_path.name)
        return True
    except Exception as exc:
        logger.warning("Remotion render error: %s", exc)
        return False


def _cut_clip(
    video_path: Path,
    start: float,
    duration: float,
    output_path: Path,
    srt_path: Path,
    scale: str | None,
    crop: str | None,
) -> bool:
    """Cut a single clip with ffmpeg, re-encoding to H.264 for Remotion compatibility.
    Returns True on success. No subtitles burned in — Remotion template handles captions."""
    if output_path.exists():
        return True

    # Build filter chain — scale/crop only, no subtitle burn-in
    filters = []
    if crop:
        filters.append(f"scale=-2:1920")
        filters.append(f"crop=1080:1920")
    elif scale:
        filters.append(f"scale={scale}:force_original_aspect_ratio=decrease")
        filters.append(f"pad={scale}:(ow-iw)/2:(oh-ih)/2")

    vf = ",".join(filters) if filters else None

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-r", "30",  # Force 30fps output — matches Remotion composition fps and prevents
                     # judder when source is 29.97fps broadcast (see Phase 2 Task 4.5 fix).
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
    clip_list = queue.get("clips", [])
    added = 0
    for clip in clips:
        if clip.get("portrait_path") or clip.get("landscape_path"):
            clip_list.append(clip)
            added += 1
    queue["clips"] = clip_list
    save_clip_queue(queue)
    logger.info("Added %d clips to social queue (total queue: %d).", added, len(clip_list))
    return added


# ─────────────────────────────────────────────
# Step 12: Move processed file on NextCloud
# ─────────────────────────────────────────────


def move_to_processed(filename: str, episode_number: int, title: str) -> None:
    """Move the processed MP4 to a named episode folder on NextCloud."""
    from integrations.nextcloud.client import move_file, upload_file

    ep_folder_name = f"Episode {episode_number} - {title}"
    # Sanitize folder name
    ep_folder_name = re.sub(r'[<>:"/\\|?*]', '', ep_folder_name).strip()
    ep_folder = f"{NC_PROCESSED_FOLDER}/{ep_folder_name}"

    try:
        # Create episode folder structure
        for folder in [NC_PROCESSED_FOLDER, ep_folder, f"{ep_folder}/Clips"]:
            try:
                create_folder(folder)
            except Exception:
                pass  # Already exists

        # Move original MP4 into the episode folder
        src = f"{NC_EPISODE_FOLDER}/{filename}"
        dst = f"{ep_folder}/{filename}"
        move_file(src, dst)
        logger.info("Moved %s to %s on NextCloud.", filename, ep_folder)
    except Exception as exc:
        logger.warning("Could not move %s to Processed: %s (non-fatal)", filename, exc)


def upload_clips_to_nextcloud(clips: list[dict], episode_number: int, title: str) -> None:
    """Upload generated clips to NextCloud under the episode's Processed folder."""
    from integrations.nextcloud.client import upload_file

    ep_folder_name = f"Episode {episode_number} - {title}"
    ep_folder_name = re.sub(r'[<>:"/\\|?*]', '', ep_folder_name).strip()
    clips_folder = f"{NC_PROCESSED_FOLDER}/{ep_folder_name}/Clips"

    # Ensure folder exists
    for folder in [NC_PROCESSED_FOLDER, f"{NC_PROCESSED_FOLDER}/{ep_folder_name}", clips_folder]:
        try:
            create_folder(folder)
        except Exception:
            pass

    uploaded = 0
    for clip in clips:
        # Upload portrait clip
        portrait = clip.get("portrait_path")
        if portrait and Path(portrait).exists():
            try:
                nc_name = f"clip_{clip['clip_index']}_{clip.get('label', 'clip')}_portrait.mp4"
                nc_name = re.sub(r'[<>:"/\\|?*]', '', nc_name)
                content = Path(portrait).read_bytes()
                upload_file(f"{clips_folder}/{nc_name}", content, "video/mp4")
                uploaded += 1
                logger.info("Uploaded clip %d portrait to NextCloud.", clip["clip_index"])
            except Exception as exc:
                logger.warning("Failed to upload clip %d portrait: %s", clip["clip_index"], exc)

        # Upload landscape clip
        landscape = clip.get("landscape_path")
        if landscape and Path(landscape).exists():
            try:
                nc_name = f"clip_{clip['clip_index']}_{clip.get('label', 'clip')}_landscape.mp4"
                nc_name = re.sub(r'[<>:"/\\|?*]', '', nc_name)
                content = Path(landscape).read_bytes()
                upload_file(f"{clips_folder}/{nc_name}", content, "video/mp4")
                uploaded += 1
            except Exception as exc:
                logger.warning("Failed to upload clip %d landscape: %s", clip["clip_index"], exc)

    logger.info("Uploaded %d clip files to NextCloud: %s", uploaded, clips_folder)


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
                video_path, transcript, clip_moments, episode_number, episode_slug,
                analysis=analysis,
            )
            results["clips_generated"] = len(clips)

            # Step 11: Queue clips for social
            if clips:
                queued = queue_clips_for_social(clips)
                results["clips_queued"] = queued

                # Step 11b: Upload clips to NextCloud
                upload_clips_to_nextcloud(clips, episode_number, title)

        # Mark as processed
        state = load_state()
        if filename not in state.get("processed", []):
            state.setdefault("processed", []).append(filename)
        state["next_episode_number"] = episode_number + 1
        save_state(state)

        # Move file on NextCloud
        move_to_processed(filename, episode_number, title)

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
# Phase 2 Rig Comparison Helper
# ─────────────────────────────────────────────


def compare_rigs_for_clip(
    raw_clip_path: Path,
    output_dir: Path,
    episode_number: int,
    episode_title: str,
    context_line: str,
    host_name: str,
    guest_name: str | None,
    transcript: dict,
    clip_start: float,
    clip_end: float,
    duration_frames: int,
) -> tuple[Path, Path]:
    """Render the same clip through BOTH rigs and return paths to both outputs.

    Used once, at the Phase 2 cutover gate, to produce a side-by-side
    comparison for human review before flipping EPISODE_RIG default.
    """
    global EPISODE_RIG
    output_dir.mkdir(parents=True, exist_ok=True)

    old_path = output_dir / f"ep{episode_number}_rucktalkclip.mp4"
    new_path = output_dir / f"ep{episode_number}_magazinerig.mp4"

    # Save and restore EPISODE_RIG so we don't leak state to the caller.
    saved = EPISODE_RIG
    try:
        EPISODE_RIG = "RuckTalkClip"
        _render_branded_clip(
            raw_clip_path, old_path, episode_number, episode_title,
            context_line, host_name, guest_name, transcript,
            clip_start, clip_end, duration_frames,
        )
        EPISODE_RIG = "MagazineRig"
        _render_branded_clip(
            raw_clip_path, new_path, episode_number, episode_title,
            context_line, host_name, guest_name, transcript,
            clip_start, clip_end, duration_frames,
        )
    finally:
        EPISODE_RIG = saved

    return old_path, new_path


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
