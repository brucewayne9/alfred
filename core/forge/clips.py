"""Forge clip ingest — fetch footage (yt-dlp: YouTube/TikTok/IG) or generate (LTX cloud)."""
from __future__ import annotations
import glob
import os
import shutil
import subprocess
import uuid
from pathlib import Path

SEARCH_N = 3

# A montage only samples a few short slices per clip, so we never need a full
# (potentially multi-minute) video. Capping each result to a 90s window is what
# keeps a 3-result search comfortably under the fetch timeout instead of trying
# to pull three whole 1080p videos and blowing past it.
SECTION_WINDOW = "*0:00-01:30"

# Player-client pool. `tv_embedded` + `mediaconnect` are the two clients that can
# still pull AGE-RESTRICTED videos without a logged-in account; the rest survive
# Google rotating which clients serve formats (the 'fetched nothing' gate). Keep
# the bypass clients FIRST so an age-gated video clears before falling through.
YT_PLAYER_CLIENTS = "tv_embedded,mediaconnect,default,web_safari,android_vr,tv"


def cookies_file() -> str | None:
    """Path to a YouTube cookies.txt (Netscape format), if one is configured.

    Order: FORGE_YTDLP_COOKIES env → config/youtube_cookies.txt. Returns None when
    neither exists, so cookie-free fetches still work for non-gated videos. A
    logged-in, age-verified export is the ONLY thing that clears age-restricted
    videos that the bypass player-clients can't.
    """
    cand = os.environ.get("FORGE_YTDLP_COOKIES")
    if cand and os.path.exists(cand):
        return cand
    repo = Path(__file__).resolve().parent.parent.parent
    default = repo / "config" / "youtube_cookies.txt"
    return str(default) if default.exists() else None


def stage_cookies(dest_dir: str | Path) -> str | None:
    """Copy the master cookies into a throwaway file for ONE yt-dlp run.

    CRITICAL: yt-dlp REWRITES the --cookies file after every run to refresh
    rotating session tokens. Pointing it at the master means concurrent fetches
    (a montage pulls several sources at once) clobber each other and can strip
    the auth cookies — silently breaking age-restricted downloads. So each run
    gets a private disposable copy; the master stays pristine for its full
    session lifetime. Returns None when no master cookies are configured.
    """
    master = cookies_file()
    if not master:
        return None
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / f"ck_{uuid.uuid4().hex[:8]}.txt"
    shutil.copyfile(master, tmp)
    return str(tmp)


def _auth_args(cookies_path: str | None = None) -> list[str]:
    """Shared yt-dlp args: player-clients + EJS challenge solver + cookies.

    `--remote-components ejs:github` pulls YouTube's JS 'n'-challenge solver
    (yt-dlp split it out of core in 2026.03). Without it the web client can only
    return storyboards ('Only images are available') and every real download
    fails 'Requested format is not available'. yt-dlp caches the solver lib under
    ~/.cache/yt-dlp after the first fetch, so this is a one-time download.

    `cookies_path` MUST be a disposable copy (see stage_cookies) — never the
    master, which yt-dlp would rewrite/clobber.
    """
    args = [
        "--extractor-args", f"youtube:player_client={YT_PLAYER_CLIENTS}",
        "--remote-components", "ejs:github",
    ]
    if cookies_path:
        args += ["--cookies", cookies_path]
    return args


def _node_path() -> str | None:
    """Locate a Node >=18 binary for yt-dlp's JS runtime.

    yt-dlp without a JS runtime falls back to limited YouTube player clients and
    intermittently returns *no* formats ('fetched nothing'). Pointing it at Node
    (EJS) restores full web-client extraction. systemd's PATH has no Node, so we
    discover the nvm binary explicitly.
    """
    p = os.environ.get("FORGE_NODE_BIN")
    if p and os.path.exists(p):
        return p
    cands = sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/v*/bin/node")), reverse=True)
    return cands[0] if cands else None


def resolve_source(spec: str) -> tuple[str, str]:
    """Map a source spec to (target, kind). kind in {'url','search'}."""
    s = (spec or "").strip()
    if s.startswith("http://") or s.startswith("https://"):
        return s, "url"
    if s.lower().startswith("search:"):
        return f"ytsearch{SEARCH_N}:{s.split(':', 1)[1].strip()}", "search"
    return f"ytsearch{SEARCH_N}:{s}", "search"


def ytdlp_cmd(target: str, out_dir: Path, cookies_path: str | None = None) -> list[str]:
    """yt-dlp command: best <=1080p mp4, no playlists beyond the search N, into out_dir."""
    out_tmpl = str(Path(out_dir) / "src_%(autonumber)s.%(ext)s")
    cmd = [
        "yt-dlp", "--no-warnings", "--no-playlist" if target.startswith("http") else "--yes-playlist",
        "-f", "bv*[height<=1080][ext=mp4]+ba/b[height<=1080]/b",
        "--merge-output-format", "mp4",
        # Only pull a short window of each result — a montage samples slices, not
        # whole videos. This is the main guard against the fetch timing out.
        "--download-sections", SECTION_WINDOW,
        "--retries", "5", "--fragment-retries", "5",
        "--socket-timeout", "30", "-N", "4",
        *_auth_args(cookies_path),
        "-o", out_tmpl, target,
    ]
    node = _node_path()
    if node:
        cmd[1:1] = ["--js-runtimes", f"node:{node}"]
    return cmd


def fetch_source(spec: str, out_dir: Path, timeout: int = 480) -> list[Path]:
    """Fetch a source spec; return the downloaded mp4 paths (may be several for a search).

    Each call downloads into its OWN subdirectory. yt-dlp's autonumber resets per
    invocation, so without isolation a second source maps to the first's filename
    and gets skipped as 'already downloaded' — which silently broke every
    multi-source montage on the 2nd source onward.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    call_dir = out_dir / f"s_{uuid.uuid4().hex[:8]}"
    call_dir.mkdir(parents=True, exist_ok=True)
    target, _kind = resolve_source(spec)
    ck = stage_cookies(call_dir)  # disposable copy — never let yt-dlp touch the master
    proc = subprocess.run(ytdlp_cmd(target, call_dir, cookies_path=ck),
                          capture_output=True, text=True, timeout=timeout)
    new = sorted(call_dir.glob("*.mp4"))
    if not new:
        detail = (proc.stderr or proc.stdout or "").strip()[-300:]
        raise RuntimeError(f"yt-dlp fetched nothing for {spec!r}: {detail}")
    return new


def generate_clip(prompt: str, out_dir: Path, duration: int = 6, *,
                  source: str = "higgsfield", start_frame: str | Path | None = None) -> Path:
    """Generate a vessel clip for Forge — Kling on Higgsfield (default) or ComfyUI emergency.

    ``source="higgsfield"`` (the DEFAULT) drives Kling v3.0 (image-to-video) on the Ultra
    subscription: Kling needs a start frame, so when none is supplied we first synthesise a
    still via the image dispatcher and feed it in. ComfyUI/LTX is NOT silently used for the
    Mainstay team — on a Higgsfield failure we only fall back when the emergency switch is
    on; otherwise we raise so the job fails visibly and we top up / re-auth.
    """
    import logging
    import sys
    repo = Path(__file__).resolve().parent.parent
    if str(repo.parent) not in sys.path:
        sys.path.insert(0, str(repo.parent))

    log = logging.getLogger(__name__)
    dest = Path(out_dir) / f"gen_{uuid.uuid4().hex}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = (source or "higgsfield").lower()

    if source != "comfyui":  # default Forge path = Higgsfield (Kling)
        from core.forge import higgsfield  # lazy: keeps SDK optional
        if higgsfield.is_available():
            frame = start_frame
            if frame is None:
                # Kling is image-to-video — needs a start frame. Synthesise one.
                from core.forge import image_generation
                frame = image_generation.render_image(prompt, 1080, 1920, source="higgsfield")
            res = higgsfield.generate_video(
                prompt, dest, start_image=frame, duration=duration, aspect_ratio="9:16",
            )
            if res and Path(res).exists():
                return Path(res)
            log.error("HIGGSFIELD video generation failed (auth/credits/empty result).")
        else:
            log.error("HIGGSFIELD CLI unavailable — cannot generate clip.")

        # Higgsfield is down — break-glass only.
        from core.forge.image_generation import comfyui_emergency_enabled
        if not comfyui_emergency_enabled():
            raise RuntimeError(
                "Higgsfield video unavailable (out of credits?) and emergency ComfyUI is OFF. "
                "Check `higgsfield account status`; touch data/forge_emergency_comfyui.flag to "
                "bring ComfyUI back."
            )
        log.warning("FORGE_EMERGENCY_COMFYUI active — serving LTX/ComfyUI fallback clip.")

    from scripts.rucktalk_common import run_comfyui_video_cloud
    res = run_comfyui_video_cloud(prompt, duration=duration)
    if not res or not Path(res).exists():
        raise RuntimeError("LTX cloud returned no video")
    dest.write_bytes(Path(res).read_bytes())
    return dest
