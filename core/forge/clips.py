"""Forge clip ingest — fetch footage (yt-dlp: YouTube/TikTok/IG) or generate (LTX cloud)."""
from __future__ import annotations
import subprocess
import uuid
from pathlib import Path

SEARCH_N = 3


def resolve_source(spec: str) -> tuple[str, str]:
    """Map a source spec to (target, kind). kind in {'url','search'}."""
    s = (spec or "").strip()
    if s.startswith("http://") or s.startswith("https://"):
        return s, "url"
    if s.lower().startswith("search:"):
        return f"ytsearch{SEARCH_N}:{s.split(':', 1)[1].strip()}", "search"
    return f"ytsearch{SEARCH_N}:{s}", "search"


def ytdlp_cmd(target: str, out_dir: Path) -> list[str]:
    """yt-dlp command: best <=1080p mp4, no playlists beyond the search N, into out_dir."""
    out_tmpl = str(Path(out_dir) / "src_%(autonumber)s.%(ext)s")
    return [
        "yt-dlp", "--no-warnings", "--no-playlist" if target.startswith("http") else "--yes-playlist",
        "-f", "bv*[height<=1080][ext=mp4]+ba/b[height<=1080]/b",
        "--merge-output-format", "mp4",
        "-o", out_tmpl, target,
    ]


def fetch_source(spec: str, out_dir: Path, timeout: int = 300) -> list[Path]:
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
    proc = subprocess.run(ytdlp_cmd(target, call_dir), capture_output=True, text=True, timeout=timeout)
    new = sorted(call_dir.glob("*.mp4"))
    if not new:
        detail = (proc.stderr or proc.stdout or "").strip()[-300:]
        raise RuntimeError(f"yt-dlp fetched nothing for {spec!r}: {detail}")
    return new


def generate_clip(prompt: str, out_dir: Path, duration: int = 6) -> Path:
    """Generate an LTX vessel clip on ComfyUI Cloud."""
    import sys
    repo = Path(__file__).resolve().parent.parent
    if str(repo.parent) not in sys.path:
        sys.path.insert(0, str(repo.parent))
    from scripts.rucktalk_common import run_comfyui_video_cloud
    res = run_comfyui_video_cloud(prompt, duration=duration)
    if not res or not Path(res).exists():
        raise RuntimeError("LTX cloud returned no video")
    dest = Path(out_dir) / f"gen_{uuid.uuid4().hex}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(res).read_bytes())
    return dest
