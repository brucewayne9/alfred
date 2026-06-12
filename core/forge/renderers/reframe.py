"""Speaker-aware reframe for 9:16 vertical clips.

Replaces the static center cover-crop with a crop that FOLLOWS the active speaker:
pan to keep the talking face framed, and cut to the other speaker when they take
over. Built in three pure, independently-tested layers plus an ffmpeg orchestrator,
with graceful fallback to a static center-crop whenever speaker data is missing
(single face, no faces, ASD unavailable) — so it never regresses today's behavior.

Active-speaker bboxes come from core.forge.renderers.asd_provider (LR-ASD). This
module is provider-agnostic: it just takes per-window active-speaker bboxes.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def _even(n: int) -> int:
    """Round down to an even integer (h264/yuv420p require even dimensions)."""
    n = int(round(n))
    return n - (n % 2)


def crop_window(
    bbox: tuple[float, float, float, float],
    src_w: int,
    src_h: int,
    ar_w: int = 9,
    ar_h: int = 16,
    headroom: float = 0.42,
) -> tuple[int, int, int, int]:
    """Compute the target-aspect crop rectangle centered on a face bbox.

    ``bbox`` is ``(fx, fy, fw, fh)`` (top-left + size) in source pixels. Returns
    ``(x, y, w, h)`` — the largest ``ar_w:ar_h`` window that fits the source,
    centered horizontally on the face, with the face placed ``headroom`` of the
    way down the frame (so eyes sit high, not dead-center), clamped to bounds,
    with even dimensions.
    """
    # Largest ar_w:ar_h region that fits the source.
    crop_w = _even(src_h * ar_w / ar_h)
    crop_h = _even(src_h)
    if crop_w > src_w:
        crop_w = _even(src_w)
        crop_h = _even(crop_w * ar_h / ar_w)

    fx, fy, fw, fh = bbox
    face_cx = fx + fw / 2
    face_cy = fy + fh / 2

    # Center horizontally on the face; shift the window down so the face sits at
    # `headroom` from the top (headroom<0.5 => face above the crop's midline).
    crop_cx = face_cx
    crop_cy = face_cy + (0.5 - headroom) * crop_h

    x = int(round(crop_cx - crop_w / 2))
    y = int(round(crop_cy - crop_h / 2))

    # Clamp to frame.
    x = max(0, min(x, src_w - crop_w))
    y = max(0, min(y, src_h - crop_h))
    return (x, y, crop_w, crop_h)


def _rle(seq: list) -> list[list]:
    """Run-length encode: -> [[value, start_idx, end_idx_exclusive], ...]."""
    runs: list[list] = []
    if not seq:
        return runs
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            runs.append([seq[start], start, i])
            start = i
    return runs


def build_segments(
    timeline: list,
    fps: float = 25.0,
    min_dwell: float = 1.2,
) -> list[tuple[float, float, object]]:
    """Merge a per-frame active-speaker ``timeline`` into stable cut segments.

    ``timeline[i]`` is the active speaker's track id at frame ``i`` (any hashable;
    ``None`` = nobody). Runs shorter than ``min_dwell`` seconds are absorbed into a
    neighbor so brief interjections/backchannels don't ping-pong the crop — the
    single biggest quality lever. Returns ``[(start_s, end_s, track_id), ...]``.
    """
    if not timeline:
        return []
    min_frames = max(1, int(round(min_dwell * fps)))
    seq = list(timeline)

    # Iteratively absorb the first too-short run into a neighbor until stable.
    changed = True
    while changed:
        changed = False
        runs = _rle(seq)
        if len(runs) <= 1:
            break
        for idx, (tid, s, e) in enumerate(runs):
            if (e - s) < min_frames:
                fill = runs[idx - 1][0] if idx > 0 else runs[idx + 1][0]
                for j in range(s, e):
                    seq[j] = fill
                changed = True
                break

    return [(s / fps, e / fps, tid) for tid, s, e in _rle(seq)]


def smooth_path(
    values: list[float],
    alpha: float = 0.12,
    dead_zone: float = 0.0,
    max_step: float | None = None,
) -> list[float]:
    """Smooth a 1-D path (crop-center x or y over frames) to kill jitter.

    EMA (``alpha`` = follow speed; lower = smoother/laggier) + a ``dead_zone`` that
    holds position on sub-threshold drift + an optional ``max_step`` per-frame
    velocity clamp so fast subject moves look intentional, not snappy. Apply once
    per axis.
    """
    out: list[float] = []
    cur: float | None = None
    for v in values:
        if cur is None:
            cur = float(v)
        else:
            target = cur if abs(v - cur) <= dead_zone else v
            nxt = cur + alpha * (target - cur)
            if max_step is not None:
                delta = nxt - cur
                if abs(delta) > max_step:
                    nxt = cur + (max_step if delta > 0 else -max_step)
            cur = nxt
        out.append(cur)
    return out


def plan_reframe(
    windows: list[dict],
    start_s: float,
    end_s: float,
    src_w: int,
    src_h: int,
    ar_w: int = 9,
    ar_h: int = 16,
) -> list[dict]:
    """Map active-speaker ``windows`` onto a clip span -> per-sub-segment crop plan.

    Each window is ``{start_s, end_s, bbox}`` in absolute source time. Windows are
    clipped to ``[start_s, end_s]``; non-overlapping ones dropped. Returns
    ``[{start_s, end_s, crop:(x,y,w,h)}, ...]`` sorted by time — one static crop per
    speaker span, with hard cuts at the boundaries.
    """
    # Clip speaker windows to the span.
    spk: list[tuple[float, float, tuple]] = []
    for w in windows:
        s = max(float(w["start_s"]), start_s)
        e = min(float(w["end_s"]), end_s)
        if e <= s:
            continue
        spk.append((s, e, crop_window(w["bbox"], src_w, src_h, ar_w, ar_h)))
    if not spk:
        return []
    spk.sort(key=lambda x: x[0])

    # Center cover-crop for gaps (before/between/after speaker windows), so the plan
    # covers the FULL [start_s, end_s] span — never shrink the clip.
    center = crop_window((src_w / 2.0, src_h / 2.0, 0, 0), src_w, src_h, ar_w, ar_h)
    plan: list[dict] = []
    cur = start_s
    for s, e, crop in spk:
        if s > cur + 1e-6:
            plan.append({"start_s": cur, "end_s": s, "crop": center})
        seg_start = max(s, cur)
        if e > seg_start + 1e-6:
            plan.append({"start_s": seg_start, "end_s": e, "crop": crop})
        cur = max(cur, e)
    if cur < end_s - 1e-6:
        plan.append({"start_s": cur, "end_s": end_s, "crop": center})
    return plan


def resolve_active_windows(
    windows: list[dict],
    fps: float = 25.0,
    min_dwell: float = 1.2,
) -> list[dict]:
    """Collapse the ASD provider's (possibly overlapping) per-track windows into a
    single non-overlapping active-speaker timeline.

    Provider returns ``{start_s,end_s,bbox,track_id}`` per track, which can overlap
    during crosstalk. Per frame the longest covering window wins; ``build_segments``
    then applies min-dwell/hysteresis so brief overlaps don't ping-pong. Returns
    non-overlapping ``[{start_s,end_s,bbox}, ...]`` ready for :func:`plan_reframe`.
    """
    if not windows:
        return []
    t0 = min(float(w["start_s"]) for w in windows)
    t1 = max(float(w["end_s"]) for w in windows)
    n = max(1, int(round((t1 - t0) * fps)))

    timeline: list = [None] * n
    for i in range(n):
        t = t0 + (i + 0.5) / fps
        covering = [w for w in windows if w["start_s"] <= t < w["end_s"]]
        if covering:
            best = max(covering, key=lambda w: w["end_s"] - w["start_s"])
            timeline[i] = best["track_id"]

    out: list[dict] = []
    for s, e, tid in build_segments(timeline, fps=fps, min_dwell=min_dwell):
        if tid is None:
            continue
        a, b = t0 + s, t0 + e
        mid = (a + b) / 2
        cand = [w for w in windows if w["track_id"] == tid and w["start_s"] <= mid < w["end_s"]]
        if not cand:
            cand = [w for w in windows if w["track_id"] == tid]
        if not cand:
            continue
        out.append({"start_s": a, "end_s": b, "bbox": cand[0]["bbox"]})
    return out


# ---- ffmpeg orchestration (verified end-to-end on real video) ----

_VF_CENTER = ("scale={w}:{h}:force_original_aspect_ratio=increase,"
              "crop={w}:{h},fps=30,format=yuv420p")


def _static_crop(src, start_s, end_s, out, w, h):
    """Today's behavior: static center cover-crop to w x h. The safe fallback."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", str(start_s), "-to", str(end_s),
         "-i", str(src), "-vf", _VF_CENTER.format(w=w, h=h),
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"static crop failed: {proc.stderr[-400:]}")
    return out


def _crop_subseg(src, p, src_w, src_h, out, w, h):
    """Cut one planned sub-segment with a fixed crop on the active speaker."""
    x, y, cw, ch = p["crop"]
    vf = (f"crop={cw}:{ch}:{x}:{y},"
          f"scale={w}:{h},fps=30,format=yuv420p,setsar=1")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", str(p["start_s"]), "-to", str(p["end_s"]),
         "-i", str(src), "-vf", vf, "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"reframe sub-seg failed: {proc.stderr[-400:]}")
    return out


def reframe_segment(
    src: str | Path,
    start_s: float,
    end_s: float,
    out_path: str | Path,
    w: int = 1080,
    h: int = 1920,
    active_bboxes: list[dict] | None = None,
    src_w: int | None = None,
    src_h: int | None = None,
) -> Path:
    """Cut ``[start_s, end_s)`` to ``w x h``, following the active speaker.

    ``active_bboxes`` are absolute-time speaker windows ``{start_s,end_s,bbox}`` from
    the ASD provider. With none (or on ANY error, or a single span), falls back to
    the static center-crop — so this never regresses today's behavior.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if active_bboxes and src_w and src_h:
            plan = plan_reframe(active_bboxes, start_s, end_s, src_w, src_h, w, h)
            # Only worth the multi-cut path when the speaker actually changes.
            if len(plan) >= 2:
                work = Path(tempfile.mkdtemp(prefix="forge_reframe_"))
                try:
                    parts = []
                    for i, p in enumerate(plan):
                        sp = work / f"sub_{i:03d}.mp4"
                        _crop_subseg(src, p, src_w, src_h, sp, w, h)
                        parts.append(sp)
                    listf = work / "concat.txt"
                    listf.write_text("".join(f"file '{p}'\n" for p in parts))
                    silent = work / "body.mp4"
                    subprocess.run(
                        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                         "-i", str(listf), "-c", "copy", str(silent)],
                        capture_output=True, text=True, check=True)
                    # Mux the original audio for the span back on.
                    proc = subprocess.run(
                        ["ffmpeg", "-y", "-v", "error",
                         "-i", str(silent),
                         "-ss", str(start_s), "-to", str(end_s), "-i", str(src),
                         "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
                         "-shortest", str(out_path)],
                        capture_output=True, text=True)
                    if proc.returncode == 0 and out_path.exists():
                        return out_path
                    # else fall through to static fallback
                finally:
                    shutil.rmtree(work, ignore_errors=True)
    except Exception:
        pass  # any failure -> safe static fallback below
    return _static_crop(src, start_s, end_s, out_path, w, h)
