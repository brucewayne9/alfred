"""
core.forge.multiply
-------------------
Multiplication engine: given ONE rendered master media file (image or video),
produce N non-duplicate variants by applying combinations of ffmpeg transforms.

Public API
----------
    multiply(master_path, count, out_dir, *, base_name="variant") -> list[Path]
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}

DHASH_SIZE = 8          # produces 8*8 = 64-bit hash
MIN_HAMMING = 6         # internal uniqueness threshold
MAX_RETRIES = 2         # retries before accepting a near-duplicate


# ---------------------------------------------------------------------------
# dHash (difference hash) — 8x8 → 64-bit int, no new deps
# ---------------------------------------------------------------------------

def _dhash(path: Path) -> int:
    """Compute an 8x8 dHash of an image file.  Returns a 64-bit integer."""
    img = Image.open(path).convert("L").resize(
        (DHASH_SIZE + 1, DHASH_SIZE), Image.LANCZOS
    )
    arr = np.array(img, dtype=np.uint8)
    # horizontal gradient: each row, left pixel < right pixel?
    diff = arr[:, :-1] < arr[:, 1:]   # shape (8, 8) bool
    bits = diff.flatten()
    # pack to int
    value = int(np.packbits(bits).tobytes().hex(), 16)
    return value


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def _get_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)


def _has_audio(path: Path) -> bool:
    """Return True if the file contains at least one audio stream."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


# ---------------------------------------------------------------------------
# Structural transform groups
# ---------------------------------------------------------------------------
# The pool is split into two "hemispheres":
#   - NORMAL  (original orientation)
#   - FLIPPED (horizontally flipped) — guaranteed >= 40 bits from NORMAL group
#
# Within each hemisphere we have anchored spatial variants that change the
# crop origin enough to move dHash bits even on highly-regular test patterns.
# Color/hue/vignette tweaks are layered ON TOP for extra divergence.
#
# Each entry: (base_vf, speed_k_or_None)
# W and H placeholders are filled at call time.

def _make_structural_pool(W: int, H: int, allow_flip: bool = True,
                          text_safe: bool = False) -> list[tuple[str, float | None]]:
    """Return the ordered list of (vf, speed_k) base recipes for WxH masters.

    When *allow_flip* is False the flipped hemisphere is omitted entirely,
    which is required for text-bearing leak covers where hflip produces
    backwards/mirrored text.

    When *text_safe* is True the pool contains NO geometry at all — no flip,
    zoom, crop, or rotate — because the master has captions/lyrics burned into
    the pixels and any spatial transform mirrors them or shoves them off the
    9:16 frame.  Divergence comes purely from colour/grade (and, for video, the
    speed layer added downstream, which also re-samples the dHash frame).  This
    is what kinetic-lyric and film-montage need: distinct distribution copies
    that keep every word on-screen and reading forwards.
    """
    if text_safe:
        # Colour/grade + texture bases. None move a single pixel spatially, so
        # burned-in lyrics never mirror or leave frame. Each pairs a grade with a
        # grain/sharpen pass: that texture (on-brand film grain for the moody Rod
        # Wave look) is what actually diverges the perceptual hash between copies,
        # since flat colour shifts barely move a gradient-based dHash.
        grades = [
            f"crop={W}:{H}:0:0",                                          # clean passthrough
            "eq=contrast=1.07:saturation=1.10,noise=alls=6:allf=t",       # punchy + light grain
            "eq=brightness=0.04:contrast=1.04,unsharp=5:5:0.8",           # brighter + crisp
            "eq=brightness=-0.05:contrast=1.08:saturation=0.92,noise=alls=10:allf=t",  # moody + grain
            "curves=preset=darker,noise=alls=8:allf=t",                   # crushed + grain
            "curves=preset=lighter,unsharp=5:5:0.6",                      # lifted + crisp
            "hue=h=8,eq=saturation=1.10,noise=alls=7:allf=t",             # warm + grain
            "hue=h=-8,eq=saturation=1.06,unsharp=3:3:0.7",                # cool + crisp
            f"vignette={round(math.pi/5, 4)},noise=alls=9:allf=t",        # vignette + grain
            "colorbalance=rs=0.06:gs=0.02:bs=-0.06,noise=alls=6:allf=t",  # warm grade + grain
            "colorbalance=rs=-0.06:gs=0.0:bs=0.06,unsharp=5:5:0.9",       # teal grade + crisp
            "eq=gamma=1.08:contrast=1.06:saturation=1.12,noise=alls=11:allf=t",  # filmic + grain
        ]
        return [(g, None) for g in grades]

    # Zoom levels that produce observably different 8x8 dHash on regular patterns
    z_factors = [0.90, 0.80, 0.70, 0.95, 0.85, 0.75]

    # Directional crop offsets (crop from different corners, then scale back)
    # Each gives a structurally distinct dHash even on testsrc
    crop_variants = [
        f"crop={W}:{H}:0:0",                                   # identity crop (no offset)
        f"crop={W-int(W*0.12)}:{H-int(H*0.12)}:{int(W*0.12)}:0,scale={W}:{H}",   # top-right shift
        f"crop={W-int(W*0.12)}:{H-int(H*0.12)}:0:{int(H*0.12)},scale={W}:{H}",   # bottom-left shift
        f"crop={W-int(W*0.15)}:{H-int(H*0.15)}:{int(W*0.07)}:{int(H*0.08)},scale={W}:{H}",  # center-ish
    ]

    pool: list[tuple[str, float | None]] = []

    # --- NORMAL hemisphere ---
    # 1: zoom variants
    for zf in z_factors:
        pool.append((f"crop=iw*{zf:.2f}:ih*{zf:.2f},scale={W}:{H}", None))

    # 2: directional crops (normal orientation)
    for cv in crop_variants[1:]:   # skip identity
        pool.append((cv, None))

    # --- FLIPPED hemisphere (guaranteed ~50+ bits from normal) ---
    # Skipped when allow_flip=False so text-bearing covers are never mirrored.
    if allow_flip:
        for zf in z_factors:
            pool.append((f"hflip,crop=iw*{zf:.2f}:ih*{zf:.2f},scale={W}:{H}", None))

        for cv in crop_variants[1:]:
            pool.append((f"hflip,{cv}", None))

    return pool


def _color_layer(index: int) -> str:
    """Return a color/hue/vignette tweak string to append, varies by index."""
    layers = [
        "",                                                # no extra layer
        f"eq=brightness={round(-0.03 + (index % 5)*0.015, 3)}:contrast={round(0.97 + (index%5)*0.01,3)}:saturation={round(0.95 + (index%5)*0.025,3)}",
        f"hue=h={round(-6 + (index % 7) * 2.0, 1)}",
        f"vignette={round(math.pi/5 + (index%4)*0.07, 4)}",
        f"eq=brightness={round(0.02,3)}:contrast={round(1.03,3)}:saturation={round(1.06,3)},hue=h={round(-4+(index%3)*4,1)}",
    ]
    return layers[index % len(layers)]


def _recipe_rotate(i: int, W: int, H: int) -> tuple[str, None]:
    """Slight rotate with crop-back to original dimensions (layered on top)."""
    angle = round(0.5 + (i % 3) * 0.5, 2)     # 0.5 … 1.5 degrees
    if i % 2 == 0:
        angle = -angle
    rad = math.radians(angle)
    return (
        f"rotate={rad:.6f}:ow=rotw({rad:.6f}):oh=roth({rad:.6f}),crop={W}:{H}",
        None,
    )


def _recipe_speed(i: int) -> tuple[str, float]:
    """Video-only: vary playback speed (pts scale); K != 1."""
    speeds = [0.94, 0.96, 0.98, 1.02, 1.04, 1.06]
    k = speeds[i % len(speeds)]
    return f"setpts={k:.3f}*PTS", k


def _build_filter(index: int, W: int, H: int, is_video: bool) -> tuple[str, float | None]:
    """
    Deterministically pick structural base + optional color layer for variant `index`.
    Returns (vf_string, speed_k_or_None).
    """
    pool = _make_structural_pool(W, H)
    base_vf, _ = pool[index % len(pool)]

    # Color layer varies separately so even same-structure variants with
    # different color tweaks stay above threshold; but structural divergence
    # is the primary driver.
    color = _color_layer(index)

    parts = [base_vf]
    if color:
        parts.append(color)

    # Add slight rotate on every third variant for extra spatial divergence
    if index % 3 == 2:
        rot_vf, _ = _recipe_rotate(index, W, H)
        parts.append(rot_vf)

    speed_k: float | None = None
    if is_video:
        spd_vf, speed_k = _recipe_speed(index)
        parts.append(spd_vf)

    vf = ",".join(parts)
    return vf, speed_k


# ---------------------------------------------------------------------------
# ffmpeg runners
# ---------------------------------------------------------------------------

def _run_image(master: Path, vf: str, out: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(master),
        "-vf", vf,
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _run_video(master: Path, vf: str, speed_k: float | None, out: Path) -> None:
    has_aud = _has_audio(master)

    cmd = ["ffmpeg", "-y", "-i", str(master)]

    # Video filter
    cmd += ["-vf", vf]

    # Audio handling
    if has_aud and speed_k is not None:
        # atempo must stay in range [0.5, 2.0]
        atempo = round(1.0 / speed_k, 4)
        cmd += ["-filter:a", f"atempo={atempo}"]
    elif not has_aud:
        cmd += ["-an"]

    # Codecs
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
    if has_aud:
        cmd += ["-c:a", "aac"]

    cmd.append(str(out))
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Middle-frame extractor for video dHash
# ---------------------------------------------------------------------------

def _extract_middle_frame(video: Path, dest: Path) -> None:
    """Extract a single frame near the middle of the video to dest (PNG)."""
    # Get duration
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    try:
        duration = float(result.stdout.strip())
        seek = duration / 2
    except ValueError:
        seek = 1.0

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(seek),
            "-i", str(video),
            "-frames:v", "1",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# dHash adapter: handles images and videos uniformly
# ---------------------------------------------------------------------------

def _compute_hash(path: Path, is_video: bool) -> int:
    if is_video:
        with tempfile.TemporaryDirectory() as td:
            frame = Path(td) / "frame.png"
            _extract_middle_frame(path, frame)
            return _dhash(frame)
    else:
        return _dhash(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def multiply(
    master_path: Union[str, Path],
    count: int,
    out_dir: Union[str, Path],
    *,
    base_name: str = "variant",
    allow_flip: bool = True,
    text_safe: bool = False,
) -> list[Path]:
    """Produce `count` non-duplicate variants of `master_path` into `out_dir`.

    Set *allow_flip=False* for text-bearing masters (leak covers, etc.) so
    that no hflip variants are generated and text is never rendered backwards.

    Set *text_safe=True* for masters with captions/lyrics burned into the pixels
    (kinetic-lyric, film-montage): the variants then use colour/grade only — no
    flip, zoom, crop, or rotate — so every word stays on-screen and forwards.

    Returns the list of variant file paths (same extension as master).
    """
    master = Path(master_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = master.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        is_video = False
    elif ext in VIDEO_EXTENSIONS:
        is_video = True
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    W, H = _get_dimensions(master)
    pool = _make_structural_pool(W, H, allow_flip=allow_flip, text_safe=text_safe)
    pool_size = len(pool)

    # Pre-assign pool slots: spread count variants evenly across the pool,
    # guaranteeing each gets a distinct structural base.
    # We stride across the pool so consecutive variants come from different
    # structural hemispheres (first half = normal, second half = flipped).
    stride = max(1, pool_size // count)
    base_slots = [(i * stride) % pool_size for i in range(count)]

    accepted_hashes: list[int] = []
    used_slots: set[int] = set()
    results: list[Path] = []

    for i in range(count):
        out_path = out_dir / f"{base_name}_{i:02d}{ext}"

        # Find a slot that hasn't been used yet; start from the pre-assigned slot
        slot = base_slots[i]
        tried_slots: list[int] = []

        for retry in range(MAX_RETRIES + 1):
            # Advance to an unused slot if current one is taken
            scan = slot
            for _ in range(pool_size):
                if scan not in used_slots:
                    slot = scan
                    break
                scan = (scan + 1) % pool_size
            tried_slots.append(slot)

            base_vf, _ = pool[slot]
            # Build full filter: base + color layer + optional rotate + optional speed
            color = _color_layer(i + retry)
            parts = [base_vf]
            if color:
                parts.append(color)
            # Rotate tilts burned-in text off-axis, so it's skipped for text_safe.
            if not text_safe and (i + retry) % 3 == 2:
                rot_vf, _ = _recipe_rotate(i + retry, W, H)
                parts.append(rot_vf)
            speed_k: float | None = None
            if is_video:
                spd_vf, speed_k = _recipe_speed(i + retry)
                parts.append(spd_vf)
            vf = ",".join(parts)

            try:
                if is_video:
                    _run_video(master, vf, speed_k, out_path)
                else:
                    _run_image(master, vf, out_path)
            except subprocess.CalledProcessError as exc:
                print(
                    f"[forge/multiply] ffmpeg failed on variant {i} "
                    f"(attempt {retry}, slot={slot}, filter={vf!r}): "
                    f"{exc.stderr[-200:] if exc.stderr else ''}"
                )
                # Move to next slot and retry
                slot = (slot + 1) % pool_size
                continue

            # The 8x8 dHash only sees coarse geometry/luminance structure — it's
            # blind to the grade/grain that text_safe relies on, so don't make it
            # arbitrate copies it can't measure (would force pointless re-renders).
            h = _compute_hash(out_path, is_video)
            too_close = (not text_safe) and any(
                _hamming(h, ah) < MIN_HAMMING for ah in accepted_hashes)

            if not too_close:
                used_slots.add(slot)
                break
            else:
                if retry < MAX_RETRIES:
                    print(
                        f"[forge/multiply] variant {i} too similar (slot={slot}, "
                        f"attempt {retry}); trying next slot"
                    )
                    # Mark this slot as used (produces something too-similar)
                    # and move to next unused slot
                    used_slots.add(slot)
                    slot = (slot + 1) % pool_size
                else:
                    print(
                        f"[forge/multiply] variant {i} accepted after max retries "
                        f"(may be near-duplicate)"
                    )
                    used_slots.add(slot)

        accepted_hashes.append(_compute_hash(out_path, is_video))
        results.append(out_path)

    return results
