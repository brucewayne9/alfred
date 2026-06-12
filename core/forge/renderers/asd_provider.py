"""Active Speaker Detection (ASD) provider for the Forge video engine.

Wraps the vendored LR-ASD pipeline (vendor/lr-asd, IJCV 2025 / CVPR 2023 by
Junhua Liao et al.) and exposes a single clean entry point:

    detect_active_speakers(video_path, work_dir=None) -> list[dict]

Each returned dict is a contiguous *active-speaker window*::

    {
        "start_s":  float,        # window start, seconds (source timeline)
        "end_s":    float,        # window end,   seconds (source timeline)
        "bbox":     (x, y, w, h), # active speaker's face box, SOURCE pixels
        "track_id": int,          # LR-ASD track index the window came from
    }

The pipeline detects faces (S3FD), tracks them across a shot, scores each
tracked face for "is this person currently talking" (audio-visual model),
then this module collapses the per-frame results into windows where a single
track is the active speaker.

------------------------------------------------------------------------------
25 fps caveat  &  how bboxes map back to SOURCE coordinates
------------------------------------------------------------------------------
LR-ASD's preprocessing re-encodes the input with ``ffmpeg ... -r 25`` BEFORE
any detection. That re-encode changes ONLY the frame rate, never the spatial
resolution -- so the bboxes stored in ``pywork/tracks.pckl`` are already in the
ORIGINAL pixel resolution of the source video. No spatial rescaling is needed.

What the 25 fps re-encode *does* change is timing: track frame indices are
indices into the 25 fps stream. We therefore convert frame index -> seconds as
``t = frame_index / 25.0``. Those seconds are correct against the SOURCE
timeline because the re-encode preserves wall-clock duration (it resamples
frames, it does not retime the content). Callers that need source *frame*
numbers at the source's native fps should multiply ``start_s``/``end_s`` by the
source fps themselves.

So: bbox is returned in source pixels (x, y, w, h) with no remapping; time
windows are in seconds and valid against the source timeline.

------------------------------------------------------------------------------
Failure behaviour
------------------------------------------------------------------------------
This module NEVER raises to the caller for an operational failure. On a missing
GPU, missing weights, a subprocess crash, no detected faces, or any unexpected
exception it logs and returns ``[]`` so the caller can fall back to a
center-crop / heuristic reframe.
"""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# LR-ASD operates entirely at 25 fps after its internal re-encode.
_ASD_FPS = 25.0

# Repo paths. asd_provider.py lives at core/forge/renderers/, repo root is 4 up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LRASD_DIR = _REPO_ROOT / "vendor" / "lr-asd"
_LRASD_SCRIPT = _LRASD_DIR / "Columbia_test.py"
_LRASD_WEIGHT = _LRASD_DIR / "weight" / "pretrain_AVA.model"

# Minimum mean score for a track to be considered "ever the active speaker"
# is implicit: per-frame score >= 0 means talking (LR-ASD convention).
_TALK_THRESHOLD = 0.0

# Smoothing / window hygiene.
_SMOOTH_HALF_WIN = 2          # +/- frames averaged when deciding talking state
_MIN_WINDOW_FRAMES = 5        # drop windows shorter than this (< 0.2s at 25fps)
_MERGE_GAP_FRAMES = 6         # bridge gaps up to this many frames in one track


def _python_exe() -> str:
    """Prefer the repo venv interpreter; fall back to the current one."""
    venv_py = _REPO_ROOT / "venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _run_lrasd(video_path: Path, run_dir: Path) -> Path:
    """Run the LR-ASD pipeline on ``video_path``.

    Lays out the demo-folder structure the script expects::

        run_dir/<name>.mp4         (input copy)
        run_dir/<name>/pywork/*.pckl  (outputs)

    Returns the ``pywork`` directory. Raises on subprocess failure (caught by
    the public wrapper).
    """
    name = "asdclip"
    staged = run_dir / f"{name}.mp4"
    shutil.copyfile(video_path, staged)

    env = dict(os.environ)
    # Make the vendored repo importable (it uses local 'model', 'ASD' imports)
    # and ensure CUDA is visible by default; caller can override.
    cmd = [
        _python_exe(),
        str(_LRASD_SCRIPT),
        "--videoName", name,
        "--videoFolder", str(run_dir),
        "--pretrainModel", str(_LRASD_WEIGHT),
    ]
    logger.info("LR-ASD: running %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(_LRASD_DIR),     # script resolves model/ relative to cwd
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60 * 30,         # 30 min hard cap
    )
    if proc.returncode != 0:
        # Surface the tail of the log for debugging.
        tail = "\n".join((proc.stdout or "").splitlines()[-25:])
        raise RuntimeError(
            f"LR-ASD exited {proc.returncode}. Last output:\n{tail}"
        )

    pywork = run_dir / name / "pywork"
    return pywork


def _load_pickles(pywork: Path):
    """Load tracks.pckl and scores.pckl. Returns (tracks, scores)."""
    tracks_p = pywork / "tracks.pckl"
    scores_p = pywork / "scores.pckl"
    if not tracks_p.exists() or not scores_p.exists():
        raise FileNotFoundError(
            f"LR-ASD outputs missing: {tracks_p.exists()=} {scores_p.exists()=}"
        )
    with open(tracks_p, "rb") as f:
        tracks = pickle.load(f)
    with open(scores_p, "rb") as f:
        scores = pickle.load(f)
    if len(tracks) != len(scores):
        logger.warning(
            "LR-ASD: tracks/scores length mismatch (%d vs %d); zipping shorter.",
            len(tracks), len(scores),
        )
    return tracks, scores


def _smoothed_active(score_arr, idx: int) -> bool:
    """True if the track is the active speaker at local frame ``idx``.

    Mirrors LR-ASD's visualization smoothing: average a small window of scores
    around ``idx`` and threshold at >= 0.
    """
    lo = max(idx - _SMOOTH_HALF_WIN, 0)
    hi = min(idx + _SMOOTH_HALF_WIN + 1, len(score_arr))
    if hi <= lo:
        return False
    window = score_arr[lo:hi]
    return (sum(window) / len(window)) >= _TALK_THRESHOLD


def _track_windows(track, score_arr, track_id: int) -> list[dict]:
    """Collapse one track's per-frame active flags into contiguous windows.

    ``track`` is ``{'track': {'frame': ndarray, 'bbox': ndarray[N,4]}, ...}``.
    ``bbox`` rows are [x1, y1, x2, y2] in SOURCE pixels.
    """
    frames = list(track["track"]["frame"])
    bboxes = track["track"]["bbox"]
    # Scores can be shorter than the track (LR-ASD truncates to min(audio,video)).
    n = min(len(frames), len(score_arr), len(bboxes))
    if n == 0:
        return []

    # Per-local-index talking flags.
    active = [_smoothed_active(score_arr, i) for i in range(n)]

    # Build raw runs of active frames (over LOCAL indices).
    runs: list[list[int]] = []
    cur: list[int] | None = None
    for i in range(n):
        if active[i]:
            if cur is None:
                cur = [i, i]
            else:
                cur[1] = i
        else:
            if cur is not None:
                runs.append(cur)
                cur = None
    if cur is not None:
        runs.append(cur)

    if not runs:
        return []

    # Merge runs separated by a small inactive gap (in LR-ASD frame numbers).
    merged: list[list[int]] = [runs[0]]
    for run in runs[1:]:
        prev = merged[-1]
        gap = frames[run[0]] - frames[prev[1]]
        if gap <= _MERGE_GAP_FRAMES:
            prev[1] = run[1]
        else:
            merged.append(run)

    windows: list[dict] = []
    for lo, hi in merged:
        if (frames[hi] - frames[lo] + 1) < _MIN_WINDOW_FRAMES:
            continue
        # Representative bbox = median over the window (robust to jitter).
        seg = bboxes[lo:hi + 1]
        xs1 = sorted(float(b[0]) for b in seg)
        ys1 = sorted(float(b[1]) for b in seg)
        xs2 = sorted(float(b[2]) for b in seg)
        ys2 = sorted(float(b[3]) for b in seg)
        m = len(seg) // 2
        x1, y1, x2, y2 = xs1[m], ys1[m], xs2[m], ys2[m]
        x = int(round(min(x1, x2)))
        y = int(round(min(y1, y2)))
        w = int(round(abs(x2 - x1)))
        h = int(round(abs(y2 - y1)))
        # Use LR-ASD frame numbers (25 fps) -> seconds on the source timeline.
        start_s = float(frames[lo]) / _ASD_FPS
        end_s = float(frames[hi] + 1) / _ASD_FPS
        windows.append({
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "bbox": (x, y, w, h),
            "track_id": track_id,
        })
    return windows


def detect_active_speakers(video_path, work_dir=None) -> list[dict]:
    """Detect active-speaker time windows in ``video_path``.

    Parameters
    ----------
    video_path : str | Path
        Path to the source video (.mp4/.avi/etc).
    work_dir : str | Path | None
        Directory for LR-ASD intermediates. If ``None`` a temp dir is created
        and removed afterwards. If provided, intermediates are KEPT (useful for
        debugging / inspecting the annotated output video).

    Returns
    -------
    list[dict]
        Active-speaker windows (see module docstring). Sorted by ``start_s``.
        Returns ``[]`` on any failure so callers can fall back to center-crop.
    """
    try:
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error("ASD: video not found: %s", video_path)
            return []
        if not _LRASD_SCRIPT.exists() or not _LRASD_WEIGHT.exists():
            logger.error(
                "ASD: vendored LR-ASD missing (script=%s weight=%s)",
                _LRASD_SCRIPT.exists(), _LRASD_WEIGHT.exists(),
            )
            return []

        # NOTE: the LR-ASD script runs with cwd=vendor/lr-asd, so run_dir MUST
        # be absolute or its internal glob(videoFolder/name.*) resolves against
        # the wrong directory and finds nothing.
        ephemeral = work_dir is None
        if ephemeral:
            run_dir = Path(tempfile.mkdtemp(prefix="asd_")).resolve()
        else:
            run_dir = (Path(work_dir) / f"asd_{uuid.uuid4().hex[:8]}").resolve()
            run_dir.mkdir(parents=True, exist_ok=True)

        try:
            pywork = _run_lrasd(video_path, run_dir)
            tracks, scores = _load_pickles(pywork)

            windows: list[dict] = []
            for tid in range(min(len(tracks), len(scores))):
                windows.extend(_track_windows(tracks[tid], scores[tid], tid))

            windows.sort(key=lambda w: (w["start_s"], w["track_id"]))
            logger.info(
                "ASD: %d track(s) -> %d active-speaker window(s) for %s",
                len(tracks), len(windows), video_path.name,
            )
            return windows
        finally:
            if ephemeral:
                shutil.rmtree(run_dir, ignore_errors=True)

    except Exception as exc:  # noqa: BLE001 - intentional catch-all
        logger.exception("ASD failed for %s: %s", video_path, exc)
        return []


__all__ = ["detect_active_speakers"]
