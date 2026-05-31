"""Film-montage renderer: hook + borrowed/generated clips -> branded 9:16 montage."""
from __future__ import annotations
import math
from pathlib import Path


def plan_segments(clip_count: int, hook_seconds: float, seg_seconds: float = 2.5) -> list[dict]:
    """Round-robin segments covering hook_seconds; last one trimmed to land exactly."""
    if clip_count <= 0:
        raise ValueError("no clips to montage")
    n = max(1, math.ceil(hook_seconds / seg_seconds))
    segs, used = [], 0.0
    for i in range(n):
        remaining = hook_seconds - used
        secs = min(seg_seconds, remaining)
        if secs <= 0.01:
            break
        segs.append({"clip_index": i % clip_count, "seconds": round(secs, 3)})
        used += secs
    return segs
