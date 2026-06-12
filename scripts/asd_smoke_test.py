#!/usr/bin/env python3
"""Smoke test for core.forge.renderers.asd_provider.detect_active_speakers.

Runs the ASD provider on the 2-person test clip and prints the returned
active-speaker windows. Confirms it returns non-empty, sensible bboxes.
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.forge.renderers.asd_provider import detect_active_speakers  # noqa: E402

TEST_VIDEO = REPO / "data" / "asd_test" / "two_person_test.mp4"


def main() -> int:
    print(f"[smoke] test video: {TEST_VIDEO}  exists={TEST_VIDEO.exists()}")
    t0 = time.time()
    windows = detect_active_speakers(TEST_VIDEO)
    dt = time.time() - t0
    print(f"[smoke] detect_active_speakers returned {len(windows)} window(s) "
          f"in {dt:.1f}s")

    if not windows:
        print("[smoke] FAIL: empty result")
        return 1

    # Print first few windows.
    for w in windows[:8]:
        x, y, ww, hh = w["bbox"]
        print(f"  track={w['track_id']}  "
              f"{w['start_s']:.2f}s -> {w['end_s']:.2f}s  "
              f"bbox=(x={x}, y={y}, w={ww}, h={hh})")

    # Sanity: bboxes inside a 1280x720 frame and positive-area.
    ok = all(
        ww > 0 and hh > 0 and 0 <= x < 1280 and 0 <= y < 720
        for (x, y, ww, hh) in (w["bbox"] for w in windows)
    )
    tracks = sorted({w["track_id"] for w in windows})
    print(f"[smoke] distinct tracks with active windows: {tracks}")
    print(f"[smoke] bbox sanity (in-frame, positive area): {ok}")

    # Which side is the active speaker? Audio came from the LEFT clip, so we
    # expect the dominant active-speaker time to be on the left half (cx<640).
    left_time = sum(w["end_s"] - w["start_s"]
                    for w in windows if w["bbox"][0] + w["bbox"][2] / 2 < 640)
    right_time = sum(w["end_s"] - w["start_s"]
                     for w in windows if w["bbox"][0] + w["bbox"][2] / 2 >= 640)
    print(f"[smoke] active-speaker seconds  LEFT={left_time:.1f}  "
          f"RIGHT={right_time:.1f}  (audio was from LEFT clip)")

    print("[smoke] PASS" if ok else "[smoke] FAIL: bbox sanity failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
