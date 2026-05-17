"""Retry the Roen batch — runs the 7 topics that weren't completed last time.

The first run completed topic[0] only ('layering beaded bracelets', pending #5).
This script runs topics 1..7 from the same TOPICS list, sequentially, with
per-topic try/except so a single failure doesn't abort the batch.
"""
from __future__ import annotations

import logging
import sys
import time
import traceback

from scripts.seo_batch_roen import TOPICS, run_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("seo_batch_roen_remaining")


def main() -> int:
    remaining = TOPICS[1:]
    total = len(remaining)
    log.info("=== Roen batch RETRY — %d remaining topics (skipping topic[0]) ===", total)
    successes: list[dict] = []
    failures: list[dict] = []
    overall_start = time.time()

    for i, t in enumerate(remaining, start=1):
        try:
            out = run_one(i, total, t)
            successes.append(out)
            log.info("[%d/%d] ✓ pending_id=%d", i, total, out["pending_id"])
        except Exception as e:
            log.error("[%d/%d] ✗ FAILED topic=%r: %s", i, total, t[0][:70], e)
            log.error(traceback.format_exc())
            failures.append({"topic": t[0], "error": str(e)})

    elapsed = time.time() - overall_start
    log.info("=== retry done in %.1fs (%.1f min) — %d ok, %d failed ===",
             elapsed, elapsed / 60, len(successes), len(failures))
    print("\n=== SUMMARY (retry) ===")
    for s in successes:
        print(f"  ✓ #{s['pending_id']:>4}  hero={s['hero_id']:>4}  {s['title'][:80]}")
    if failures:
        print("\n=== FAILURES (retry) ===")
        for f in failures:
            print(f"  ✗ {f['topic'][:70]}\n    {f['error']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
