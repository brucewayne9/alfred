"""Mainstay Forge — viral clip scoring ("Auto-Clips").

Phase 08: rank the best standalone clip moments in a transcribed source and
score each 0-100 — the Opus-Clip-style behaviour Forge was missing.

Design notes
------------
* Judge model is ``kimi-k2.6:cloud`` via local Ollama (``localhost:11434``) —
  same instance the embeddings already use. $0 marginal (flat Ollama Cloud
  subscription), GPU untouched. Swappable via ``FORGE_JUDGE_MODEL``.
* The judge is prompted as a viral short-form editor for a music artist's
  social team — the rubric is music/artist-tuned, NOT generic.
* Kimi is shakier than Claude on strict JSON, so ``parse_candidates`` is
  defensive: it tolerates prose/markdown wrapping, alternate container keys,
  clamps scores, and drops malformed entries rather than raising.
* Timestamps from the model are snapped to real word boundaries (from the
  Whisper word timings already stored) so cuts land clean.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

from core.forge.db import _conn, init_db
from core.forge.ingest import get_segments

logger = logging.getLogger(__name__)

# Judge model — Kimi via local Ollama. $0 marginal, GPU-free. Override with
# FORGE_JUDGE_MODEL to A/B (e.g. glm-5.1:cloud, or a local instruct model).
DEFAULT_JUDGE_MODEL = "kimi-k2.6:cloud"
OLLAMA_URL = "http://localhost:11434/api/chat"
_DEFAULT_MAX_CLIPS = 20

# Music/artist-tuned rubric — NOT generic talking-head virality. This is what
# pops on a Rod Wave interview: vulnerable beats, origin story, quotable bars.
_SYSTEM_PROMPT = (
    "You are a viral short-form editor for a music artist's social team. You "
    "read interview transcripts and pick the moments that will pop as 9:16 "
    "TikTok/Reels/Shorts clips.\n\n"
    "Score each moment 0-100 on: hook strength in the first 3 seconds (does it "
    "stop the scroll), emotional intensity (vulnerable admissions, origin-story "
    "beats, raw moments), quotability (a line a fan would screenshot), story "
    "completeness (setup and payoff inside the clip), standalone clarity (makes "
    "sense with zero prior context), and controversy/hot-take that drives "
    "comments without being a liability.\n\n"
    "Favour complete thoughts. A great clip is 15-60 seconds. Pick start/end "
    "times that begin on a clean sentence and end on a payoff."
)

_USER_TEMPLATE = (
    "Here is the transcript, one line per segment as [start_seconds] (speaker) text:\n\n"
    "{transcript}\n\n"
    "Return the {max_clips} best standalone clip moments as a JSON array. Each "
    "element must be an object with exactly these keys:\n"
    '  "start_s" (number, seconds), "end_s" (number, seconds),\n'
    '  "score" (integer 0-100), "hook" (the exact opening line),\n'
    '  "emotion" (one of: vulnerable, hot-take, origin-story, quotable, funny, aspirational),\n'
    '  "reason" (one sentence: why it pops), "caption" (a suggested social caption).\n'
    "Return ONLY the JSON array, nothing else."
)

# Container keys a model might wrap the list in instead of returning a bare array.
_CONTAINER_KEYS = ("moments", "clips", "candidates", "results")


def _coerce_list(data) -> list:
    """Pull the moment list out of whatever shape the model returned."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in _CONTAINER_KEYS:
            val = data.get(key)
            if isinstance(val, list):
                return val
    return []


def _extract_json(raw: str):
    """Best-effort: load JSON from a model reply that may be wrapped in prose
    or ```json fences. Returns the decoded object, or None."""
    if not raw or not raw.strip():
        return None
    # Strip markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(raw)
    # Also try the widest [...] or {...} slice — models love trailing chatter.
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = raw.find(opener), raw.rfind(closer)
        if 0 <= i < j:
            candidates.append(raw[i:j + 1])
    for text in candidates:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def build_prompt(transcript_text: str, max_clips: int = _DEFAULT_MAX_CLIPS) -> list[dict]:
    """Build the chat messages for the judge."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(
            transcript=transcript_text, max_clips=max_clips)},
    ]


def _ollama_chat(messages: list[dict], model: str) -> str:
    """Call Ollama /api/chat and return the assistant text.

    ``think=False`` + a generous ``num_predict`` because Kimi's reasoning mode
    otherwise burns the token budget thinking with empty content (see
    core/seo/content/blog_planner.py). ``format=json`` nudges clean output.
    Retries once on transport/decode failure.
    """
    import requests  # noqa: PLC0415

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.4, "num_predict": 4096},
    }
    last_err = None
    for attempt in range(2):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=180)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "") or ""
        except Exception as e:  # noqa: BLE001 — transport/JSON, retry then surface
            last_err = e
            logger.warning("judge call failed (attempt %d/2): %s", attempt + 1, e)
    raise RuntimeError(f"judge call failed: {last_err}")


def score_source(source_id: str, max_clips: int = _DEFAULT_MAX_CLIPS,
                 chat_fn=None, judge_model: str | None = None,
                 now: int | None = None) -> list[dict]:
    """Score a transcribed source into ranked clip candidates.

    Pipeline: load segments → render transcript → judge → parse → snap each
    candidate to real word edges → persist (replacing prior scores). Returns
    the stored candidates, highest score first. ``chat_fn`` is injectable for
    testing; it defaults to the real Kimi/Ollama call.
    """
    segments = get_segments(source_id)
    if not segments:
        logger.info("score_source: no segments for %s — nothing to score", source_id)
        return []
    model = judge_model or os.environ.get("FORGE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    chat = chat_fn or _ollama_chat
    messages = build_prompt(build_transcript_text(segments), max_clips)
    raw = chat(messages, model)
    candidates = [snap_to_words(c, segments) for c in parse_candidates(raw)]
    save_candidates(source_id, candidates, judge_model=model, now=now)
    logger.info("score_source: %s -> %d candidates (judge=%s)",
                source_id, len(candidates), model)
    return get_candidates(source_id)


def save_candidates(source_id: str, candidates: list[dict],
                    judge_model: str = "", now: int | None = None) -> int:
    """Persist scored candidates for a source, replacing any prior scores.

    Delete-before-insert so re-scoring a source never leaves stale rows.
    Returns the number of candidates stored.
    """
    init_db()
    ts = int(time.time()) if now is None else now
    with _conn() as c:
        c.execute("DELETE FROM clip_candidates WHERE source_id = ?", (source_id,))
        for cand in candidates:
            c.execute(
                """
                INSERT INTO clip_candidates
                    (source_id, start_s, end_s, score, hook, emotion, reason,
                     caption, judge_model, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    float(cand["start_s"]), float(cand["end_s"]),
                    int(cand.get("score", 0)),
                    cand.get("hook", ""), cand.get("emotion", ""),
                    cand.get("reason", ""), cand.get("caption", ""),
                    judge_model, ts,
                ),
            )
    return len(candidates)


def get_candidates(source_id: str) -> list[dict]:
    """Return a source's clip candidates, highest score first."""
    init_db()
    with _conn() as c:
        rows = c.execute(
            """
            SELECT id, source_id, start_s, end_s, score, hook, emotion, reason,
                   caption, judge_model, rendered, posted, created_at
              FROM clip_candidates
             WHERE source_id = ?
             ORDER BY score DESC, start_s ASC
            """,
            (source_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def build_transcript_text(segments: list[dict]) -> str:
    """Render segments as one ``[start_s] (speaker) text`` line each.

    Start times are in seconds so the judge returns timestamps in the same
    units we can snap against. Speaker tag is omitted when unknown.
    """
    lines = []
    for seg in segments:
        start = float(seg.get("start_s", 0.0))
        speaker = seg.get("speaker")
        tag = f"({speaker}) " if speaker else ""
        lines.append(f"[{start:.1f}] {tag}{seg.get('text', '')}".rstrip())
    return "\n".join(lines)


def snap_to_words(candidate: dict, segments: list[dict]) -> dict:
    """Snap a candidate's ``start_s``/``end_s`` to the nearest real word edges.

    Uses the Whisper word timings already stored on the segments so a cut never
    lands mid-word. No-op when the source has no word timings. Returns a new
    dict; all non-timestamp fields are preserved.
    """
    starts: list[float] = []
    ends: list[float] = []
    for seg in segments:
        for w in seg.get("words") or []:
            try:
                starts.append(float(w["start"]))
                ends.append(float(w["end"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not starts or not ends:
        return candidate
    out = dict(candidate)
    out["start_s"] = min(starts, key=lambda s: abs(s - float(candidate["start_s"])))
    out["end_s"] = min(ends, key=lambda e: abs(e - float(candidate["end_s"])))
    return out


def parse_candidates(raw: str) -> list[dict]:
    """Parse a judge reply into validated clip candidates.

    Defensive by design (Kimi is shaky on strict JSON): tolerates prose/fence
    wrapping and alternate container keys, clamps scores to 0-100, and drops
    entries without a valid forward time range rather than raising.
    """
    items = _coerce_list(_extract_json(raw))
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            start_s = float(item["start_s"])
            end_s = float(item["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_s <= start_s:
            continue
        try:
            score = int(round(float(item.get("score", 0))))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(100, score))
        out.append({
            "start_s": start_s,
            "end_s": end_s,
            "score": score,
            "hook": str(item.get("hook") or ""),
            "emotion": str(item.get("emotion") or ""),
            "reason": str(item.get("reason") or ""),
            "caption": str(item.get("caption") or ""),
        })
    return out
