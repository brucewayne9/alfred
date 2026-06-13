"""Mainstay Forge — topic-targeted segment retrieval.

Phase 11: windowing + bge-m3 embedding + source-scoped ranked query.

Design notes
------------
* One shared ChromaDB collection ``forge_segments`` — sources scoped via
  ``source_id`` metadata, NOT one collection per source (avoids ChromaDB
  overhead and keeps query interface simple).
* Windows are 20-40 s of adjacent segments merged by duration, NOT split on
  speaker boundaries — a short interjection flips speaker but not topic.
* bge-m3 is pinned — never swapped for a :cloud model (CLAUDE.md whitelist).
* Speaker ``None`` is coerced to ``""`` before upsert — ChromaDB metadata
  rejects ``None`` values.
* ``embed_source_windows`` deletes-before-upsert so re-embedding never leaves
  orphan win_ids.
* Cosine distance d ∈ [0, 2] is inverted to score ∈ [0, 1] via
  ``score = round(1.0 - d / 2.0, 4)``.  Raw distances are NEVER returned.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from core.memory.store import get_client

if TYPE_CHECKING:
    import chromadb

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "forge_segments"

# ---------------------------------------------------------------------------
# Collection accessor
# ---------------------------------------------------------------------------


def _get_collection() -> "chromadb.Collection":
    """Return (or create) the shared forge_segments ChromaDB collection.

    Uses an OllamaEmbeddingFunction backed by bge-m3:latest.  The function is
    attached at creation time so ChromaDB uses it for both upsert and query.
    """
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction  # noqa: PLC0415

    ef = OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name="bge-m3:latest",
    )
    return get_client().get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef,
    )


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def build_windows(
    segments: list[dict],
    target_dur_s: float = 30.0,
    max_dur_s: float = 45.0,
    min_dur_s: float = 10.0,
) -> list[dict]:
    """Merge adjacent segments into topic-coherent windows.

    Grows a buffer of adjacent segments.  Flushes the buffer into a window
    when adding the *next* segment would push the buffer past ``max_dur_s``
    AND the buffer is already >= ``min_dur_s``.  The trailing buffer is always
    emitted (even if shorter than ``min_dur_s``).

    Each returned window dict has:
        source_id, seq_start, seq_end, start_s, end_s, text, speaker, win_id

    ``win_id`` is set to ``f"{source_id}_w{seq_start:04d}"`` by this function.
    ``speaker`` is the dominant speaker over all segments in the window
    (``collections.Counter``); ``None`` speaker values are coerced to ``""``.

    Returns an empty list for empty input.
    """
    if not segments:
        return []

    windows: list[dict] = []
    buf: list[dict] = []

    def _flush(buf: list[dict]) -> dict:
        texts = " ".join((s.get("text") or "").strip() for s in buf).strip()
        speakers = [s.get("speaker") or "" for s in buf]
        dominant_speaker = Counter(speakers).most_common(1)[0][0]
        source_id = buf[0].get("source_id", "")
        seq_start = int(buf[0].get("seq", 0))
        seq_end = int(buf[-1].get("seq", seq_start))
        win_id = f"{source_id}_w{seq_start:04d}"
        return {
            "source_id": source_id,
            "win_id": win_id,
            "seq_start": seq_start,
            "seq_end": seq_end,
            "start_s": float(buf[0].get("start_s", 0.0)),
            "end_s": float(buf[-1].get("end_s", 0.0)),
            "text": texts,
            "speaker": dominant_speaker,
        }

    for seg in segments:
        if not buf:
            buf.append(seg)
            continue

        buf_dur = float(buf[-1].get("end_s", 0.0)) - float(buf[0].get("start_s", 0.0))
        new_end = float(seg.get("end_s", 0.0))
        new_dur = new_end - float(buf[0].get("start_s", 0.0))

        # Flush when adding this segment would exceed max_dur AND buffer is
        # already at or beyond min_dur.
        if new_dur > max_dur_s and buf_dur >= min_dur_s:
            windows.append(_flush(buf))
            buf = [seg]
        else:
            buf.append(seg)

    if buf:
        windows.append(_flush(buf))

    return windows


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


def upsert_windows(source_id: str, windows: list[dict], org: str = "mainstay") -> None:
    """Upsert a list of window dicts into the shared collection.

    Coerces ``speaker`` to ``""`` when ``None`` — ChromaDB metadata does not
    accept ``None`` values.

    Args:
        source_id: The source this window set belongs to.
        windows:   Window dicts produced by :func:`build_windows`.
        org:       Organisation slug stored as ``org_id`` metadata; used by
                   :func:`search_segments` to scope queries to one tenant.
    """
    if not windows:
        return
    col = _get_collection()
    col.upsert(
        ids=[w["win_id"] for w in windows],
        documents=[w["text"] for w in windows],
        metadatas=[
            {
                "source_id": source_id,
                "org_id": org,
                "start_s": float(w["start_s"]),
                "end_s": float(w["end_s"]),
                "speaker": w.get("speaker") or "",
                "seq_start": int(w["seq_start"]),
                "seq_end": int(w["seq_end"]),
            }
            for w in windows
        ],
    )


# ---------------------------------------------------------------------------
# High-level embed entry point
# ---------------------------------------------------------------------------


def embed_source_windows(source_id: str) -> int:
    """Build windows for *source_id*, delete old windows, upsert fresh ones.

    Loads segments via ``ingest.get_segments`` (lazy import to avoid circular
    dependency).  Returns the number of windows upserted (0 if no segments).

    Delete-before-upsert ensures no orphan win_ids survive a re-embed.

    The source's ``org_id`` is looked up from the DB and forwarded to
    :func:`upsert_windows` so embedded windows carry the correct tenant tag.
    """
    # Lazy import to break circular: ingest imports nothing from search;
    # search imports ingest only inside this function.
    from core.forge import ingest as _ingest  # noqa: PLC0415

    segments = _ingest.get_segments(source_id)
    if not segments:
        logger.warning("embed_source_windows: no segments for source %s", source_id)
        return 0

    # Attach source_id to each segment dict if missing (get_segments doesn't
    # include it in the row because it's implied by the query parameter).
    for seg in segments:
        if "source_id" not in seg:
            seg["source_id"] = source_id

    windows = build_windows(segments)
    if not windows:
        return 0

    col = _get_collection()

    # Delete-before-upsert: removes any windows from a previous windowing run
    # whose win_ids may no longer exist (e.g. after a segment-count change).
    try:
        col.delete(where={"source_id": {"$eq": source_id}})
    except Exception:
        # Collection may be empty on first embed — delete on empty is fine in
        # recent ChromaDB versions but some older builds raise; swallow it.
        logger.debug("embed_source_windows: delete-before-upsert swallowed an exception for %s", source_id)

    # Thread org_id from the source row so windows are correctly tenant-tagged.
    source_row = _ingest.get_source(source_id)
    org = (source_row or {}).get("org_id") or "mainstay"

    upsert_windows(source_id, windows, org=org)
    logger.info(
        "embed_source_windows: %s -> %d windows embedded",
        source_id,
        len(windows),
    )
    return len(windows)


# ---------------------------------------------------------------------------
# has_windows
# ---------------------------------------------------------------------------


def has_windows(source_id: str) -> bool:
    """Return True if *source_id* has at least one embedded window."""
    col = _get_collection()
    res = col.get(where={"source_id": {"$eq": source_id}}, limit=1)
    return len(res["ids"]) > 0


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def search_segments(
    source_id: str | None = None,
    query: str = "",
    top_k: int = 10,
    speaker: str | None = None,
    score_threshold: float = 0.45,
    org: str | None = None,
) -> list[dict]:
    """Return ranked windows matching *query*, optionally scoped by org and/or source.

    Scores are inverted cosine distances in [0, 1] (higher = more relevant).
    Raw ChromaDB distances are NEVER returned to callers.

    Args:
        source_id:       Scope results to this source (optional).
        query:           Natural-language topic string.
        top_k:           Maximum number of results to consider from ChromaDB.
        speaker:         If truthy, restrict to windows dominated by this speaker.
        score_threshold: Discard windows with score < threshold.
        org:             If set, restrict results to this org's windows only.

    Returns:
        List of dicts sorted by score descending::

            {start_s, end_s, text, speaker, score, seq_start, seq_end}

        Empty list if the collection is empty or no windows exceed the threshold.
    """
    col = _get_collection()
    total = col.count()
    if total == 0:
        return []

    # Build org+source where clause first, then layer in speaker if needed.
    # ChromaDB requires $and when combining multiple conditions.
    if source_id and org:
        base_where: dict | None = {"$and": [{"source_id": source_id}, {"org_id": org}]}
    elif source_id:
        base_where = {"source_id": source_id}
    elif org:
        base_where = {"org_id": org}
    else:
        base_where = None

    if speaker:
        speaker_clause: dict = {"speaker": {"$eq": speaker}}
        if base_where is None:
            where: dict | None = speaker_clause
        elif "$and" in base_where:
            # Already a $and list — append speaker clause.
            where = {"$and": base_where["$and"] + [speaker_clause]}
        else:
            where = {"$and": [base_where, speaker_clause]}
    else:
        where = base_where

    n_results = min(top_k, total)
    raw = col.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    results: list[dict] = []
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        # Cosine distance d ∈ [0, 2] → relevance score ∈ [0, 1].
        score = round(1.0 - dist / 2.0, 4)
        if score < score_threshold:
            continue
        results.append(
            {
                "start_s": meta.get("start_s"),
                "end_s": meta.get("end_s"),
                "text": doc,
                "speaker": meta.get("speaker", ""),
                "score": score,
                "seq_start": meta.get("seq_start"),
                "seq_end": meta.get("seq_end"),
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results
