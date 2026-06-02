# Phase 11: Topic-targeted Segment Retrieval — Research

**Researched:** 2026-06-01
**Domain:** Semantic vector search over transcript segments — ChromaDB, bge-m3 embeddings, windowing/merging, API shape
**Confidence:** HIGH (all findings grounded in live codebase + runtime-verified)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TOPIC-01 | Enter a topic → ranked list of matching source segments with in/out timestamps + relevance scores | ChromaDB `collection.query()` with `where={"source_id":...}` returns ranked results with cosine distances; window records carry `start_s`/`end_s` |
| TOPIC-02 | Preview transcript text of each matched segment before building anything | Text is stored in ChromaDB `documents` field and in `transcript_segments` table; the search endpoint returns it inline — no second request needed |
| TOPIC-03 | Deselect or trim matched segments before assembly | The search response is a ranked list of window objects; Phase 13 UI holds operator selection state client-side and passes only the kept windows to Phase 12 |
</phase_requirements>

---

## Summary

All infrastructure for semantic segment retrieval already exists in the codebase and is verified live:

- **Corpus:** `transcript_segments` table in `data/forge_live.db` (columns: `source_id, seq, start_s, end_s, text, speaker, words`). `get_segments(source_id)` in `core/forge/ingest.py:213` returns the full ordered list.
- **Embedding model:** `bge-m3:latest` is live on `localhost:11434` (1024-dim, ~0.33s per 50 segments batch). Never delete it (CLAUDE.md rule).
- **Vector store:** `chromadb==1.4.1` with `PersistentClient` at `data/chromadb/` is already wired via `core/memory/store.py:get_collection()`. End-to-end test (upsert + where-filtered query + speaker filter) verified in this research session.
- **API home:** `core/api/forge.py` — add `GET /forge/sources/{source_id}/search?q=...&speaker=A&top_k=10`.

The make-or-break design choice is **windowing before embedding**: raw Whisper `medium` segments are often 2–8s (one sentence) — too short to be meaningful for clip assembly. Merge adjacent segments into ~20–40s windows before embedding. Each window is one ChromaDB document. The operator sees windows, not raw segments — a match is always a usable span.

**Primary recommendation:** Embed windows (not raw segments) at ingest completion time as an async post-step. Search is then instant: embed the query string, call `collection.query()` with `where=source_id`, return top-k windows with scores.

---

## 1. Corpus: What Phase 10 Built

### `transcript_segments` table (db.py:73-83)
```sql
CREATE TABLE IF NOT EXISTS transcript_segments (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    seq       INTEGER NOT NULL,   -- 0-based, ordered
    start_s   REAL NOT NULL,
    end_s     REAL NOT NULL,
    text      TEXT NOT NULL,
    speaker   TEXT,               -- 'A' | 'B' | NULL
    words     TEXT                -- JSON: [{"word","start","end"}, ...]
);
```

### `get_segments()` callable (ingest.py:213)
```python
def get_segments(source_id: str) -> list[dict]:
    """Return all transcript segments ordered by seq. Words JSON-decoded."""
```
Returns dicts with keys: `seq, start_s, end_s, text, speaker, words`.

### Segment granularity reality
Whisper `medium` with `vad_filter=True` produces sentences/phrases — typically 2–8s each, occasionally up to 15s for long utterances. At 5s average, a 90-minute interview yields ~1080 raw segments. These are **too granular to embed individually** for topic retrieval: a single sentence rarely captures a complete topic; similarity scores will be noisy; and a 3s match is not a usable clip span.

---

## 2. Embedding Callable: bge-m3 via Ollama

### What's live
`bge-m3:latest` is running on `localhost:11434`. Verified 2026-06-01:
- Endpoint: `POST http://localhost:11434/api/embed` with `{"model":"bge-m3:latest","input":[...list...]}`
- Response: `{"embeddings": [[float, ...], ...]}`
- Dimensions: **1024**
- Batch speed: 50 segments in **~0.33s** (fast enough that 240 windows embed in ~1.6s)

### ChromaDB's `OllamaEmbeddingFunction`
```python
# core/memory/store.py uses chromadb.PersistentClient already
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# Signature (verified against chromadb==1.4.1 source):
OllamaEmbeddingFunction(
    url='http://localhost:11434',   # BASE URL, not /api/embeddings
    model_name='bge-m3:latest',
    timeout=60,
)
```
This is the preferred wrapper — pass it to `get_or_create_collection()` and ChromaDB handles calling `/api/embeddings` automatically on `add()`/`query(query_texts=...)`.

**Alternative (manual):** Call `/api/embed` directly for batch upsert (faster control) and pass `embeddings=[...]` to `collection.upsert()` instead of `documents=[...]`. Both paths work; `OllamaEmbeddingFunction` is cleaner for the query side.

### Rule: do NOT delete bge-* models
CLAUDE.md explicitly prohibits deleting `bge-*`, `nomic-*`, `*-embed-*` from Ollama. `bge-m3` is also used by Grey Matter (LightRAG on 117). Treat it as always-present.

---

## 3. ChromaDB Upsert/Query Pattern

### Collection per source vs shared collection
Use a **single shared collection** named `forge_segments` with `source_id` in metadata. Reason: creating a collection per source would result in hundreds of tiny collections that fragment the HNSW index; ChromaDB's `where` filter handles source scoping efficiently at any realistic scale (thousands of windows per source, dozens of sources).

### Collection creation (once, at startup or first use)
```python
# In a new core/forge/search.py
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from core.memory.store import get_client

_COLLECTION_NAME = "forge_segments"

def _get_collection():
    ef = OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name="bge-m3:latest",
        timeout=120,
    )
    return get_client().get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef,
    )
```

### Upsert (at embed time)
```python
def upsert_windows(source_id: str, windows: list[dict]) -> None:
    """Embed and store transcript windows for a source.

    Each window dict: {win_id, source_id, start_s, end_s, text, speaker, seq_start, seq_end}
    """
    col = _get_collection()
    col.upsert(
        ids=[w["win_id"] for w in windows],
        documents=[w["text"] for w in windows],       # bge-m3 embeds these
        metadatas=[{
            "source_id": w["source_id"],
            "start_s":   w["start_s"],
            "end_s":     w["end_s"],
            "speaker":   w.get("speaker") or "",      # chroma requires str, not None
            "seq_start": w["seq_start"],
            "seq_end":   w["seq_end"],
        } for w in windows],
    )
```
`win_id` format: `"{source_id}_w{seq_start:04d}"` — stable across re-embeds (idempotent upsert).

### Query (at search time)
```python
def search_segments(
    source_id: str,
    query: str,
    top_k: int = 10,
    speaker: str | None = None,
    score_threshold: float = 0.5,
) -> list[dict]:
    """Return ranked windows matching query for a source.

    cosine distance 0 = identical, 2 = opposite. We convert to score = 1 - dist/2.
    Threshold of 0.5 = cosine distance <= 1.0 (i.e. not purely opposite).
    """
    col = _get_collection()
    count = col.count()
    if count == 0:
        return []

    where: dict = {"source_id": {"$eq": source_id}}
    if speaker:
        where = {"$and": [{"source_id": {"$eq": source_id}}, {"speaker": {"$eq": speaker}}]}

    res = col.query(
        query_texts=[query],
        n_results=min(top_k, count),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for i, doc in enumerate(res["documents"][0]):
        dist = res["distances"][0][i]
        score = round(1.0 - dist / 2.0, 4)   # cosine: dist in [0,2], score in [0,1]
        if score < score_threshold:
            continue
        meta = res["metadatas"][0][i]
        results.append({
            "start_s":   meta["start_s"],
            "end_s":     meta["end_s"],
            "text":      doc,
            "speaker":   meta["speaker"],
            "score":     score,
            "seq_start": meta["seq_start"],
            "seq_end":   meta["seq_end"],
        })

    return sorted(results, key=lambda r: r["score"], reverse=True)
```

**Verified behavior (2026-06-01):**
- `where={"source_id": {"$eq": source_id}}` scopes to one source — works
- `where={"$and": [...]}` compound filter (source + speaker) — works
- `n_results > col.count()` — ChromaDB 1.4.1 auto-caps, no error
- Cosine distance 0.31 for Broadway query on Broadway document (good separation from 0.63 for off-topic)

---

## 4. Windowing / Merging Design — The Precision Crux

### Why windowing is mandatory
A raw Whisper segment for "he talked about Broadway" might be 4 seconds of transcript text. Embedding it in isolation gives poor recall (the embedding captures only one sentence's context) and poor usability (a 4s clip is not a usable output for Phase 12). Target windows of **20–40 seconds** of continuous speech.

### Windowing algorithm
```python
def build_windows(
    segments: list[dict],
    target_dur_s: float = 30.0,
    max_dur_s: float = 45.0,
    min_dur_s: float = 10.0,
) -> list[dict]:
    """Merge adjacent segments into topic-coherent windows.

    Strategy:
    - Grow a window by accumulating adjacent segments until adding the next
      segment would exceed max_dur_s OR a speaker change occurs (optional break).
    - If window duration < min_dur_s, absorb across the speaker change anyway.
    - Emit the window when it crosses target_dur_s or hits max_dur_s.

    Each window = one ChromaDB document. Text is the concatenated segment texts.
    """
    if not segments:
        return []

    windows = []
    buf_segs = [segments[0]]

    for seg in segments[1:]:
        buf_dur = buf_segs[-1]["end_s"] - buf_segs[0]["start_s"]
        next_dur = seg["end_s"] - buf_segs[0]["start_s"]

        # Flush if adding this segment exceeds max_dur
        if next_dur > max_dur_s and buf_dur >= min_dur_s:
            windows.append(_make_window(buf_segs))
            buf_segs = [seg]
        else:
            buf_segs.append(seg)

    if buf_segs:
        windows.append(_make_window(buf_segs))

    return windows


def _make_window(segs: list[dict]) -> dict:
    # Dominant speaker = most segments' speaker label
    from collections import Counter
    speakers = [s.get("speaker") or "" for s in segs]
    dominant = Counter(speakers).most_common(1)[0][0]
    return {
        "source_id":  segs[0]["source_id"] if "source_id" in segs[0] else "",
        "seq_start":  segs[0]["seq"],
        "seq_end":    segs[-1]["seq"],
        "start_s":    segs[0]["start_s"],
        "end_s":      segs[-1]["end_s"],
        "text":       " ".join(s["text"].strip() for s in segs),
        "speaker":    dominant,
        "win_id":     "",  # filled by caller: f"{source_id}_w{seq_start:04d}"
    }
```

**Window count for a 90-min interview:** ~1080 raw segments at 5s avg → ~135 windows at 40s each. Embedding 135 windows: ~0.9s. Perfectly tractable.

**Speaker in window metadata:** Set to the dominant speaker in the window's segments. This makes `speaker="A"` filter meaningful even though windows span multiple segments — it filters to windows where speaker A is the primary voice.

### Do not split on speaker boundaries by default
For interviews, a 5-word interjection ("yeah, exactly, right") causes a speaker flip but the topic hasn't changed. The min_dur_s guard absorbs these naturally. Only flush on speaker change if the buffer is already ≥ target_dur_s.

---

## 5. When to Embed: The Trigger Decision

### Recommendation: embed as a post-step immediately after transcription completes

**Option A — extend the `ingest_transcribe` job handler** (ingest.py:249 `transcribe_handler`): After `save_segments()` completes, call `embed_source_windows(source_id)`. The transcription handler already has the segment list in memory.

**Option B — separate `embed_segments` job type**: Enqueue after transcription completes. Keeps concerns separate but adds a job to the queue.

**Option C — on first search (lazy)**: Embed when the operator first queries the source. Zero overhead if the source is never searched, but adds 1–2s latency to the first search.

**Recommend Option A** for v1: the embed step is fast (~1.6s for 240 windows), has no failure mode independent of transcription, and means the source is search-ready as soon as `status=done`. The handler just calls a `search.embed_source_windows(source_id)` function after `save_segments()`.

**Backfill:** For sources transcribed before Phase 11 ships, a backfill endpoint or script calls `embed_source_windows(source_id)` for all `status=done` sources. Given zero completed transcriptions in the live DB right now (the one existing source is `status=error`), there is nothing to backfill at ship time.

**Re-embed on retry:** `upsert_windows()` is idempotent (same `win_id` keys). Re-embedding a source replaces its windows cleanly.

---

## 6. API Shape

### New endpoint: `GET /forge/sources/{source_id}/search`

Add to `core/api/forge.py` inside `register(app)`:

```python
@app.get("/forge/sources/{source_id}/search")
async def search_source_segments(
    source_id: str,
    q: str = Query(..., description="Topic or theme to search for"),
    top_k: int = Query(default=10, ge=1, le=50),
    speaker: str | None = Query(default=None, description="'A' or 'B' to filter by speaker"),
    threshold: float = Query(default=0.45, ge=0.0, le=1.0),
    user: dict = Depends(require_auth),
):
    """Return ranked windows of the source that match the topic query.

    Response: {"source_id": ..., "query": ..., "results": [{
        "start_s", "end_s", "text", "speaker", "score",
        "seq_start", "seq_end"
    }]}
    """
    from core.forge import ingest, search as forge_search
    source = ingest.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="source not found")
    if source["status"] != "done":
        raise HTTPException(status_code=409, detail=f"source not ready: {source['status']}")

    # Lazy backfill: embed on first search if windows not yet in ChromaDB
    if not forge_search.has_windows(source_id):
        forge_search.embed_source_windows(source_id)

    results = forge_search.search_segments(
        source_id=source_id,
        query=q,
        top_k=top_k,
        speaker=speaker or None,
        score_threshold=threshold,
    )
    return {"source_id": source_id, "query": q, "results": results}
```

**Response object per result:**
```json
{
  "start_s": 142.5,
  "end_s": 178.3,
  "text": "Yeah Broadway man that's where I really grew up artistically...",
  "speaker": "A",
  "score": 0.74,
  "seq_start": 28,
  "seq_end": 35
}
```

`score` is in [0.0, 1.0] where 1.0 = perfect match. The operator sees this as a relevance percentage in the UI.

### Supporting function: `has_windows(source_id)`
```python
def has_windows(source_id: str) -> bool:
    """True if source has at least one window in ChromaDB."""
    col = _get_collection()
    res = col.get(where={"source_id": {"$eq": source_id}}, limit=1)
    return len(res["ids"]) > 0
```

### No new job type needed
Search is synchronous and fast (~0.1s for query embed + ChromaDB lookup). No queue involvement. The lazy embed step above is also fast enough to run inline (1–2s on first search).

---

## 7. Precision Strategy

### Score threshold
The cosine distance range verified in testing:
- On-topic match: distance ~0.31, score ~0.845
- Off-topic:     distance ~0.63, score ~0.685

A threshold of **0.45** (cosine distance ≤ 1.10) is permissive but filters clearly-unrelated segments. Start here; tighten to 0.55 if the operator reports noise.

The operator-readable transcript text in the results (TOPIC-02) is the primary precision safety net — the operator reads each match and deselects bad ones (TOPIC-03). Do not over-tune the threshold to the point of dropping genuine but weakly-phrased matches.

### top_k default
Return **10 results** by default. A 90-min interview has ~135 windows; 10 is ~7% of the corpus. Enough to find the good moments without overwhelming the operator. Expose `top_k` as a query param so the operator can widen to 20 if the first pass is too narrow.

### Speaker filter
Pass `speaker=A` or `speaker=B` in the query. ChromaDB `$and` filter verified. This maps to the `speaker` metadata field on windows (dominant speaker in the window's segments). The UI should let the operator pick "Filter by speaker" with A/B labels (or ideally map those labels to the actual speaker name if the operator has identified them — out of scope for Phase 11 but the metadata is ready for Phase 13 to surface).

### Operator preview as safety net (TOPIC-02)
The response always includes the full `text` of each matching window. The UI renders this as a collapsible transcript excerpt below each result card. The operator can read it before confirming. This means precision thresholds can be loose — a false positive is caught at glance, not at clip-cut time.

### Query phrasing matters
`bge-m3` is a dense retrieval model — it handles paraphrases well but does NOT do keyword exact-match. "the part where he talks about Broadway" works; so does "Broadway theater". Do not tell the operator they need precise phrasing. The model handles natural language.

---

## 8. New Files to Create

```
core/forge/
└── search.py          # NEW: build_windows(), embed_source_windows(), search_segments(),
                       #      has_windows(), upsert_windows(), _get_collection()

core/api/forge.py      # MODIFY: add GET /forge/sources/{source_id}/search
core/forge/ingest.py   # MODIFY: call search.embed_source_windows(source_id) after
                       #   save_segments() in transcribe_handler (line ~461)
```

No new dependencies. `chromadb`, `OllamaEmbeddingFunction`, `requests`, `collections.Counter` are all already available.

---

## 9. Common Pitfalls (Precision Failure Modes)

### Pitfall 1: Embedding raw segments instead of windows
**What goes wrong:** A 3s segment "Broadway, man" embeds poorly — one phrase has no surrounding context. Retrieval returns a disconnected 3s span that is useless for Phase 12 and misleads the operator.
**How to avoid:** Always window first. Minimum window = 10s; target = 30s.

### Pitfall 2: `speaker=None` stored as None in ChromaDB metadata
**What goes wrong:** ChromaDB metadata values must be strings, ints, or floats — `None` raises a validation error silently or causes incorrect `where` filter matches.
**How to avoid:** Coerce `speaker` to `""` (empty string) when storing. The speaker filter uses `speaker="A"` or `speaker="B"` explicitly; empty string never matches either.

### Pitfall 3: `n_results` > collection size with a `where` filter
**What goes wrong:** ChromaDB 1.4.1 auto-caps `n_results` to the total collection count (not the filtered subset count). If a source has 30 windows but `n_results=50` and the collection has 100 total docs, you get at most 50 results but only 30 will have `source_id=this_source`. No error, but the extra 20 results are from other sources — they pass the `where` filter so this doesn't actually happen, but if the filter is wrong the bleed-through is silent.
**How to avoid:** Use the correct `where={"source_id": {"$eq": source_id}}` filter. Verify the filter syntax matches the verified pattern above.

### Pitfall 4: Re-embedding on every transcribe retry doubles windows
**What goes wrong:** If `embed_source_windows` is called twice for the same source (e.g., restart + re-enqueue), `upsert` with the same `win_id` keys is idempotent — no duplication. But if the windowing changes (different segment list), old `win_id`s from a prior windowing that no longer exist will remain as orphans in ChromaDB.
**How to avoid:** Before upserting, delete all existing windows for the source: `col.delete(where={"source_id": {"$eq": source_id}})`. Then upsert fresh. This is safe because upsert always follows transcription completion.

### Pitfall 5: Query embedding calls a :cloud model accidentally
**What goes wrong:** CLAUDE.md says new app builds should default to `:cloud` models, but embedding is a special case — bge-m3 is local, fast, and must not be replaced with a cloud LLM. `OllamaEmbeddingFunction` hardcodes `bge-m3:latest`; as long as the collection is created with this function, queries use the same model automatically.
**How to avoid:** The `_get_collection()` function in `search.py` pins `model_name='bge-m3:latest'`. Never pass `query_texts` to a collection that was created with a different embedding function.

### Pitfall 6: ChromaDB `where` on a collection with 0 documents
**What goes wrong:** `col.query(where=..., n_results=k)` on an empty collection raises an error.
**How to avoid:** `has_windows()` guard before every query. The search endpoint already checks `source["status"] == "done"` which implies transcription completed, but the `has_windows()` lazy-backfill path handles the edge case.

### Pitfall 7: Cosine distance vs cosine similarity confusion
**What goes wrong:** ChromaDB `hnsw:space=cosine` returns `distances` in [0, 2] where 0 = identical. If you present `distance` directly as a "score" the operator sees 0 = best, which is confusing and inverted.
**How to avoid:** Convert: `score = 1.0 - distance / 2.0`. Score 1.0 = perfect match, 0.0 = perfectly opposite, 0.5 = orthogonal. Return `score` to the API; the UI displays it as a percentage.

---

## 10. Architecture: What Phase 10 Left Ready

The `transcribe_handler` in `ingest.py:249` ends at line 476 with:
```python
save_segments(source_id, labelled_clean)
update_source(source_id, status="done", ...)
return {"transcript_id": source_id, "segment_count": len(labelled_clean), "resumed": False}
```

Phase 11 adds one line between `save_segments()` and `update_source()`:
```python
# ingest.py — after save_segments(), before update_source(status="done")
from core.forge import search as _search
_search.embed_source_windows(source_id)   # ~1.6s for a 90-min interview; fast enough to inline
```

This keeps the "source is search-ready immediately when status=done" guarantee without a separate job.

---

## 11. RuckTalk Reuse Check

RuckTalk has no semantic transcript search. The pipeline (`scripts/rucktalk_episode_pipeline.py`) uses transcript segments only for SRT/caption generation via regex range checks on `start`/`end` timestamps (lines 846–848, 1011–1016). No vector store, no embeddings, no topic retrieval. Nothing to lift.

The RuckTalk episode pipeline does confirm the Whisper segment dict shape: `{"start": float, "end": float, "text": str}` — which is the same shape Phase 10 stores in `transcript_segments` (as `start_s`, `end_s`, `text`).

---

## Sources

### Primary (HIGH confidence — live codebase + runtime verified)
- `core/forge/ingest.py` — `get_segments()` at line 213, `transcribe_handler()` at line 249, `save_segments()` at line 85
- `core/forge/db.py:73-83` — `transcript_segments` table schema (confirmed live)
- `core/memory/store.py` — `get_client()` (`PersistentClient` at `data/chromadb/`), `get_collection()` pattern
- `core/api/forge.py` — existing endpoints, `require_auth` dependency pattern
- `core/forge/handlers.py:105-111` — handler registration pattern
- Runtime verification (2026-06-01):
  - `bge-m3:latest` live at `localhost:11434`, 1024-dim, 50-segment batch in 0.33s
  - `chromadb==1.4.1` in venv; `OllamaEmbeddingFunction(url='http://localhost:11434', model_name='bge-m3:latest')` works
  - `collection.upsert()` + `collection.query(where={"$and":[...]})` with source_id + speaker filter — verified end-to-end
  - Cosine distances: on-topic 0.31 (score 0.845), off-topic 0.63 (score 0.685)
  - `n_results` > count auto-capped by ChromaDB 1.4.1, no error

### Secondary (MEDIUM confidence — verified installs + docs)
- `chromadb.utils.embedding_functions.OllamaEmbeddingFunction` — signature `(url, model_name, timeout)` verified against installed package
- `OllamaEmbeddingFunction` calls `/api/embeddings` internally (not `/api/embed`) — both endpoints respond correctly

---

## Metadata

**Confidence breakdown:**
- Transcript corpus shape: HIGH — read live source, verified DB schema
- bge-m3 embedding: HIGH — live API test, confirmed 1024-dim, batch speed measured
- ChromaDB upsert/query/filter: HIGH — end-to-end test in this research session
- Windowing design: MEDIUM-HIGH — algorithm is standard sliding-window, parameters (20–40s) based on whisper segment behavior knowledge; exact values should be tunable at query time
- Precision thresholds: MEDIUM — cosine distances verified on two-doc test; real-world threshold needs one real interview to validate

**Research date:** 2026-06-01
**Valid until:** 2026-09-01 (chromadb and bge-m3 are stable; OllamaEmbeddingFunction API is unlikely to change)
