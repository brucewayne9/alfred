"""Pick the best existing URL on a site to target a given keyword.

Strategy (cheap, no embeddings for POC):
1. Token-overlap score between keyword and each URL's (title + slug + excerpt)
2. Tie-break favors product pages > posts > pages (transactional intent default)
3. Returns None if no URL scores above a minimum threshold (signals "create new")
"""
from __future__ import annotations

import re
from typing import Optional

from core.seo.keywords.url_inventory import SiteUrl

WORD_RE = re.compile(r"[a-z0-9]+")

# Stopwords intentionally tiny — keep substantive words doing the matching.
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "with",
    "is", "are", "be", "by", "at", "vs",
}

# Type ordering preference when token-overlap ties.
_TYPE_RANK = {
    "product": 0,
    "post":    1,
    "page":    2,
}

# Minimum overlap required to call it a match. Below this we say "create new".
MIN_OVERLAP_RATIO = 0.34  # ≥1 of 3 tokens, or ≥2 of 5, etc.


def _tokenize(s: str) -> set[str]:
    return {w for w in WORD_RE.findall((s or "").lower()) if w not in _STOPWORDS and len(w) > 1}


def _score_url(keyword_tokens: set[str], url: SiteUrl) -> float:
    haystack = " ".join([url.title or "", url.slug or "", url.excerpt or ""])
    url_tokens = _tokenize(haystack)
    if not url_tokens or not keyword_tokens:
        return 0.0
    overlap = keyword_tokens & url_tokens
    if not overlap:
        return 0.0
    # Recall against the keyword tokens (we want most of the keyword's words to land)
    recall = len(overlap) / len(keyword_tokens)
    # Precision tiny bonus (favors tight pages over kitchen-sink ones)
    precision = len(overlap) / len(url_tokens)
    return recall + 0.1 * precision


def pick_target_url(keyword: str, urls: list[SiteUrl]) -> Optional[str]:
    """Best target URL for a keyword, or None if no good match."""
    kw_tokens = _tokenize(keyword)
    if not kw_tokens:
        return None

    scored: list[tuple[float, int, SiteUrl]] = []
    for u in urls:
        s = _score_url(kw_tokens, u)
        if s <= 0:
            continue
        type_rank = _TYPE_RANK.get(u.post_type, 9)
        scored.append((s, type_rank, u))

    if not scored:
        return None

    scored.sort(key=lambda t: (-t[0], t[1]))  # highest score first, then preferred type
    best_score, _, best_url = scored[0]

    # Require minimum overlap to count as a real target.
    recall_proxy = best_score  # recall dominates; precision contribution is small
    if recall_proxy < MIN_OVERLAP_RATIO:
        return None
    return best_url.url
