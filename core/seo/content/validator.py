"""Validates LLM-generated content against the brand profile.

Three checks per the spec:
  1. never_say regex scan (word-boundary, case-insensitive)
  2. Flesch reading ease within ±tolerance of brand target
  3. Primary keyword presence in first 100 words

Plus a length sanity check vs the per-content-type target_words
(±50% of target).

Returns ValidationResult — caller decides whether to retry,
fall back to a stricter prompt, or surface in the queue with the
failure reason for human review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.seo.content.types import canonicalize
from core.seo.sites.profile import BrandProfile

_WORD_RE = re.compile(r"\b[\w']+\b")
_SENTENCE_RE = re.compile(r"[.!?]+\s+|[.!?]+$")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


@dataclass
class ValidationResult:
    ok: bool
    flesch: float
    word_count: int
    sentence_count: int
    issues: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover
        return self.ok


def count_syllables(word: str) -> int:
    """Heuristic syllable count. Good enough for Flesch scoring on prose."""
    word = word.lower().strip("'.,;:!?\"")
    if not word:
        return 0
    groups = _VOWEL_GROUP_RE.findall(word)
    n = len(groups)
    # Silent trailing 'e' (but not 'le' which is its own syllable)
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    return max(1, n)


def flesch_reading_ease(text: str) -> tuple[float, int, int]:
    """Return (flesch_score, word_count, sentence_count).

    Flesch Reading Ease = 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    """
    words = _WORD_RE.findall(text)
    word_count = len(words)
    if word_count == 0:
        return (0.0, 0, 0)

    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    sentence_count = max(1, len(sentences))

    syllable_total = sum(count_syllables(w) for w in words)
    if syllable_total == 0:
        return (0.0, word_count, sentence_count)

    score = (
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (syllable_total / word_count)
    )
    return (round(score, 1), word_count, sentence_count)


def _never_say_violations(text: str, never_say: list[str]) -> list[str]:
    """Return list of phrases from never_say that appear in text.

    Single-word entries (e.g. "I", "we", "our") use word-boundary regex so
    we don't false-positive on "I" inside "Italy" or "our" inside "outsourced".
    Multi-word entries (e.g. "handcrafted with love") use case-insensitive
    substring containment, since whole-phrase matches are unambiguous.
    Symbols like "♥" use plain containment.
    """
    hits: list[str] = []
    for phrase in never_say:
        if not phrase:
            continue
        is_alphabetic_single_word = phrase.isalpha() and " " not in phrase
        if is_alphabetic_single_word:
            pat = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            if pat.search(text):
                hits.append(phrase)
        else:
            if phrase.lower() in text.lower():
                hits.append(phrase)
    return hits


def validate_draft(
    text: str,
    profile: BrandProfile,
    *,
    content_type: str,
    target_keyword: str | None = None,
) -> ValidationResult:
    """Run all checks against a draft.

    Args:
        text: the LLM-generated body
        profile: loaded BrandProfile
        content_type: 'blog' | 'cluster' | 'product_enrichment' | 'ad_landing'
        target_keyword: the primary keyword the brief targeted; if not given,
            defaults to first primary keyword on the brand profile.
    """
    issues: list[str] = []

    # 1. never_say
    bad_phrases = _never_say_violations(text, profile.never_say)
    if bad_phrases:
        issues.append(f"never_say violations: {', '.join(bad_phrases)}")

    # 2. Flesch — prefer per-content-type target/tolerance from
    # content_type_preferences, fall back to voice-level defaults.
    ct_for_pref = canonicalize(content_type)
    pref_for_flesch = profile.content_type_preferences.get(ct_for_pref, {}) or {}
    flesch, word_count, sentence_count = flesch_reading_ease(text)
    target = float(pref_for_flesch.get("flesch_target", profile.voice.get("flesch_target", 70)))
    tolerance = float(pref_for_flesch.get("flesch_tolerance", profile.voice.get("flesch_tolerance", 8)))
    if word_count >= 30:  # don't grade tiny snippets
        if abs(flesch - target) > tolerance:
            direction = "too dense" if flesch < target else "too breezy"
            issues.append(
                f"Flesch {flesch:.1f} outside target {target}±{tolerance} ({direction})"
            )

    # 3. Keyword presence in first 100 words. Skipped per content type for
    # types where the keyword is structural (product copy serves the product
    # name, not a head SEO term).
    skip_keyword_check = pref_for_flesch.get("skip_keyword_check", False)

    #
    # Accept either:
    #   (a) the full target keyword phrase verbatim, OR
    #   (b) at least ceil(N * 0.75) significant tokens (>2 chars) of the
    #       keyword phrase appear in the first 100 words, after a light
    #       suffix-strip (-s/-ing/-ed/-es). Catches paraphrased headlines
    #       like "what an evil eye bracelet means" for target keyword
    #       "evil eye bracelet meaning".
    primary = profile.target_keywords.get("primary", [])
    keyword = target_keyword or (primary[0] if primary else None)
    if keyword and not skip_keyword_check:
        first_100_lower = " ".join(_WORD_RE.findall(text)[:100]).lower()
        verbatim_hit = keyword.lower() in first_100_lower
        tokens = [t for t in _WORD_RE.findall(keyword.lower()) if len(t) > 2]

        def _stem(w: str) -> str:
            for suf in ("ings", "ing", "ies", "ied", "ed", "es", "s"):
                if len(w) > len(suf) + 2 and w.endswith(suf):
                    return w[: -len(suf)]
            return w

        first_100_stems = {_stem(t) for t in _WORD_RE.findall(first_100_lower)}
        token_stems = [_stem(t) for t in tokens]
        present = sum(1 for t in token_stems if t in first_100_stems)
        threshold = -(-len(token_stems) * 3 // 4)  # ceil(N * 0.75)
        token_hit = bool(token_stems) and present >= threshold

        if not (verbatim_hit or token_hit):
            issues.append(
                f"primary keyword '{keyword}' (or its tokens) missing from first 100 words"
            )

    # 4. Length sanity vs target. Use canonical content type so brand.yaml
    # files that still say cluster_pages still resolve.
    ct = canonicalize(content_type)
    pref = profile.content_type_preferences.get(ct, {}) or {}
    target_words = pref.get("target_words")
    if target_words:
        # Tighter window for short copy (ad/product), loose for long-form.
        if target_words <= 250:
            lo, hi = int(target_words * 0.5), int(target_words * 1.7)
        else:
            lo, hi = int(target_words * 0.55), int(target_words * 1.5)
        if word_count < lo or word_count > hi:
            issues.append(
                f"length {word_count} words outside {lo}-{hi} window for {ct}"
            )

    return ValidationResult(
        ok=(len(issues) == 0),
        flesch=flesch,
        word_count=word_count,
        sentence_count=sentence_count,
        issues=issues,
    )
