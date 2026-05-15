"""LLM writer for the SEO content engine.

Builds prompts in three parts:
  [system]   Brand voice constraints (perspective, tone, never_say, always)
  [examples] Few-shot voice samples (3-4 from the site's voice_examples_dir)
  [user]     Brief: write {content_type} on {topic} targeting {keyword}

Calls Ollama at 105:11434 with kimi-k2.6:cloud for long-form content,
gemma4:31b-cloud for short structured outputs.

Reads BOTH message.content AND message.reasoning per
[[feedback_kimi_reasoning_field_quirk]] — Kimi reasoning models on Ollama
return their output in `message.reasoning`, not `message.content`, and a
naive proxy that reads only content gets empty strings silently.

Retry-on-validation-fail: caller (queue.publisher or smoke_test) decides the
loop. Writer's job is single-shot generation + telemetry.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from core.seo.content.types import canonicalize
from core.seo.content.validator import ValidationResult, validate_draft
from core.seo.sites.profile import BrandProfile, VoiceExample, sample_voice_examples

log = logging.getLogger(__name__)

OLLAMA_URL = "http://75.43.156.105:11434/v1/chat/completions"
DEFAULT_MODEL_LONG = "kimi-k2.6:cloud"
DEFAULT_MODEL_SHORT = "gemma4:31b-cloud"
DEFAULT_TIMEOUT = 180  # kimi cloud reasoning runs long for cluster pages

# Map (canonical) content type → preferred model
MODEL_BY_TYPE: dict[str, str] = {
    "blog": DEFAULT_MODEL_LONG,
    "cluster": DEFAULT_MODEL_LONG,
    "product_enrichment": DEFAULT_MODEL_LONG,
    "ad_landing": DEFAULT_MODEL_SHORT,
}

# Per-type structural guidance the writer adds to the system prompt. Keep terse
# — Kimi follows section lists very literally and over-prescription kills voice.
PROMPT_HINTS: dict[str, str] = {
    "blog": (
        "This is a blog post. Open with a concrete observation, not a thesis. "
        "Body is 5-7 short paragraphs, ~600 words total. No list-of-tips "
        "structure. Close with one line that points the reader back to the "
        "studio or a specific piece."
    ),
    "cluster": (
        "This is a topic cluster page (~800 words) meant to rank for a head-term "
        "and educate. Mix narrative paragraphs with two or three short H2 "
        "sections covering meaning/origin, style, and care. Avoid 'ultimate "
        "guide' phrasing. Aim for an intelligent friend explaining the topic, "
        "not a textbook. Keep sentences short — under 18 words on average."
    ),
    "product_enrichment": (
        "This is product enrichment copy that will be appended to a WooCommerce "
        "product page. DO NOT include an H1. Structure: (1) opening story "
        "paragraph 3-4 sentences about the piece, (2) '## Materials & care' "
        "with 3-4 sentences, (3) '## Pairs well with' with 3-4 short bullets. "
        "Total length ~180 words. Tactile, specific, no marketing puffery."
    ),
    "ad_landing": (
        "This is an ad landing page (paid traffic from IG/Pinterest), ~300 "
        "words. Open with the offer or hook in the first sentence. Body is 4-5 "
        "short paragraphs covering: what the piece is, why now, what it's like "
        "to wear, what to do next. Voice stays calm — no caps, no exclamation "
        "marks, no urgency hype unless it's literally true."
    ),
}

_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^```[a-z]*\n|\n```$", re.MULTILINE)


@dataclass
class Brief:
    """A single content brief — input to the writer."""
    topic: str
    content_type: str          # blog | cluster | product_enrichment | ad_landing
    target_keyword: str
    audience: Optional[str] = None
    title_hint: Optional[str] = None
    extra_keywords: list[str] = field(default_factory=list)
    source_signal: Optional[str] = None  # why this brief exists (GSC gap, manual, etc)


@dataclass
class GeneratedDraft:
    title: str
    body: str
    model: str
    content_type: str
    target_keyword: str
    latency_s: float
    raw_response: str
    validation: Optional[ValidationResult] = None


def _format_examples_block(examples: list[VoiceExample]) -> str:
    if not examples:
        return "(no voice examples available)"
    parts: list[str] = []
    for i, e in enumerate(examples, start=1):
        label = e.example_type.replace("_", " ")
        parts.append(f"--- Example {i} ({label}) ---\n{e.body}\n")
    return "\n".join(parts)


def build_system_prompt(profile: BrandProfile, content_type: str) -> str:
    voice = profile.voice
    perspective = voice.get("perspective", "third_person").replace("_", " ")
    descriptors = ", ".join(voice.get("descriptors", []) or [])
    tagline = voice.get("tagline", "")

    ct = canonicalize(content_type)
    pref = profile.content_type_preferences.get(ct, {}) or {}
    target_words = pref.get("target_words", 500)
    sections = pref.get("sections", []) or []

    never_say_block = "\n".join(f"  - {p}" for p in profile.never_say) or "  (none)"
    always_block = "\n".join(f"  - {p}" for p in profile.always) or "  (none)"
    locked_block = "\n".join(f"  - {p}" for p in profile.locked_decisions) or "  (none)"
    sections_block = ", ".join(sections) if sections else "(no constraint)"

    type_hint = PROMPT_HINTS.get(ct, "")

    # Product enrichment doesn't use an H1 (it's appended to a WC product page).
    output_format_block = (
        "OUTPUT FORMAT\n"
        "- Do NOT include an H1 — this copy will be appended to an existing product page.\n"
        "- Start with the story paragraph directly. Use H2 (`## Heading`) for the "
        "Materials & care and Pairs-well-with sections.\n"
        "- Plain Markdown. No emojis. No code fences."
    ) if ct == "product_enrichment" else (
        "OUTPUT FORMAT\n"
        "- Start with a single H1 heading on its own line: `# Your Title Here`\n"
        "- Body follows, in plain Markdown. Short paragraphs. No bullet lists unless the content type explicitly calls for them. No emojis.\n"
        "- Do not wrap your response in code fences. Output the heading + body directly."
    )

    return f"""You are writing for {profile.display_name} ({profile.domain}).

Brand: {profile.brand_one_liner}
Tagline: {tagline}

VOICE
- Perspective: {perspective}
- Descriptors: {descriptors}
- Reading level target: Flesch reading ease ~{voice.get("flesch_target", 70)} (clean, accessible prose)
- Tone energy: {voice.get("energy", "low")}

AUDIENCE
{profile.target_audience}

CONTENT TYPE: {ct}
- Aim for ~{target_words} words.
- Section types you may draw from: {sections_block}.
{type_hint}

NEVER SAY (these phrases will fail validation and be sent back):
{never_say_block}

ALWAYS:
{always_block}

LOCKED BRAND DECISIONS:
{locked_block}

{output_format_block}
"""


def build_user_prompt(brief: Brief, profile: BrandProfile) -> str:
    extra = ""
    if brief.extra_keywords:
        extra = f"\nSecondary keywords to weave in naturally: {', '.join(brief.extra_keywords)}"
    title_hint = f"\nTitle direction (you can refine): {brief.title_hint}" if brief.title_hint else ""
    return f"""Write a {brief.content_type} for {profile.display_name} on this topic:

TOPIC: {brief.topic}

PRIMARY KEYWORD (must appear in the first 100 words): {brief.target_keyword}{extra}{title_hint}

Match the voice of the examples above precisely. Do not break any NEVER SAY rule.
Do not use any first-person pronouns. Do not name the founder.
"""


def _split_title_and_body(text: str) -> tuple[str, str]:
    """Extract H1 title; return (title, body_without_h1)."""
    text = _CODE_FENCE_RE.sub("", text).strip()
    m = _HEADING_RE.search(text)
    if not m:
        # No H1 — synthesize a short title from the first line
        first_line = text.split("\n", 1)[0].strip()
        title = first_line[:80] if first_line else "Untitled"
        body = text[len(first_line):].lstrip("\n")
        return title, body
    title = m.group(1).strip()
    body = (text[: m.start()] + text[m.end():]).strip()
    return title, body


def generate(
    brief: Brief,
    profile: BrandProfile,
    *,
    model: Optional[str] = None,
    temperature: float = 0.6,
    timeout: int = DEFAULT_TIMEOUT,
    examples_override: Optional[list[VoiceExample]] = None,
) -> GeneratedDraft:
    """Single-shot LLM call. Returns a GeneratedDraft (no validation run here)."""
    ct = canonicalize(brief.content_type)
    chosen_model = model or MODEL_BY_TYPE.get(ct, DEFAULT_MODEL_LONG)

    # Bias the few-shot pool toward the matching content type when we have
    # examples of that type on disk.
    type_to_example_kind = {
        "blog": "blog_post",
        "cluster": "cluster_page_intro",
        "product_enrichment": "product_description",
        "ad_landing": "product_description",
    }
    prefer = type_to_example_kind.get(ct)
    examples = examples_override if examples_override is not None else sample_voice_examples(
        profile, prefer_type=prefer
    )

    system_prompt = build_system_prompt(profile, brief.content_type)
    examples_block = _format_examples_block(examples)
    user_prompt = build_user_prompt(brief, profile)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Voice reference (study before writing):\n\n{examples_block}\n\n---\n\n{user_prompt}"},
    ]

    log.info(
        "writer: model=%s type=%s topic=%r kw=%r examples=%d",
        chosen_model, brief.content_type, brief.topic[:60], brief.target_keyword, len(examples),
    )

    start = time.monotonic()
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": chosen_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    latency = time.monotonic() - start

    msg = (data.get("choices") or [{}])[0].get("message", {})
    # Kimi reasoning quirk: read both fields, prefer content if present.
    raw = (msg.get("content") or msg.get("reasoning") or "").strip()
    if not raw:
        raise RuntimeError(
            f"writer: empty content+reasoning from {chosen_model} (response keys={list(data.keys())})"
        )

    title, body = _split_title_and_body(raw)

    return GeneratedDraft(
        title=title,
        body=body,
        model=chosen_model,
        content_type=brief.content_type,
        target_keyword=brief.target_keyword,
        latency_s=round(latency, 2),
        raw_response=raw,
    )


def generate_with_retry(
    brief: Brief,
    profile: BrandProfile,
    *,
    max_retries: int = 1,
    fallback_model: Optional[str] = None,
    **gen_kwargs,
) -> GeneratedDraft:
    """Generate, validate, retry once with stricter prompt on failure.

    For longform types (blog/cluster/product_enrichment), retry stays on
    Kimi — Gemma is too short and too dense for these. For ad_landing,
    fallback_model defaults to Gemma since ad copy is short structured.

    Returns the LAST attempt's GeneratedDraft with .validation populated, even
    if validation never passed — caller decides whether to enqueue with
    needs_manual_review or hard-reject.
    """
    ct = canonicalize(brief.content_type)
    if fallback_model is None:
        fallback_model = DEFAULT_MODEL_SHORT if ct == "ad_landing" else DEFAULT_MODEL_LONG

    last: Optional[GeneratedDraft] = None
    for attempt in range(max_retries + 1):
        kwargs = dict(gen_kwargs)
        if attempt > 0:
            # Strict retry: lower temperature + (maybe) different model.
            kwargs["temperature"] = 0.3
            kwargs.setdefault("model", fallback_model)
        draft = generate(brief, profile, **kwargs)
        # Validate on the body (title is excluded from word window deliberately —
        # an H1 like "What an Evil Eye Bracelet Means" doesn't count toward
        # the body's first-100-words keyword check; the LLM should weave the
        # keyword into the opening paragraph too).
        # Keep it inclusive: pass title + body so the keyword check is lenient.
        full = f"{draft.title}\n\n{draft.body}"
        draft.validation = validate_draft(
            full,
            profile,
            content_type=brief.content_type,
            target_keyword=brief.target_keyword,
        )
        last = draft
        if draft.validation.ok:
            return draft
        log.warning(
            "writer attempt %d/%d failed validation: %s",
            attempt + 1, max_retries + 1, draft.validation.issues,
        )
    assert last is not None
    return last
