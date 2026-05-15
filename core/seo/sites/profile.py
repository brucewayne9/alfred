"""Brand profile loader for the SEO content engine.

Each site has:
  - data/seo/sites/<slug>/brand.yaml          (single YAML, voice + audience + keywords + never_say)
  - data/seo/sites/<slug>/voice_examples/*.md (5-10 in-voice pieces, frontmatter optional)

The writer pipeline calls load_profile(slug) once per generation, then
sample_voice_examples(profile, n) to pull few-shot context for the LLM prompt.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SITES_ROOT = PROJECT_ROOT / "data" / "seo" / "sites"

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


class BrandProfileNotFound(FileNotFoundError):
    pass


@dataclass(frozen=True)
class VoiceExample:
    path: Path
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def example_type(self) -> str:
        return str(self.metadata.get("type", "unknown"))


@dataclass(frozen=True)
class BrandProfile:
    slug: str
    domain: str
    display_name: str
    brand_one_liner: str
    voice: dict[str, Any]
    never_say: list[str]
    always: list[str]
    target_audience: str
    target_keywords: dict[str, list[str]]
    locked_decisions: list[str]
    content_type_preferences: dict[str, Any]
    voice_examples_dir: Path
    voice_examples_per_call: int
    raw: dict[str, Any]

    def keywords_for(self, content_type: str) -> list[str]:
        """Return a flat dedup'd list of keywords relevant to a content type."""
        primary = self.target_keywords.get("primary", [])
        local = self.target_keywords.get("local", [])
        long_tail = self.target_keywords.get("long_tail", [])
        # Cluster pages and blog favor long-tail; ad landing favors primary;
        # product enrichment uses primary + local.
        if content_type in {"cluster", "cluster_pages", "blog"}:
            ordered = long_tail + primary + local
        elif content_type in {"ad_landing", "ad"}:
            ordered = primary + long_tail
        else:
            ordered = primary + local + long_tail
        seen: set[str] = set()
        out: list[str] = []
        for k in ordered:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out


def _parse_voice_example(path: Path) -> VoiceExample:
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = text[m.end():].strip()
    else:
        meta = {}
        body = text.strip()
    return VoiceExample(path=path, body=body, metadata=meta)


def load_profile(slug: str, sites_root: Path | None = None) -> BrandProfile:
    """Load the brand profile YAML for a site slug.

    Raises BrandProfileNotFound if the site directory or brand.yaml is missing.
    """
    root = sites_root or SITES_ROOT
    yaml_path = root / slug / "brand.yaml"
    if not yaml_path.is_file():
        raise BrandProfileNotFound(f"missing brand.yaml for slug={slug} at {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    examples_dir_str = raw.get("voice_examples_dir") or f"data/seo/sites/{slug}/voice_examples/"
    examples_dir = PROJECT_ROOT / examples_dir_str if not Path(examples_dir_str).is_absolute() else Path(examples_dir_str)

    return BrandProfile(
        slug=raw.get("slug", slug),
        domain=raw["domain"],
        display_name=raw.get("display_name", slug),
        brand_one_liner=raw.get("brand_one_liner", ""),
        voice=raw.get("voice", {}),
        never_say=list(raw.get("never_say", [])),
        always=list(raw.get("always", [])),
        target_audience=raw.get("target_audience", "").strip(),
        target_keywords=raw.get("target_keywords", {}) or {},
        locked_decisions=list(raw.get("locked_decisions", [])),
        content_type_preferences=raw.get("content_type_preferences", {}) or {},
        voice_examples_dir=examples_dir,
        voice_examples_per_call=int(raw.get("voice_examples_per_call", 4)),
        raw=raw,
    )


def list_voice_examples(profile: BrandProfile) -> list[VoiceExample]:
    """Load every voice example file for a profile (no sampling)."""
    if not profile.voice_examples_dir.is_dir():
        return []
    paths = sorted(profile.voice_examples_dir.glob("*.md"))
    return [_parse_voice_example(p) for p in paths]


def sample_voice_examples(
    profile: BrandProfile,
    n: int | None = None,
    *,
    prefer_type: str | None = None,
    rng: random.Random | None = None,
) -> list[VoiceExample]:
    """Sample N voice examples for few-shot prompting.

    If prefer_type is given, all examples of that type are surfaced first
    (caps at n), then random others fill any remaining slots. This lets the
    writer bias toward "show me product copy" when generating product
    enrichment, without losing exposure to other in-voice forms.
    """
    examples = list_voice_examples(profile)
    if not examples:
        return []
    n = n or profile.voice_examples_per_call
    rng = rng or random.Random()

    if prefer_type:
        preferred = [e for e in examples if e.example_type == prefer_type]
        others = [e for e in examples if e.example_type != prefer_type]
        rng.shuffle(preferred)
        rng.shuffle(others)
        chosen = preferred[:n]
        if len(chosen) < n:
            chosen += others[: n - len(chosen)]
        return chosen

    pool = list(examples)
    rng.shuffle(pool)
    return pool[:n]
