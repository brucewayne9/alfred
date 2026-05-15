"""Canonical content type vocabulary for the SEO content engine.

Single source of truth — keep writer.MODEL_BY_TYPE,
profile.keywords_for(), validator length-window, and the admin UI all
reading from CONTENT_TYPES.

Includes a small backwards-compat alias map for older brand.yaml files
that used `cluster_pages` instead of `cluster`.
"""
from __future__ import annotations

CONTENT_TYPES: tuple[str, ...] = (
    "blog",
    "cluster",
    "product_enrichment",
    "ad_landing",
)

CONTENT_TYPE_LABELS: dict[str, str] = {
    "blog": "Blog post",
    "cluster": "Topic cluster page",
    "product_enrichment": "Product enrichment",
    "ad_landing": "Ad landing page",
}

# Older brand.yaml files used these names; canonicalize on load.
_ALIASES: dict[str, str] = {
    "cluster_pages": "cluster",
    "cluster_page": "cluster",
    "ad": "ad_landing",
    "landing": "ad_landing",
    "product": "product_enrichment",
}


def canonicalize(content_type: str) -> str:
    """Map an alias to its canonical name. Unknown types pass through unchanged."""
    if not content_type:
        return content_type
    return _ALIASES.get(content_type, content_type)


def is_known(content_type: str) -> bool:
    return canonicalize(content_type) in CONTENT_TYPES
