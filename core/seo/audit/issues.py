# core/seo/audit/issues.py
"""AuditIssue dataclass + stable fingerprint helper.

The fingerprint is what lets us tell "same issue, still open" from "new
issue with same type on the same page". It's stored on
seo_audit_issues.issue_fingerprint and participates in the UNIQUE
constraint (site_id, page_url, issue_type, issue_fingerprint).

Convention: 16-char hex slice of a sha256 over whatever piece of the
issue uniquely identifies the SPECIFIC instance (image src, target URL,
external link href, etc.). For issues that only have one instance per
page (e.g. missing title, no h1), fingerprint is the literal string
"page".
"""
from __future__ import annotations

import dataclasses as dc
import hashlib
from typing import Any, Optional


# Mapping of DataForSEO On-Page check names to our issue_type taxonomy +
# severity. Adapt as the real API surfaces new check names.
DFS_CHECK_TO_ISSUE: dict[str, tuple[str, str]] = {
    "no_h1_tag":                       ("h1_missing",             "critical"),
    "duplicate_title_tag":             ("duplicate_title",        "warning"),
    "duplicate_meta_descriptions":     ("duplicate_meta_desc",    "warning"),
    "no_title":                        ("missing_title",          "critical"),
    "no_description":                  ("missing_meta_desc",      "warning"),
    "size_greater_than_3mb":           ("page_too_large",         "warning"),
    "has_render_blocking_resources":   ("render_blocking",        "info"),
    "low_content_rate":                ("thin_content",           "warning"),
    "has_meta_refresh_redirect":       ("meta_refresh",           "warning"),
    "no_image_alt":                    ("missing_alt_text",       "warning"),
    "no_image_title":                  ("missing_image_title",    "info"),
    "broken_redirect":                 ("broken_redirect",        "critical"),
    "is_broken":                       ("broken_link",            "critical"),
    "is_4xx_code":                     ("http_error",             "critical"),
    "is_5xx_code":                     ("http_error",             "critical"),
    "canonical_to_redirect":           ("canonical_to_redirect",  "warning"),
    "has_links_to_redirect":           ("link_to_redirect",       "info"),
    "irrelevant_description":          ("irrelevant_meta_desc",   "info"),
    "irrelevant_title":                ("irrelevant_title",       "info"),
    "seo_friendly_url_relative_length": ("url_too_long",          "info"),
}

# Fingerprints for these issue_types depend on a specific field in the
# detail_payload. Anything not in this map falls back to the page-level
# fingerprint of "page".
FINGERPRINT_KEY_BY_TYPE: dict[str, str] = {
    "missing_alt_text":      "image_src",
    "missing_image_title":   "image_src",
    "broken_link":           "target_url",
    "broken_redirect":       "target_url",
    "link_to_redirect":      "target_url",
    "http_error":            "target_url",
    "missing_schema_product": "page",
    "missing_schema_article": "page",
    "missing_schema_localbusiness": "page",
}


@dc.dataclass
class AuditIssue:
    """In-flight representation before persistence.

    page_url     — absolute URL of the page where the issue was found
    issue_type   — our taxonomy (see DFS_CHECK_TO_ISSUE values)
    severity     — "critical" | "warning" | "info"
    detail       — short human label (rendered in dashboard list)
    detail_payload — JSON-serializable dict with everything else
    fingerprint  — populated via IssueFingerprint.for_issue if omitted
    """

    page_url: str
    issue_type: str
    severity: str
    detail: Optional[str] = None
    detail_payload: dict[str, Any] = dc.field(default_factory=dict)
    fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = IssueFingerprint.for_issue(
                self.issue_type, self.detail_payload
            )


class IssueFingerprint:
    """Stable short-id generator for issue dedup."""

    @staticmethod
    def _hash16(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def for_issue(cls, issue_type: str, detail_payload: dict[str, Any]) -> str:
        """Return a 16-char fingerprint for the issue instance.

        Page-level issues (no per-target field) return the literal "page".
        Issues whose fingerprint key is missing from payload also fall back
        to "page" so we don't silently lose them.
        """
        key = FINGERPRINT_KEY_BY_TYPE.get(issue_type)
        if not key or key == "page":
            return "page"
        value = detail_payload.get(key)
        if not value:
            return "page"
        return cls._hash16(str(value))
