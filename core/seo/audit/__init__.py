# core/seo/audit/__init__.py
"""SEO Site Audit module.

Drives a full on-page audit for one SeoSite via DataForSEO On-Page API,
emits one row per detected issue into seo_audit_issues (de-duped via a
stable fingerprint), and offers an opt-in alt-text backfill pass via the
105 Ollama vision model.

Public surface:
    run_site_audit(site_id, max_pages=100, with_alt_backfill=False) -> AuditRun
    AuditIssue, IssueFingerprint
    generate_alt_text(image_bytes_or_url)

See spec in tasking prompt 2026-05-16.
"""

from core.seo.audit.issues import AuditIssue, IssueFingerprint  # noqa: F401
from core.seo.audit.runner import AuditRun, run_site_audit  # noqa: F401
