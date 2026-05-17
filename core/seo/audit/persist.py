# core/seo/audit/persist.py
"""Persist a batch of detected AuditIssues for one site.

Single entry point: persist_issues(site_id, detected). Handles:
 - upsert against UNIQUE (site_id, page_url, issue_type, issue_fingerprint)
 - last_detected_at bump on existing open rows
 - fixed_at marking for previously-open rows not seen this run
 - returns the counts the runner needs for AuditRun summary
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.seo.audit.issues import AuditIssue
from core.seo.db import SessionLocal
from core.seo.models import SeoAuditIssue

logger = logging.getLogger(__name__)


def persist_issues(site_id: int, detected: Iterable[AuditIssue]) -> dict[str, int]:
    """Reconcile detected issues against currently-open rows in the DB.

    Steps:
      1. Snapshot all currently-open issues for site_id (fixed_at IS NULL)
      2. For each detected issue: upsert. New row -> insert. Existing open
         row -> update last_detected_at and detail/payload (the issue is
         still there, but maybe its detail changed slightly).
      3. Any pre-snapshot open row not seen this run -> mark fixed_at=now.

    Returns counts: detected, new, resolved, still_open.
    """
    now = dt.datetime.now(dt.timezone.utc)
    detected_list = list(detected)
    detected_keys: set[tuple[str, str, str]] = {
        (i.page_url, i.issue_type, i.fingerprint or "page") for i in detected_list
    }

    new_count = 0
    still_open = 0

    with SessionLocal() as s:
        # Snapshot pre-existing open issues for this site.
        open_rows = list(
            s.scalars(
                select(SeoAuditIssue).where(
                    SeoAuditIssue.site_id == site_id,
                    SeoAuditIssue.fixed_at.is_(None),
                )
            ).all()
        )
        existing_keys = {
            (r.page_url, r.issue_type, r.issue_fingerprint): r for r in open_rows
        }

        # Upsert each detected.
        for issue in detected_list:
            key = (issue.page_url, issue.issue_type, issue.fingerprint or "page")
            stmt = (
                pg_insert(SeoAuditIssue)
                .values(
                    site_id=site_id,
                    page_url=issue.page_url,
                    issue_type=issue.issue_type,
                    severity=issue.severity,
                    detail=issue.detail,
                    detail_payload=issue.detail_payload or {},
                    first_detected_at=now,
                    last_detected_at=now,
                    fixed_at=None,
                    issue_fingerprint=issue.fingerprint or "page",
                )
                .on_conflict_do_update(
                    index_elements=[
                        "site_id", "page_url", "issue_type", "issue_fingerprint",
                    ],
                    set_=dict(
                        last_detected_at=now,
                        detail=issue.detail,
                        detail_payload=issue.detail_payload or {},
                        severity=issue.severity,
                        # If a row was previously marked fixed and the issue
                        # came back, clear fixed_at so the dashboard re-opens
                        # it instead of orphaning a closed row.
                        fixed_at=None,
                    ),
                )
            )
            s.execute(stmt)
            if key in existing_keys:
                still_open += 1
            else:
                new_count += 1

        # Resolve issues that disappeared this run.
        resolved = 0
        for key, row in existing_keys.items():
            if key not in detected_keys:
                row.fixed_at = now
                row.fix_method = row.fix_method or "auto_resolved"
                resolved += 1
        s.commit()

    return {
        "detected": len(detected_list),
        "new": new_count,
        "resolved": resolved,
        "still_open": still_open,
    }


def log_api_cost(
    *,
    api_name: str,
    endpoint: str,
    cost_usd: float,
    site_id: int | None,
    purpose: str,
    meta: dict | None = None,
) -> None:
    """Append a cost row to seo_api_costs. Best-effort; never raises."""
    from core.seo.models import SeoApiCost

    try:
        with SessionLocal() as s:
            s.add(
                SeoApiCost(
                    api_name=api_name,
                    endpoint=endpoint,
                    cost_usd=cost_usd,
                    site_id=site_id,
                    purpose=purpose,
                    meta=meta or {},
                )
            )
            s.commit()
    except Exception:
        logger.exception("failed to log api cost (non-fatal)")
