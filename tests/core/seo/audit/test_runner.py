# tests/core/seo/audit/test_runner.py
"""Tests for the SEO site-audit runner.

Stubs the DataForSEO client + the network-bound schema/alt helpers so the
suite runs offline against the live Postgres instance.
"""
from __future__ import annotations

import datetime as dt
from typing import Any
from unittest.mock import patch

import pytest

from core.seo.audit.issues import AuditIssue, IssueFingerprint
from core.seo.audit.persist import persist_issues
from core.seo.audit.runner import run_site_audit
from core.seo.db import SessionLocal
from core.seo.models import SeoAuditIssue, SeoSite
from core.seo.sites.registry import deactivate_site, register_site

SLUG = "roen-audit-test"


# ----------------------------------------------------------------- fixtures


def _purge() -> None:
    """Hard-clean any prior fixture rows."""
    with SessionLocal() as s:
        for site in s.query(SeoSite).filter_by(slug=SLUG).all():
            s.query(SeoAuditIssue).filter_by(site_id=site.id).delete()
            s.delete(site)
        s.commit()


@pytest.fixture
def site():
    _purge()
    site = register_site(
        slug=SLUG,
        domain="roen-audit.invalid",
        display_name="Roen Audit Test",
        wp_rest_url="https://x/wp-json",
        business_type="LocalBusiness",
    )
    yield site
    deactivate_site(SLUG)
    _purge()


class _StubDfsClient:
    """Mimics the slice of DataForSEOClient the runner uses.

    The real client starts total_cost_usd at 0.0 and accumulates per call.
    Mirror that: the stub adds `cost_per_call` on each summary/pages fetch
    so the runner's (after - before) delta is non-zero, matching prod.
    """

    def __init__(self, summary: dict, pages: list[dict], cost: float = 0.10):
        self._summary = summary
        self._pages = pages
        self._cost_per_call = cost
        self.total_cost_usd = 0.0

    def onpage_summary(self, target: str, max_crawl_pages: int = 100) -> tuple[str, dict]:
        self.total_cost_usd += self._cost_per_call
        return "stub-task-001", self._summary

    def onpage_tasks_ready(self) -> list[str]:
        return ["stub-task-001"]

    def onpage_pages(self, task_id: str, *, limit: int = 100, offset: int = 0):
        return list(self._pages)


def _make_summary(pages_crawled: int = 2) -> dict:
    return {
        "task_id": "stub-task-001",
        "domain": {"name": "roen-audit.invalid"},
        "crawl_progress": "finished",
        "crawl_status": {"pages_crawled": pages_crawled},
        "page_metrics": {"onpage_score": 78.0, "checks": {}},
    }


def _make_pages_with_issues() -> list[dict]:
    """Two pages, several checks each, with image + link rows."""
    return [
        {
            "url": "https://roen-audit.invalid/",
            "checks": {
                "no_h1_tag": True,
                "no_description": True,
                "size_greater_than_3mb": False,
            },
            "images": [
                {"src": "https://cdn/img-a.jpg", "alt": None, "title": None},
                {"src": "https://cdn/img-b.jpg", "alt": "fine", "title": "fine"},
            ],
            "links": [
                {"link_to": "https://broken.example/x",
                 "is_broken": True, "status_code": 404},
            ],
        },
        {
            "url": "https://roen-audit.invalid/about/",
            "checks": {
                "no_title": True,
                "duplicate_meta_descriptions": True,
            },
            "images": [],
            "links": [],
        },
    ]


# ----------------------------------------------------------------- tests


def test_fingerprint_stable_for_same_input():
    fp1 = IssueFingerprint.for_issue("missing_alt_text", {"image_src": "https://x/y.jpg"})
    fp2 = IssueFingerprint.for_issue("missing_alt_text", {"image_src": "https://x/y.jpg"})
    assert fp1 == fp2 and len(fp1) == 16


def test_fingerprint_page_for_page_level_issue():
    assert IssueFingerprint.for_issue("missing_title", {}) == "page"
    assert IssueFingerprint.for_issue("h1_missing", {"anything": 1}) == "page"


def test_fingerprint_differs_for_different_targets():
    a = IssueFingerprint.for_issue("broken_link", {"target_url": "https://a/"})
    b = IssueFingerprint.for_issue("broken_link", {"target_url": "https://b/"})
    assert a != b


def test_persist_inserts_then_dedups_then_resolves(site):
    """Three-call lifecycle: insert, repeat (dedup), then disappear (resolve)."""
    issues_round_1 = [
        AuditIssue(
            page_url="https://roen-audit.invalid/",
            issue_type="h1_missing",
            severity="critical",
            detail="page is missing h1",
        ),
        AuditIssue(
            page_url="https://roen-audit.invalid/",
            issue_type="missing_alt_text",
            severity="warning",
            detail="image missing alt: cdn/img-a.jpg",
            detail_payload={"image_src": "https://cdn/img-a.jpg"},
        ),
    ]
    counts_1 = persist_issues(site.id, issues_round_1)
    assert counts_1["new"] == 2
    assert counts_1["resolved"] == 0
    assert counts_1["still_open"] == 0

    # Same issues again — should NOT duplicate rows.
    counts_2 = persist_issues(site.id, issues_round_1)
    assert counts_2["new"] == 0
    assert counts_2["still_open"] == 2

    with SessionLocal() as s:
        rows = s.query(SeoAuditIssue).filter_by(site_id=site.id).all()
        assert len(rows) == 2
        assert all(r.fixed_at is None for r in rows)
        assert all(r.last_detected_at is not None for r in rows)

    # Drop one of them this run — it should get marked fixed.
    counts_3 = persist_issues(site.id, [issues_round_1[0]])
    assert counts_3["resolved"] == 1
    assert counts_3["still_open"] == 1
    with SessionLocal() as s:
        alt_row = (
            s.query(SeoAuditIssue)
            .filter_by(site_id=site.id, issue_type="missing_alt_text")
            .one()
        )
        assert alt_row.fixed_at is not None
        assert alt_row.fix_method == "auto_resolved"


def test_persist_reopens_previously_fixed_issue(site):
    issue = AuditIssue(
        page_url="https://roen-audit.invalid/",
        issue_type="h1_missing",
        severity="critical",
    )
    persist_issues(site.id, [issue])      # insert
    persist_issues(site.id, [])           # resolve
    counts = persist_issues(site.id, [issue])  # comes back
    assert counts["new"] == 1  # was closed, counts as new since not in open snapshot

    with SessionLocal() as s:
        row = (
            s.query(SeoAuditIssue)
            .filter_by(site_id=site.id, issue_type="h1_missing")
            .one()
        )
        assert row.fixed_at is None


def test_run_site_audit_end_to_end_with_stub_client(site):
    """End-to-end with a stubbed DFS client + schema probe stubbed empty."""
    stub = _StubDfsClient(
        summary=_make_summary(pages_crawled=2),
        pages=_make_pages_with_issues(),
        cost=0.0345,
    )

    with patch(
        "core.seo.audit.runner._schema_issues_for_site", return_value=[]
    ):
        run = run_site_audit(site.id, max_pages=50, dfs_client=stub)

    assert run.pages_crawled == 2
    assert run.dfs_cost_usd == pytest.approx(0.0345, rel=1e-3)
    # h1_missing, missing_meta_desc, missing_alt_text(a), missing_image_title(a),
    # missing_title, duplicate_meta_desc, broken_link  -> 7
    assert run.issues_detected >= 6
    assert run.issues_new == run.issues_detected
    assert run.issues_resolved == 0
    assert run.errors == []

    with SessionLocal() as s:
        rows = s.query(SeoAuditIssue).filter_by(site_id=site.id).all()
        types = {r.issue_type for r in rows}
        assert "h1_missing" in types
        assert "missing_alt_text" in types
        assert "broken_link" in types
        assert "missing_title" in types

    # Second identical run should add zero new rows.
    with patch(
        "core.seo.audit.runner._schema_issues_for_site", return_value=[]
    ):
        run2 = run_site_audit(site.id, max_pages=50, dfs_client=stub)
    assert run2.issues_new == 0
    assert run2.issues_resolved == 0
    assert run2.issues_still_open == run.issues_detected


def test_run_site_audit_resolves_issues_that_disappear(site):
    """Round 1 detects issues; round 2 returns empty pages -> all resolved."""
    stub_full = _StubDfsClient(_make_summary(2), _make_pages_with_issues())
    with patch("core.seo.audit.runner._schema_issues_for_site", return_value=[]):
        first = run_site_audit(site.id, dfs_client=stub_full)
    assert first.issues_detected > 0

    stub_empty = _StubDfsClient(_make_summary(2), pages=[])
    with patch("core.seo.audit.runner._schema_issues_for_site", return_value=[]):
        second = run_site_audit(site.id, dfs_client=stub_empty)

    assert second.issues_resolved == first.issues_detected
    assert second.issues_new == 0
    assert second.issues_still_open == 0
