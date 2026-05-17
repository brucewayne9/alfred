# core/seo/audit/runner.py
"""Orchestrates one full audit run for a single SeoSite.

End-to-end flow:
  1. Resolve site_id -> SeoSite (need domain + business_type)
  2. Post DataForSEO On-Page task, poll until ready, fetch summary
  3. Pull per-page detail via /v3/on_page/pages (extends the sister-agent
     client if the method is missing; calls DFS directly via the same
     auth path)
  4. Map DFS checks -> AuditIssue list using DFS_CHECK_TO_ISSUE
  5. Augment with our own structured-data checks (JSON-LD presence on
     homepage / product / blog pages)
  6. Reconcile with seo_audit_issues via persist_issues
  7. Log spend in seo_api_costs
  8. Optionally backfill alt text on up to 10 missing-alt images
  9. Return AuditRun summary
"""
from __future__ import annotations

import dataclasses as dc
import datetime as dt
import json
import logging
import re
import uuid
from typing import Any, Optional

import httpx

from core.seo.audit.altext import generate_alt_text
from core.seo.audit.issues import DFS_CHECK_TO_ISSUE, AuditIssue
from core.seo.audit.persist import log_api_cost, persist_issues
from core.seo.db import SessionLocal
from core.seo.models import SeoSite

logger = logging.getLogger(__name__)

# Don't run alt-text generation on every missing-alt image — too slow/expensive.
DEFAULT_ALT_BACKFILL_CAP = 10

# Schema (JSON-LD) regex — pull out <script type="application/ld+json"> blocks.
_JSONLD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)


@dc.dataclass
class AuditRun:
    """Summary returned to the caller after one run."""

    run_id: str
    site_id: int
    started_at: dt.datetime
    finished_at: dt.datetime
    pages_crawled: int
    issues_detected: int
    issues_new: int
    issues_resolved: int
    issues_still_open: int
    dfs_cost_usd: float
    alt_texts_generated: int = 0
    errors: list[str] = dc.field(default_factory=list)


# --------------------------------------------------------------------- helpers


def _get_site(site_id: int) -> SeoSite:
    with SessionLocal() as s:
        site = s.get(SeoSite, site_id)
        if not site:
            raise ValueError(f"site not found: id={site_id}")
        # Detach so callers can read attrs after the session closes.
        s.expunge(site)
        return site


def _build_dfs_client():
    """Construct the sister-agent's DataForSEOClient. Imported lazily so
    tests can stub the whole runner without the sister-agent code in place."""
    from integrations.dataforseo.client import DataForSEOClient
    return DataForSEOClient()


def _onpage_pages(client, task_id: str, *, limit: int = 100, offset: int = 0) -> list[dict]:
    """Fetch per-page details + checks for a completed on-page task.

    The sister-agent client only specced onpage_summary. If it has
    onpage_pages, use it; otherwise call /v3/on_page/pages directly with
    the same auth pattern (login/password basic auth).
    """
    # Prefer the typed method if the sister agent added it.
    if hasattr(client, "onpage_pages"):
        return list(client.onpage_pages(task_id, limit=limit, offset=offset))

    # Fallback: call DFS directly using the client's auth path.
    path = "/on_page/pages"
    payload = [{"id": task_id, "limit": limit, "offset": offset}]
    data = client._request("POST", path, json_body=payload)

    pages: list[dict] = []
    for task in data.get("tasks") or []:
        for r in task.get("result") or []:
            for item in r.get("items") or []:
                if item:
                    pages.append(item)
    return pages


def _checks_from_page(page: dict) -> dict[str, bool]:
    """A DFS page item has `checks` (dict[str, bool]) for boolean flags.
    Older shapes may nest under `onpage_score` siblings — be liberal."""
    checks = page.get("checks") or {}
    if isinstance(checks, dict):
        return {k: bool(v) for k, v in checks.items()}
    return {}


def _images_from_page(page: dict) -> list[dict]:
    """Image entries on a page result; presence/format may vary by DFS."""
    imgs = page.get("images") or []
    return [i for i in imgs if isinstance(i, dict)]


def _links_from_page(page: dict) -> list[dict]:
    """Outbound link entries on a page result."""
    links = page.get("links") or []
    return [l for l in links if isinstance(l, dict)]


def _issues_from_page_checks(page: dict) -> list[AuditIssue]:
    """Yield page-level AuditIssues based on DFS boolean checks on this page."""
    out: list[AuditIssue] = []
    url = page.get("url") or ""
    if not url:
        return out

    checks = _checks_from_page(page)
    for dfs_check, flagged in checks.items():
        if not flagged:
            continue
        mapping = DFS_CHECK_TO_ISSUE.get(dfs_check)
        if not mapping:
            continue
        issue_type, severity = mapping

        # Some checks emit per-target rows (broken_link, no_image_alt).
        # These get handled below in dedicated loops to capture the target.
        # Skip them here so we don't create a duplicate page-level row.
        if issue_type in {
            "broken_link", "broken_redirect", "link_to_redirect", "http_error",
            "missing_alt_text", "missing_image_title",
        }:
            continue

        out.append(
            AuditIssue(
                page_url=url,
                issue_type=issue_type,
                severity=severity,
                detail=dfs_check.replace("_", " "),
                detail_payload={"dfs_check": dfs_check},
            )
        )
    return out


def _issues_from_page_images(page: dict) -> list[AuditIssue]:
    """One issue per image lacking alt / title (so fingerprint can pin to src)."""
    out: list[AuditIssue] = []
    url = page.get("url") or ""
    for img in _images_from_page(page):
        src = img.get("src") or img.get("url") or ""
        if not src:
            continue
        alt = img.get("alt")
        title = img.get("title")
        # DFS sometimes returns these as None to mean "missing"; sometimes ""
        if not alt:
            out.append(AuditIssue(
                page_url=url,
                issue_type="missing_alt_text",
                severity="warning",
                detail=f"image missing alt: {src}",
                detail_payload={"image_src": src},
            ))
        if not title:
            out.append(AuditIssue(
                page_url=url,
                issue_type="missing_image_title",
                severity="info",
                detail=f"image missing title: {src}",
                detail_payload={"image_src": src},
            ))
    return out


def _issues_from_page_links(page: dict) -> list[AuditIssue]:
    """One issue per broken / redirect link found in page outlinks."""
    out: list[AuditIssue] = []
    url = page.get("url") or ""
    for link in _links_from_page(page):
        target = link.get("link_to") or link.get("url") or link.get("target_url")
        if not target:
            continue

        if link.get("is_broken"):
            out.append(AuditIssue(
                page_url=url,
                issue_type="broken_link",
                severity="critical",
                detail=f"broken link -> {target}",
                detail_payload={
                    "target_url": target,
                    "status_code": link.get("status_code"),
                },
            ))
        if link.get("is_4xx_code") or link.get("is_5xx_code"):
            out.append(AuditIssue(
                page_url=url,
                issue_type="http_error",
                severity="critical",
                detail=f"http error -> {target}",
                detail_payload={
                    "target_url": target,
                    "status_code": link.get("status_code"),
                },
            ))
        if link.get("broken_redirect"):
            out.append(AuditIssue(
                page_url=url,
                issue_type="broken_redirect",
                severity="critical",
                detail=f"broken redirect -> {target}",
                detail_payload={"target_url": target},
            ))
        if link.get("has_links_to_redirect") or link.get("link_to_redirect"):
            out.append(AuditIssue(
                page_url=url,
                issue_type="link_to_redirect",
                severity="info",
                detail=f"link to redirect -> {target}",
                detail_payload={"target_url": target},
            ))
    return out


# ----------------------------------------------------- structured-data probes


def _fetch_jsonld_blocks(page_url: str, *, timeout: float = 15.0) -> list[dict]:
    """GET page_url, extract all parsed JSON-LD script blocks.

    Returns a flat list of parsed JSON objects. Silent on fetch/parse errors —
    a site we can't reach is the SEO issue, not the audit module's bug.
    """
    try:
        resp = httpx.get(page_url, timeout=timeout, follow_redirects=True)
        if resp.status_code >= 400:
            logger.info("schema-probe: %s -> HTTP %s", page_url, resp.status_code)
            return []
        html = resp.text
    except Exception as e:
        logger.info("schema-probe: %s fetch failed: %s", page_url, e)
        return []

    out: list[dict] = []
    for match in _JSONLD_RE.finditer(html):
        block = match.group(1).strip()
        if not block:
            continue
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            # Some sites emit invalid JSON-LD; try a forgiving cleanup.
            try:
                cleaned = re.sub(r",(\s*[}\]])", r"\1", block)
                data = json.loads(cleaned)
            except Exception:
                continue
        if isinstance(data, list):
            for d in data:
                if isinstance(d, dict):
                    out.append(d)
        elif isinstance(data, dict):
            # @graph idiom: top-level dict with nested @graph list
            if isinstance(data.get("@graph"), list):
                for d in data["@graph"]:
                    if isinstance(d, dict):
                        out.append(d)
            else:
                out.append(data)
    return out


def _has_schema_type(blocks: list[dict], wanted: str) -> bool:
    """True iff any block declares @type == wanted (or includes it in a list)."""
    wanted_lc = wanted.lower()
    for b in blocks:
        t = b.get("@type")
        if isinstance(t, str) and t.lower() == wanted_lc:
            return True
        if isinstance(t, list) and any(
            isinstance(x, str) and x.lower() == wanted_lc for x in t
        ):
            return True
    return False


def _schema_issues_for_site(site: SeoSite, pages: list[dict]) -> list[AuditIssue]:
    """Probe homepage / product / blog pages for required JSON-LD types."""
    out: list[AuditIssue] = []

    homepage_url = f"https://{site.domain.rstrip('/')}/"
    home_blocks = _fetch_jsonld_blocks(homepage_url)

    if (site.business_type or "").lower() == "localbusiness":
        if not _has_schema_type(home_blocks, "LocalBusiness"):
            out.append(AuditIssue(
                page_url=homepage_url,
                issue_type="missing_schema_localbusiness",
                severity="critical",
                detail="homepage missing LocalBusiness JSON-LD",
                detail_payload={"expected_type": "LocalBusiness"},
            ))

    # Probe a handful of product + blog pages (cap to keep it cheap).
    product_pages = [p for p in pages if "/product/" in (p.get("url") or "")][:5]
    blog_pages = [p for p in pages if "/blog/" in (p.get("url") or "")][:5]

    for p in product_pages:
        url = p.get("url")
        if not url:
            continue
        blocks = _fetch_jsonld_blocks(url)
        if not _has_schema_type(blocks, "Product"):
            out.append(AuditIssue(
                page_url=url,
                issue_type="missing_schema_product",
                severity="warning",
                detail="product page missing Product JSON-LD",
                detail_payload={"expected_type": "Product"},
            ))

    for p in blog_pages:
        url = p.get("url")
        if not url:
            continue
        blocks = _fetch_jsonld_blocks(url)
        if not (_has_schema_type(blocks, "Article") or _has_schema_type(blocks, "BlogPosting")):
            out.append(AuditIssue(
                page_url=url,
                issue_type="missing_schema_article",
                severity="warning",
                detail="blog page missing Article JSON-LD",
                detail_payload={"expected_type": "Article"},
            ))

    return out


# ------------------------------------------------------------ alt backfill


def _run_alt_backfill(issues: list[AuditIssue], *, cap: int) -> int:
    """Mutate up to `cap` missing_alt_text issues, attaching suggested_alt
    to their detail_payload. Returns count of suggestions generated."""
    generated = 0
    for issue in issues:
        if generated >= cap:
            break
        if issue.issue_type != "missing_alt_text":
            continue
        src = (issue.detail_payload or {}).get("image_src")
        if not src:
            continue
        suggested = generate_alt_text(src)
        if suggested:
            issue.detail_payload = dict(issue.detail_payload or {})
            issue.detail_payload["suggested_alt"] = suggested
            generated += 1
    return generated


# -------------------------------------------------------------------- public


def run_site_audit(
    site_id: int,
    max_pages: int = 100,
    *,
    with_alt_backfill: bool = False,
    alt_backfill_cap: int = DEFAULT_ALT_BACKFILL_CAP,
    dfs_client: Optional[Any] = None,
) -> AuditRun:
    """Run one full audit for the given site. Returns an AuditRun summary.

    `dfs_client` is injectable for testing — in production we build one
    from settings on demand.
    """
    run_id = uuid.uuid4().hex[:12]
    started_at = dt.datetime.now(dt.timezone.utc)
    errors: list[str] = []

    site = _get_site(site_id)
    target = site.domain
    logger.info(
        "audit start run_id=%s site=%s domain=%s max_pages=%s",
        run_id, site.slug, target, max_pages,
    )

    client = dfs_client if dfs_client is not None else _build_dfs_client()
    cost_before = float(getattr(client, "total_cost_usd", 0.0))

    pages: list[dict] = []
    summary: dict = {}
    try:
        # Post task + poll for summary in one blocking call. The convenience
        # helper returns the task_id so we can chain the per-page fetch
        # without re-discovering it from /tasks_ready (which drains on read).
        result = client.onpage_summary(target=target, max_crawl_pages=max_pages)
        if isinstance(result, tuple):
            task_id, summary = result
        else:
            # Legacy mock that returns summary only — best-effort recovery
            summary = result
            task_id = (
                summary.get("task_id")
                or (summary.get("domain") or {}).get("task_id")
            )
            if not task_id:
                ready = client.onpage_tasks_ready()
                task_id = ready[-1] if ready else None

        if task_id:
            pages = _onpage_pages(client, task_id, limit=max_pages)
        else:
            errors.append("could not resolve task_id to fetch per-page detail")
    except Exception as e:
        logger.exception("audit: DFS call failed for site=%s", site.slug)
        errors.append(f"dfs_call_failed: {e}")

    # ---- map DFS payload -> AuditIssues -----------------------------------
    detected: list[AuditIssue] = []
    for page in pages:
        detected.extend(_issues_from_page_checks(page))
        detected.extend(_issues_from_page_images(page))
        detected.extend(_issues_from_page_links(page))

    # ---- structured-data probes ------------------------------------------
    try:
        detected.extend(_schema_issues_for_site(site, pages))
    except Exception as e:
        logger.exception("audit: schema probe failed for site=%s", site.slug)
        errors.append(f"schema_probe_failed: {e}")

    # ---- optional alt-text backfill --------------------------------------
    alt_count = 0
    if with_alt_backfill and detected:
        try:
            alt_count = _run_alt_backfill(detected, cap=alt_backfill_cap)
        except Exception as e:
            logger.exception("audit: alt backfill failed for site=%s", site.slug)
            errors.append(f"alt_backfill_failed: {e}")

    # ---- persist + reconcile ---------------------------------------------
    counts = persist_issues(site_id, detected)

    # ---- spend log -------------------------------------------------------
    dfs_cost = float(getattr(client, "total_cost_usd", 0.0)) - cost_before
    if dfs_cost > 0:
        log_api_cost(
            api_name="dataforseo",
            endpoint="/on_page/summary+pages",
            cost_usd=dfs_cost,
            site_id=site_id,
            purpose="site_audit",
            meta={"run_id": run_id, "max_pages": max_pages, "pages_returned": len(pages)},
        )

    finished_at = dt.datetime.now(dt.timezone.utc)
    pages_crawled = (
        (summary.get("crawl_status") or {}).get("pages_crawled")
        if isinstance(summary.get("crawl_status"), dict)
        else None
    ) or len(pages)

    run = AuditRun(
        run_id=run_id,
        site_id=site_id,
        started_at=started_at,
        finished_at=finished_at,
        pages_crawled=int(pages_crawled or 0),
        issues_detected=counts["detected"],
        issues_new=counts["new"],
        issues_resolved=counts["resolved"],
        issues_still_open=counts["still_open"],
        dfs_cost_usd=round(dfs_cost, 6),
        alt_texts_generated=alt_count,
        errors=errors,
    )
    logger.info(
        "audit done run_id=%s site=%s detected=%d new=%d resolved=%d cost=$%.4f errs=%d",
        run_id, site.slug, run.issues_detected, run.issues_new,
        run.issues_resolved, run.dfs_cost_usd, len(errors),
    )
    return run
