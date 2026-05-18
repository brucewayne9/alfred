# core/api/seo_admin.py
"""FastAPI routes for /admin/seo/*. Auth-gated via the existing JWT cookie.

Screens:
  /admin/seo                              cross-site dashboard (KPI strip per site)
  /admin/seo/sites/{slug}                 per-site overview (KPIs + top movers + nav)
  /admin/seo/sites/{slug}/keywords        sortable keyword table
  /admin/seo/sites/{slug}/rankings        ranking history with sparkline + drill-down
  /admin/seo/sites/{slug}/audit           audit issues grouped by severity
  /admin/seo/spend                        cross-site DataForSEO spend breakdown
  /admin/seo/pending                      approval queue (existing)
  /admin/seo/new                          new brief form (existing)
"""
from __future__ import annotations

import datetime as dt
import html
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy import text

from core.security.auth import get_current_user, require_auth
from core.seo.content.types import CONTENT_TYPE_LABELS, CONTENT_TYPES, canonicalize
from core.seo.content.writer import Brief, generate_with_retry
from core.seo.dashboard import site_kpis, top_movers, spend_breakdown
from core.seo.db import SessionLocal
from core.seo.images.selector import compose_blog_images
from core.seo.models import SeoAuditIssue, SeoKeyword, SeoRankingDaily
from core.seo.queue.pending import (
    approve_and_publish,
    enqueue_draft,
    list_pending,
    reject,
)
from core.seo.sites.profile import BrandProfileNotFound, load_profile
from core.seo.sites.registry import get_site_by_slug, list_sites

logger = logging.getLogger(__name__)


# =============================================================================
# Shared CSS — every page sources from this so visual style stays consistent.
# =============================================================================
_BASE_CSS = """
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1180px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }
  h1 { font-weight: 200; letter-spacing: 1px; margin: 0 0 4px; }
  h2 { font-weight: 300; letter-spacing: 0.5px; margin: 24px 0 12px; font-size: 19px; }
  h3 { font-weight: 600; font-size: 15px; margin: 16px 0 8px; }
  a  { color: #1a1a1a; text-decoration: none; border-bottom: 1px dotted #999; }
  a:hover { color: #B85C3D; border-color: #B85C3D; }
  .muted { color: #999; font-size: 13px; }
  .nav { margin-bottom: 18px; font-size: 13px; }
  .nav a { margin-right: 14px; }
  .nav .active { color: #B85C3D; border-bottom: 1px solid #B85C3D; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid #eee; vertical-align: top; }
  th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; color: #666; font-weight: 600; }
  tr:hover td { background: #fafafa; }
  /* KPI strip */
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 10px; margin: 14px 0 28px; }
  .kpi { border: 1px solid #eee; border-radius: 6px; padding: 12px 14px; background: #fff; }
  .kpi .label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.7px; color: #888; }
  .kpi .value { font-size: 26px; font-weight: 200; margin-top: 2px; letter-spacing: -0.5px; }
  .kpi .sub   { font-size: 12px; color: #777; margin-top: 2px; }
  .kpi.warn .value { color: #a06700; }
  .kpi.err  .value { color: #a02020; }
  .kpi.ok   .value { color: #2a7a4a; }
  /* Tags */
  .tag { display: inline-block; font-size: 11px; padding: 3px 8px; border-radius: 4px;
         text-transform: uppercase; letter-spacing: 0.4px; margin-right: 4px; }
  .tag.site { background: #B85C3D; color: white; }
  .tag.type { background: #f0f0f0; color: #444; }
  .tag.sev-warning { background: #fff5e0; color: #a06700; }
  .tag.sev-info    { background: #eef4fb; color: #2c5a8a; }
  .tag.sev-error   { background: #ffe5e5; color: #a02020; }
  /* Delta arrows in rank columns */
  .up   { color: #2a7a4a; font-weight: 600; }
  .down { color: #a02020; font-weight: 600; }
  .flat { color: #999; }
  /* Bar chart for spend */
  .barchart { display: flex; align-items: flex-end; gap: 3px; height: 80px;
              border-bottom: 1px solid #eee; padding-bottom: 4px; margin: 8px 0; }
  .bar { background: #B85C3D; opacity: 0.85; min-width: 6px; border-radius: 2px 2px 0 0;
         transition: opacity 0.15s; }
  .bar:hover { opacity: 1; }
  /* Sparkline column */
  .spark { vertical-align: middle; }
"""


# =============================================================================
# Formatters
# =============================================================================
def _format_when(dt_val: datetime | None) -> str:
    if not dt_val:
        return "—"
    delta = datetime.now(timezone.utc) - dt_val
    if delta.total_seconds() < 60:
        return "just now"
    if delta.total_seconds() < 3600:
        return f"{int(delta.total_seconds() / 60)}m ago"
    if delta.total_seconds() < 86400:
        return f"{int(delta.total_seconds() / 3600)}h ago"
    return dt_val.strftime("%b %d %H:%M")


def _format_rank(p) -> str:
    if p is None:
        return '<span class="flat">—</span>'
    p = int(p)
    if p <= 3:
        return f'<strong class="up">{p}</strong>'
    if p <= 10:
        return f'<strong>{p}</strong>'
    return str(p)


def _format_delta(delta) -> str:
    if delta is None or delta == 0:
        return '<span class="flat">—</span>'
    if delta > 0:
        return f'<span class="up">+{int(delta)} ↑</span>'
    return f'<span class="down">{int(delta)} ↓</span>'


def _money(usd: float) -> str:
    if usd < 1.0:
        return f"${usd:.4f}"
    if usd < 100.0:
        return f"${usd:.2f}"
    return f"${usd:,.2f}"


# =============================================================================
# Shared nav
# =============================================================================
def _site_nav(slug: str, active: str) -> str:
    """In-site nav strip rendered on each per-site page."""
    items = [
        ("overview", f"/admin/seo/sites/{slug}", "Overview"),
        ("keywords", f"/admin/seo/sites/{slug}/keywords", "Keywords"),
        ("rankings", f"/admin/seo/sites/{slug}/rankings", "Rankings"),
        ("audit",    f"/admin/seo/sites/{slug}/audit",    "Audit"),
    ]
    parts = []
    for key, href, label in items:
        cls = ' class="active"' if key == active else ""
        parts.append(f'<a href="{href}"{cls}>{label}</a>')
    return (
        '<div class="nav">'
        '<a href="/admin/seo">← all sites</a>'
        f'<span style="color:#ccc">·</span> {" ".join(parts)}'
        '<span style="color:#ccc">·</span> '
        '<a href="/admin/seo/pending">queue</a>'
        '<a href="/admin/seo/spend">spend</a>'
        '<a href="/admin/seo/new">+ new brief</a>'
        '</div>'
    )


def _top_nav() -> str:
    """Cross-site nav for non-site pages."""
    return (
        '<div class="nav">'
        '<a href="/admin/seo" class="active">sites</a>'
        '<a href="/admin/seo/pending">queue</a>'
        '<a href="/admin/seo/spend">spend</a>'
        '<a href="/admin/seo/new">+ new brief</a>'
        '</div>'
    )


# =============================================================================
# Render: cross-site dashboard (/admin/seo)
# =============================================================================
def _render_dashboard(sites: list) -> str:
    # For each site, pull a thin KPI strip — keyword count + top-10 + open issues.
    rows = []
    for s in sites:
        try:
            k = site_kpis(s.id)
            kw_summary = (
                f"{k.keyword_count} kw · top10 {k.top10_count} · top3 {k.top3_count}"
                if k.keyword_count else
                '<span class="muted">no keywords</span>'
            )
            issues_summary = (
                f'<span class="tag sev-warning">{k.warning_issues} warn</span> '
                f'<span class="tag sev-info">{k.info_issues} info</span>'
                if k.open_issues else '<span class="muted">none</span>'
            )
            spend_str = _money(k.spend_mtd_usd) if k.spend_mtd_usd else '<span class="muted">$0</span>'
        except Exception as e:
            logger.exception("kpis failed for site=%s", s.slug)
            kw_summary = issues_summary = f'<span class="muted">err: {html.escape(str(e)[:30])}</span>'
            spend_str = "—"

        rows.append(f"""
        <tr>
          <td><a href="/admin/seo/sites/{s.slug}"><strong>{html.escape(s.display_name)}</strong></a><br>
              <span class="muted">{html.escape(s.domain)}</span></td>
          <td>{kw_summary}</td>
          <td>{issues_summary}</td>
          <td>{spend_str}</td>
          <td class="muted">{_format_when(s.updated_at)}</td>
        </tr>""")
    rows_html = "\n".join(rows) or (
        '<tr><td colspan=5 class=muted>No sites registered yet. '
        'Run scripts/seo_init_roen.py to add the first.</td></tr>'
    )

    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>SEO — sites</title>
<style>{_BASE_CSS}</style></head><body>
{_top_nav()}
<h1>alfred's seo</h1>
<p class="muted">{len(sites)} site{'' if len(sites) == 1 else 's'} · KPIs reflect current state · spend = month-to-date</p>
<table>
  <thead><tr><th>Site</th><th>Keywords</th><th>Open issues</th><th>Spend (MTD)</th><th>Updated</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</body></html>"""


# =============================================================================
# Render: per-site overview (/admin/seo/sites/{slug})
# =============================================================================
def _render_site_overview(site, kpis, gainers, losers) -> str:
    def kpi_card(label: str, value: str, sub: str = "", cls: str = "") -> str:
        sub_html = f'<div class="sub">{sub}</div>' if sub else ""
        cls_html = f" {cls}" if cls else ""
        return f'<div class="kpi{cls_html}"><div class="label">{label}</div><div class="value">{value}</div>{sub_html}</div>'

    avg_rank_str = f"{kpis.avg_rank:.1f}" if kpis.avg_rank is not None else "—"
    last_capture = kpis.last_rank_capture.isoformat() if kpis.last_rank_capture else "never"

    kpi_strip = "".join([
        kpi_card("keywords", str(kpis.keyword_count), f"avg rank {avg_rank_str}"),
        kpi_card("ranked top 100", str(kpis.ranked_count),
                 f"of {kpis.keyword_count}",
                 cls="ok" if kpis.ranked_count else ""),
        kpi_card("top 10", str(kpis.top10_count), "first page"),
        kpi_card("top 3", str(kpis.top3_count), "visible above fold",
                 cls="ok" if kpis.top3_count else ""),
        kpi_card("open issues", str(kpis.open_issues),
                 f"{kpis.warning_issues}w · {kpis.info_issues}i · {kpis.error_issues}e",
                 cls="warn" if kpis.warning_issues else ("err" if kpis.error_issues else "")),
        kpi_card("shipped (queue)", str(kpis.content_shipped_total),
                 f"{kpis.content_shipped_30d} last 30d"),
        kpi_card("pending drafts", str(kpis.pending_drafts), "awaiting approval",
                 cls="warn" if kpis.pending_drafts else ""),
        kpi_card("spend (MTD)", _money(kpis.spend_mtd_usd), f"as of {dt.date.today().isoformat()}"),
    ])

    def movers_table(rows: list, kind: str) -> str:
        if not rows:
            return f'<p class="muted">No {kind} yet — need at least 2 rank captures to compute deltas.</p>'
        body = "".join(f"""
        <tr>
          <td>{html.escape(m.keyword)}</td>
          <td>{_format_rank(m.current_rank)}</td>
          <td>{_format_rank(m.prior_rank)}</td>
          <td>{_format_delta(m.delta)}</td>
        </tr>""" for m in rows)
        return f"""<table>
          <thead><tr><th>Keyword</th><th>Now</th><th>Prior</th><th>Δ</th></tr></thead>
          <tbody>{body}</tbody></table>"""

    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>{html.escape(site.display_name)} — SEO</title>
<style>{_BASE_CSS}</style></head><body>
{_site_nav(site.slug, "overview")}
<h1>{html.escape(site.display_name)} <span class="muted" style="font-size:14px">{html.escape(site.domain)}</span></h1>
<p class="muted">Last rank capture: {last_capture} · GSC: {'✓' if site.gsc_property else '—'} · GA4: {'✓' if site.ga4_property_id else '—'} · Brand profile: {'✓' if site.brand_profile_path else '—'}</p>
<div class="kpis">{kpi_strip}</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
  <div>
    <h2>top gainers</h2>
    {movers_table(gainers, "gainers")}
  </div>
  <div>
    <h2>top losers</h2>
    {movers_table(losers, "losers")}
  </div>
</div>
</body></html>"""


# =============================================================================
# Render: keywords table (/admin/seo/sites/{slug}/keywords)
# =============================================================================
def _render_keywords_screen(site, keywords: list) -> str:
    rows = []
    for k in keywords:
        target = (k.target_url or "")
        target_short = target.replace("https://", "").replace("http://", "")
        if len(target_short) > 50:
            target_short = "…" + target_short[-50:]
        rows.append(f"""
        <tr>
          <td><strong>{html.escape(k.keyword)}</strong>
              {(' <span class="tag type">' + html.escape(k.search_intent) + '</span>') if k.search_intent else ''}</td>
          <td>{k.search_volume or '—'}</td>
          <td>{k.keyword_difficulty if k.keyword_difficulty is not None else '—'}</td>
          <td>{_format_rank(k.current_rank)}</td>
          <td>{html.escape(target_short) if target_short else '<span class="muted">—</span>'}</td>
          <td class="muted">{_format_when(k.rank_checked_at)}</td>
        </tr>""")
    body = "\n".join(rows) or '<tr><td colspan=6 class=muted>No active keywords. Run scripts/seo_run_keyword_engine.py for this site.</td></tr>'
    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>{html.escape(site.display_name)} — keywords</title>
<style>{_BASE_CSS}</style></head><body>
{_site_nav(site.slug, "keywords")}
<h1>{html.escape(site.display_name)} — keywords</h1>
<p class="muted">{len(keywords)} active keyword{'' if len(keywords) == 1 else 's'} · sorted by search volume · column-sort coming when the SPA lands.</p>
<table>
  <thead><tr><th>Keyword</th><th>Vol/mo</th><th>KD</th><th>Rank</th><th>Target URL</th><th>Checked</th></tr></thead>
  <tbody>{body}</tbody>
</table>
</body></html>"""


# =============================================================================
# Render: rankings (/admin/seo/sites/{slug}/rankings)
# =============================================================================
def _sparkline(positions: list, width: int = 80, height: int = 18) -> str:
    """Mini SVG sparkline of position history. Lower y = better rank.
    positions: list of (date, position) tuples in chronological order.
    """
    if len(positions) < 2:
        return '<span class="muted">—</span>'
    # Map: None (not in top-100) → y = bottom (101). Cap viz at 100 for shape.
    vals = [(p if p is not None else 101) for _, p in positions]
    n = len(vals)
    vmin, vmax = min(vals), max(vals)
    span = max(1, vmax - vmin)
    pts = []
    for i, v in enumerate(vals):
        x = (i / (n - 1)) * width
        # Invert y so lower rank = higher pixel position
        y = ((v - vmin) / span) * height
        pts.append(f"{x:.1f},{y:.1f}")
    path = "M" + " L".join(pts)
    return (
        f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<path d="{path}" stroke="#B85C3D" stroke-width="1.5" fill="none"/>'
        '</svg>'
    )


def _render_rankings_screen(site, history_by_keyword: dict) -> str:
    """history_by_keyword: dict[str, list[tuple[date, Optional[int]]]] sorted asc by date."""
    if not history_by_keyword:
        return f"""<!doctype html><html><head><meta charset=utf-8><title>{html.escape(site.display_name)} — rankings</title>
<style>{_BASE_CSS}</style></head><body>
{_site_nav(site.slug, "rankings")}
<h1>{html.escape(site.display_name)} — rankings</h1>
<p class="muted">No rank captures yet. Run <code>python -m scripts.seo_rank_tracker --site {site.slug}</code> to seed the first capture.</p>
</body></html>"""

    captures_count = len(next(iter(history_by_keyword.values())))
    rows = []
    for keyword in sorted(history_by_keyword.keys()):
        hist = history_by_keyword[keyword]
        current = hist[-1][1]
        prior = hist[-2][1] if len(hist) >= 2 else None
        delta = (prior - current) if (prior is not None and current is not None) else None
        rows.append(f"""
        <tr>
          <td><strong>{html.escape(keyword)}</strong></td>
          <td>{_sparkline(hist)}</td>
          <td>{_format_rank(current)}</td>
          <td>{_format_rank(prior)}</td>
          <td>{_format_delta(delta)}</td>
          <td class="muted">{len(hist)} pts</td>
        </tr>""")
    body = "\n".join(rows)
    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>{html.escape(site.display_name)} — rankings</title>
<style>{_BASE_CSS}</style></head><body>
{_site_nav(site.slug, "rankings")}
<h1>{html.escape(site.display_name)} — rankings</h1>
<p class="muted">{len(history_by_keyword)} tracked keyword{'' if len(history_by_keyword) == 1 else 's'} · {captures_count} capture{'' if captures_count == 1 else 's'} so far · sparkline lights up after 2+ pulls (weekly cron runs Sundays 9 AM ET).</p>
<table>
  <thead><tr><th>Keyword</th><th>Trend</th><th>Now</th><th>Prior</th><th>Δ</th><th></th></tr></thead>
  <tbody>{body}</tbody>
</table>
</body></html>"""


# =============================================================================
# Render: audit (/admin/seo/sites/{slug}/audit)
# =============================================================================
def _render_audit_screen(site, issues_by_severity: dict) -> str:
    total = sum(len(v) for v in issues_by_severity.values())
    if not total:
        return f"""<!doctype html><html><head><meta charset=utf-8><title>{html.escape(site.display_name)} — audit</title>
<style>{_BASE_CSS}</style></head><body>
{_site_nav(site.slug, "audit")}
<h1>{html.escape(site.display_name)} — audit</h1>
<p class="muted">No open audit issues. Either a clean site (unlikely) or the audit hasn't run — try <code>python -m scripts.seo_run_audit --site {site.slug}</code>.</p>
</body></html>"""

    sections = []
    severity_order = ["error", "warning", "info"]
    for sev in severity_order:
        issues = issues_by_severity.get(sev, [])
        if not issues:
            continue
        # Group issues by type within the severity
        by_type: dict[str, list] = {}
        for iss in issues:
            by_type.setdefault(iss.issue_type, []).append(iss)
        # Sort types by count desc
        type_blocks = []
        for itype, rows in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            # Show first 12 pages per type to keep page sane
            page_rows = []
            for r in rows[:12]:
                url_short = (r.page_url or "").replace("https://", "").replace("http://", "")
                if len(url_short) > 70:
                    url_short = url_short[:67] + "…"
                detail = html.escape((r.detail or "")[:120])
                page_rows.append(f'<tr><td class="muted">{html.escape(url_short)}</td><td class="muted">{detail}</td></tr>')
            more = ""
            if len(rows) > 12:
                more = f'<tr><td colspan=2 class="muted">+{len(rows) - 12} more page(s) with this issue…</td></tr>'
            type_blocks.append(f"""
            <h3><span class="tag sev-{sev}">{sev}</span> {html.escape(itype)} <span class="muted">({len(rows)} page{'' if len(rows) == 1 else 's'})</span></h3>
            <table><tbody>{''.join(page_rows)}{more}</tbody></table>""")
        sections.append("\n".join(type_blocks))

    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>{html.escape(site.display_name)} — audit</title>
<style>{_BASE_CSS}</style></head><body>
{_site_nav(site.slug, "audit")}
<h1>{html.escape(site.display_name)} — audit</h1>
<p class="muted">{total} open issue{'' if total == 1 else 's'} · grouped by severity → type. Re-audit with <code>python -m scripts.seo_run_audit --site {site.slug}</code>.</p>
{''.join(sections)}
</body></html>"""


# =============================================================================
# Render: spend (/admin/seo/spend)
# =============================================================================
def _render_spend_screen(sb) -> str:
    # Pure SVG bar chart for daily spend
    chart_html = ""
    if sb.daily_30d:
        max_val = max((d.total_usd for d in sb.daily_30d), default=0.0)
        if max_val > 0:
            bars = []
            for d in sb.daily_30d:
                height_pct = (d.total_usd / max_val) * 100
                title = f"{d.day.isoformat()}: {_money(d.total_usd)}"
                bars.append(f'<div class="bar" style="height:{height_pct:.0f}%" title="{title}"></div>')
            chart_html = f"""
            <div class="barchart">{''.join(bars)}</div>
            <p class="muted" style="text-align:right;font-size:11px;">peak day: {_money(max_val)} · {len(sb.daily_30d)} active day{'' if len(sb.daily_30d) == 1 else 's'} in last 30</p>
            """

    def cost_table(rows: list) -> str:
        if not rows:
            return '<p class="muted">No spend in last 30 days.</p>'
        body = "".join(f"""
        <tr><td>{html.escape(r.label)}</td><td>{r.call_count}</td><td>{_money(r.total_usd)}</td></tr>
        """ for r in rows)
        return f"<table><thead><tr><th>Label</th><th>Calls</th><th>Total</th></tr></thead><tbody>{body}</tbody></table>"

    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>SEO — spend</title>
<style>{_BASE_CSS}</style></head><body>
{_top_nav()}
<h1>seo — api spend</h1>
<p class="muted">Last 30 days · DataForSEO + content writer + any other tracked APIs · drawn from seo_api_costs.</p>

<div class="kpis">
  <div class="kpi"><div class="label">total (last 30d)</div><div class="value">{_money(sb.total_30d_usd)}</div></div>
  <div class="kpi"><div class="label">month-to-date</div><div class="value">{_money(sb.total_mtd_usd)}</div></div>
  <div class="kpi"><div class="label">active days</div><div class="value">{len(sb.daily_30d)}</div><div class="sub">of last 30</div></div>
</div>

<h2>daily spend (last 30 days)</h2>
{chart_html or '<p class="muted">No spend recorded yet.</p>'}

<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
  <div>
    <h2>by purpose</h2>
    {cost_table(sb.by_purpose)}
  </div>
  <div>
    <h2>by site</h2>
    {cost_table(sb.by_site)}
  </div>
</div>
</body></html>"""


# =============================================================================
# Render: approval queue (unchanged from prior — kept verbatim except CSS)
# =============================================================================
def _render_pending_queue(items: list, sites_by_id: dict) -> str:
    cards: list[str] = []
    for it in items:
        site = sites_by_id.get(it.site_id)
        site_label = html.escape(site.display_name if site else f"site #{it.site_id}")
        body = it.body_payload or {}
        title = html.escape(body.get("title") or it.title or "Untitled")
        preview = html.escape((body.get("body") or "")[:240]).replace("\n", " ")
        if len(body.get("body") or "") > 240:
            preview += "…"
        ctype = html.escape(it.content_type or "?")
        signal = ""
        if it.source_signal:
            sig_text = ", ".join(f"{k}={v}" for k, v in (it.source_signal or {}).items() if v)
            signal = f'<div class="muted" style="font-size:12px;margin-bottom:12px;">signal: {html.escape(sig_text)}</div>'
        validation = body.get("validation") or {}
        v_badge = ""
        if validation:
            ok = validation.get("ok")
            issues = validation.get("issues") or []
            if ok:
                v_badge = f'<span class="tag" style="background:#e8f5e8;color:#2a7a4a">validated · Flesch {validation.get("flesch")} · {validation.get("word_count")}w</span>'
            else:
                v_badge = (
                    f'<span class="tag sev-warning">needs review · {", ".join(html.escape(i) for i in issues)}</span>'
                )
        cards.append(f"""
<div style="border:1px solid #eee;border-radius:8px;padding:16px 18px;margin-bottom:16px;background:#fff;">
  <div>
    <span class="tag site">{site_label}</span>
    <span class="tag type">{ctype}</span>
    {v_badge}
  </div>
  <h3>{title}</h3>
  <p style="color:#555;line-height:1.5;font-size:14px;margin:4px 0 12px;">{preview}</p>
  {signal}
  <form method="post" action="/admin/seo/pending/{it.id}/approve" style="display:inline-flex;gap:6px;align-items:center;margin:4px 6px 0 0;">
    <button type="submit" style="background:#2a7a4a;color:white;border:0;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:13px;" name="publish_status" value="draft">✓ Approve → WP draft</button>
    <button type="submit" style="background:#1a5a36;color:white;border:0;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:13px;" name="publish_status" value="publish">✓ Approve → publish live</button>
  </form>
  <form method="post" action="/admin/seo/pending/{it.id}/reject" style="display:inline-flex;gap:6px;align-items:center;margin:4px 6px 0 0;">
    <input name="reason" placeholder="reject reason (optional)" style="padding:6px 10px;border:1px solid #ddd;border-radius:4px;font-size:13px;min-width:220px;" />
    <button type="submit" style="background:#ffe5e5;color:#a02020;border:0;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:13px;">✗ Reject</button>
  </form>
</div>""")
    cards_html = "\n".join(cards) or '<p class="muted">Nothing pending. Generated drafts land here for approval.</p>'

    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>SEO — pending approvals</title>
<style>{_BASE_CSS}</style></head><body>
{_top_nav()}
<h1>seo — pending approvals</h1>
<p class="muted">{len(items)} draft{'' if len(items) == 1 else 's'} awaiting Mike. Approve → WP draft is the safe default; live publish skips preview.</p>
{cards_html}
</body></html>"""


def _render_new_brief_form(sites: list, *, default_site: str = "", default_ct: str = "blog") -> str:
    site_options = []
    for s in sites:
        sel = " selected" if s.slug == default_site else ""
        site_options.append(
            f'<option value="{html.escape(s.slug)}"{sel}>{html.escape(s.display_name)} ({html.escape(s.domain)})</option>'
        )
    site_options_html = "\n".join(site_options) or '<option value="">No sites with brand profiles yet</option>'

    ct_options = []
    for ct in CONTENT_TYPES:
        sel = " selected" if ct == default_ct else ""
        label = CONTENT_TYPE_LABELS.get(ct, ct)
        ct_options.append(f'<option value="{ct}"{sel}>{html.escape(label)}</option>')
    ct_options_html = "\n".join(ct_options)

    form_styles = """
      form { background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 24px; margin-top: 16px; }
      label { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px; color: #666; margin: 16px 0 6px; }
      input[type=text], textarea, select { width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; font-family: inherit; box-sizing: border-box; }
      textarea { min-height: 80px; resize: vertical; }
      .hint { color: #999; font-size: 12px; margin-top: 4px; }
      .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
      .warn { background: #fff5e0; color: #a06700; padding: 10px 14px; border-radius: 4px; font-size: 13px; margin: 16px 0; }
      .submit { font-size: 14px; padding: 10px 20px; border-radius: 4px; cursor: pointer; border: 0; background: #B85C3D; color: white; margin-top: 24px; font-weight: 600; }
    """

    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>SEO — new brief</title>
<style>{_BASE_CSS}{form_styles}</style></head><body>
{_top_nav()}
<h1>seo — new brief</h1>
<p class="muted">Hand a topic to the writer. Generation runs synchronously (30–180s) — leave the tab open. Draft lands in the pending queue.</p>

<form method="post" action="/admin/seo/new" onsubmit="document.getElementById('go').disabled=true; document.getElementById('go').textContent='Generating… leave tab open (30-180s)';">
  <div class="row2">
    <div>
      <label>Site</label>
      <select name="site_slug" required>{site_options_html}</select>
    </div>
    <div>
      <label>Content type</label>
      <select name="content_type" required>{ct_options_html}</select>
    </div>
  </div>

  <label>Topic</label>
  <textarea name="topic" required placeholder="e.g. How to choose a beaded bracelet that fits your everyday style"></textarea>
  <div class="hint">A sentence or two telling the writer what to write about. Be specific.</div>

  <label>Primary target keyword</label>
  <input type="text" name="target_keyword" required placeholder="e.g. beaded bracelet" />
  <div class="hint">Must appear in the first 100 words (skipped automatically for product enrichment).</div>

  <div class="row2">
    <div>
      <label>Title hint <span class="muted">(optional)</span></label>
      <input type="text" name="title_hint" placeholder="LLM may refine this" />
    </div>
    <div>
      <label>Secondary keywords <span class="muted">(optional, comma-sep)</span></label>
      <input type="text" name="extra_keywords" placeholder="layering, gift, atlanta" />
    </div>
  </div>

  <label>Audience override <span class="muted">(optional)</span></label>
  <input type="text" name="audience" placeholder="defaults to brand profile audience" />

  <div class="warn">⏳ Click once. The page hangs while Kimi generates — that's normal. You'll land on the pending queue when it's done.</div>

  <button type="submit" id="go" class="submit">Generate draft</button>
</form>
</body></html>"""


def _render_brief_error(brief: Brief, err: str) -> str:
    return f"""<!doctype html><html><head><meta charset=utf-8><title>SEO — brief error</title>
<style>{_BASE_CSS} pre{{background:#fff5e0;padding:14px;border-radius:4px;overflow:auto;font-size:12px}}</style>
</head><body>
{_top_nav()}
<h1>generation failed</h1>
<p>The writer threw an error mid-generation. Brief details:</p>
<pre>topic: {html.escape(brief.topic)}
content_type: {html.escape(brief.content_type)}
target_keyword: {html.escape(brief.target_keyword)}

error: {html.escape(err)}</pre>
<p class="muted">Try again, or check <code>journalctl -u alfred -n 50</code> for the full traceback.</p>
</body></html>"""


# =============================================================================
# Data-access helpers used by routes (pure SQL — no DFS / no LLM calls)
# =============================================================================
def _load_keywords(site_id: int) -> list:
    with SessionLocal() as s:
        return s.execute(
            text("""
                SELECT * FROM seo_keywords
                WHERE site_id = :sid AND status = 'active'
                ORDER BY search_volume DESC NULLS LAST, keyword ASC
            """),
            {"sid": site_id},
        ).all()


def _load_rankings_history(site_id: int) -> dict:
    """Return {keyword: [(date, position), ...]} sorted asc by date."""
    with SessionLocal() as s:
        rows = s.execute(
            text("""
                SELECT query, position, captured_at
                FROM seo_rankings_daily
                WHERE site_id = :sid
                ORDER BY query, captured_at
            """),
            {"sid": site_id},
        ).all()
    history: dict[str, list] = {}
    for r in rows:
        history.setdefault(r.query, []).append(
            (r.captured_at, int(r.position) if r.position is not None else None)
        )
    return history


def _load_audit_issues(site_id: int) -> dict:
    """Return {severity: [SeoAuditIssue, ...]} for open issues only."""
    with SessionLocal() as s:
        rows = s.execute(
            text("""
                SELECT * FROM seo_audit_issues
                WHERE site_id = :sid AND fixed_at IS NULL
                ORDER BY severity, issue_type, page_url
            """),
            {"sid": site_id},
        ).all()
    by_sev: dict[str, list] = {}
    for r in rows:
        by_sev.setdefault(r.severity, []).append(r)
    return by_sev


# =============================================================================
# Route registration
# =============================================================================
def register(app: FastAPI) -> None:

    # ---- Cross-site dashboard ----
    @app.get("/admin/seo", response_class=HTMLResponse)
    @app.get("/admin/seo/", response_class=HTMLResponse)
    async def admin_seo_index(user: dict | None = Depends(get_current_user)):
        if user is None:
            return RedirectResponse(url="/?returnTo=/admin/seo", status_code=303)
        sites = list_sites()
        return HTMLResponse(_render_dashboard(sites))

    @app.get("/admin/seo/sites")
    async def admin_seo_sites_json(user: dict = Depends(require_auth)):
        sites = list_sites()
        return JSONResponse([{
            "id": s.id,
            "slug": s.slug,
            "domain": s.domain,
            "display_name": s.display_name,
            "business_type": s.business_type,
            "status": s.status,
            "gsc_property": s.gsc_property,
            "ga4_property_id": s.ga4_property_id,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        } for s in sites])

    # ---- Per-site screens ----
    @app.get("/admin/seo/sites/{slug}", response_class=HTMLResponse)
    async def admin_seo_site_detail(slug: str, user: dict | None = Depends(get_current_user)):
        if user is None:
            return RedirectResponse(url=f"/?returnTo=/admin/seo/sites/{slug}", status_code=303)
        site = get_site_by_slug(slug)
        if not site:
            raise HTTPException(status_code=404, detail="site not found")
        kpis = site_kpis(site.id)
        gainers, losers = top_movers(site.id, limit=5)
        return HTMLResponse(_render_site_overview(site, kpis, gainers, losers))

    @app.get("/admin/seo/sites/{slug}/keywords", response_class=HTMLResponse)
    async def admin_seo_keywords(slug: str, user: dict | None = Depends(get_current_user)):
        if user is None:
            return RedirectResponse(url=f"/?returnTo=/admin/seo/sites/{slug}/keywords", status_code=303)
        site = get_site_by_slug(slug)
        if not site:
            raise HTTPException(status_code=404, detail="site not found")
        keywords = _load_keywords(site.id)
        return HTMLResponse(_render_keywords_screen(site, keywords))

    @app.get("/admin/seo/sites/{slug}/rankings", response_class=HTMLResponse)
    async def admin_seo_rankings(slug: str, user: dict | None = Depends(get_current_user)):
        if user is None:
            return RedirectResponse(url=f"/?returnTo=/admin/seo/sites/{slug}/rankings", status_code=303)
        site = get_site_by_slug(slug)
        if not site:
            raise HTTPException(status_code=404, detail="site not found")
        history = _load_rankings_history(site.id)
        return HTMLResponse(_render_rankings_screen(site, history))

    @app.get("/admin/seo/sites/{slug}/audit", response_class=HTMLResponse)
    async def admin_seo_audit(slug: str, user: dict | None = Depends(get_current_user)):
        if user is None:
            return RedirectResponse(url=f"/?returnTo=/admin/seo/sites/{slug}/audit", status_code=303)
        site = get_site_by_slug(slug)
        if not site:
            raise HTTPException(status_code=404, detail="site not found")
        issues = _load_audit_issues(site.id)
        return HTMLResponse(_render_audit_screen(site, issues))

    # ---- Cross-site spend ----
    @app.get("/admin/seo/spend", response_class=HTMLResponse)
    async def admin_seo_spend(user: dict | None = Depends(get_current_user)):
        if user is None:
            return RedirectResponse(url="/?returnTo=/admin/seo/spend", status_code=303)
        sb = spend_breakdown()
        return HTMLResponse(_render_spend_screen(sb))

    # ---- Approval queue (existing) ----
    @app.get("/admin/seo/pending", response_class=HTMLResponse)
    async def admin_seo_pending(user: dict | None = Depends(get_current_user)):
        if user is None:
            return RedirectResponse(url="/?returnTo=/admin/seo/pending", status_code=303)
        sites_by_id = {s.id: s for s in list_sites(include_inactive=True)}
        items = list_pending()
        return HTMLResponse(_render_pending_queue(items, sites_by_id))

    @app.post("/admin/seo/pending/{pending_id}/approve")
    async def admin_seo_pending_approve(
        pending_id: int,
        publish_status: str = Form("draft"),
        user: dict = Depends(require_auth),
    ):
        decided_by = user.get("email") or user.get("sub") or "admin"
        try:
            res = approve_and_publish(
                pending_id, decided_by=decided_by, publish_status=publish_status,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        if res.outcome == "publish_failed":
            return RedirectResponse(
                url=f"/admin/seo/pending?err={html.escape(res.error or 'publish failed', quote=True)}",
                status_code=303,
            )
        return RedirectResponse(
            url=f"/admin/seo/pending?ok=approved&wp_id={res.wp_post_id}",
            status_code=303,
        )

    @app.post("/admin/seo/pending/{pending_id}/reject")
    async def admin_seo_pending_reject(
        pending_id: int,
        reason: str = Form(""),
        user: dict = Depends(require_auth),
    ):
        decided_by = user.get("email") or user.get("sub") or "admin"
        try:
            reject(pending_id, decided_by=decided_by, reason=reason or None)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return RedirectResponse(url="/admin/seo/pending?ok=rejected", status_code=303)

    @app.get("/admin/seo/new", response_class=HTMLResponse)
    async def admin_seo_new_form(
        site: str = "",
        ct: str = "blog",
        user: dict | None = Depends(get_current_user),
    ):
        if user is None:
            return RedirectResponse(url="/?returnTo=/admin/seo/new", status_code=303)
        sites = list_sites()
        candidate_sites = [s for s in sites if s.brand_profile_path]
        return HTMLResponse(_render_new_brief_form(candidate_sites, default_site=site, default_ct=ct))

    @app.post("/admin/seo/new", response_class=HTMLResponse)
    async def admin_seo_new_submit(
        site_slug: str = Form(...),
        content_type: str = Form(...),
        topic: str = Form(...),
        target_keyword: str = Form(...),
        title_hint: str = Form(""),
        extra_keywords: str = Form(""),
        audience: str = Form(""),
        user: dict = Depends(require_auth),
    ):
        site = get_site_by_slug(site_slug)
        if not site:
            raise HTTPException(status_code=404, detail=f"unknown site: {site_slug}")
        if content_type not in CONTENT_TYPES:
            raise HTTPException(status_code=400, detail=f"unknown content_type: {content_type}")
        try:
            profile = load_profile(site_slug)
        except BrandProfileNotFound:
            raise HTTPException(status_code=400, detail=f"site {site_slug} has no brand profile yet")

        extras = [k.strip() for k in extra_keywords.split(",") if k.strip()]
        brief = Brief(
            topic=topic.strip(),
            content_type=content_type,
            target_keyword=target_keyword.strip(),
            audience=audience.strip() or None,
            title_hint=title_hint.strip() or None,
            extra_keywords=extras,
            source_signal="manual_brief",
        )

        decided_by = user.get("email") or user.get("sub") or "admin"
        logger.info("manual brief: site=%s ct=%s topic=%r by=%s",
                    site_slug, content_type, brief.topic[:60], decided_by)

        try:
            draft = generate_with_retry(brief, profile)
        except Exception as e:
            logger.exception("manual brief generation failed")
            return HTMLResponse(_render_brief_error(brief, str(e)), status_code=500)

        ct = canonicalize(brief.content_type)
        image_meta = {}
        if ct in {"blog", "cluster"}:
            try:
                imaged = compose_blog_images(
                    draft.body, site,
                    topic=brief.topic, target_keyword=brief.target_keyword,
                    inline_count=2, use_comfyui_hero=True,
                )
                draft.body = imaged.body
                image_meta = {
                    "featured_image_id": imaged.featured_image_id,
                    "featured_image_url": imaged.featured_image_url,
                    "image_ids": imaged.all_image_ids,
                    "inline_image_ids": imaged.inline_image_ids,
                }
                logger.info("manual brief: composed blog images, hero=%d inline=%d",
                            imaged.featured_image_id, len(imaged.inline_image_ids))
            except Exception:
                logger.exception("blog image composition failed; continuing text-only")
        elif ct == "ad_landing":
            try:
                imaged = compose_blog_images(
                    draft.body, site,
                    topic=brief.topic, target_keyword=brief.target_keyword,
                    inline_count=0, use_comfyui_hero=True,
                )
                draft.body = imaged.body
                image_meta = {
                    "featured_image_id": imaged.featured_image_id,
                    "featured_image_url": imaged.featured_image_url,
                    "image_ids": imaged.all_image_ids,
                    "inline_image_ids": [],
                }
            except Exception:
                logger.exception("ad_landing hero generation failed; continuing text-only")

        result = enqueue_draft(
            site.id,
            draft=draft,
            source_signal={
                "manual_by": decided_by,
                "topic": brief.topic[:120],
                "target_keyword": brief.target_keyword,
                **{k: v for k, v in image_meta.items() if k in {"featured_image_id"}},
            },
            meta_description=None,
        )
        if image_meta:
            from core.seo.models import SeoPending
            with SessionLocal() as s:
                row = s.get(SeoPending, result.pending_id)
                if row and row.body_payload is not None:
                    payload = dict(row.body_payload)
                    payload.update(image_meta)
                    payload["body"] = draft.body
                    row.body_payload = payload
                    s.commit()
        return RedirectResponse(
            url=f"/admin/seo/pending?ok=generated&pending_id={result.pending_id}",
            status_code=303,
        )
