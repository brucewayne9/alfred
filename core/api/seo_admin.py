# core/api/seo_admin.py
"""FastAPI routes for /admin/seo/*. Auth-gated via the existing JWT cookie.

Phase 1 routes: dashboard, site list, site detail. Approval queue, content
preview, and editing land in Plan 2.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from core.security.auth import get_current_user, require_auth
from core.seo.sites.registry import get_site_by_slug, list_sites

logger = logging.getLogger(__name__)


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


def _render_dashboard(sites: list) -> str:
    rows = []
    for s in sites:
        gsc = "✓" if s.gsc_property else "—"
        ga4 = "✓" if s.ga4_property_id else "—"
        brand = "✓" if s.brand_profile_path else "—"
        rows.append(f"""
        <tr>
          <td><a href="/admin/seo/sites/{s.slug}"><strong>{s.display_name}</strong></a><br><span class="muted">{s.domain}</span></td>
          <td>{s.business_type}</td>
          <td class="status-cell">{gsc}</td>
          <td class="status-cell">{ga4}</td>
          <td class="status-cell">{brand}</td>
          <td>{_format_when(s.updated_at)}</td>
        </tr>""")
    rows_html = "\n".join(rows) or '<tr><td colspan=6 class=muted>No sites registered yet. Run scripts/seo_init_roen.py to add the first.</td></tr>'

    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>SEO — sites</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }}
  h1 {{ font-weight: 200; letter-spacing: 1px; }}
  .muted {{ color: #999; font-size: 13px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid #eee; vertical-align: top; }}
  th {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; color: #666; }}
  .status-cell {{ text-align: center; color: #2a7a4a; font-weight: 600; }}
</style></head><body>
<h1>seo — cross-site dashboard</h1>
<p class="muted">{len(sites)} active sites · Plan 1 foundation · live data widgets arrive in Plan 2</p>
<table>
  <thead><tr><th>Site</th><th>Type</th><th>GSC</th><th>GA4</th><th>Brand</th><th>Updated</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<p class="muted" style="margin-top:24px">Each ✓ means that integration is configured. Empty cells need OAuth (GSC/GA4) or a brand profile YAML.</p>
</body></html>"""


def register(app: FastAPI) -> None:
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

    @app.get("/admin/seo/sites/{slug}", response_class=HTMLResponse)
    async def admin_seo_site_detail(slug: str, user: dict | None = Depends(get_current_user)):
        if user is None:
            return RedirectResponse(url=f"/?returnTo=/admin/seo/sites/{slug}", status_code=303)
        site = get_site_by_slug(slug)
        if not site:
            raise HTTPException(status_code=404, detail="site not found")
        return HTMLResponse(f"""<!doctype html>
<html><head><meta charset=utf-8><title>{site.display_name} — SEO</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;}}h1{{font-weight:200;}}dt{{font-weight:600;margin-top:8px;}}dd{{margin-left:0;color:#444;}}</style>
</head><body>
<p><a href="/admin/seo">&larr; all sites</a></p>
<h1>{site.display_name}</h1>
<dl>
  <dt>Domain</dt><dd>{site.domain}</dd>
  <dt>WP REST</dt><dd>{site.wp_rest_url}</dd>
  <dt>Business type</dt><dd>{site.business_type}</dd>
  <dt>GSC property</dt><dd>{site.gsc_property or '<em>not set</em>'}</dd>
  <dt>GA4 property</dt><dd>{site.ga4_property_id or '<em>not set</em>'}</dd>
  <dt>Brand profile</dt><dd>{site.brand_profile_path or '<em>not set</em>'}</dd>
  <dt>Status</dt><dd>{site.status}</dd>
</dl>
<p class="muted" style="margin-top:24px;color:#999;">Data widgets (queries, pages, CWV, backlinks) land in Plan 2.</p>
</body></html>""")
