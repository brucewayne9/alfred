# core/api/seo_admin.py
"""FastAPI routes for /admin/seo/*. Auth-gated via the existing JWT cookie.

Phase 1: dashboard, site list, site detail.
Plan 2: approval queue (/admin/seo/pending) for human review of generated drafts.
"""
from __future__ import annotations

import html
import logging
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from core.security.auth import get_current_user, require_auth
from core.seo.content.types import CONTENT_TYPE_LABELS, CONTENT_TYPES
from core.seo.content.writer import Brief, generate_with_retry
from core.seo.queue.pending import (
    approve_and_publish,
    enqueue_draft,
    get_pending,
    list_pending,
    reject,
)
from core.seo.sites.profile import BrandProfileNotFound, load_profile
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

    # ----- Plan 2: Approval queue -----

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
        # Filter to sites that actually have a brand profile path on disk —
        # we can't generate copy for sites without voice encoding.
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

        result = enqueue_draft(
            site.id,
            draft=draft,
            source_signal={
                "manual_by": decided_by,
                "topic": brief.topic[:120],
                "target_keyword": brief.target_keyword,
            },
        )
        return RedirectResponse(
            url=f"/admin/seo/pending?ok=generated&pending_id={result.pending_id}",
            status_code=303,
        )


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
            signal = f'<div class="signal">signal: {html.escape(sig_text)}</div>'
        validation = body.get("validation") or {}
        v_badge = ""
        if validation:
            ok = validation.get("ok")
            issues = validation.get("issues") or []
            if ok:
                v_badge = f'<span class="vb ok">validated · Flesch {validation.get("flesch")} · {validation.get("word_count")}w</span>'
            else:
                v_badge = (
                    f'<span class="vb warn">needs review · {", ".join(html.escape(i) for i in issues)}</span>'
                )
        cards.append(f"""
<div class="card">
  <div class="row">
    <span class="tag site">{site_label}</span>
    <span class="tag type">{ctype}</span>
    {v_badge}
  </div>
  <h3>{title}</h3>
  <p class="preview">{preview}</p>
  {signal}
  <form method="post" action="/admin/seo/pending/{it.id}/approve" class="actions">
    <button type="submit" class="approve" name="publish_status" value="draft">✓ Approve → WP draft</button>
    <button type="submit" class="approve publish" name="publish_status" value="publish">✓ Approve → publish live</button>
  </form>
  <form method="post" action="/admin/seo/pending/{it.id}/reject" class="actions">
    <input name="reason" placeholder="reject reason (optional)" />
    <button type="submit" class="reject">✗ Reject</button>
  </form>
</div>""")
    cards_html = "\n".join(cards) or '<p class="muted">Nothing pending. Generated drafts land here for approval.</p>'

    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>SEO — pending approvals</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 920px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }}
  h1 {{ font-weight: 200; letter-spacing: 1px; }}
  .muted {{ color: #999; font-size: 13px; }}
  nav a {{ color: #1a1a1a; text-decoration: none; border-bottom: 1px dotted #999; }}
  .card {{ border: 1px solid #eee; border-radius: 8px; padding: 16px 18px; margin-bottom: 16px; background: #fff; }}
  .row {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
  .tag {{ display: inline-block; font-size: 11px; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.4px; }}
  .tag.site {{ background: #B85C3D; color: white; }}
  .tag.type {{ background: #f0f0f0; color: #444; }}
  .vb {{ font-size: 11px; padding: 3px 8px; border-radius: 4px; }}
  .vb.ok {{ background: #e8f5e8; color: #2a7a4a; }}
  .vb.warn {{ background: #fff5e0; color: #a06700; }}
  h3 {{ margin: 12px 0 4px; font-weight: 600; font-size: 17px; }}
  .preview {{ color: #555; line-height: 1.5; font-size: 14px; margin: 4px 0 12px; }}
  .signal {{ font-size: 12px; color: #888; margin-bottom: 12px; }}
  .actions {{ display: inline-flex; gap: 6px; align-items: center; margin: 4px 6px 0 0; }}
  .actions input {{ padding: 6px 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; min-width: 220px; }}
  button {{ font-size: 13px; padding: 6px 14px; border-radius: 4px; cursor: pointer; border: 0; }}
  .approve {{ background: #2a7a4a; color: white; }}
  .approve.publish {{ background: #1a5a36; }}
  .reject {{ background: #ffe5e5; color: #a02020; }}
</style></head><body>
<p><nav><a href="/admin/seo">&larr; sites</a> · <a href="/admin/seo/new">+ new brief</a></nav></p>
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

    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>SEO — new brief</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 720px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }}
  h1 {{ font-weight: 200; letter-spacing: 1px; }}
  .muted {{ color: #999; font-size: 13px; }}
  nav a {{ color: #1a1a1a; text-decoration: none; border-bottom: 1px dotted #999; margin-right: 12px; }}
  form {{ background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 24px; margin-top: 16px; }}
  label {{ display: block; font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px; color: #666; margin: 16px 0 6px; }}
  input[type=text], textarea, select {{ width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; font-family: inherit; box-sizing: border-box; }}
  textarea {{ min-height: 80px; resize: vertical; }}
  .hint {{ color: #999; font-size: 12px; margin-top: 4px; }}
  .row2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  button {{ font-size: 14px; padding: 10px 20px; border-radius: 4px; cursor: pointer; border: 0; background: #B85C3D; color: white; margin-top: 24px; font-weight: 600; }}
  button:disabled {{ opacity: 0.6; cursor: wait; }}
  .warn {{ background: #fff5e0; color: #a06700; padding: 10px 14px; border-radius: 4px; font-size: 13px; margin: 16px 0; }}
</style></head><body>
<p><nav><a href="/admin/seo">&larr; sites</a><a href="/admin/seo/pending">pending queue</a></nav></p>
<h1>seo — new brief</h1>
<p class="muted">Hand a topic to the writer. Generation runs synchronously and can take 30–180 seconds — leave the tab open. The draft lands in the pending queue when done.</p>

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

  <button type="submit" id="go">Generate draft</button>
</form>
</body></html>"""


def _render_brief_error(brief: Brief, err: str) -> str:
    return f"""<!doctype html><html><head><meta charset=utf-8><title>SEO — brief error</title>
<style>body{{font-family:-apple-system,sans-serif;max-width:720px;margin:24px auto;padding:0 16px}}
h1{{font-weight:200}} pre{{background:#fff5e0;padding:14px;border-radius:4px;overflow:auto;font-size:12px}}</style>
</head><body>
<p><a href="/admin/seo/new">&larr; back to brief form</a></p>
<h1>generation failed</h1>
<p>The writer threw an error mid-generation. Brief details:</p>
<pre>topic: {html.escape(brief.topic)}
content_type: {html.escape(brief.content_type)}
target_keyword: {html.escape(brief.target_keyword)}

error: {html.escape(err)}</pre>
<p class="muted">Try again, or check <code>journalctl -u alfred -n 50</code> for the full traceback.</p>
</body></html>"""
