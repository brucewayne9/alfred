"""Admin /admin/roen/social-pending — FB Page draft approval gate.

Mirrors the rucktalk admin pattern. Reads pending drafts written by the
Roen Telegram publish flow and lets Mike approve / reject each one.
Approval posts to the roenhandmade Facebook Page via meta_roen.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from core.jewelry.social_queue import (
    approve_and_post,
    list_pending,
    list_recent_decisions,
    reject,
)
from core.security.auth import get_current_user, require_auth

logger = logging.getLogger(__name__)


def _format_when(iso: str) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        if delta.total_seconds() < 60:
            return "just now"
        if delta.total_seconds() < 3600:
            return f"{int(delta.total_seconds() / 60)}m ago"
        if delta.total_seconds() < 86400:
            return f"{int(delta.total_seconds() / 3600)}h ago"
        return dt.strftime("%b %d %H:%M")
    except Exception:
        return iso


def _esc(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render(pending: list[dict], decisions: list[dict]) -> str:
    pending_cards = []
    for d in pending:
        did = d.get("draft_id", "")
        name = _esc(d.get("product_name", ""))
        price = _esc(d.get("product_price", ""))
        url = _esc(d.get("product_url", ""))
        img = _esc(d.get("image_url", ""))
        caption = _esc(d.get("caption", "")).replace("\n", "<br>")
        when = _format_when(d.get("created_at", ""))
        pending_cards.append(f"""
        <div class="card">
          <img src="{img}" alt="{name}" />
          <div class="meta">
            <div class="title">{name}</div>
            <div class="sub">{price} · <a href="{url}" target="_blank" rel="noopener">view on site</a></div>
            <div class="when">submitted {when}</div>
            <div class="caption">{caption}</div>
            <div class="actions">
              <form method="post" action="/admin/roen/decide" style="display:inline;">
                <input type="hidden" name="draft_id" value="{did}" />
                <input type="hidden" name="action" value="approve" />
                <button class="btn btn-approve">Approve &amp; post to FB + IG →</button>
              </form>
              <form method="post" action="/admin/roen/decide" style="display:inline;">
                <input type="hidden" name="draft_id" value="{did}" />
                <input type="hidden" name="action" value="reject" />
                <button class="btn btn-reject">Reject</button>
              </form>
            </div>
          </div>
        </div>""")

    if not pending_cards:
        pending_html = '<div class="empty">No drafts in queue. Sarah hasn\'t published anything that needs your eyes.</div>'
    else:
        pending_html = "\n".join(pending_cards)

    decisions_rows = []
    for d in decisions:
        status = d.get("status", "?")
        name = _esc(d.get("product_name", ""))
        when = _format_when(d.get("decided_at") or d.get("created_at", ""))
        fb_id = d.get("fb_post_id")
        ig_id = d.get("ig_media_id")
        links = []
        if fb_id:
            links.append(f'<a href="https://www.facebook.com/{fb_id}" target="_blank" rel="noopener">FB</a>')
        if ig_id:
            links.append(f'<a href="https://www.instagram.com/" target="_blank" rel="noopener">IG</a>')
        err = _esc(d.get("error") or d.get("fb_error") or d.get("ig_error") or d.get("reason") or "")
        cell = " · ".join(links) if links else err
        decisions_rows.append(
            f"<tr><td>{when}</td><td class=\"st-{status}\">{status}</td>"
            f"<td>{name}</td><td>{cell}</td></tr>"
        )
    decisions_html = "\n".join(decisions_rows) or "<tr><td colspan=4 class=muted>no history yet</td></tr>"

    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>Roen — social queue</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 980px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }}
  h1 {{ font-weight: 200; letter-spacing: 1px; }}
  h2 {{ font-weight: 400; margin-top: 36px; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
  .card {{ display: grid; grid-template-columns: 180px 1fr; gap: 16px; border: 1px solid #e5e5e5; border-radius: 6px; padding: 12px; margin-bottom: 14px; }}
  .card img {{ width: 180px; height: 180px; object-fit: cover; border-radius: 4px; }}
  .title {{ font-size: 18px; font-weight: 600; }}
  .sub {{ color: #666; margin-top: 2px; }}
  .when {{ color: #999; font-size: 13px; margin-top: 4px; }}
  .caption {{ background: #fafafa; padding: 10px; border-radius: 4px; margin-top: 10px; white-space: pre-wrap; line-height: 1.4; }}
  .actions {{ margin-top: 12px; }}
  .btn {{ border: none; padding: 8px 14px; border-radius: 4px; cursor: pointer; font-size: 14px; margin-right: 8px; }}
  .btn-approve {{ background: #B85C3D; color: white; }}
  .btn-reject {{ background: #eee; color: #333; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; }}
  .st-posted {{ color: #2a7; font-weight: 600; }}
  .st-rejected {{ color: #999; }}
  .st-failed {{ color: #c33; font-weight: 600; }}
  .empty, .muted {{ color: #999; }}
</style></head><body>
<h1>roen — social queue</h1>
<h2>Pending ({len(pending)})</h2>
{pending_html}
<h2>Recent decisions</h2>
<table>
  <thead><tr><th>When</th><th>Status</th><th>Product</th><th>Link / note</th></tr></thead>
  <tbody>{decisions_html}</tbody>
</table>
</body></html>"""


def register(app: FastAPI) -> None:
    @app.get("/admin/roen", response_class=HTMLResponse)
    @app.get("/admin/roen/social-pending", response_class=HTMLResponse)
    async def admin_roen_pending(user: dict | None = Depends(get_current_user)):
        if user is None:
            return RedirectResponse(url="/?returnTo=/admin/roen/social-pending", status_code=303)
        try:
            pending = list_pending()
        except Exception:
            logger.exception("roen admin: list_pending failed")
            pending = []
        try:
            decisions = list_recent_decisions(limit=50)
        except Exception:
            logger.exception("roen admin: recent_decisions failed")
            decisions = []
        return HTMLResponse(_render(pending, decisions))

    @app.post("/admin/roen/decide")
    async def admin_roen_decide(
        draft_id: str = Form(...),
        action: str = Form(...),
        user: dict = Depends(require_auth),
    ):
        if action == "approve":
            result = approve_and_post(draft_id)
            if not result:
                return RedirectResponse(url="/admin/roen/social-pending?err=not_found", status_code=303)
            logger.info("roen approve %s by %s — status=%s", draft_id, user.get("username"), result.get("status"))
        elif action == "reject":
            result = reject(draft_id, reason=f"rejected by {user.get('username','admin')}")
            if not result:
                return RedirectResponse(url="/admin/roen/social-pending?err=not_found", status_code=303)
            logger.info("roen reject %s by %s", draft_id, user.get("username"))
        else:
            return RedirectResponse(url="/admin/roen/social-pending?err=bad_action", status_code=303)
        return RedirectResponse(url="/admin/roen/social-pending", status_code=303)
