"""Fan out an AI Audit submission to Twenty CRM, Brevo, email, and Telegram in parallel.

Single entrypoint: `process_audit_lead(lead, savings, booking_url) -> dict`

All integrations are wrapped in run_in_executor so one slow/failing call does not
block the others. Each task's outcome is captured into a results dict so the
caller (the FastAPI endpoint) can decide how much to surface vs swallow.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import requests

from config.settings import settings
from core.audit.calculator import fit_tier_for
from core.audit.models import AIAuditLeadRequest, INDUSTRY_PRESETS, SavingsBreakdown
from core.audit.pdf_report import generate_audit_pdf
from integrations.base_crm import client as crm
from integrations.brevo import client as brevo
from integrations.email.client import EmailClient

logger = logging.getLogger(__name__)

TWENTY_BASE_UI = "https://crm.groundrushlabs.com"

CONSUMER_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "aol.com"}


# ============================================================================
# Individual fan-out tasks (sync; called via run_in_executor)
# ============================================================================

def _twenty_upsert(lead: AIAuditLeadRequest, savings: SavingsBreakdown) -> dict:
    """Find-or-create Person + append a Note with the calc inputs and savings.

    Idempotent on email: dup submission with same email appends a 2nd Note
    rather than creating a 2nd Person.
    """
    existing = crm.search_people(lead.email, limit=5)
    person_id: str | None = None
    for p in existing:
        if p.get("email", "").lower() == lead.email.lower():
            person_id = p.get("id")
            break

    if person_id:
        action = "updated"
        logger.info(f"Twenty: found existing person {person_id} for {lead.email}; appending note")
    else:
        person = crm.create_person(
            first_name=lead.first_name,
            last_name=lead.last_name,
            email=lead.email,
            phone=lead.phone,
            job_title="",
        )
        person_id = person.get("id")
        action = "created"
        logger.info(f"Twenty: created person {person_id} for {lead.email}")

    if not person_id:
        raise RuntimeError("Twenty CRM did not return a person ID")

    note_title = f"AI Savings Audit — ${savings.annual_total:,.0f}/yr ({fit_tier_for(savings)} fit)"
    note_body = _format_note_body(lead, savings)
    crm.create_note_for_person(title=note_title, person_id=person_id, body=note_body)

    return {
        "status": "ok",
        "action": action,
        "person_id": person_id,
        "person_url": f"{TWENTY_BASE_UI}/object/person/{person_id}",
    }


def _format_note_body(lead: AIAuditLeadRequest, savings: SavingsBreakdown) -> str:
    """Markdown-flavored note body — Twenty renders these as plain text but the
    structure still scans cleanly in the UI."""
    parts = [
        f"**Submitted:** {lead.first_name} {lead.last_name}",
        f"**Company:** {lead.company_name}" + (f" ({lead.company_website})" if lead.company_website else ""),
        f"**Email:** {lead.email}",
        f"**Phone:** {lead.phone}",
        "",
        f"**Industry:** {lead.industry} ({INDUSTRY_PRESETS[lead.industry]:.2f}× multiplier)",
        f"**Headcount:** {lead.headcount}",
        f"**Hourly rate:** ${lead.hourly_rate:.0f}",
        f"**Hours/wk repeatable:** {lead.hours_per_week}",
        f"**Missed calls/mo:** {lead.missed_calls_monthly}",
        f"**Avg customer value:** ${lead.avg_customer_value:.0f}",
    ]
    if lead.support_tickets_monthly:
        parts.append(f"**Support tickets/mo:** {lead.support_tickets_monthly}")
    if lead.monthly_leads:
        parts.append(f"**Inbound leads/mo:** {lead.monthly_leads}")

    parts += [
        "",
        "## Savings",
        f"- Annual: ${savings.annual_total:,.0f}",
        f"- Monthly: ${savings.monthly_total:,.0f}",
        f"- Labor: ${savings.labor_savings:,.0f}",
        f"- Support: ${savings.support_savings:,.0f}",
        f"- Lead/missed-call: ${savings.lead_speed_savings:,.0f}",
        f"- Hours reclaimed/yr: {savings.hours_reclaimed:,.0f}",
        f"- Payback: {savings.payback_months} months",
        "",
        f"**Fit tier:** {fit_tier_for(savings)}",
    ]

    if lead.utm_source or lead.utm_medium or lead.utm_campaign:
        parts.append("")
        parts.append("## Tracking")
        for k in ("utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"):
            v = getattr(lead, k, None)
            if v:
                parts.append(f"- {k}: {v}")
        if lead.referrer:
            parts.append(f"- referrer: {lead.referrer}")
        if lead.page_url:
            parts.append(f"- page_url: {lead.page_url}")
        if lead.variant:
            parts.append(f"- variant: {lead.variant}")

    return "\n".join(parts)


def _brevo_upsert(lead: AIAuditLeadRequest, savings: SavingsBreakdown) -> dict:
    """Add or update the lead in Brevo, attached to the AI-audit list (if configured)."""
    list_id = settings.brevo_ai_audit_list_id
    list_ids = [list_id] if list_id else None
    # Standard attributes only — custom AI_AUDIT_* attrs require predefining in
    # Brevo's contact attributes panel. Phase 2 (#11 nurture sequence) covers
    # creating those + the list itself.
    attributes = {
        "FIRSTNAME": lead.first_name,
        "LASTNAME": lead.last_name,
        "COMPANY": lead.company_name,
    }

    return brevo.add_or_update_contact(
        email=lead.email,
        attributes=attributes,
        list_ids=list_ids,
    )


def _send_audit_email(
    lead: AIAuditLeadRequest, savings: SavingsBreakdown, booking_url: str, pdf_bytes: bytes
) -> dict:
    """Email the PDF to the lead from alfred@groundrushinc.com."""
    subject = f"Your AI Savings Audit — ${savings.annual_total:,.0f}/yr"
    body_html = _audit_email_html(lead, savings, booking_url)

    return EmailClient().send_email(
        account="alfred-gw",
        to=lead.email,
        subject=subject,
        body=body_html,
        html=True,
        attachments=[{
            "filename": f"AI-Savings-Audit-{lead.company_name.replace(' ', '-')}.pdf",
            "content": pdf_bytes,
            "mimetype": "application/pdf",
        }],
        reply_to="hello@groundrushlabs.com",
    )


def _audit_email_html(lead: AIAuditLeadRequest, savings: SavingsBreakdown, booking_url: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<body style="font-family: -apple-system, system-ui, sans-serif; color: #111; max-width: 600px; margin: 0 auto; padding: 24px; line-height: 1.55;">
  <p style="font-size: 11px; letter-spacing: 0.16em; color: #FF6B35; text-transform: uppercase; font-weight: 700;">GroundRush · AI Savings Audit</p>
  <h1 style="font-size: 24px; line-height: 1.2; margin: 6px 0 16px 0;">Hi {lead.first_name} — your numbers are attached.</h1>

  <p>Quick recap of what the calculator surfaced for {lead.company_name}:</p>

  <table style="border-collapse: collapse; margin: 16px 0; width: 100%;">
    <tr>
      <td style="padding: 12px 16px; background: #fafafa; border: 1px solid #eee; width: 50%;">
        <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.08em;">Annual</div>
        <div style="font-size: 22px; font-weight: 800; color: #FF6B35;">${savings.annual_total:,.0f}</div>
      </td>
      <td style="padding: 12px 16px; background: #fafafa; border: 1px solid #eee;">
        <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.08em;">Monthly</div>
        <div style="font-size: 22px; font-weight: 800; color: #111;">${savings.monthly_total:,.0f}</div>
      </td>
    </tr>
  </table>

  <p>The full breakdown — labor reclamation, missed-call rescue, lead-speed lift — is in the attached PDF. It's a 4-page audit, no fluff.</p>

  <p><strong>What I'd do next, if I were you:</strong> book 15 minutes. I walk through your numbers, show you which agent plugs the biggest leak first for {lead.company_name}, and if it's a fit we scope a custom build.</p>

  <p style="background: #fafafa; border-left: 3px solid #FF6B35; padding: 12px 14px; font-size: 14px; color: #444;">
    <strong>What it costs if you say yes:</strong> $997 one-time setup + $697/mo retainer. We do the build, the training, the ongoing tuning. No pitch deck, no pressure.
  </p>

  <p style="margin: 28px 0;">
    <a href="{booking_url}" style="display: inline-block; background: #FF6B35; color: #000; font-weight: 700; padding: 14px 24px; border-radius: 999px; text-decoration: none;">Book your 15-min AI Audit Call →</a>
  </p>

  <p style="font-size: 13px; color: #666; margin-top: 32px;">
    — Mike Johnson<br>
    GroundRush<br>
    <a href="mailto:hello@groundrushlabs.com" style="color: #666;">hello@groundrushlabs.com</a>
  </p>
</body>
</html>"""


def _telegram_ping(lead: AIAuditLeadRequest, savings: SavingsBreakdown, twenty_url: str | None) -> dict:
    """Notify Mike on Telegram. Suppress when fit is 'low' to avoid noise."""
    fit = fit_tier_for(savings)
    if fit == "low":
        logger.info(f"Skipping Telegram for low-fit lead {lead.email}")
        return {"status": "skipped", "reason": "low-fit"}

    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        return {"status": "skipped", "reason": "telegram-not-configured"}

    prefix = "🔥 " if fit == "high" else ""
    is_consumer = lead.email.split("@", 1)[-1] in CONSUMER_DOMAINS
    consumer_tag = " · consumer-email" if is_consumer else ""

    text = (
        f"{prefix}*New AI Audit lead* — {fit.upper()} fit{consumer_tag}\n"
        f"\n"
        f"*{lead.first_name} {lead.last_name}* · {lead.company_name}\n"
        f"📧 {lead.email}\n"
        f"📞 {lead.phone}\n"
        f"🏢 {lead.industry} · {lead.headcount} people\n"
        f"\n"
        f"💰 *${savings.annual_total:,.0f}/yr* (${savings.monthly_total:,.0f}/mo)\n"
        f"⏱  Payback: {savings.payback_months} months\n"
    )
    if twenty_url:
        text += f"\n[Open in Twenty]({twenty_url})"
    if lead.utm_source:
        text += f"\n_via {lead.utm_source}"
        if lead.utm_campaign:
            text += f" / {lead.utm_campaign}"
        text += "_"

    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        },
        timeout=15,
    )
    if resp.status_code != 200:
        logger.error(f"Telegram ping failed {resp.status_code}: {resp.text[:300]}")
        return {"status": "error", "code": resp.status_code, "body": resp.text[:300]}
    return {"status": "sent"}


# ============================================================================
# Async coordinator
# ============================================================================

async def _to_thread(fn, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(None, lambda: fn(*args, **kwargs))


async def process_audit_lead(
    lead: AIAuditLeadRequest,
    savings: SavingsBreakdown,
    booking_url: str,
) -> dict[str, Any]:
    """Fan out one lead submission across all integrations. Returns per-integration results."""
    # Generate the PDF first (synchronous, ~1-2s in a thread).
    pdf_task = asyncio.create_task(_to_thread(generate_audit_pdf, lead, savings, booking_url))

    # Twenty + Brevo can run in parallel.
    twenty_task = asyncio.create_task(_to_thread(_twenty_upsert, lead, savings))
    brevo_task = asyncio.create_task(_to_thread(_brevo_upsert, lead, savings))

    twenty_result, brevo_result, pdf_bytes = await asyncio.gather(
        twenty_task, brevo_task, pdf_task, return_exceptions=True
    )

    # Email depends on the PDF bytes; Telegram depends on the Twenty URL.
    email_task: asyncio.Task | None = None
    if not isinstance(pdf_bytes, BaseException):
        email_task = asyncio.create_task(
            _to_thread(_send_audit_email, lead, savings, booking_url, pdf_bytes)
        )

    twenty_url = (
        twenty_result.get("person_url")
        if isinstance(twenty_result, dict)
        else None
    )
    telegram_task = asyncio.create_task(_to_thread(_telegram_ping, lead, savings, twenty_url))

    email_result: Any
    if email_task is not None:
        email_result, telegram_result = await asyncio.gather(
            email_task, telegram_task, return_exceptions=True
        )
    else:
        telegram_result = await telegram_task
        email_result = {"status": "skipped", "reason": f"pdf-failed: {pdf_bytes!r}"}

    def _normalize(name: str, val: Any) -> dict:
        if isinstance(val, BaseException):
            logger.exception(f"Audit fan-out: {name} failed", exc_info=val)
            return {"status": "error", "error": f"{type(val).__name__}: {val}"}
        return val if isinstance(val, dict) else {"status": "ok", "value": val}

    results = {
        "twenty": _normalize("twenty", twenty_result),
        "brevo": _normalize("brevo", brevo_result),
        "email": _normalize("email", email_result),
        "telegram": _normalize("telegram", telegram_result),
        "pdf_bytes": (
            len(pdf_bytes) if isinstance(pdf_bytes, (bytes, bytearray)) else None
        ),
    }

    return results
