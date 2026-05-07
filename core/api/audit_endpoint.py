"""POST /api/forms/ai-audit — public lead-capture endpoint for the AI Savings Calculator.

Registered onto the main FastAPI app via `register(app, limiter)` from main.py.

Flow:
  1. Validate input via AIAuditLeadRequest Pydantic
  2. Compute savings via core.audit.calculator
  3. Fan out to Twenty + Brevo + Email (with PDF attachment) + Telegram in parallel
  4. Return booking link + savings summary so the frontend can unblur the result
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError
from slowapi import Limiter

from config.settings import settings
from core.audit.calculator import compute_savings, fit_tier_for
from core.audit.lead_router import process_audit_lead
from core.audit.models import AIAuditLeadRequest, AIAuditResponse, SavingsBreakdown

logger = logging.getLogger(__name__)


def _booking_url() -> str:
    return settings.cal_booking_link


def register(app: FastAPI, limiter: Limiter) -> None:
    """Mount audit endpoints on the FastAPI app."""

    @app.post("/api/forms/ai-audit", response_model=AIAuditResponse)
    @limiter.limit("10/hour")
    async def submit_ai_audit(request: Request) -> AIAuditResponse:
        try:
            payload = await request.json()
        except Exception as e:
            logger.warning(f"AI audit: invalid JSON: {e}")
            raise HTTPException(status_code=400, detail="Body must be valid JSON")

        try:
            lead = AIAuditLeadRequest(**payload)
        except ValidationError as ve:
            raise HTTPException(status_code=422, detail=ve.errors())

        savings = compute_savings(lead)
        booking_url = _booking_url()
        fit = fit_tier_for(savings)

        # Fan out — never raise from individual integration failures.
        try:
            results = await process_audit_lead(lead, savings, booking_url)
        except Exception as e:
            # Catastrophic failure of the coordinator itself (not individual tasks).
            logger.exception("AI audit fan-out coordinator failed", exc_info=e)
            results = {"twenty": {"status": "error"}, "brevo": {"status": "error"},
                       "email": {"status": "error"}, "telegram": {"status": "error"}}

        # Log a one-line summary so it's easy to scan in journalctl.
        logger.info(
            f"ai-audit lead: {lead.email} · {lead.company_name} · "
            f"${savings.annual_total:,.0f}/yr ({fit}) · "
            f"twenty={results.get('twenty', {}).get('status')} "
            f"brevo={results.get('brevo', {}).get('status')} "
            f"email={results.get('email', {}).get('status')} "
            f"telegram={results.get('telegram', {}).get('status')}"
        )

        message = (
            "Your AI Savings Audit is on its way to your inbox. "
            "Book your 15-min audit call when you're ready."
        )

        return AIAuditResponse(
            status="ok",
            booking_url=booking_url,
            savings=savings,
            fit_tier=fit,
            message=message,
        )

    @app.get("/api/forms/ai-audit/health")
    async def ai_audit_health():
        """Simple health check — confirms the module loads + booking URL is set."""
        return {
            "status": "ok",
            "booking_url_configured": bool(_booking_url()),
            "brevo_list_id_configured": bool(settings.brevo_ai_audit_list_id),
        }
