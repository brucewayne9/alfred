"""Render the AI Savings Audit as a branded PDF using WeasyPrint."""
from __future__ import annotations

import base64
import logging
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from core.audit.models import AIAuditLeadRequest, INDUSTRY_PRESETS, SavingsBreakdown

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path("/home/aialfred/alfred/core/audit/templates")
LOGO_PATH = Path("/home/aialfred/alfred/static/grl-site/grl-logo.png")

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
)

_INDUSTRY_LABELS = {
    "services": "Professional services",
    "ecommerce": "E-commerce / Retail",
    "saas": "SaaS / Software",
    "trades": "Trades / Field services",
    "real_estate": "Real estate",
    "agency": "Agency / Marketing",
}


def _logo_b64() -> str:
    """Return base64 of the GR logo so WeasyPrint inlines it (no network fetch)."""
    if not LOGO_PATH.exists():
        logger.warning(f"Logo missing at {LOGO_PATH}; PDF will render without it")
        return ""
    return base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")


def generate_audit_pdf(
    lead: AIAuditLeadRequest,
    savings: SavingsBreakdown,
    booking_url: str,
) -> bytes:
    """Render the lead's AI Savings Audit as a multi-page PDF, returning raw bytes."""
    template = _env.get_template("audit_report.html")
    html_str = template.render(
        # contact
        first_name=lead.first_name,
        last_name=lead.last_name,
        email=lead.email,
        company_name=lead.company_name,
        # inputs
        industry_label=_INDUSTRY_LABELS.get(lead.industry, lead.industry.title()),
        headcount=lead.headcount,
        hourly_rate=lead.hourly_rate,
        hours_per_week=lead.hours_per_week,
        missed_calls_monthly=lead.missed_calls_monthly,
        avg_customer_value=lead.avg_customer_value,
        support_tickets_monthly=lead.support_tickets_monthly,
        monthly_leads=lead.monthly_leads,
        # math
        savings=savings,
        # branding / cta
        logo_b64=_logo_b64(),
        booking_url=booking_url,
        today_human=datetime.now().strftime("%B %d, %Y"),
    )
    return HTML(string=html_str, base_url=str(TEMPLATE_DIR)).write_pdf()


def write_audit_pdf(
    out_path: Path,
    lead: AIAuditLeadRequest,
    savings: SavingsBreakdown,
    booking_url: str,
) -> Path:
    """Render and write to disk. Returns out_path."""
    out_path = Path(out_path)
    out_path.write_bytes(generate_audit_pdf(lead, savings, booking_url))
    return out_path
