"""Pydantic models for the AI Savings Audit funnel."""

import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator


_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


INDUSTRY_PRESETS = {
    "services": 1.00,
    "ecommerce": 1.15,
    "saas": 1.25,
    "trades": 0.90,
    "real_estate": 1.10,
    "agency": 1.20,
}


class AIAuditLeadRequest(BaseModel):
    """Submitted from the AI Savings Calculator landing page."""

    # Calculator inputs
    industry: str = Field(default="services", description="One of INDUSTRY_PRESETS keys")
    headcount: int = Field(ge=1, le=200)
    hourly_rate: float = Field(ge=15, le=150)
    hours_per_week: float = Field(ge=1, le=40)
    missed_calls_monthly: int = Field(ge=0, le=5000, default=0)
    avg_customer_value: float = Field(ge=0, le=100000, default=500)

    # Optional advanced inputs (kept for parity with original calculator)
    support_tickets_monthly: int = Field(ge=0, le=5000, default=0)
    monthly_leads: int = Field(ge=0, le=5000, default=0)

    # Contact (required)
    first_name: str = Field(min_length=1, max_length=80)
    last_name: str = Field(min_length=1, max_length=80)
    email: str = Field(min_length=5, max_length=200)
    phone: str = Field(min_length=7, max_length=30)
    company_name: str = Field(min_length=1, max_length=120)
    company_website: Optional[str] = Field(default=None, max_length=200)

    # Tracking (hidden form fields)
    utm_source: Optional[str] = Field(default=None, max_length=120)
    utm_medium: Optional[str] = Field(default=None, max_length=120)
    utm_campaign: Optional[str] = Field(default=None, max_length=120)
    utm_term: Optional[str] = Field(default=None, max_length=120)
    utm_content: Optional[str] = Field(default=None, max_length=120)
    referrer: Optional[str] = Field(default=None, max_length=500)
    page_url: Optional[str] = Field(default=None, max_length=500)
    variant: Optional[str] = Field(default=None, max_length=10)

    @field_validator("email")
    @classmethod
    def email_must_be_valid(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("invalid email address")
        return v

    @field_validator("industry")
    @classmethod
    def industry_must_be_known(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in INDUSTRY_PRESETS:
            raise ValueError(f"industry must be one of {list(INDUSTRY_PRESETS.keys())}")
        return v

    @field_validator("phone")
    @classmethod
    def normalize_phone(cls, v: str) -> str:
        # Keep digits and leading +; strip everything else
        cleaned = "".join(c for c in v if c.isdigit() or c == "+")
        if len(cleaned) < 7:
            raise ValueError("phone too short after normalization")
        return cleaned


class SavingsBreakdown(BaseModel):
    annual_total: float
    monthly_total: float
    labor_savings: float
    support_savings: float
    lead_speed_savings: float
    hours_reclaimed: float
    leads_recaptured: float
    payback_months: int                # Months until setup fee pays itself back from monthly_net
    industry_multiplier: float
    # GroundRush hybrid-pricing economics — what the customer actually nets
    gr_setup_fee: float = 0.0          # One-time setup fee
    gr_monthly_fee: float = 0.0        # Recurring monthly retainer
    monthly_net: float = 0.0           # monthly_total - gr_monthly_fee (can be negative)
    annual_net_year_one: float = 0.0   # monthly_net*12 - setup_fee
    annual_net_recurring: float = 0.0  # monthly_net*12 (year 2+)


class AIAuditResponse(BaseModel):
    status: str
    booking_url: str
    savings: SavingsBreakdown
    fit_tier: str  # "low" | "mid" | "high"
    message: str
