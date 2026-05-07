"""Pure-Python port of the AI savings math.

Mirrors the JavaScript `compute()` in static/ai-savings-calc/index.html so the
PDF report, the API response, and the live frontend all show identical numbers.
"""

from core.audit.models import AIAuditLeadRequest, INDUSTRY_PRESETS, SavingsBreakdown


WEEKS_PER_YEAR = 50
AUTOMATION_RATE = 0.60      # Share of repetitive hours AI can absorb
TICKET_COST = 4.20          # Avg human ticket cost ($)
DEFLECTION_RATE = 0.55      # Share of tier-1 tickets resolved by AI
LEAD_RECOVERY_RATE = 0.22   # Share of slow-response leads AI rescues
LEAD_CLOSE_RATE = 0.08      # Conversion of recaptured leads to closed deals
DEFAULT_DEAL_VALUE = 250.0  # Used when monthly_leads is supplied without avg_customer_value
MISSED_CALL_VALUE_FACTOR = 0.30  # Share of missed calls that would have converted

# Cost of deploying a Ground Rush AI engagement, used for payback math
DEPLOY_COST_FLOOR = 15000
DEPLOY_COST_PCT_OF_SAVINGS = 0.18


def compute_savings(req: AIAuditLeadRequest) -> SavingsBreakdown:
    """Compute the savings breakdown for a lead submission.

    All figures are annualized and in USD (the funnel only ships USD for v1;
    multi-currency lives in Phase 2).
    """
    multiplier = INDUSTRY_PRESETS[req.industry]

    # 1. Labor reclamation
    labor = (
        req.hours_per_week
        * WEEKS_PER_YEAR
        * req.hourly_rate
        * AUTOMATION_RATE
        * req.headcount
    )

    # 2. Support deflection (only if they gave us tickets)
    support = req.support_tickets_monthly * 12 * TICKET_COST * DEFLECTION_RATE

    # 3a. Lead-speed lift from inbound forms
    lead_speed = (
        req.monthly_leads
        * 12
        * LEAD_RECOVERY_RATE
        * req.avg_customer_value
        * LEAD_CLOSE_RATE
    )

    # 3b. Missed-call rescue (the dealengine.io lever — high impact for SMB)
    missed_call_value = (
        req.missed_calls_monthly
        * 12
        * MISSED_CALL_VALUE_FACTOR
        * req.avg_customer_value
    )
    lead_total = lead_speed + missed_call_value

    subtotal = (labor + support + lead_total) * multiplier

    hours_reclaimed = req.hours_per_week * WEEKS_PER_YEAR * AUTOMATION_RATE * req.headcount
    leads_recaptured = req.monthly_leads * 12 * LEAD_RECOVERY_RATE

    deploy_cost = max(DEPLOY_COST_FLOOR, subtotal * DEPLOY_COST_PCT_OF_SAVINGS)
    payback_months = (
        max(2, round((deploy_cost / subtotal) * 12)) if subtotal > 0 else 0
    )

    return SavingsBreakdown(
        annual_total=round(subtotal, 2),
        monthly_total=round(subtotal / 12, 2) if subtotal else 0.0,
        labor_savings=round(labor * multiplier, 2),
        support_savings=round(support * multiplier, 2),
        lead_speed_savings=round(lead_total * multiplier, 2),
        hours_reclaimed=round(hours_reclaimed, 1),
        leads_recaptured=round(leads_recaptured, 0),
        payback_months=payback_months,
        industry_multiplier=multiplier,
    )


def fit_tier_for(savings: SavingsBreakdown) -> str:
    """Bucket savings into low/mid/high for routing + Telegram urgency."""
    monthly = savings.monthly_total
    if monthly < 1000:
        return "low"
    if monthly >= 5000:
        return "high"
    return "mid"
