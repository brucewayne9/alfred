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

# GroundRush hybrid pricing — single source of truth for the calculator + PDF + email
GR_SETUP_FEE = 997      # One-time setup
GR_MONTHLY_FEE = 697    # Recurring retainer

# Fit tier thresholds — based on monthly savings vs the monthly fee
# A "high" fit needs ~3× the fee in savings (Darryn's healthy-customer ratio)
HIGH_FIT_MONTHLY_FLOOR = 5000   # ≥ $5k/mo savings = strong fit, 🔥 ping
MID_FIT_MONTHLY_FLOOR = 2100    # $2.1k–$5k/mo = workable, normal ping
# Below MID_FIT_MONTHLY_FLOOR (~3× monthly fee), customer can't profit on the
# subscription. We still capture the lead but mark it "low" (no Telegram).


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

    monthly_total = round(subtotal / 12, 2) if subtotal else 0.0

    # Hybrid-pricing economics: customer pays GR_MONTHLY_FEE/mo + GR_SETUP_FEE once.
    # Net to customer = their monthly savings minus our monthly fee.
    monthly_net = monthly_total - GR_MONTHLY_FEE
    if monthly_net > 0:
        # Setup fee pays back as the customer accrues monthly_net surplus
        setup_payback_months = max(1, round(GR_SETUP_FEE / monthly_net))
    else:
        # Customer can't profit on the subscription — math is upside-down
        setup_payback_months = 0

    annual_net_year_one = round(monthly_net * 12 - GR_SETUP_FEE, 2)
    annual_net_recurring = round(monthly_net * 12, 2)

    return SavingsBreakdown(
        annual_total=round(subtotal, 2),
        monthly_total=monthly_total,
        labor_savings=round(labor * multiplier, 2),
        support_savings=round(support * multiplier, 2),
        lead_speed_savings=round(lead_total * multiplier, 2),
        hours_reclaimed=round(hours_reclaimed, 1),
        leads_recaptured=round(leads_recaptured, 0),
        payback_months=setup_payback_months,
        industry_multiplier=multiplier,
        # New hybrid-pricing fields
        gr_setup_fee=GR_SETUP_FEE,
        gr_monthly_fee=GR_MONTHLY_FEE,
        monthly_net=round(monthly_net, 2),
        annual_net_year_one=annual_net_year_one,
        annual_net_recurring=annual_net_recurring,
    )


def fit_tier_for(savings: SavingsBreakdown) -> str:
    """Bucket savings into low/mid/high for routing + Telegram urgency.

    Uses the customer's monthly savings against our monthly fee. If their
    savings can't cover ~3× our fee (MID_FIT_MONTHLY_FLOOR = $2,100/mo), the
    subscription is upside-down for them — mark "low" and don't ping Mike.
    """
    monthly = savings.monthly_total
    if monthly < MID_FIT_MONTHLY_FLOOR:
        return "low"
    if monthly >= HIGH_FIT_MONTHLY_FLOOR:
        return "high"
    return "mid"
