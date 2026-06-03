"""
Roen Handmade friends & family coupons.

Two halves:
  - parse_coupon_request(text)  — pure parser, no I/O. Turns Sarah's plain-English
    request ("20% off Brittany good for a month") into a CouponSpec.
  - create/list/delete coupons  — thin wrappers over wp-cli, reusing the same
    SSH-into-the-container helper the /orders flow uses (core.jewelry.orders._wp).

The phrase space is small and bounded, so parsing is deterministic regex — no LLM
call. Faster, free, and predictable for Sarah.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

from core.jewelry.orders import _wp  # shared SSH+docker wp-cli helper

logger = logging.getLogger(__name__)


class CouponError(RuntimeError):
    """Raised when a wp-cli coupon operation fails."""


@dataclass
class CouponSpec:
    discount_type: str               # "percent" | "fixed_cart"
    amount: float                    # 20, 15, 100, or 0 (free-shipping-only)
    code: Optional[str]              # normalized code, or None if Sarah hasn't named it yet
    usage_limit: Optional[int] = None   # 1 == single use; None == unlimited
    free_shipping: bool = False
    expiry_date: Optional[str] = None   # "YYYY-MM-DD" or None


# ----------------------- parsing -----------------------

# Words dropped when extracting the code name (the leftover after discount/modifier
# tokens are stripped). "my" is intentionally kept — "my sister" -> MY-SISTER.
_STOPWORDS = {
    "create", "creates", "make", "makes", "made", "me", "a", "an", "the",
    "coupon", "coupons", "code", "for", "to", "off", "please", "new", "good",
    "expires", "expire", "in", "of", "and", "with",
}

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_WORD_NUMS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
              "five": 5, "six": 6}


def normalize_code(name: str) -> str:
    """Uppercase, hyphenate spaces, strip anything that isn't A-Z 0-9 - _."""
    s = name.strip().upper()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^A-Z0-9_-]", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def _extract_expiry(text: str, today: date) -> tuple[Optional[str], str]:
    """Pull an expiry out of `text`. Returns (iso_date_or_None, text_without_match)."""
    # Absolute ISO: "expires 2026-12-25"
    m = re.search(r"expires?\s+(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1), text[:m.start()] + " " + text[m.end():]

    # Absolute month name: "expires June 30"
    m = re.search(r"expires?\s+([a-z]+)\s+(\d{1,2})", text)
    if m and m.group(1) in _MONTHS:
        month = _MONTHS[m.group(1)]
        day = int(m.group(2))
        year = today.year
        try:
            d = date(year, month, day)
            if d < today:
                d = date(year + 1, month, day)
            return d.isoformat(), text[:m.start()] + " " + text[m.end():]
        except ValueError:
            pass

    # Relative: "good for a month" / "30 days" / "a week" / "in 2 weeks"
    m = re.search(r"\b(\d+|a|an|one|two|three|four|five|six)\s+(day|week|month)s?\b", text)
    if m:
        n_raw = m.group(1)
        n = int(n_raw) if n_raw.isdigit() else _WORD_NUMS.get(n_raw, 1)
        unit = m.group(2)
        days = {"day": 1, "week": 7, "month": 30}[unit] * n
        d = today + timedelta(days=days)
        return d.isoformat(), text[:m.start()] + " " + text[m.end():]

    return None, text


def parse_coupon_request(text: str, today: Optional[date] = None) -> Optional[CouponSpec]:
    """Parse a coupon request. Returns None if no discount intent is present
    (so the bot can ask "what discount?"). A spec with code=None means Sarah
    gave a valid discount but hasn't named it yet."""
    today = today or date.today()
    remaining = " " + (text or "").lower().strip() + " "

    free_shipping = False
    if "free shipping" in remaining:
        free_shipping = True
        remaining = remaining.replace("free shipping", " ")

    discount_type: Optional[str] = None
    amount: Optional[float] = None

    # percent: "20%" or "20 percent"
    m = re.search(r"(\d{1,3})\s*(?:%|percent)", remaining)
    if m:
        discount_type = "percent"
        amount = float(m.group(1))
        remaining = remaining[:m.start()] + " " + remaining[m.end():]
    else:
        # dollar: "$15" or "15 dollars"
        m = re.search(r"\$\s*(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s*dollars?", remaining)
        if m:
            discount_type = "fixed_cart"
            amount = float(m.group(1) or m.group(2))
            remaining = remaining[:m.start()] + " " + remaining[m.end():]

    # "free" / "comp" with no explicit number == 100% off
    if discount_type is None and not free_shipping:
        if re.search(r"\bfree\b", remaining) or re.search(r"\bcomp(?:ed|s)?\b", remaining):
            discount_type = "percent"
            amount = 100.0
            remaining = re.sub(r"\bfree\b", " ", remaining)
            remaining = re.sub(r"\bcomp(?:ed|s)?\b", " ", remaining)

    if discount_type is None:
        if free_shipping:
            discount_type = "percent"
            amount = 0.0
        else:
            return None

    # usage limit
    usage_limit: Optional[int] = None
    if (re.search(r"\bone[\s-]?time\b", remaining)
            or "single use" in remaining or "single-use" in remaining
            or re.search(r"\bonce\b", remaining)):
        usage_limit = 1
        remaining = re.sub(r"\bone[\s-]?time\b", " ", remaining)
        remaining = remaining.replace("single use", " ").replace("single-use", " ")
        remaining = re.sub(r"\bonce\b", " ", remaining)

    # expiry
    expiry_date, remaining = _extract_expiry(remaining, today)

    # whatever's left, minus stopwords and stray punctuation, is the code name
    cleaned = re.sub(r"[^\w\s-]", " ", remaining)
    words = [w for w in cleaned.split() if w not in _STOPWORDS]
    code = normalize_code(" ".join(words)) if words else None
    code = code or None  # empty string -> None

    return CouponSpec(
        discount_type=discount_type,
        amount=amount,
        code=code,
        usage_limit=usage_limit,
        free_shipping=free_shipping,
        expiry_date=expiry_date,
    )


# ----------------------- wp-cli wrappers -----------------------

def _fmt_amount(amount: float) -> str:
    """WooCommerce wants a plain decimal string; drop the trailing .0 for whole numbers."""
    if amount == int(amount):
        return str(int(amount))
    return f"{amount:.2f}"


def coupon_exists(code: str) -> bool:
    """True if a coupon with this exact code already exists."""
    rc, out, err = _wp(["wc", "shop_coupon", "list", "--code=" + code,
                        "--field=id", "--user=1"], timeout=20)
    if rc != 0:
        raise CouponError(f"coupon list failed: {err.strip()[:200]}")
    return bool(out.strip())


def create_coupon(spec: CouponSpec) -> str:
    """Create the coupon in WooCommerce. Returns the new coupon id. Caller must
    have set spec.code (uniqueness checked separately via coupon_exists)."""
    if not spec.code:
        raise CouponError("coupon has no code")
    args = [
        "wc", "shop_coupon", "create",
        "--code=" + spec.code,
        "--discount_type=" + spec.discount_type,
        "--amount=" + _fmt_amount(spec.amount),
        "--individual_use=true",
        "--user=1", "--porcelain",
    ]
    if spec.free_shipping:
        args.append("--free_shipping=true")
    if spec.usage_limit:
        args.append("--usage_limit=" + str(spec.usage_limit))
    if spec.expiry_date:
        args.append("--date_expires=" + spec.expiry_date)
    rc, out, err = _wp(args, timeout=30)
    if rc != 0:
        raise CouponError(f"coupon create failed: {err.strip()[:200]}")
    return out.strip()


@dataclass
class CouponRow:
    id: int
    code: str
    discount_type: str
    amount: str
    free_shipping: bool
    usage_limit: Optional[int]
    usage_count: int
    date_expires: Optional[str]


def list_coupons() -> List[CouponRow]:
    """All coupons, newest first."""
    rc, out, err = _wp([
        "wc", "shop_coupon", "list", "--user=1", "--format=json",
        "--fields=id,code,discount_type,amount,free_shipping,usage_limit,usage_count,date_expires",
        "--orderby=id", "--order=desc",
    ], timeout=25)
    if rc != 0:
        raise CouponError(f"coupon list failed: {err.strip()[:200]}")
    try:
        raw = json.loads(out) if out.strip() else []
    except json.JSONDecodeError as e:
        raise CouponError(f"could not parse coupon list: {e}; head={out[:200]!r}")
    rows: List[CouponRow] = []
    for c in raw:
        exp = c.get("date_expires") or None
        if isinstance(exp, str) and exp:
            exp = exp.split("T")[0]
        rows.append(CouponRow(
            id=int(c.get("id", 0)),
            code=str(c.get("code", "")),
            discount_type=str(c.get("discount_type", "")),
            amount=str(c.get("amount", "0")),
            free_shipping=str(c.get("free_shipping", "")).lower() in ("1", "true", "yes"),
            usage_limit=int(c["usage_limit"]) if c.get("usage_limit") not in (None, "", "0") else None,
            usage_count=int(c.get("usage_count") or 0),
            date_expires=exp,
        ))
    return rows


def delete_coupon(code: str) -> bool:
    """Permanently delete the coupon with this code. Returns False if not found."""
    rc, out, err = _wp(["wc", "shop_coupon", "list", "--code=" + code,
                        "--field=id", "--user=1"], timeout=20)
    if rc != 0:
        raise CouponError(f"coupon lookup failed: {err.strip()[:200]}")
    cid = out.strip().splitlines()[0].strip() if out.strip() else ""
    if not cid:
        return False
    rc, out, err = _wp(["wc", "shop_coupon", "delete", cid, "--force=true", "--user=1"], timeout=25)
    if rc != 0:
        raise CouponError(f"coupon delete failed: {err.strip()[:200]}")
    return True
