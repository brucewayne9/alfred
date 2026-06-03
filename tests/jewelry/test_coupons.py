"""Parser tests for the Roen bot friends & family coupon flow.

parse_coupon_request turns Sarah's plain-English request into a CouponSpec.
Pure function, no I/O — every example phrase from the design doc is covered here.
"""
from __future__ import annotations

from datetime import date

import pytest

from core.jewelry.coupons import (
    CouponSpec,
    normalize_code,
    parse_coupon_request,
)

TODAY = date(2026, 6, 3)


# ---------------- normalize_code ----------------

@pytest.mark.parametrize("raw,expected", [
    ("Brittany", "BRITTANY"),
    (" brittany ", "BRITTANY"),
    ("my sister", "MY-SISTER"),
    ("Brittany!", "BRITTANY"),
    ("Best Friend 4 Ever", "BEST-FRIEND-4-EVER"),
    ("José", "JOS"),  # non-ascii stripped
])
def test_normalize_code(raw, expected):
    assert normalize_code(raw) == expected


# ---------------- discount type + amount ----------------

def test_percent_off():
    spec = parse_coupon_request("create a coupon 20% off Brittany")
    assert spec.discount_type == "percent"
    assert spec.amount == 20
    assert spec.code == "BRITTANY"
    assert spec.usage_limit is None
    assert spec.free_shipping is False
    assert spec.expiry_date is None


def test_percent_word_form():
    spec = parse_coupon_request("create a coupon 20 percent off Brittany")
    assert spec.discount_type == "percent"
    assert spec.amount == 20
    assert spec.code == "BRITTANY"


def test_fixed_dollar_off():
    spec = parse_coupon_request("create a coupon $100 off Mom")
    assert spec.discount_type == "fixed_cart"
    assert spec.amount == 100
    assert spec.code == "MOM"


def test_fixed_dollar_word_form():
    spec = parse_coupon_request("$15 off Mom")
    assert spec.discount_type == "fixed_cart"
    assert spec.amount == 15
    assert spec.code == "MOM"


def test_free_means_hundred_percent():
    spec = parse_coupon_request("create a free coupon Sarah")
    assert spec.discount_type == "percent"
    assert spec.amount == 100
    assert spec.code == "SARAH"


def test_hundred_percent_off():
    spec = parse_coupon_request("100% off Sarah")
    assert spec.discount_type == "percent"
    assert spec.amount == 100
    assert spec.code == "SARAH"


def test_free_shipping():
    spec = parse_coupon_request("create a coupon free shipping Kelly")
    assert spec.free_shipping is True
    assert spec.amount == 0
    assert spec.code == "KELLY"


# ---------------- name handling ----------------

def test_missing_name_returns_spec_with_none_code():
    spec = parse_coupon_request("create a 20% off coupon")
    assert spec.discount_type == "percent"
    assert spec.amount == 20
    assert spec.code is None


def test_name_with_spaces_before_discount():
    spec = parse_coupon_request("create a coupon for my sister 20% off")
    assert spec.amount == 20
    assert spec.code == "MY-SISTER"


# ---------------- no discount intent ----------------

def test_no_discount_returns_none():
    assert parse_coupon_request("create a coupon") is None
    assert parse_coupon_request("make me a coupon please") is None


# ---------------- usage limit ----------------

def test_one_time_sets_single_use():
    spec = parse_coupon_request("20% off Brittany one time")
    assert spec.usage_limit == 1
    assert spec.code == "BRITTANY"


def test_single_use_phrase():
    spec = parse_coupon_request("free shipping Mom single use")
    assert spec.usage_limit == 1
    assert spec.free_shipping is True
    assert spec.code == "MOM"


def test_unlimited_by_default():
    spec = parse_coupon_request("20% off Brittany")
    assert spec.usage_limit is None


# ---------------- expiry ----------------

def test_expiry_good_for_a_month():
    spec = parse_coupon_request("20% off Brittany good for a month", today=TODAY)
    assert spec.expiry_date == "2026-07-03"


def test_expiry_n_days():
    spec = parse_coupon_request("20% off Kelly 30 days", today=TODAY)
    assert spec.expiry_date == "2026-07-03"


def test_expiry_a_week():
    spec = parse_coupon_request("20% off Kelly good for a week", today=TODAY)
    assert spec.expiry_date == "2026-06-10"


def test_expiry_absolute_month_name():
    spec = parse_coupon_request("20% off Kelly expires June 30", today=TODAY)
    assert spec.expiry_date == "2026-06-30"
    assert spec.code == "KELLY"


def test_expiry_absolute_iso():
    spec = parse_coupon_request("20% off Kelly expires 2026-12-25", today=TODAY)
    assert spec.expiry_date == "2026-12-25"


def test_expiry_does_not_pollute_code():
    spec = parse_coupon_request("20% off Brittany good for a month", today=TODAY)
    assert spec.code == "BRITTANY"


# ---------------- combined ----------------

def test_everything_at_once():
    spec = parse_coupon_request(
        "create a coupon 25% off Brittany one time good for a week", today=TODAY
    )
    assert spec.discount_type == "percent"
    assert spec.amount == 25
    assert spec.code == "BRITTANY"
    assert spec.usage_limit == 1
    assert spec.expiry_date == "2026-06-10"
