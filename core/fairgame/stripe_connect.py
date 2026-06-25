"""Fair Game — Stripe Connect Express + held-payment escrow.

The ticket moves on Ticketmaster's rails (see tm_transfer.py); the money moves
here on Stripe. They never touch: Fair Game HOLDS the buyer's funds (a manual-
capture PaymentIntent) until the TM transfer is confirmed, then releases the
seat price to the seller and keeps Rod's flat $5 fee. If the transfer never
completes, the held funds are refunded to the buyer and the seller gets nothing.

>>> DEFAULTS TO A SIMULATOR. <<<
If ``FAIRGAME_STRIPE_SIM=1`` (the default whenever ``FAIRGAME_STRIPE_KEY`` is
unset) NOTHING hits the network — the escrow state machine is modelled entirely
against the local sqlite tables so the resale flow can be built, tested, and
demoed with no keys. Set ``FAIRGAME_STRIPE_KEY`` (a *test-mode* key) AND
``FAIRGAME_STRIPE_SIM=0`` to drive the real Stripe SDK; the live branch uses
manual-capture PaymentIntents (hold), capture (release), and refund.

Money is integer cents everywhere. Every function is idempotent — calling it
twice for the same order/seller is a no-op that returns the same row/ref.

Schema is owned by the Foundation phase (core/fairgame/db.py); this module only
reads/writes the ``connect_accounts`` and ``orders`` tables, never alters them.

Escrow state machine (orders.state). ``release``/``refund`` accept the orders
module's ``'paid'`` vocabulary as well as our own ``'held'`` so the orders flow
can drive a payout/refund directly, with no observable intermediate state:
    (no row | pending) --create_held_payment--> 'held'
            'held' | 'paid' --release_to_seller--> 'released'
            'held' | 'paid' --refund-----------> 'refunded'
"""
from __future__ import annotations

import os
import time
import uuid

from . import db


class StripeError(Exception):
    """Raised on an illegal escrow transition or a Stripe configuration error."""


def _sim_mode() -> bool:
    """True when the in-memory/db-backed simulator should be used.

    Simulator is the default: it is ON unless the operator both provides a real
    key (``FAIRGAME_STRIPE_KEY``) and explicitly opts out (``FAIRGAME_STRIPE_SIM=0``).
    """
    sim = os.environ.get("FAIRGAME_STRIPE_SIM")
    if sim is not None:
        return sim.strip() not in ("0", "false", "False", "")
    # No explicit flag: simulate unless a real key is present.
    return not os.environ.get("FAIRGAME_STRIPE_KEY")


# Public alias used by the API layer — keeps the underscore internals clean.
is_sim = _sim_mode


def create_unlock_checkout(amount_cents: int) -> str:
    """Create a Stripe Checkout Session for the $1 Discover unlock (live mode only).

    Returns the hosted checkout URL. Only called when ``is_sim()`` is False.
    Raises ``StripeError`` if ``FAIRGAME_STRIPE_KEY`` is unset.
    """
    stripe = _stripe()
    base = os.environ.get(
        "FAIRGAME_PUBLIC_BASE", "https://aialfred.groundrushcloud.com"
    ).rstrip("/")
    session = stripe.checkout.Session.create(
        mode="payment",
        line_items=[{
            "price_data": {
                "currency": "usd",
                "unit_amount": amount_cents,
                "product_data": {"name": "Fans First — Verified Ticket Access"},
            },
            "quantity": 1,
        }],
        success_url=f"{base}/fairgame/app/discover.html?unlocked=1",
        cancel_url=f"{base}/fairgame/app/discover.html",
    )
    return session["url"]


def create_credit_checkout(pack_id: str, credits: int, amount_cents: int,
                           fan_id: str, label: str) -> str:
    """Checkout Session for a Discover credit pack (live mode only). Carries the
    fan + pack in metadata so the success handler can grant the credits."""
    stripe = _stripe()
    base = os.environ.get(
        "FAIRGAME_PUBLIC_BASE", "https://aialfred.groundrushcloud.com"
    ).rstrip("/")
    session = stripe.checkout.Session.create(
        mode="payment",
        line_items=[{
            "price_data": {
                "currency": "usd",
                "unit_amount": amount_cents,
                "product_data": {"name": f"Fans First — {label} ({credits} credits)"},
            },
            "quantity": 1,
        }],
        metadata={"fan_id": fan_id, "pack": pack_id, "credits": str(credits)},
        success_url=f"{base}/fairgame/app/discover.html?credits_session={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base}/fairgame/app/discover.html",
    )
    return session["url"]


def retrieve_checkout(session_id: str) -> dict:
    """Verify a credit-pack checkout with Stripe. Returns
    {paid, fan_id, pack, credits, amount_cents}."""
    stripe = _stripe()
    s = stripe.checkout.Session.retrieve(session_id)
    md = s.get("metadata") or {}
    return {
        "paid": s.get("payment_status") == "paid",
        "fan_id": md.get("fan_id"),
        "pack": md.get("pack"),
        "credits": int(md.get("credits") or 0),
        "amount_cents": s.get("amount_total"),
    }


def _stripe():
    """Return a configured live Stripe client, or raise if no key is set."""
    key = os.environ.get("FAIRGAME_STRIPE_KEY")
    if not key:
        raise StripeError("FAIRGAME_STRIPE_KEY required for live Stripe mode")
    import stripe  # local import: never needed in sim mode
    stripe.api_key = key
    return stripe


def _row_to_dict(row):
    return dict(row) if row else None


# --------------------------------------------------------------------------- #
# Seller onboarding (Stripe Connect Express)
# --------------------------------------------------------------------------- #

def onboard_seller(fan_id: str) -> dict:
    """Create (or return) the seller's Connect Express account + onboarding URL.

    Idempotent: a fan keeps one Connect account for life. Returns a dict with the
    ``connect_accounts`` row fields plus an ``onboarding_url`` the fan visits to
    finish Stripe KYC. In sim mode the account id and URL are synthesized; in live
    mode this calls Stripe Account + AccountLink create.
    """
    if not fan_id:
        raise StripeError("fan_id required")
    now = int(time.time())
    with db.connect() as c:
        row = c.execute(
            "SELECT * FROM connect_accounts WHERE fan_id=?", (fan_id,)
        ).fetchone()
        if row is not None:
            acct = row["stripe_account_id"]
            onboarded = row["onboarded"]
        else:
            if _sim_mode():
                acct = "acct_sim_" + uuid.uuid4().hex[:16]
            else:
                stripe = _stripe()
                created = stripe.Account.create(
                    type="express",
                    capabilities={"transfers": {"requested": True}},
                    metadata={"fairgame_fan_id": fan_id},
                )
                acct = created["id"]
            onboarded = 0
            c.execute(
                "INSERT INTO connect_accounts(fan_id,stripe_account_id,onboarded,created_at) "
                "VALUES(?,?,?,?)",
                (fan_id, acct, onboarded, now),
            )
        out = c.execute(
            "SELECT * FROM connect_accounts WHERE fan_id=?", (fan_id,)
        ).fetchone()

    result = dict(out)
    if _sim_mode():
        result["onboarding_url"] = f"https://connect.fairgame.test/onboard/{acct}"
    else:
        stripe = _stripe()
        base = os.environ.get(
            "FAIRGAME_PUBLIC_BASE", "https://aialfred.groundrushcloud.com"
        ).rstrip("/")
        link = stripe.AccountLink.create(
            account=acct,
            refresh_url=f"{base}/fairgame/app/onboard?retry=1",
            return_url=f"{base}/fairgame/app/onboard?done=1",
            type="account_onboarding",
        )
        result["onboarding_url"] = link["url"]
    return result


def mark_onboarded(fan_id: str) -> dict | None:
    """Flip a seller's Connect account to onboarded (idempotent).

    In live mode the source of truth is Stripe's ``account.updated`` webhook
    (charges_enabled); this helper is the seam the webhook/admin calls.
    """
    with db.connect() as c:
        c.execute(
            "UPDATE connect_accounts SET onboarded=1 WHERE fan_id=?", (fan_id,)
        )
        row = c.execute(
            "SELECT * FROM connect_accounts WHERE fan_id=?", (fan_id,)
        ).fetchone()
    return _row_to_dict(row)


def get_account(fan_id: str) -> dict | None:
    with db.connect() as c:
        row = c.execute(
            "SELECT * FROM connect_accounts WHERE fan_id=?", (fan_id,)
        ).fetchone()
    return _row_to_dict(row)


# --------------------------------------------------------------------------- #
# Held-payment escrow (manual-capture PaymentIntent)
# --------------------------------------------------------------------------- #

def _get_order(c, order_id: str):
    return c.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()


def create_held_payment(
    order_id: str,
    amount_cents: int,
    buyer_ref: str | None = None,
    listing_id: str | None = None,
    buyer_fan_id: str | None = None,
) -> dict:
    """Hold the buyer's funds (manual-capture) — money is captured but not paid out.

    Creates the order row if it doesn't exist yet, sets its state to ``'held'``
    and records the payment ref. Idempotent: if the order is already held (or
    further along), the existing payment ref is returned unchanged. Returns the
    order row as a dict.

    Live mode creates a PaymentIntent with ``capture_method='manual'`` so the
    funds are authorized + captured into Fair Game's balance but held until the
    TM transfer confirms (then ``release_to_seller`` captures, or ``refund``
    cancels). Sim mode synthesizes the ref.
    """
    if not order_id:
        raise StripeError("order_id required")
    amount = int(amount_cents)
    if amount < 0:
        raise StripeError("amount_cents must be non-negative")
    now = int(time.time())

    with db.connect() as c:
        row = _get_order(c, order_id)
        # Already held (or further along) -> idempotent no-op.
        if row is not None and row["state"] in ("held", "released", "refunded"):
            return dict(row)

        if _sim_mode():
            payment_ref = "pi_sim_" + uuid.uuid4().hex[:20]
        else:
            stripe = _stripe()
            intent = stripe.PaymentIntent.create(
                amount=amount,
                currency="usd",
                capture_method="manual",
                metadata={
                    "fairgame_order_id": order_id,
                    "fairgame_buyer_ref": buyer_ref or "",
                },
            )
            payment_ref = intent["id"]

        if row is None:
            c.execute(
                "INSERT INTO orders(id,listing_id,buyer_fan_id,amount_cents,state,"
                "payment_ref,transfer_ref,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    order_id, listing_id, buyer_fan_id, amount, "held",
                    payment_ref, None, now, now,
                ),
            )
        else:
            c.execute(
                "UPDATE orders SET amount_cents=?,state='held',payment_ref=?,updated_at=? "
                "WHERE id=?",
                (amount, payment_ref, now, order_id),
            )
        out = _get_order(c, order_id)
    return dict(out)


def release_to_seller(order_id: str) -> dict:
    """Release held funds to the seller — the escrow payout (idempotent).

    Only a held order can be released; this is what the TM-transfer-confirmed
    trigger calls. Marks the order ``'released'``. Raises if the order doesn't
    exist or was refunded. Idempotent on an already-released order.

    Live mode captures the manual-capture PaymentIntent (and, in marketplace
    setups, the transfer to the seller's Connect account fires off the captured
    balance). Sim mode just advances state.
    """
    if not order_id:
        raise StripeError("order_id required")
    now = int(time.time())
    with db.connect() as c:
        row = _get_order(c, order_id)
        if row is None:
            raise StripeError("no order to release")
        if row["state"] == "released":
            return dict(row)
        # Accept both the orders-module 'paid' vocabulary and our own legacy
        # 'held' so the payout can be driven directly with no intermediate state.
        if row["state"] not in ("held", "paid"):
            raise StripeError(f"cannot release order in state '{row['state']}'")

        if not _sim_mode():
            stripe = _stripe()
            stripe.PaymentIntent.capture(row["payment_ref"])

        c.execute(
            "UPDATE orders SET state='released',updated_at=? WHERE id=?",
            (now, order_id),
        )
        out = _get_order(c, order_id)
    return dict(out)


def refund(order_id: str) -> dict:
    """Refund the buyer — the fraud wall when a transfer never completes (idempotent).

    A held order is refunded (buyer made whole, seller gets nothing). Marks the
    order ``'refunded'``. Raises if the order doesn't exist or was already
    released. Idempotent on an already-refunded order.

    Live mode cancels the uncaptured PaymentIntent (or issues a Refund if it was
    already captured). Sim mode just advances state.
    """
    if not order_id:
        raise StripeError("order_id required")
    now = int(time.time())
    with db.connect() as c:
        row = _get_order(c, order_id)
        if row is None:
            raise StripeError("no order to refund")
        if row["state"] == "refunded":
            return dict(row)
        if row["state"] == "released":
            raise StripeError("cannot refund an already-released order")
        # Accept both the orders-module 'paid' vocabulary and our own legacy
        # 'held' so the refund can be driven directly with no intermediate state.
        if row["state"] not in ("held", "paid"):
            raise StripeError(f"cannot refund order in state '{row['state']}'")

        if not _sim_mode():
            stripe = _stripe()
            try:
                stripe.PaymentIntent.cancel(row["payment_ref"])
            except Exception:
                # Already captured -> issue a real refund instead.
                stripe.Refund.create(payment_intent=row["payment_ref"])

        c.execute(
            "UPDATE orders SET state='refunded',updated_at=? WHERE id=?",
            (now, order_id),
        )
        out = _get_order(c, order_id)
    return dict(out)


def get_order(order_id: str) -> dict | None:
    with db.connect() as c:
        return _row_to_dict(_get_order(c, order_id))
