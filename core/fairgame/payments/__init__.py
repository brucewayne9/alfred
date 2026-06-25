"""Fans First payment-processor abstraction.

One env var — ``FAIRGAME_PROCESSOR`` — selects the active payment processor:

    FAIRGAME_PROCESSOR=stripe   (default; fully implemented via stripe_connect)
    FAIRGAME_PROCESSOR=square    (stubbed — fill in square_proc.py)
    FAIRGAME_PROCESSOR=paypal    (stubbed — fill in paypal_proc.py)

Every call site in the app goes through ``get_processor()`` instead of touching
``stripe_connect`` directly, so switching processors is a config flip plus
implementing the chosen stub. Stripe stays the default and its behaviour is
byte-for-byte unchanged (the Stripe processor delegates straight to the existing
``stripe_connect`` module).

The money interface every processor must satisfy:

    is_sim() -> bool
    create_unlock_checkout(amount_cents) -> url
    create_credit_checkout(pack_id, credits, amount_cents, fan_id, label) -> url
    retrieve_checkout(session_id) -> dict
    onboard_seller(fan_id) -> dict
    mark_onboarded(fan_id) -> dict | None
    get_account(fan_id) -> dict | None
    create_held_payment(...) -> dict        # escrow HOLD of buyer funds
    release_to_seller(order_id) -> dict      # release escrow to seller
    refund(order_id) -> dict                 # refund the held payment
    get_order(order_id) -> dict | None
"""
from __future__ import annotations

import os


class ProcessorNotConfigured(RuntimeError):
    """Raised when a selected processor has no working integration yet."""


def active_processor_name() -> str:
    return os.environ.get("FAIRGAME_PROCESSOR", "stripe").strip().lower() or "stripe"


def get_processor():
    """Return the active payment processor (module or instance).

    Cheap to call per-request; no shared mutable state. Selection is read from
    the environment each call so the processor can be flipped with a restart.
    """
    name = active_processor_name()
    if name == "stripe":
        # The Stripe implementation already lives in stripe_connect — return it
        # directly so the Stripe path is identical to before this abstraction.
        from .. import stripe_connect
        return stripe_connect
    if name == "square":
        from .square_proc import SquareProcessor
        return SquareProcessor()
    if name == "paypal":
        from .paypal_proc import PayPalProcessor
        return PayPalProcessor()
    raise ProcessorNotConfigured(
        f"Unknown FAIRGAME_PROCESSOR={name!r}. Use stripe | square | paypal."
    )
