"""Square payment processor — STUB.

Square is Mike's intended first-choice processor, but there is no Square
integration yet. This file is the skeleton: implement each method and set
``FAIRGAME_PROCESSOR=square`` to go live. Until then every money operation
raises ``ProcessorNotConfigured`` so nothing silently no-ops.

Square mapping notes (for whoever wires this up):
  - Checkout / hosted payment ........ Square Checkout API (Payment Links)
  - Escrow "hold then release" ....... Square has no native marketplace escrow.
        Options: (a) delayed-capture — authorize on purchase, capture on
        transfer-confirm, void/refund on failure; or (b) Square + a separate
        payout step. Mirror the Stripe semantics in create_held_payment /
        release_to_seller / refund.
  - Seller onboarding ................ Square OAuth connect for seller accounts.

Required env (set these before flipping FAIRGAME_PROCESSOR=square):
  FAIRGAME_SQUARE_ACCESS_TOKEN   - Square access token (server-side)
  FAIRGAME_SQUARE_LOCATION_ID    - Square location id
  FAIRGAME_SQUARE_APP_ID         - Square application id
  FAIRGAME_SQUARE_ENV            - "sandbox" | "production" (default sandbox)
"""
from __future__ import annotations

import os

from . import ProcessorNotConfigured

REQUIRED_ENV = (
    "FAIRGAME_SQUARE_ACCESS_TOKEN",
    "FAIRGAME_SQUARE_LOCATION_ID",
    "FAIRGAME_SQUARE_APP_ID",
)


class SquareProcessor:
    name = "square"

    def _not_ready(self, op: str):
        missing = [e for e in REQUIRED_ENV if not os.environ.get(e)]
        hint = f" Missing env: {', '.join(missing)}." if missing else ""
        raise ProcessorNotConfigured(
            f"Square processor is stubbed — '{op}' is not implemented yet.{hint} "
            f"Implement core/fairgame/payments/square_proc.py, or set "
            f"FAIRGAME_PROCESSOR=stripe to keep using Stripe."
        )

    # --- sim / mode -------------------------------------------------------
    def is_sim(self) -> bool:
        # No sandbox bridge yet; treat as never-sim so callers don't assume
        # a working simulator exists for Square.
        return False

    # --- checkout ---------------------------------------------------------
    def create_unlock_checkout(self, amount_cents: int) -> str:
        self._not_ready("create_unlock_checkout")

    def create_credit_checkout(self, pack_id: str, credits: int,
                               amount_cents: int, fan_id: str, label: str) -> str:
        self._not_ready("create_credit_checkout")

    def retrieve_checkout(self, session_id: str) -> dict:
        self._not_ready("retrieve_checkout")

    # --- seller onboarding -----------------------------------------------
    def onboard_seller(self, fan_id: str) -> dict:
        self._not_ready("onboard_seller")

    def mark_onboarded(self, fan_id: str):
        self._not_ready("mark_onboarded")

    def get_account(self, fan_id: str):
        self._not_ready("get_account")

    # --- escrow money flow ------------------------------------------------
    def create_held_payment(self, *args, **kwargs) -> dict:
        self._not_ready("create_held_payment")

    def release_to_seller(self, order_id: str) -> dict:
        self._not_ready("release_to_seller")

    def refund(self, order_id: str) -> dict:
        self._not_ready("refund")

    def get_order(self, order_id: str):
        self._not_ready("get_order")
