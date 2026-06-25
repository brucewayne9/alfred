"""PayPal payment processor — STUB.

Fallback processor. No integration yet. Implement each method and set
``FAIRGAME_PROCESSOR=paypal`` to go live. Until then every money operation
raises ``ProcessorNotConfigured`` so nothing silently no-ops.

PayPal mapping notes (for whoever wires this up):
  - Checkout / hosted payment ........ PayPal Orders v2 (Create/Capture Order)
  - Escrow "hold then release" ....... Authorize on purchase (intent=AUTHORIZE),
        capture on transfer-confirm, void on failure. For seller payouts use
        PayPal Payouts API, or Marketplaces/Platform Commerce for split flows.
  - Seller onboarding ................ PayPal Partner Referrals (Connected Path).

Required env (set these before flipping FAIRGAME_PROCESSOR=paypal):
  FAIRGAME_PAYPAL_CLIENT_ID   - REST app client id
  FAIRGAME_PAYPAL_SECRET      - REST app secret
  FAIRGAME_PAYPAL_ENV         - "sandbox" | "live" (default sandbox)
"""
from __future__ import annotations

import os

from . import ProcessorNotConfigured

REQUIRED_ENV = (
    "FAIRGAME_PAYPAL_CLIENT_ID",
    "FAIRGAME_PAYPAL_SECRET",
)


class PayPalProcessor:
    name = "paypal"

    def _not_ready(self, op: str):
        missing = [e for e in REQUIRED_ENV if not os.environ.get(e)]
        hint = f" Missing env: {', '.join(missing)}." if missing else ""
        raise ProcessorNotConfigured(
            f"PayPal processor is stubbed — '{op}' is not implemented yet.{hint} "
            f"Implement core/fairgame/payments/paypal_proc.py, or set "
            f"FAIRGAME_PROCESSOR=stripe to keep using Stripe."
        )

    def is_sim(self) -> bool:
        return False

    def create_unlock_checkout(self, amount_cents: int) -> str:
        self._not_ready("create_unlock_checkout")

    def create_credit_checkout(self, pack_id: str, credits: int,
                               amount_cents: int, fan_id: str, label: str) -> str:
        self._not_ready("create_credit_checkout")

    def retrieve_checkout(self, session_id: str) -> dict:
        self._not_ready("retrieve_checkout")

    def onboard_seller(self, fan_id: str) -> dict:
        self._not_ready("onboard_seller")

    def mark_onboarded(self, fan_id: str):
        self._not_ready("mark_onboarded")

    def get_account(self, fan_id: str):
        self._not_ready("get_account")

    def create_held_payment(self, *args, **kwargs) -> dict:
        self._not_ready("create_held_payment")

    def release_to_seller(self, order_id: str) -> dict:
        self._not_ready("release_to_seller")

    def refund(self, order_id: str) -> dict:
        self._not_ready("refund")

    def get_order(self, order_id: str):
        self._not_ready("get_order")
