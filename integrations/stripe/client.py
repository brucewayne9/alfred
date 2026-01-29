"""Stripe API client for payments, customers, subscriptions, and more."""

import logging
from typing import Any
from datetime import datetime

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

API_KEY = settings.stripe_api_key
BASE_URL = "https://api.stripe.com/v1"


def _headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}


def _get(path: str, params: dict = None) -> Any:
    resp = requests.get(f"{BASE_URL}{path}", headers=_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, data: dict = None) -> Any:
    resp = requests.post(f"{BASE_URL}{path}", headers=_headers(), data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _delete(path: str) -> Any:
    resp = requests.delete(f"{BASE_URL}{path}", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


# ==================== Connection Check ====================

def is_connected() -> bool:
    if not API_KEY:
        return False
    try:
        resp = requests.get(f"{BASE_URL}/balance", headers=_headers(), timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


# ==================== Balance ====================

def get_balance() -> dict:
    """Get account balance."""
    data = _get("/balance")
    available = data.get("available", [])
    pending = data.get("pending", [])

    def format_amount(amounts):
        result = []
        for a in amounts:
            currency = a.get("currency", "usd").upper()
            amount = a.get("amount", 0) / 100  # Convert from cents
            result.append({"currency": currency, "amount": amount})
        return result

    return {
        "available": format_amount(available),
        "pending": format_amount(pending),
        "livemode": data.get("livemode", False),
    }


def list_payouts(limit: int = 20) -> list[dict]:
    """List recent payouts."""
    data = _get("/payouts", {"limit": limit})
    return [{
        "id": p.get("id"),
        "amount": p.get("amount", 0) / 100,
        "currency": p.get("currency", "usd").upper(),
        "status": p.get("status"),
        "arrival_date": datetime.fromtimestamp(p.get("arrival_date", 0)).isoformat() if p.get("arrival_date") else None,
        "created": datetime.fromtimestamp(p.get("created", 0)).isoformat() if p.get("created") else None,
    } for p in data.get("data", [])]


# ==================== Payments (Charges & Payment Intents) ====================

def list_payments(limit: int = 20, customer: str = None) -> list[dict]:
    """List recent payments/charges."""
    params = {"limit": limit}
    if customer:
        params["customer"] = customer
    data = _get("/charges", params)
    return [{
        "id": c.get("id"),
        "amount": c.get("amount", 0) / 100,
        "currency": c.get("currency", "usd").upper(),
        "status": c.get("status"),
        "paid": c.get("paid"),
        "refunded": c.get("refunded"),
        "customer": c.get("customer"),
        "description": c.get("description"),
        "receipt_email": c.get("receipt_email"),
        "created": datetime.fromtimestamp(c.get("created", 0)).isoformat() if c.get("created") else None,
    } for c in data.get("data", [])]


def get_payment(charge_id: str) -> dict:
    """Get details of a specific charge."""
    c = _get(f"/charges/{charge_id}")
    return {
        "id": c.get("id"),
        "amount": c.get("amount", 0) / 100,
        "amount_refunded": c.get("amount_refunded", 0) / 100,
        "currency": c.get("currency", "usd").upper(),
        "status": c.get("status"),
        "paid": c.get("paid"),
        "refunded": c.get("refunded"),
        "customer": c.get("customer"),
        "description": c.get("description"),
        "receipt_email": c.get("receipt_email"),
        "receipt_url": c.get("receipt_url"),
        "payment_method": c.get("payment_method"),
        "created": datetime.fromtimestamp(c.get("created", 0)).isoformat() if c.get("created") else None,
    }


def search_payments(query: str, limit: int = 20) -> list[dict]:
    """Search payments using Stripe's search API."""
    data = _get("/charges/search", {"query": query, "limit": limit})
    return [{
        "id": c.get("id"),
        "amount": c.get("amount", 0) / 100,
        "currency": c.get("currency", "usd").upper(),
        "status": c.get("status"),
        "customer": c.get("customer"),
        "description": c.get("description"),
        "created": datetime.fromtimestamp(c.get("created", 0)).isoformat() if c.get("created") else None,
    } for c in data.get("data", [])]


def refund_payment(charge_id: str, amount: float = None, reason: str = None) -> dict:
    """Issue a refund for a charge. Amount in dollars (omit for full refund)."""
    data = {"charge": charge_id}
    if amount is not None:
        data["amount"] = int(amount * 100)  # Convert to cents
    if reason:
        data["reason"] = reason  # duplicate, fraudulent, or requested_by_customer

    r = _post("/refunds", data)
    return {
        "id": r.get("id"),
        "amount": r.get("amount", 0) / 100,
        "currency": r.get("currency", "usd").upper(),
        "status": r.get("status"),
        "charge": r.get("charge"),
        "reason": r.get("reason"),
    }


def list_refunds(limit: int = 20, charge: str = None) -> list[dict]:
    """List refunds."""
    params = {"limit": limit}
    if charge:
        params["charge"] = charge
    data = _get("/refunds", params)
    return [{
        "id": r.get("id"),
        "amount": r.get("amount", 0) / 100,
        "currency": r.get("currency", "usd").upper(),
        "status": r.get("status"),
        "charge": r.get("charge"),
        "reason": r.get("reason"),
        "created": datetime.fromtimestamp(r.get("created", 0)).isoformat() if r.get("created") else None,
    } for r in data.get("data", [])]


# ==================== Customers ====================

def list_customers(limit: int = 20, email: str = None) -> list[dict]:
    """List customers."""
    params = {"limit": limit}
    if email:
        params["email"] = email
    data = _get("/customers", params)
    return [{
        "id": c.get("id"),
        "email": c.get("email"),
        "name": c.get("name"),
        "phone": c.get("phone"),
        "balance": c.get("balance", 0) / 100,
        "currency": c.get("currency"),
        "created": datetime.fromtimestamp(c.get("created", 0)).isoformat() if c.get("created") else None,
    } for c in data.get("data", [])]


def search_customers(query: str, limit: int = 20) -> list[dict]:
    """Search customers."""
    data = _get("/customers/search", {"query": query, "limit": limit})
    return [{
        "id": c.get("id"),
        "email": c.get("email"),
        "name": c.get("name"),
        "phone": c.get("phone"),
        "created": datetime.fromtimestamp(c.get("created", 0)).isoformat() if c.get("created") else None,
    } for c in data.get("data", [])]


def get_customer(customer_id: str) -> dict:
    """Get customer details."""
    c = _get(f"/customers/{customer_id}")
    return {
        "id": c.get("id"),
        "email": c.get("email"),
        "name": c.get("name"),
        "phone": c.get("phone"),
        "description": c.get("description"),
        "balance": c.get("balance", 0) / 100,
        "currency": c.get("currency"),
        "default_source": c.get("default_source"),
        "created": datetime.fromtimestamp(c.get("created", 0)).isoformat() if c.get("created") else None,
    }


def create_customer(email: str, name: str = None, phone: str = None, description: str = None) -> dict:
    """Create a new customer."""
    data = {"email": email}
    if name:
        data["name"] = name
    if phone:
        data["phone"] = phone
    if description:
        data["description"] = description

    c = _post("/customers", data)
    return {
        "id": c.get("id"),
        "email": c.get("email"),
        "name": c.get("name"),
        "phone": c.get("phone"),
    }


def update_customer(customer_id: str, email: str = None, name: str = None,
                    phone: str = None, description: str = None) -> dict:
    """Update a customer."""
    data = {}
    if email:
        data["email"] = email
    if name:
        data["name"] = name
    if phone:
        data["phone"] = phone
    if description:
        data["description"] = description

    c = _post(f"/customers/{customer_id}", data)
    return {
        "id": c.get("id"),
        "email": c.get("email"),
        "name": c.get("name"),
        "phone": c.get("phone"),
    }


def delete_customer(customer_id: str) -> dict:
    """Delete a customer."""
    _delete(f"/customers/{customer_id}")
    return {"deleted": True, "id": customer_id}


# ==================== Subscriptions ====================

def list_subscriptions(limit: int = 20, customer: str = None, status: str = None) -> list[dict]:
    """List subscriptions. Status: active, past_due, canceled, all, etc."""
    params = {"limit": limit}
    if customer:
        params["customer"] = customer
    if status:
        params["status"] = status
    data = _get("/subscriptions", params)
    return [{
        "id": s.get("id"),
        "customer": s.get("customer"),
        "status": s.get("status"),
        "current_period_start": datetime.fromtimestamp(s.get("current_period_start", 0)).isoformat() if s.get("current_period_start") else None,
        "current_period_end": datetime.fromtimestamp(s.get("current_period_end", 0)).isoformat() if s.get("current_period_end") else None,
        "cancel_at_period_end": s.get("cancel_at_period_end"),
        "items": [{
            "price_id": item.get("price", {}).get("id") if isinstance(item.get("price"), dict) else item.get("price"),
            "quantity": item.get("quantity"),
        } for item in s.get("items", {}).get("data", [])],
        "created": datetime.fromtimestamp(s.get("created", 0)).isoformat() if s.get("created") else None,
    } for s in data.get("data", [])]


def get_subscription(subscription_id: str) -> dict:
    """Get subscription details."""
    s = _get(f"/subscriptions/{subscription_id}")
    return {
        "id": s.get("id"),
        "customer": s.get("customer"),
        "status": s.get("status"),
        "current_period_start": datetime.fromtimestamp(s.get("current_period_start", 0)).isoformat() if s.get("current_period_start") else None,
        "current_period_end": datetime.fromtimestamp(s.get("current_period_end", 0)).isoformat() if s.get("current_period_end") else None,
        "cancel_at_period_end": s.get("cancel_at_period_end"),
        "canceled_at": datetime.fromtimestamp(s.get("canceled_at", 0)).isoformat() if s.get("canceled_at") else None,
        "items": [{
            "price_id": item.get("price", {}).get("id") if isinstance(item.get("price"), dict) else item.get("price"),
            "quantity": item.get("quantity"),
        } for item in s.get("items", {}).get("data", [])],
    }


def create_subscription(customer_id: str, price_id: str, quantity: int = 1) -> dict:
    """Create a subscription for a customer."""
    data = {
        "customer": customer_id,
        "items[0][price]": price_id,
        "items[0][quantity]": quantity,
    }
    s = _post("/subscriptions", data)
    return {
        "id": s.get("id"),
        "customer": s.get("customer"),
        "status": s.get("status"),
    }


def cancel_subscription(subscription_id: str, at_period_end: bool = True) -> dict:
    """Cancel a subscription. Set at_period_end=False for immediate cancellation."""
    if at_period_end:
        s = _post(f"/subscriptions/{subscription_id}", {"cancel_at_period_end": "true"})
    else:
        s = _delete(f"/subscriptions/{subscription_id}")
    return {
        "id": s.get("id"),
        "status": s.get("status"),
        "cancel_at_period_end": s.get("cancel_at_period_end"),
    }


def resume_subscription(subscription_id: str) -> dict:
    """Resume a subscription scheduled for cancellation."""
    s = _post(f"/subscriptions/{subscription_id}", {"cancel_at_period_end": "false"})
    return {
        "id": s.get("id"),
        "status": s.get("status"),
        "cancel_at_period_end": s.get("cancel_at_period_end"),
    }


# ==================== Products & Prices ====================

def list_products(limit: int = 20, active: bool = None) -> list[dict]:
    """List products."""
    params = {"limit": limit}
    if active is not None:
        params["active"] = "true" if active else "false"
    data = _get("/products", params)
    return [{
        "id": p.get("id"),
        "name": p.get("name"),
        "description": p.get("description"),
        "active": p.get("active"),
        "default_price": p.get("default_price"),
        "created": datetime.fromtimestamp(p.get("created", 0)).isoformat() if p.get("created") else None,
    } for p in data.get("data", [])]


def get_product(product_id: str) -> dict:
    """Get product details."""
    p = _get(f"/products/{product_id}")
    return {
        "id": p.get("id"),
        "name": p.get("name"),
        "description": p.get("description"),
        "active": p.get("active"),
        "default_price": p.get("default_price"),
        "images": p.get("images", []),
        "metadata": p.get("metadata", {}),
    }


def create_product(name: str, description: str = None, active: bool = True) -> dict:
    """Create a product."""
    data = {"name": name, "active": "true" if active else "false"}
    if description:
        data["description"] = description
    p = _post("/products", data)
    return {
        "id": p.get("id"),
        "name": p.get("name"),
        "description": p.get("description"),
        "active": p.get("active"),
    }


def update_product(product_id: str, name: str = None, description: str = None, active: bool = None) -> dict:
    """Update a product."""
    data = {}
    if name:
        data["name"] = name
    if description:
        data["description"] = description
    if active is not None:
        data["active"] = "true" if active else "false"
    p = _post(f"/products/{product_id}", data)
    return {
        "id": p.get("id"),
        "name": p.get("name"),
        "description": p.get("description"),
        "active": p.get("active"),
    }


def list_prices(limit: int = 20, product: str = None, active: bool = None) -> list[dict]:
    """List prices."""
    params = {"limit": limit}
    if product:
        params["product"] = product
    if active is not None:
        params["active"] = "true" if active else "false"
    data = _get("/prices", params)
    return [{
        "id": p.get("id"),
        "product": p.get("product"),
        "active": p.get("active"),
        "currency": p.get("currency", "usd").upper(),
        "unit_amount": p.get("unit_amount", 0) / 100 if p.get("unit_amount") else None,
        "recurring": p.get("recurring"),
        "type": p.get("type"),
    } for p in data.get("data", [])]


def create_price(product_id: str, unit_amount: float, currency: str = "usd",
                 recurring_interval: str = None) -> dict:
    """Create a price. For subscriptions, set recurring_interval to 'month' or 'year'."""
    data = {
        "product": product_id,
        "unit_amount": int(unit_amount * 100),  # Convert to cents
        "currency": currency.lower(),
    }
    if recurring_interval:
        data["recurring[interval]"] = recurring_interval

    p = _post("/prices", data)
    return {
        "id": p.get("id"),
        "product": p.get("product"),
        "unit_amount": p.get("unit_amount", 0) / 100,
        "currency": p.get("currency", "usd").upper(),
        "recurring": p.get("recurring"),
    }


# ==================== Invoices ====================

def list_invoices(limit: int = 20, customer: str = None, status: str = None) -> list[dict]:
    """List invoices. Status: draft, open, paid, uncollectible, void."""
    params = {"limit": limit}
    if customer:
        params["customer"] = customer
    if status:
        params["status"] = status
    data = _get("/invoices", params)
    return [{
        "id": i.get("id"),
        "number": i.get("number"),
        "customer": i.get("customer"),
        "customer_email": i.get("customer_email"),
        "status": i.get("status"),
        "amount_due": i.get("amount_due", 0) / 100,
        "amount_paid": i.get("amount_paid", 0) / 100,
        "currency": i.get("currency", "usd").upper(),
        "due_date": datetime.fromtimestamp(i.get("due_date", 0)).isoformat() if i.get("due_date") else None,
        "hosted_invoice_url": i.get("hosted_invoice_url"),
        "created": datetime.fromtimestamp(i.get("created", 0)).isoformat() if i.get("created") else None,
    } for i in data.get("data", [])]


def get_invoice(invoice_id: str) -> dict:
    """Get invoice details."""
    i = _get(f"/invoices/{invoice_id}")
    return {
        "id": i.get("id"),
        "number": i.get("number"),
        "customer": i.get("customer"),
        "customer_email": i.get("customer_email"),
        "status": i.get("status"),
        "amount_due": i.get("amount_due", 0) / 100,
        "amount_paid": i.get("amount_paid", 0) / 100,
        "amount_remaining": i.get("amount_remaining", 0) / 100,
        "currency": i.get("currency", "usd").upper(),
        "due_date": datetime.fromtimestamp(i.get("due_date", 0)).isoformat() if i.get("due_date") else None,
        "hosted_invoice_url": i.get("hosted_invoice_url"),
        "invoice_pdf": i.get("invoice_pdf"),
        "lines": [{
            "description": line.get("description"),
            "amount": line.get("amount", 0) / 100,
            "quantity": line.get("quantity"),
        } for line in i.get("lines", {}).get("data", [])],
    }


def create_invoice(customer_id: str, description: str = None, days_until_due: int = 30) -> dict:
    """Create a draft invoice for a customer."""
    data = {
        "customer": customer_id,
        "collection_method": "send_invoice",
        "days_until_due": days_until_due,
    }
    if description:
        data["description"] = description

    i = _post("/invoices", data)
    return {
        "id": i.get("id"),
        "number": i.get("number"),
        "customer": i.get("customer"),
        "status": i.get("status"),
    }


def add_invoice_item(invoice_id: str, description: str, amount: float, quantity: int = 1) -> dict:
    """Add a line item to a draft invoice."""
    invoice = _get(f"/invoices/{invoice_id}")
    customer_id = invoice.get("customer")

    data = {
        "customer": customer_id,
        "invoice": invoice_id,
        "description": description,
        "unit_amount": int(amount * 100),
        "quantity": quantity,
    }
    item = _post("/invoiceitems", data)
    return {
        "id": item.get("id"),
        "description": item.get("description"),
        "amount": item.get("amount", 0) / 100,
    }


def finalize_invoice(invoice_id: str) -> dict:
    """Finalize a draft invoice."""
    i = _post(f"/invoices/{invoice_id}/finalize", {})
    return {
        "id": i.get("id"),
        "number": i.get("number"),
        "status": i.get("status"),
        "hosted_invoice_url": i.get("hosted_invoice_url"),
    }


def send_invoice(invoice_id: str) -> dict:
    """Send an invoice to the customer."""
    i = _post(f"/invoices/{invoice_id}/send", {})
    return {
        "id": i.get("id"),
        "number": i.get("number"),
        "status": i.get("status"),
        "sent": True,
    }


def void_invoice(invoice_id: str) -> dict:
    """Void an invoice."""
    i = _post(f"/invoices/{invoice_id}/void", {})
    return {
        "id": i.get("id"),
        "status": i.get("status"),
    }


# ==================== Payment Links ====================

def list_payment_links(limit: int = 20, active: bool = None) -> list[dict]:
    """List payment links."""
    params = {"limit": limit}
    if active is not None:
        params["active"] = "true" if active else "false"
    data = _get("/payment_links", params)
    return [{
        "id": p.get("id"),
        "url": p.get("url"),
        "active": p.get("active"),
        "line_items": p.get("line_items", {}).get("data", []),
    } for p in data.get("data", [])]


def create_payment_link(price_id: str, quantity: int = 1) -> dict:
    """Create a payment link for a price."""
    data = {
        "line_items[0][price]": price_id,
        "line_items[0][quantity]": quantity,
    }
    p = _post("/payment_links", data)
    return {
        "id": p.get("id"),
        "url": p.get("url"),
        "active": p.get("active"),
    }


def deactivate_payment_link(payment_link_id: str) -> dict:
    """Deactivate a payment link."""
    p = _post(f"/payment_links/{payment_link_id}", {"active": "false"})
    return {
        "id": p.get("id"),
        "url": p.get("url"),
        "active": p.get("active"),
    }


# ==================== Revenue Summary ====================

def get_revenue_summary() -> dict:
    """Get a summary of recent revenue."""
    # Get recent successful charges
    charges = _get("/charges", {"limit": 100, "status": "succeeded"})

    total_revenue = 0
    currency_totals = {}

    for c in charges.get("data", []):
        amount = c.get("amount", 0) / 100
        currency = c.get("currency", "usd").upper()
        if c.get("refunded"):
            amount -= c.get("amount_refunded", 0) / 100

        total_revenue += amount
        currency_totals[currency] = currency_totals.get(currency, 0) + amount

    # Get active subscriptions
    subs = _get("/subscriptions", {"limit": 100, "status": "active"})
    mrr = 0
    for s in subs.get("data", []):
        for item in s.get("items", {}).get("data", []):
            price = item.get("price", {})
            if isinstance(price, dict) and price.get("recurring"):
                amount = (price.get("unit_amount", 0) / 100) * item.get("quantity", 1)
                interval = price.get("recurring", {}).get("interval")
                if interval == "year":
                    amount = amount / 12
                mrr += amount

    return {
        "recent_revenue": total_revenue,
        "by_currency": currency_totals,
        "active_subscriptions": len(subs.get("data", [])),
        "estimated_mrr": round(mrr, 2),
    }
