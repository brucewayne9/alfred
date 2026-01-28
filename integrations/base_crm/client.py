"""Twenty CRM API client. Wraps REST endpoints for people, companies, opportunities, notes, and tasks."""

import logging
from typing import Any

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = settings.base_crm_url.rstrip("/")
API_KEY = settings.base_crm_api_key


def _headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _get(path: str, params: dict | None = None) -> Any:
    resp = requests.get(f"{BASE_URL}{path}", headers=_headers(), params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict) -> Any:
    resp = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _patch(path: str, body: dict) -> Any:
    resp = requests.patch(f"{BASE_URL}{path}", headers=_headers(), json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _delete(path: str) -> Any:
    resp = requests.delete(f"{BASE_URL}{path}", headers=_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json()


# ==================== People ====================

def _format_person(p: dict) -> dict:
    name = p.get("name", {})
    emails = p.get("emails", {})
    phones = p.get("phones", {})
    return {
        "id": p.get("id"),
        "first_name": name.get("firstName", ""),
        "last_name": name.get("lastName", ""),
        "email": emails.get("primaryEmail", ""),
        "phone": phones.get("primaryPhoneNumber", ""),
        "job_title": p.get("jobTitle", ""),
        "city": p.get("city", ""),
        "company_id": p.get("companyId"),
    }


def list_people(limit: int = 20) -> list[dict]:
    data = _get("/rest/people", {"limit": limit})
    return [_format_person(p) for p in data.get("data", {}).get("people", [])]


def search_people(query: str, limit: int = 10) -> list[dict]:
    q = query.lower()
    data = _get("/rest/people", {"limit": 100})
    results = []
    for p in data.get("data", {}).get("people", []):
        name = p.get("name", {})
        full = f"{name.get('firstName', '')} {name.get('lastName', '')}".lower()
        email = (p.get("emails", {}) or {}).get("primaryEmail", "").lower()
        if q in full or q in email:
            results.append(_format_person(p))
            if len(results) >= limit:
                break
    return results


def get_person(person_id: str) -> dict:
    data = _get(f"/rest/people/{person_id}")
    return _format_person(data.get("data", {}).get("person", data.get("data", {})))


def create_person(first_name: str, last_name: str, email: str = "", phone: str = "",
                   job_title: str = "", city: str = "", company_id: str = "") -> dict:
    body: dict[str, Any] = {
        "name": {"firstName": first_name, "lastName": last_name},
    }
    if email:
        body["emails"] = {"primaryEmail": email}
    if phone:
        body["phones"] = {"primaryPhoneNumber": phone, "primaryPhoneCountryCode": "US", "primaryPhoneCallingCode": "+1"}
    if job_title:
        body["jobTitle"] = job_title
    if city:
        body["city"] = city
    if company_id:
        body["companyId"] = company_id
    data = _post("/rest/people", body)
    return _format_person(data.get("data", {}).get("createPerson", data.get("data", {})))


def update_person(person_id: str, first_name: str = "", last_name: str = "",
                   email: str = "", phone: str = "", job_title: str = "", city: str = "") -> dict:
    body: dict[str, Any] = {}
    if first_name or last_name:
        name: dict[str, str] = {}
        if first_name:
            name["firstName"] = first_name
        if last_name:
            name["lastName"] = last_name
        body["name"] = name
    if email:
        body["emails"] = {"primaryEmail": email}
    if phone:
        body["phones"] = {"primaryPhoneNumber": phone, "primaryPhoneCountryCode": "US", "primaryPhoneCallingCode": "+1"}
    if job_title:
        body["jobTitle"] = job_title
    if city:
        body["city"] = city
    data = _patch(f"/rest/people/{person_id}", body)
    return _format_person(data.get("data", {}).get("updatePerson", data.get("data", {})))


def delete_person(person_id: str) -> dict:
    data = _delete(f"/rest/people/{person_id}")
    return {"deleted": True, "id": person_id}


# ==================== Companies ====================

def _format_company(c: dict) -> dict:
    domain = c.get("domainName", {})
    addr = c.get("address", {})
    arr = c.get("annualRecurringRevenue", {})
    amount_micros = arr.get("amountMicros")
    return {
        "id": c.get("id"),
        "name": c.get("name", ""),
        "domain": domain.get("primaryLinkUrl", "") if isinstance(domain, dict) else str(domain),
        "employees": c.get("employees"),
        "city": addr.get("addressCity", "") if isinstance(addr, dict) else "",
        "annual_revenue": float(amount_micros) / 1_000_000 if amount_micros else None,
    }


def list_companies(limit: int = 20) -> list[dict]:
    data = _get("/rest/companies", {"limit": limit})
    return [_format_company(c) for c in data.get("data", {}).get("companies", [])]


def search_companies(query: str, limit: int = 10) -> list[dict]:
    q = query.lower()
    data = _get("/rest/companies", {"limit": 100})
    results = []
    for c in data.get("data", {}).get("companies", []):
        if q in (c.get("name") or "").lower():
            results.append(_format_company(c))
            if len(results) >= limit:
                break
    return results


def get_company(company_id: str) -> dict:
    data = _get(f"/rest/companies/{company_id}")
    return _format_company(data.get("data", {}).get("company", data.get("data", {})))


def create_company(name: str, domain: str = "", employees: int = 0, city: str = "") -> dict:
    body: dict[str, Any] = {"name": name}
    if domain:
        body["domainName"] = {"primaryLinkUrl": domain}
    if employees:
        body["employees"] = employees
    if city:
        body["address"] = {"addressCity": city}
    data = _post("/rest/companies", body)
    return _format_company(data.get("data", {}).get("createCompany", data.get("data", {})))


def update_company(company_id: str, name: str = "", domain: str = "",
                    employees: int = 0, city: str = "") -> dict:
    body: dict[str, Any] = {}
    if name:
        body["name"] = name
    if domain:
        body["domainName"] = {"primaryLinkUrl": domain}
    if employees:
        body["employees"] = employees
    if city:
        body["address"] = {"addressCity": city}
    data = _patch(f"/rest/companies/{company_id}", body)
    return _format_company(data.get("data", {}).get("updateCompany", data.get("data", {})))


def delete_company(company_id: str) -> dict:
    _delete(f"/rest/companies/{company_id}")
    return {"deleted": True, "id": company_id}


# ==================== Opportunities (Deals) ====================

def _format_opportunity(o: dict) -> dict:
    amt = o.get("amount", {})
    amount_micros = amt.get("amountMicros")
    return {
        "id": o.get("id"),
        "name": o.get("name", ""),
        "amount": float(amount_micros) / 1_000_000 if amount_micros else None,
        "currency": amt.get("currencyCode", "USD"),
        "stage": o.get("stage", ""),
        "close_date": o.get("closeDate"),
        "company_id": o.get("companyId"),
        "contact_id": o.get("pointOfContactId"),
    }


def list_opportunities(limit: int = 20) -> list[dict]:
    data = _get("/rest/opportunities", {"limit": limit})
    return [_format_opportunity(o) for o in data.get("data", {}).get("opportunities", [])]


def get_opportunity(opp_id: str) -> dict:
    data = _get(f"/rest/opportunities/{opp_id}")
    return _format_opportunity(data.get("data", {}).get("opportunity", data.get("data", {})))


def create_opportunity(name: str, stage: str = "MEETING", amount: float = 0,
                        company_id: str = "", contact_id: str = "", close_date: str = "") -> dict:
    body: dict[str, Any] = {"name": name, "stage": stage}
    if amount:
        body["amount"] = {"amountMicros": str(int(amount * 1_000_000)), "currencyCode": "USD"}
    if company_id:
        body["companyId"] = company_id
    if contact_id:
        body["pointOfContactId"] = contact_id
    if close_date:
        body["closeDate"] = close_date
    data = _post("/rest/opportunities", body)
    return _format_opportunity(data.get("data", {}).get("createOpportunity", data.get("data", {})))


def search_opportunities(query: str, limit: int = 10) -> list[dict]:
    q = query.lower()
    data = _get("/rest/opportunities", {"limit": 100})
    results = []
    for o in data.get("data", {}).get("opportunities", []):
        name = (o.get("name") or "").lower()
        stage = (o.get("stage") or "").lower()
        if q in name or q in stage:
            results.append(_format_opportunity(o))
            if len(results) >= limit:
                break
    return results


def update_opportunity_stage(opp_id: str, stage: str) -> dict:
    data = _patch(f"/rest/opportunities/{opp_id}", {"stage": stage})
    return _format_opportunity(data.get("data", {}).get("updateOpportunity", data.get("data", {})))


def delete_opportunity(opp_id: str) -> dict:
    _delete(f"/rest/opportunities/{opp_id}")
    return {"deleted": True, "id": opp_id}


def pipeline_summary() -> dict:
    data = _get("/rest/opportunities", {"limit": 100})
    opps = data.get("data", {}).get("opportunities", [])
    stages: dict[str, dict] = {}
    for o in opps:
        stage = o.get("stage", "UNKNOWN")
        amt = o.get("amount", {})
        micros = amt.get("amountMicros")
        value = float(micros) / 1_000_000 if micros else 0
        if stage not in stages:
            stages[stage] = {"count": 0, "total_value": 0}
        stages[stage]["count"] += 1
        stages[stage]["total_value"] += value
    total_value = sum(s["total_value"] for s in stages.values())
    total_deals = sum(s["count"] for s in stages.values())
    return {"stages": stages, "total_deals": total_deals, "total_value": total_value}


# ==================== Tasks ====================

def _format_task(t: dict) -> dict:
    return {
        "id": t.get("id"),
        "title": t.get("title", ""),
        "status": t.get("status", ""),
        "due_date": t.get("dueAt"),
        "body": (t.get("bodyV2") or {}).get("markdown", "") if isinstance(t.get("bodyV2"), dict) else "",
    }


def list_tasks(limit: int = 20) -> list[dict]:
    data = _get("/rest/tasks", {"limit": limit})
    return [_format_task(t) for t in data.get("data", {}).get("tasks", [])]


def create_task(title: str, status: str = "TODO", due_date: str = "", body: str = "") -> dict:
    payload: dict[str, Any] = {"title": title, "status": status}
    if due_date:
        payload["dueAt"] = due_date
    if body:
        payload["bodyV2"] = {"markdown": body}
    data = _post("/rest/tasks", payload)
    return _format_task(data.get("data", {}).get("createTask", data.get("data", {})))


def update_task(task_id: str, title: str = "", status: str = "", due_date: str = "") -> dict:
    body: dict[str, Any] = {}
    if title:
        body["title"] = title
    if status:
        body["status"] = status
    if due_date:
        body["dueAt"] = due_date
    data = _patch(f"/rest/tasks/{task_id}", body)
    return _format_task(data.get("data", {}).get("updateTask", data.get("data", {})))


def delete_task(task_id: str) -> dict:
    _delete(f"/rest/tasks/{task_id}")
    return {"deleted": True, "id": task_id}


# ==================== Notes ====================

def create_note(title: str, body: str = "") -> dict:
    payload: dict[str, Any] = {"title": title}
    if body:
        payload["bodyV2"] = {"markdown": body}
    data = _post("/rest/notes", payload)
    note_data = data.get("data", {}).get("createNote", data.get("data", {}))
    return {"id": note_data.get("id"), "title": title}


def create_note_for_person(title: str, person_id: str, body: str = "") -> dict:
    note = create_note(title, body)
    _post("/rest/noteTargets", {"noteId": note["id"], "personId": person_id})
    note["linked_to"] = {"person_id": person_id}
    return note


def create_note_for_company(title: str, company_id: str, body: str = "") -> dict:
    note = create_note(title, body)
    _post("/rest/noteTargets", {"noteId": note["id"], "companyId": company_id})
    note["linked_to"] = {"company_id": company_id}
    return note


def create_note_for_opportunity(title: str, opportunity_id: str, body: str = "") -> dict:
    note = create_note(title, body)
    _post("/rest/noteTargets", {"noteId": note["id"], "opportunityId": opportunity_id})
    note["linked_to"] = {"opportunity_id": opportunity_id}
    return note


# ==================== Task Linking ====================

def create_task_for_person(title: str, person_id: str, status: str = "TODO",
                            due_date: str = "") -> dict:
    task = create_task(title, status, due_date)
    _post("/rest/taskTargets", {"taskId": task["id"], "personId": person_id})
    task["linked_to"] = {"person_id": person_id}
    return task


def create_task_for_company(title: str, company_id: str, status: str = "TODO",
                             due_date: str = "") -> dict:
    task = create_task(title, status, due_date)
    _post("/rest/taskTargets", {"taskId": task["id"], "companyId": company_id})
    task["linked_to"] = {"company_id": company_id}
    return task


def create_task_for_opportunity(title: str, opportunity_id: str, status: str = "TODO",
                                 due_date: str = "") -> dict:
    task = create_task(title, status, due_date)
    _post("/rest/taskTargets", {"taskId": task["id"], "opportunityId": opportunity_id})
    task["linked_to"] = {"opportunity_id": opportunity_id}
    return task


# ==================== Connection Check ====================

def is_connected() -> bool:
    if not API_KEY:
        return False
    try:
        resp = requests.get(f"{BASE_URL}/rest/people?limit=1", headers=_headers(), timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
