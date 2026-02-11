"""Twenty CRM API client. Wraps REST endpoints for people, companies, opportunities, notes, and tasks."""

import logging
from difflib import SequenceMatcher
from typing import Any

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = settings.base_crm_url.rstrip("/")
API_KEY = settings.base_crm_api_key


def _fuzzy_match(query: str, text: str, threshold: float = 0.6) -> bool:
    """Check if query fuzzy-matches text. Returns True if substring match or similarity >= threshold."""
    if query in text:
        return True
    for word in text.split():
        if SequenceMatcher(None, query, word).ratio() >= threshold:
            return True
    return False


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
        if _fuzzy_match(q, full) or _fuzzy_match(q, email):
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
                   email: str = "", phone: str = "", job_title: str = "", city: str = "",
                   company_id: str = "") -> dict:
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
    if company_id:
        body["companyId"] = company_id
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
        if _fuzzy_match(q, (c.get("name") or "").lower()):
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
        if _fuzzy_match(q, name) or _fuzzy_match(q, stage):
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


# ==================== Notes (Read/Update) ====================

def _format_note(n: dict) -> dict:
    return {
        "id": n.get("id"),
        "title": n.get("title", ""),
        "body": (n.get("bodyV2") or {}).get("markdown", "") if isinstance(n.get("bodyV2"), dict) else "",
        "created_at": n.get("createdAt"),
    }


def list_notes(limit: int = 20) -> list[dict]:
    data = _get("/rest/notes", {"limit": limit})
    return [_format_note(n) for n in data.get("data", {}).get("notes", [])]


def get_note(note_id: str) -> dict:
    data = _get(f"/rest/notes/{note_id}")
    return _format_note(data.get("data", {}).get("note", data.get("data", {})))


def update_note(note_id: str, title: str = "", body: str = "") -> dict:
    payload: dict[str, Any] = {}
    if title:
        payload["title"] = title
    if body:
        payload["bodyV2"] = {"markdown": body}
    data = _patch(f"/rest/notes/{note_id}", payload)
    return _format_note(data.get("data", {}).get("updateNote", data.get("data", {})))


def delete_note(note_id: str) -> dict:
    _delete(f"/rest/notes/{note_id}")
    return {"deleted": True, "id": note_id}


# ==================== Activities/Timeline ====================

def _format_activity(a: dict) -> dict:
    return {
        "id": a.get("id"),
        "type": a.get("type", ""),
        "title": a.get("title", ""),
        "body": a.get("body", ""),
        "due_at": a.get("dueAt"),
        "completed_at": a.get("completedAt"),
        "created_at": a.get("createdAt"),
        "person_id": a.get("personId"),
        "company_id": a.get("companyId"),
    }


def list_activities(limit: int = 20) -> list[dict]:
    """List recent activities/timeline events."""
    try:
        data = _get("/rest/activities", {"limit": limit})
        return [_format_activity(a) for a in data.get("data", {}).get("activities", [])]
    except Exception as e:
        logger.warning(f"Activities endpoint may not exist: {e}")
        return []


# ==================== Metadata/Schema ====================

def get_metadata_objects() -> list[dict]:
    """Get all available objects and their metadata."""
    try:
        data = _get("/rest/metadata/objects")
        objects = data.get("data", {}).get("objects", [])
        return [{"name": o.get("nameSingular"), "plural": o.get("namePlural"),
                 "description": o.get("description")} for o in objects]
    except Exception as e:
        logger.warning(f"Metadata endpoint error: {e}")
        return []


def get_object_fields(object_name: str) -> list[dict]:
    """Get fields for a specific object type."""
    try:
        data = _get(f"/rest/metadata/objects/{object_name}")
        obj = data.get("data", {}).get("object", {})
        fields = obj.get("fields", [])
        return [{"name": f.get("name"), "type": f.get("type"),
                 "label": f.get("label")} for f in fields]
    except Exception as e:
        logger.warning(f"Object metadata error: {e}")
        return []


# ==================== Search ====================

def search_all(query: str, limit: int = 10) -> dict:
    """Search across all record types."""
    results = {
        "people": search_people(query, limit),
        "companies": search_companies(query, limit),
        "opportunities": search_opportunities(query, limit),
    }
    return results


# ==================== Connection Check ====================

def is_connected() -> bool:
    if not API_KEY:
        return False
    try:
        resp = requests.get(f"{BASE_URL}/rest/people?limit=1", headers=_headers(), timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def get_crm_summary() -> dict:
    """Get a summary of CRM data counts."""
    try:
        people = _get("/rest/people", {"limit": 1})
        companies = _get("/rest/companies", {"limit": 1})
        opportunities = _get("/rest/opportunities", {"limit": 1})
        tasks = _get("/rest/tasks", {"limit": 1})

        return {
            "people_count": len(people.get("data", {}).get("people", [])),
            "companies_count": len(companies.get("data", {}).get("companies", [])),
            "opportunities_count": len(opportunities.get("data", {}).get("opportunities", [])),
            "tasks_count": len(tasks.get("data", {}).get("tasks", [])),
            "connected": True,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


# ============ Workflows ============

def _format_workflow(w: dict) -> dict:
    """Format a workflow for display."""
    return {
        "id": w.get("id"),
        "name": w.get("name"),
        "statuses": w.get("statuses", []),
        "position": w.get("position"),
        "lastPublishedVersionId": w.get("lastPublishedVersionId"),
        "createdAt": w.get("createdAt"),
        "updatedAt": w.get("updatedAt"),
    }


def list_workflows(limit: int = 20) -> list[dict]:
    """List all workflows."""
    resp = _get("/rest/workflows", {"limit": limit})
    workflows = resp.get("data", {}).get("workflows", [])
    return [_format_workflow(w) for w in workflows]


def get_workflow(workflow_id: str) -> dict:
    """Get a specific workflow by ID."""
    resp = _get(f"/rest/workflows/{workflow_id}")
    return _format_workflow(resp.get("data", {}).get("workflow", resp))


def create_workflow(name: str, statuses: list = None) -> dict:
    """Create a new workflow.

    Args:
        name: Workflow name
        statuses: List of status strings (e.g., ["ACTIVE", "INACTIVE"])
    """
    body = {"name": name}
    if statuses:
        body["statuses"] = statuses
    resp = _post("/rest/workflows", body)
    return _format_workflow(resp.get("data", {}).get("createWorkflow", resp))


def update_workflow(workflow_id: str, name: str = None, statuses: list = None) -> dict:
    """Update a workflow."""
    body = {}
    if name:
        body["name"] = name
    if statuses:
        body["statuses"] = statuses
    if not body:
        return {"error": "No fields to update"}
    resp = _patch(f"/rest/workflows/{workflow_id}", body)
    return _format_workflow(resp.get("data", {}).get("updateWorkflow", resp))


def delete_workflow(workflow_id: str) -> dict:
    """Delete a workflow."""
    return _delete(f"/rest/workflows/{workflow_id}")


# ============ Workflow Versions ============

def _format_workflow_version(v: dict) -> dict:
    """Format a workflow version for display."""
    return {
        "id": v.get("id"),
        "name": v.get("name"),
        "status": v.get("status"),
        "trigger": v.get("trigger"),
        "steps": v.get("steps"),
        "workflowId": v.get("workflowId"),
        "createdAt": v.get("createdAt"),
    }


def list_workflow_versions(workflow_id: str = None, limit: int = 20) -> list[dict]:
    """List workflow versions, optionally filtered by workflow."""
    params = {"limit": limit}
    if workflow_id:
        params["filter"] = f"workflowId[eq]:{workflow_id}"
    resp = _get("/rest/workflowVersions", params)
    versions = resp.get("data", {}).get("workflowVersions", [])
    return [_format_workflow_version(v) for v in versions]


def get_workflow_version(version_id: str) -> dict:
    """Get a specific workflow version."""
    resp = _get(f"/rest/workflowVersions/{version_id}")
    return _format_workflow_version(resp.get("data", {}).get("workflowVersion", resp))


def create_workflow_version(
    workflow_id: str,
    name: str,
    trigger: dict = None,
    steps: dict = None,
    status: str = "DRAFT"
) -> dict:
    """Create a new workflow version.

    Args:
        workflow_id: ID of parent workflow
        name: Version name
        trigger: Trigger configuration (e.g., {"type": "MANUAL"})
        steps: Steps configuration
        status: DRAFT or ACTIVE
    """
    body = {
        "workflowId": workflow_id,
        "name": name,
        "status": status,
    }
    if trigger:
        body["trigger"] = trigger
    if steps:
        body["steps"] = steps
    resp = _post("/rest/workflowVersions", body)
    return _format_workflow_version(resp.get("data", {}).get("createWorkflowVersion", resp))


def activate_workflow_version(version_id: str) -> dict:
    """Activate a workflow version (set status to ACTIVE)."""
    resp = _patch(f"/rest/workflowVersions/{version_id}", {"status": "ACTIVE"})
    return _format_workflow_version(resp.get("data", {}).get("updateWorkflowVersion", resp))


def deactivate_workflow_version(version_id: str) -> dict:
    """Deactivate a workflow version (set status to DRAFT)."""
    resp = _patch(f"/rest/workflowVersions/{version_id}", {"status": "DRAFT"})
    return _format_workflow_version(resp.get("data", {}).get("updateWorkflowVersion", resp))


def update_workflow_version(version_id: str, trigger: dict = None, steps: list = None, name: str = None) -> dict:
    """Update a workflow version's trigger, steps, or name.

    Args:
        version_id: The workflow version ID
        trigger: Trigger configuration dict
        steps: Steps configuration list
        name: New name for the version
    """
    body = {}
    if trigger:
        body["trigger"] = trigger
    if steps:
        body["steps"] = steps
    if name:
        body["name"] = name
    if not body:
        return {"error": "No fields to update"}
    resp = _patch(f"/rest/workflowVersions/{version_id}", body)
    return _format_workflow_version(resp.get("data", {}).get("updateWorkflowVersion", resp))


def set_workflow_trigger(workflow_id: str, trigger_type: str, entity: str = None) -> dict:
    """Set a trigger on a workflow's draft version.

    Args:
        workflow_id: The workflow ID
        trigger_type: Type of trigger - 'MANUAL', 'RECORD_CREATED', 'RECORD_UPDATED', 'RECORD_DELETED'
        entity: For record triggers, the entity type (e.g., 'person', 'company', 'opportunity')

    Returns:
        Updated workflow version
    """
    # Find the draft version for this workflow
    versions = list_workflow_versions(workflow_id)
    draft_version = None
    for v in versions:
        if v.get("status") == "DRAFT":
            draft_version = v
            break

    if not draft_version:
        return {"error": "No draft version found. Create a new version first."}

    # Build trigger based on type
    if trigger_type == "MANUAL":
        trigger = {
            "name": "Launch manually",
            "type": "MANUAL",
            "settings": {
                "outputSchema": {}
            }
        }
    elif trigger_type in ["RECORD_CREATED", "RECORD_UPDATED", "RECORD_DELETED"]:
        if not entity:
            return {"error": f"Entity required for {trigger_type} trigger (e.g., 'person', 'company')"}
        event_name = f"{entity}.{trigger_type.split('_')[1].lower()}"
        trigger = {
            "name": f"When {entity} is {trigger_type.split('_')[1].lower()}",
            "type": "DATABASE_EVENT",
            "settings": {
                "eventName": event_name,
                "outputSchema": {}
            }
        }
    else:
        return {"error": f"Unknown trigger type: {trigger_type}. Use MANUAL, RECORD_CREATED, RECORD_UPDATED, or RECORD_DELETED"}

    return update_workflow_version(draft_version["id"], trigger=trigger)


# ============ Workflow Runs ============

def _format_workflow_run(r: dict) -> dict:
    """Format a workflow run for display."""
    return {
        "id": r.get("id"),
        "name": r.get("name"),
        "status": r.get("status"),
        "workflowId": r.get("workflowId"),
        "workflowVersionId": r.get("workflowVersionId"),
        "enqueuedAt": r.get("enqueuedAt"),
        "startedAt": r.get("startedAt"),
        "endedAt": r.get("endedAt"),
        "state": r.get("state"),
    }


def list_workflow_runs(workflow_id: str = None, status: str = None, limit: int = 20) -> list[dict]:
    """List workflow runs.

    Args:
        workflow_id: Filter by workflow
        status: Filter by status (PENDING, RUNNING, COMPLETED, FAILED)
        limit: Max results
    """
    params = {"limit": limit}
    filters = []
    if workflow_id:
        filters.append(f"workflowId[eq]:{workflow_id}")
    if status:
        filters.append(f"status[eq]:{status}")
    if filters:
        params["filter"] = ",".join(filters)
    resp = _get("/rest/workflowRuns", params)
    runs = resp.get("data", {}).get("workflowRuns", [])
    return [_format_workflow_run(r) for r in runs]


def get_workflow_run(run_id: str) -> dict:
    """Get a specific workflow run."""
    resp = _get(f"/rest/workflowRuns/{run_id}")
    return _format_workflow_run(resp.get("data", {}).get("workflowRun", resp))


def trigger_workflow(workflow_id: str, name: str = None) -> dict:
    """Manually trigger a workflow run.

    Args:
        workflow_id: The workflow to trigger
        name: Optional name for this run
    """
    body = {"workflowId": workflow_id}
    if name:
        body["name"] = name
    resp = _post("/rest/workflowRuns", body)
    return _format_workflow_run(resp.get("data", {}).get("createWorkflowRun", resp))
