#!/usr/bin/env python3
"""Twenty CRM - CLI tool for OpenClaw"""
import os, sys, json, urllib.request, urllib.parse

def _load_api_key():
    """Load API key from env var first, then fall back to openclaw.json."""
    key = os.environ.get("BASE_CRM_API_KEY", "")
    if key:
        return key
    config_paths = [
        os.path.expanduser("~/.openclaw/openclaw.json"),
        "/home/brucewayne9/.openclaw/openclaw.json",
    ]
    for path in config_paths:
        try:
            with open(path) as f:
                data = json.load(f)
                key = data.get("env", {}).get("vars", {}).get("BASE_CRM_API_KEY", "")
                if key:
                    return key
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return ""

BASE_URL = os.environ.get("BASE_CRM_URL", "https://crm.groundrushlabs.com")
API_KEY = _load_api_key()

def _slim(obj):
    """Remove null/empty values recursively to reduce output size."""
    if isinstance(obj, dict):
        return {k: _slim(v) for k, v in obj.items()
                if v is not None and v != "" and v != [] and v != {}}
    if isinstance(obj, list):
        return [_slim(i) for i in obj]
    return obj

def _request(endpoint, method="GET", data=None):
    url = f"{BASE_URL}/rest/{endpoint}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return _slim(json.loads(resp.read().decode()))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:500]}"}
    except Exception as e:
        return {"error": str(e)}

def _gql(query, variables=None):
    """Execute a GraphQL query against Twenty CRM."""
    url = f"{BASE_URL}/graphql"
    data = {"query": query}
    if variables:
        data["variables"] = variables
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"errors": [{"message": f"HTTP {e.code}: {e.read().decode()[:300]}"}]}
    except Exception as e:
        return {"errors": [{"message": str(e)}]}

def _fmt_person(p):
    """Format a person record to concise output."""
    name = p.get("name", {})
    emails = p.get("emails", {})
    phones = p.get("phones", {})
    out = {"id": p.get("id")}
    fn = name.get("firstName", "")
    ln = name.get("lastName", "")
    if fn or ln:
        out["name"] = f"{fn} {ln}".strip()
    email = emails.get("primaryEmail", "")
    if email:
        out["email"] = email
    extra = emails.get("additionalEmails")
    if extra:
        out["additionalEmails"] = extra
    phone = phones.get("primaryPhoneNumber", "")
    if phone:
        out["phone"] = phone
        cc = phones.get("primaryPhoneCountryCode", "")
        if cc:
            out["phoneCountry"] = cc
    if p.get("jobTitle"):
        out["jobTitle"] = p["jobTitle"]
    if p.get("city"):
        out["city"] = p["city"]
    if p.get("companyId"):
        out["companyId"] = p["companyId"]
    return out

def list_people(limit=20):
    """List people using GraphQL for proper pagination."""
    result = _gql("""
    query($first: Int) {
      people(first: $first) {
        edges {
          node {
            id
            name { firstName lastName }
            emails { primaryEmail additionalEmails }
            phones { primaryPhoneNumber primaryPhoneCountryCode }
            jobTitle
            city
            companyId
          }
        }
        totalCount
      }
    }
    """, {"first": min(limit, 100)})

    if "errors" in result:
        return {"error": result["errors"][0]["message"]}

    edges = result.get("data", {}).get("people", {}).get("edges", [])
    total = result.get("data", {}).get("people", {}).get("totalCount", 0)
    people = [_fmt_person(e["node"]) for e in edges]
    return {"people": people, "totalCount": total}

def search_people(query):
    """Search contacts by name, email, phone, OR company name.

    When a query matches a company name, returns people linked to that company.
    This allows searches like 'My Hands Car Wash' to find Leshuan Williams (CEO).
    """
    parts = query.strip().split()

    # --- Step 1: Search companies matching the query ---
    company_ids = set()
    comp_filters = []
    for part in parts:
        comp_filters.append({"name": {"like": f"%{part}%"}})

    comp_result = _gql("""
    query($filter: CompanyFilterInput, $first: Int) {
      companies(filter: $filter, first: $first) {
        edges { node { id name } }
      }
    }
    """, {"filter": {"or": comp_filters}, "first": 20})

    if "data" in comp_result:
        for edge in comp_result["data"].get("companies", {}).get("edges", []):
            co = edge["node"]
            # Only include companies where ALL query parts match (not just one)
            co_name = (co.get("name") or "").lower()
            if all(p.lower() in co_name for p in parts):
                company_ids.add(co["id"])

    # --- Step 2: Search people by name/email (original logic) ---
    filters = []
    for part in parts:
        pattern = f"%{part}%"
        filters.extend([
            {"name": {"firstName": {"like": pattern}}},
            {"name": {"lastName": {"like": pattern}}},
            {"emails": {"primaryEmail": {"like": pattern}}},
        ])

    # Also add filters for people linked to matching companies
    for cid in company_ids:
        filters.append({"companyId": {"eq": cid}})

    result = _gql("""
    query($filter: PersonFilterInput, $first: Int) {
      people(filter: $filter, first: $first) {
        edges {
          node {
            id
            name { firstName lastName }
            emails { primaryEmail additionalEmails }
            phones { primaryPhoneNumber primaryPhoneCountryCode }
            jobTitle
            city
            companyId
          }
        }
        totalCount
      }
    }
    """, {
        "filter": {"or": filters},
        "first": 500,
    })

    if "errors" in result:
        return {"error": result["errors"][0]["message"], "query": query}

    edges = result.get("data", {}).get("people", {}).get("edges", [])
    total = result.get("data", {}).get("people", {}).get("totalCount", 0)
    people_raw = [e["node"] for e in edges]

    # Score and rank results - company matches and exact name matches first
    def score(p):
        s = 0
        fn = (p.get("name", {}).get("firstName") or "").lower()
        ln = (p.get("name", {}).get("lastName") or "").lower()
        em = (p.get("emails", {}).get("primaryEmail") or "").lower()
        q = query.lower()
        full = f"{fn} {ln}".strip()
        # Boost people linked to a matching company
        if p.get("companyId") in company_ids:
            s += 200
        if full == q:
            s += 100
        elif all(part.lower() in full for part in parts):
            s += 50
        for part in parts:
            pl = part.lower()
            if fn == pl or ln == pl:
                s += 20
            elif pl in fn or pl in ln:
                s += 10
            elif pl in em:
                s += 5
        if p.get("jobTitle"):
            s += 2
        if p.get("phones", {}).get("primaryPhoneNumber"):
            s += 2
        return s

    people_raw.sort(key=score, reverse=True)
    people = [_fmt_person(p) for p in people_raw]

    count = len(people)
    result_dict = {
        "people": people,
        "results": people,
        "count": count,
        "total": total,
        "query": query,
        "matched_companies": list(company_ids) if company_ids else None,
    }
    if count == 500:
        result_dict["truncated"] = True
    return result_dict

def get_person(person_id):
    return _request(f"people/{person_id}")

def create_person(data):
    return _request("people", "POST", data)

def update_person(person_id, data):
    return _request(f"people/{person_id}", "PATCH", data)

def delete_person(person_id):
    return _request(f"people/{person_id}", "DELETE")

def list_companies(limit=20):
    """List companies using GraphQL for proper pagination."""
    result = _gql("""
    query($first: Int) {
      companies(first: $first) {
        edges {
          node {
            id
            name
            domainName { primaryLinkUrl }
            employees
            address { addressCity addressState addressCountry }
          }
        }
        totalCount
      }
    }
    """, {"first": min(limit, 100)})

    if "errors" in result:
        return {"error": result["errors"][0]["message"]}

    edges = result.get("data", {}).get("companies", {}).get("edges", [])
    total = result.get("data", {}).get("companies", {}).get("totalCount", 0)
    companies = [_slim(e["node"]) for e in edges]
    return {"companies": companies, "totalCount": total}

def search_companies(query):
    """Search companies using server-side GraphQL filtering."""
    parts = query.strip().split()
    filters = []
    for part in parts:
        pattern = f"%{part}%"
        filters.append({"name": {"like": pattern}})

    result = _gql("""
    query($filter: CompanyFilterInput, $first: Int) {
      companies(filter: $filter, first: $first) {
        edges {
          node {
            id
            name
            domainName { primaryLinkUrl }
            employees
            address { addressCity addressState addressCountry }
          }
        }
        totalCount
      }
    }
    """, {
        "filter": {"or": filters},
        "first": 50,
    })

    if "errors" in result:
        return {"error": result["errors"][0]["message"], "query": query}

    edges = result.get("data", {}).get("companies", {}).get("edges", [])
    total = result.get("data", {}).get("companies", {}).get("totalCount", 0)
    companies = [_slim(e["node"]) for e in edges]

    return {"companies": companies, "results": companies, "count": len(companies), "total": total, "query": query}

def get_company(company_id):
    return _request(f"companies/{company_id}")

def create_company(data):
    return _request("companies", "POST", data)

def list_opportunities(limit=20):
    return _request(f"opportunities?limit={limit}")

def pipeline():
    data = _request("opportunities?limit=100")
    if "error" in data:
        return data
    opps = data if isinstance(data, list) else data.get("data", data.get("opportunities", []))
    if not isinstance(opps, list):
        return data
    stages = {}
    for o in opps:
        stage = o.get("stage", "Unknown")
        if stage not in stages:
            stages[stage] = {"count": 0, "total_amount": 0}
        stages[stage]["count"] += 1
        stages[stage]["total_amount"] += o.get("amount", 0) or 0
    return {"pipeline": stages, "total_deals": len(opps)}

def create_note(person_id, body):
    """Create a note with bodyV2 markdown format (fixed from legacy 'body' field)."""
    return _request("notes", "POST", {"bodyV2": {"markdown": body}})

def create_linked_note(person_id, body_markdown, title=""):
    """Create a note linked to a contact. Rollback on link failure (no orphans).

    Step 1: Create the note. Step 2: Create noteTarget linking note to person.
    If step 2 fails, step 1 is deleted immediately (no orphaned records).
    """
    # Step 1: Create the note
    data = {"bodyV2": {"markdown": body_markdown}}
    if title:
        data["title"] = title
    note_resp = _request("notes", "POST", data)
    if "error" in note_resp:
        return {"error": f"Note creation failed: {note_resp['error']}"}

    note_id = note_resp.get("data", {}).get("createNote", {}).get("id")
    if not note_id:
        return {"error": "Note creation returned no ID"}

    # Step 2: Link note to contact
    target_resp = _request("noteTargets", "POST", {"noteId": note_id, "personId": person_id})
    if "error" in target_resp:
        # Rollback: delete the orphaned note
        _request(f"notes/{note_id}", "DELETE")
        return {"error": f"Note created but link to contact failed — note deleted (no orphan). Detail: {target_resp['error']}"}

    target_id = target_resp.get("data", {}).get("createNoteTarget", {}).get("id")
    return {"success": True, "noteId": note_id, "targetId": target_id}

def create_linked_task(person_id, title, status="TODO"):
    """Create a task linked to a contact. Rollback on link failure (no orphans).

    Step 1: Create the task. Step 2: Create taskTarget linking task to person.
    If step 2 fails, step 1 is deleted immediately (no orphaned records).
    """
    # Step 1: Create the task
    task_resp = _request("tasks", "POST", {"title": title, "status": status})
    if "error" in task_resp:
        return {"error": f"Task creation failed: {task_resp['error']}"}

    task_id = task_resp.get("data", {}).get("createTask", {}).get("id")
    if not task_id:
        return {"error": "Task creation returned no ID"}

    # Step 2: Link task to contact
    target_resp = _request("taskTargets", "POST", {"taskId": task_id, "personId": person_id})
    if "error" in target_resp:
        # Rollback: delete the orphaned task
        _request(f"tasks/{task_id}", "DELETE")
        return {"error": f"Task created but link to contact failed — task deleted (no orphan). Detail: {target_resp['error']}"}

    target_id = target_resp.get("data", {}).get("createTaskTarget", {}).get("id")
    return {"success": True, "taskId": task_id, "targetId": target_id}

def list_tasks(limit=20):
    return _request(f"tasks?limit={limit}")

def create_task(data):
    return _request("tasks", "POST", data)

def update_task(task_id, data):
    """Update a task - supports bodyV2.markdown for rich text body."""
    return _request(f"tasks/{task_id}", "PATCH", data)

def delete_task(task_id):
    return _request(f"tasks/{task_id}", "DELETE")

def get_task(task_id):
    return _request(f"tasks/{task_id}")

def create_note_v2(body_markdown, title=""):
    """Create a note with markdown body (uses bodyV2 format)."""
    data = {"bodyV2": {"markdown": body_markdown}}
    if title:
        data["title"] = title
    return _request("notes", "POST", data)

def update_note(note_id, body_markdown):
    """Update a note's body."""
    return _request(f"notes/{note_id}", "PATCH", {"bodyV2": {"markdown": body_markdown}})


def _print_search_results(result, query):
    """Print search_people results in human-readable format for CLI output."""
    if "error" in result:
        print(json.dumps(result, indent=2))
        return

    count = result.get("count", 0)
    if count == 0:
        print(f'No contacts found matching "{query}". Would you like me to create a new contact?')
    elif count == 1:
        # Single result — print directly without disambiguation
        print(json.dumps(result, indent=2))
    else:
        # Multiple results — numbered list for user disambiguation
        people = result.get("people", [])
        print(f'Found {count} contacts matching "{query}":')
        for i, p in enumerate(people, 1):
            name = p.get("name", "Unknown")
            email = p.get("email", "")
            pid = p.get("id", "")
            if email:
                print(f'{i}. {name} — {email} (id: {pid})')
            else:
                print(f'{i}. {name} (id: {pid})')
        print("Reply with the number of the contact you want.")
        if result.get("truncated"):
            print(f"(Note: showing first 500 of {result.get('total', count)} total matches)")


def main():
    if len(sys.argv) < 2:
        print("Usage: crm.py <command> [args]")
        print("Commands: search <name>, get-person <id>, create-person <json>,")
        print("          list-people, list-companies, search-companies <name>,")
        print("          get-company <id>, create-company <json>,")
        print("          pipeline, list-opportunities, create-note <person_id> <body>,")
        print("          create-linked-note <person_id> <body_markdown> [title],")
        print("          create-linked-task <person_id> <title> [status],")
        print("          list-tasks, create-task, update-task, delete-task, get-task,")
        print("          create-note-v2, update-note")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd in ("search", "search-people") and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        result = search_people(query)
        _print_search_results(result, query)
    elif cmd == "get-person" and len(sys.argv) > 2:
        print(json.dumps(get_person(sys.argv[2]), indent=2))
    elif cmd == "create-person" and len(sys.argv) > 2:
        print(json.dumps(create_person(json.loads(sys.argv[2])), indent=2))
    elif cmd == "update-person" and len(sys.argv) > 3:
        print(json.dumps(update_person(sys.argv[2], json.loads(sys.argv[3])), indent=2))
    elif cmd == "delete-person" and len(sys.argv) > 2:
        print(json.dumps(delete_person(sys.argv[2]), indent=2))
    elif cmd == "list-people":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        print(json.dumps(list_people(limit), indent=2))
    elif cmd == "list-companies":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        print(json.dumps(list_companies(limit), indent=2))
    elif cmd == "search-companies" and len(sys.argv) > 2:
        print(json.dumps(search_companies(" ".join(sys.argv[2:])), indent=2))
    elif cmd == "get-company" and len(sys.argv) > 2:
        print(json.dumps(get_company(sys.argv[2]), indent=2))
    elif cmd == "create-company" and len(sys.argv) > 2:
        print(json.dumps(create_company(json.loads(sys.argv[2])), indent=2))
    elif cmd == "pipeline":
        print(json.dumps(pipeline(), indent=2))
    elif cmd == "list-opportunities":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        print(json.dumps(list_opportunities(limit), indent=2))
    elif cmd == "create-note" and len(sys.argv) > 3:
        print(json.dumps(create_note(sys.argv[2], " ".join(sys.argv[3:])), indent=2))
    elif cmd == "create-linked-note" and len(sys.argv) > 3:
        person_id = sys.argv[2]
        body_markdown = sys.argv[3]
        title = sys.argv[4] if len(sys.argv) > 4 else ""
        print(json.dumps(create_linked_note(person_id, body_markdown, title), indent=2))
    elif cmd == "create-linked-task" and len(sys.argv) > 3:
        person_id = sys.argv[2]
        title = sys.argv[3]
        status = sys.argv[4] if len(sys.argv) > 4 else "TODO"
        print(json.dumps(create_linked_task(person_id, title, status), indent=2))
    elif cmd == "list-tasks":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        print(json.dumps(list_tasks(limit), indent=2))
    elif cmd == "create-task" and len(sys.argv) > 2:
        print(json.dumps(create_task(json.loads(sys.argv[2])), indent=2))
    elif cmd == "update-task" and len(sys.argv) > 3:
        print(json.dumps(update_task(sys.argv[2], json.loads(sys.argv[3])), indent=2))
    elif cmd == "delete-task" and len(sys.argv) > 2:
        print(json.dumps(delete_task(sys.argv[2]), indent=2))
    elif cmd == "get-task" and len(sys.argv) > 2:
        print(json.dumps(get_task(sys.argv[2]), indent=2))
    elif cmd == "create-note-v2" and len(sys.argv) > 2:
        title = sys.argv[3] if len(sys.argv) > 3 else ""
        print(json.dumps(create_note_v2(sys.argv[2], title), indent=2))
    elif cmd == "update-note" and len(sys.argv) > 3:
        print(json.dumps(update_note(sys.argv[2], sys.argv[3]), indent=2))
    else:
        print(f"Unknown command or missing args: {cmd}")
        print("Commands: search, search-people, get-person, create-person, update-person, delete-person,")
        print("          list-people, list-companies, search-companies, get-company,")
        print("          create-company, pipeline, list-opportunities, create-note,")
        print("          create-linked-note, create-linked-task,")
        print("          list-tasks, create-task, update-task, delete-task, get-task,")
        print("          create-note-v2, update-note")
        sys.exit(1)

if __name__ == "__main__":
    main()
