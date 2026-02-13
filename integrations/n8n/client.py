"""n8n Workflow Automation API client."""

import json
import logging
from typing import Any

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = settings.n8n_url.rstrip("/") if settings.n8n_url else ""
API_KEY = settings.n8n_api_key


def _headers() -> dict:
    return {"X-N8N-API-KEY": API_KEY, "Content-Type": "application/json"}


def _get(path: str, params: dict | None = None) -> Any:
    resp = requests.get(f"{BASE_URL}/api/v1{path}", headers=_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict) -> Any:
    resp = requests.post(f"{BASE_URL}/api/v1{path}", headers=_headers(), json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _patch(path: str, body: dict) -> Any:
    resp = requests.patch(f"{BASE_URL}/api/v1{path}", headers=_headers(), json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _delete(path: str) -> Any:
    resp = requests.delete(f"{BASE_URL}/api/v1{path}", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


# ==================== Connection Check ====================

def is_connected() -> bool:
    if not API_KEY or not BASE_URL:
        return False
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/workflows?limit=1", headers=_headers(), timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ==================== Workflows ====================

def _format_workflow(w: dict) -> dict:
    """Format workflow for display."""
    return {
        "id": w.get("id"),
        "name": w.get("name", ""),
        "active": w.get("active", False),
        "created_at": w.get("createdAt"),
        "updated_at": w.get("updatedAt"),
        "tags": [t.get("name") for t in w.get("tags", [])],
        "nodes_count": len(w.get("nodes", [])),
    }


def list_workflows(limit: int = 50, active_only: bool = False) -> list[dict]:
    """List all workflows."""
    data = _get("/workflows", {"limit": limit})
    workflows = data.get("data", [])
    if active_only:
        workflows = [w for w in workflows if w.get("active")]
    return [_format_workflow(w) for w in workflows]


def get_workflow(workflow_id: str) -> dict:
    """Get full workflow details including nodes and connections."""
    data = _get(f"/workflows/{workflow_id}")
    return data


def get_workflow_summary(workflow_id: str) -> dict:
    """Get workflow summary with node list."""
    data = _get(f"/workflows/{workflow_id}")
    nodes = data.get("nodes", [])
    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "active": data.get("active"),
        "nodes": [{"name": n.get("name"), "type": n.get("type")} for n in nodes],
        "connections_count": sum(len(c) for c in data.get("connections", {}).values()),
    }


def create_workflow(name: str, nodes: list[dict], connections: dict, active: bool = False) -> dict:
    """Create a new workflow with nodes and connections.

    Note: n8n's public API treats 'active' as read-only on creation.
    Workflows are always created inactive. Use activate_workflow() after.
    """
    body = {
        "name": name,
        "nodes": nodes,
        "connections": connections,
        "settings": {
            "executionOrder": "v1"
        },
    }
    data = _post("/workflows", body)
    # Activate separately if requested (API doesn't allow it on creation)
    if active and data.get("id"):
        try:
            activate_workflow(data["id"])
            data["active"] = True
        except Exception:
            pass
    return _format_workflow(data)


def update_workflow(workflow_id: str, name: str = None, nodes: list[dict] = None,
                    connections: dict = None, active: bool = None) -> dict:
    """Update an existing workflow."""
    body = {}
    if name is not None:
        body["name"] = name
    if nodes is not None:
        body["nodes"] = nodes
    if connections is not None:
        body["connections"] = connections
    if active is not None:
        body["active"] = active
    data = _patch(f"/workflows/{workflow_id}", body)
    return _format_workflow(data)


def delete_workflow(workflow_id: str) -> dict:
    """Delete a workflow."""
    _delete(f"/workflows/{workflow_id}")
    return {"deleted": True, "id": workflow_id}


def activate_workflow(workflow_id: str) -> dict:
    """Activate a workflow."""
    data = _post(f"/workflows/{workflow_id}/activate", {})
    return _format_workflow(data)


def deactivate_workflow(workflow_id: str) -> dict:
    """Deactivate a workflow."""
    data = _post(f"/workflows/{workflow_id}/deactivate", {})
    return _format_workflow(data)


# ==================== Executions ====================

def execute_workflow(workflow_id: str, data: dict = None) -> dict:
    """Execute a workflow by triggering its webhook (if it has one).

    n8n's public API does not support direct workflow execution.
    This finds the workflow's webhook trigger and POSTs to it instead.
    """
    # Get workflow to find webhook path
    workflow = get_workflow(workflow_id)
    nodes = workflow.get("nodes", [])
    webhook_node = None
    for node in nodes:
        node_type = node.get("type", "")
        if "webhook" in node_type.lower():
            webhook_node = node
            break

    if not webhook_node:
        return {
            "success": False,
            "error": "Workflow has no webhook trigger. It can only be executed from the n8n UI or via its own trigger (schedule, form, etc.).",
        }

    # Build webhook URL
    path = webhook_node.get("parameters", {}).get("path", "")
    if not path:
        return {"success": False, "error": "Webhook node has no path configured."}

    is_active = workflow.get("active", False)
    if is_active:
        webhook_url = f"{BASE_URL}/webhook/{path}"
    else:
        webhook_url = f"{BASE_URL}/webhook-test/{path}"

    # Trigger the webhook
    resp = requests.post(webhook_url, json=data or {}, timeout=30)
    return {
        "success": resp.status_code < 400,
        "status_code": resp.status_code,
        "webhook_url": webhook_url,
        "response": resp.text[:500] if resp.text else "",
    }


def duplicate_workflow(workflow_id: str, new_name: str = None) -> dict:
    """Duplicate an existing workflow with all its nodes and connections.

    Fetches the full workflow and creates a new copy with a new name.
    The copy is created in inactive state.
    """
    source = get_workflow(workflow_id)
    name = new_name or f"{source.get('name', 'Workflow')} (Copy)"

    # Strip IDs from nodes so n8n assigns new ones
    nodes = []
    for node in source.get("nodes", []):
        clean_node = {k: v for k, v in node.items() if k != "id"}
        nodes.append(clean_node)

    body = {
        "name": name,
        "nodes": nodes,
        "connections": source.get("connections", {}),
        "settings": source.get("settings", {}),
    }
    # staticData can be large; only include if present
    if source.get("staticData"):
        body["staticData"] = source["staticData"]
    data = _post("/workflows", body)
    return _format_workflow(data)


def get_executions(workflow_id: str = None, limit: int = 20) -> list[dict]:
    """Get execution history."""
    params = {"limit": limit}
    if workflow_id:
        params["workflowId"] = workflow_id
    data = _get("/executions", params)
    return [{
        "id": e.get("id"),
        "workflow_id": e.get("workflowId"),
        "finished": e.get("finished"),
        "mode": e.get("mode"),
        "started_at": e.get("startedAt"),
        "stopped_at": e.get("stoppedAt"),
        "status": e.get("status"),
    } for e in data.get("data", [])]


# ==================== Workflow Builder Helpers ====================

def build_node(node_type: str, name: str, position: list[int], parameters: dict = None,
               credentials: dict = None) -> dict:
    """Build a node definition."""
    node = {
        "name": name,
        "type": node_type,
        "position": position,
        "typeVersion": 1,
        "parameters": parameters or {},
    }
    if credentials:
        node["credentials"] = credentials
    return node


def build_webhook_trigger(name: str = "Webhook", path: str = "webhook", method: str = "POST") -> dict:
    """Build a webhook trigger node."""
    return build_node(
        "n8n-nodes-base.webhook",
        name,
        [250, 300],
        {"path": path, "httpMethod": method, "responseMode": "onReceived"}
    )


def build_schedule_trigger(name: str = "Schedule", cron: str = "0 9 * * *") -> dict:
    """Build a schedule/cron trigger node."""
    return build_node(
        "n8n-nodes-base.scheduleTrigger",
        name,
        [250, 300],
        {"rule": {"interval": [{"field": "cronExpression", "expression": cron}]}}
    )


def build_http_request(name: str, url: str, method: str = "GET", body: dict = None,
                       headers: dict = None, position: list[int] = None) -> dict:
    """Build an HTTP request node."""
    params = {
        "url": url,
        "method": method,
        "options": {},
    }
    if body:
        params["bodyParameters"] = {"parameters": [{"name": k, "value": v} for k, v in body.items()]}
    if headers:
        params["headerParameters"] = {"parameters": [{"name": k, "value": v} for k, v in headers.items()]}
    return build_node("n8n-nodes-base.httpRequest", name, position or [450, 300], params)


def build_slack_message(name: str, channel: str, message: str, position: list[int] = None) -> dict:
    """Build a Slack message node."""
    return build_node(
        "n8n-nodes-base.slack",
        name,
        position or [650, 300],
        {
            "resource": "message",
            "operation": "post",
            "channel": channel,
            "text": message,
        }
    )


def build_email_send(name: str, to: str, subject: str, body: str, position: list[int] = None) -> dict:
    """Build an email send node (Gmail)."""
    return build_node(
        "n8n-nodes-base.gmail",
        name,
        position or [650, 300],
        {
            "resource": "message",
            "operation": "send",
            "sendTo": to,
            "subject": subject,
            "message": body,
        }
    )


def build_code_node(name: str, code: str, position: list[int] = None) -> dict:
    """Build a code/function node."""
    return build_node(
        "n8n-nodes-base.code",
        name,
        position or [450, 300],
        {"jsCode": code}
    )


def build_if_node(name: str, condition_value1: str, operation: str, condition_value2: str,
                  position: list[int] = None) -> dict:
    """Build an IF condition node."""
    return build_node(
        "n8n-nodes-base.if",
        name,
        position or [450, 300],
        {
            "conditions": {
                "string": [{
                    "value1": condition_value1,
                    "operation": operation,
                    "value2": condition_value2,
                }]
            }
        }
    )


def build_set_node(name: str, values: dict, position: list[int] = None) -> dict:
    """Build a Set node to set values."""
    return build_node(
        "n8n-nodes-base.set",
        name,
        position or [450, 300],
        {
            "values": {
                "string": [{"name": k, "value": v} for k, v in values.items()]
            }
        }
    )


def build_connections(node_sequence: list[str]) -> dict:
    """Build connections dict from a sequence of node names."""
    connections = {}
    for i in range(len(node_sequence) - 1):
        connections[node_sequence[i]] = {
            "main": [[{"node": node_sequence[i + 1], "type": "main", "index": 0}]]
        }
    return connections


# ==================== Workflow Templates ====================

def create_webhook_to_slack_workflow(name: str, webhook_path: str, slack_channel: str,
                                      message_template: str) -> dict:
    """Create a simple webhook -> Slack notification workflow."""
    nodes = [
        build_webhook_trigger("Webhook", webhook_path),
        build_slack_message("Send Slack", slack_channel, message_template, [450, 300]),
    ]
    connections = build_connections(["Webhook", "Send Slack"])
    return create_workflow(name, nodes, connections)


def create_scheduled_http_workflow(name: str, cron: str, url: str, method: str = "GET") -> dict:
    """Create a scheduled HTTP request workflow."""
    nodes = [
        build_schedule_trigger("Schedule", cron),
        build_http_request("HTTP Request", url, method, position=[450, 300]),
    ]
    connections = build_connections(["Schedule", "HTTP Request"])
    return create_workflow(name, nodes, connections)


def create_workflow_from_description(name: str, description: str) -> dict:
    """
    Create a workflow from a natural language description.
    This generates a basic workflow structure that can be refined.
    Returns the workflow JSON structure for review before creation.
    """
    # Parse common patterns from description
    desc_lower = description.lower()

    nodes = []
    node_names = []
    x_pos = 250

    # Determine trigger type
    if any(word in desc_lower for word in ["every day", "daily", "every hour", "hourly", "schedule", "cron", "at "]):
        # Extract schedule if possible
        cron = "0 9 * * *"  # Default: 9 AM daily
        if "every hour" in desc_lower or "hourly" in desc_lower:
            cron = "0 * * * *"
        elif "every minute" in desc_lower:
            cron = "* * * * *"
        elif "midnight" in desc_lower:
            cron = "0 0 * * *"
        nodes.append(build_schedule_trigger("Schedule Trigger", cron))
        node_names.append("Schedule Trigger")
    elif any(word in desc_lower for word in ["webhook", "when called", "api endpoint", "http trigger"]):
        path = name.lower().replace(" ", "-")
        nodes.append(build_webhook_trigger("Webhook Trigger", path))
        node_names.append("Webhook Trigger")
    else:
        # Default to manual trigger (webhook)
        nodes.append(build_webhook_trigger("Webhook Trigger", name.lower().replace(" ", "-")))
        node_names.append("Webhook Trigger")

    x_pos += 200

    # Determine actions
    if any(word in desc_lower for word in ["slack", "send to slack", "slack message", "notify slack"]):
        nodes.append(build_slack_message("Slack Notification", "#general",
                                         "Workflow triggered: {{$json.message}}", [x_pos, 300]))
        node_names.append("Slack Notification")
        x_pos += 200

    if any(word in desc_lower for word in ["email", "send email", "mail"]):
        nodes.append(build_email_send("Send Email", "{{$json.email}}",
                                      "Notification", "{{$json.message}}", [x_pos, 300]))
        node_names.append("Send Email")
        x_pos += 200

    if any(word in desc_lower for word in ["http", "api call", "request", "fetch", "call url"]):
        nodes.append(build_http_request("HTTP Request", "https://api.example.com/endpoint",
                                        "GET", position=[x_pos, 300]))
        node_names.append("HTTP Request")
        x_pos += 200

    # If no specific actions detected, add a placeholder
    if len(nodes) == 1:
        nodes.append(build_set_node("Process Data", {"status": "processed"}, [x_pos, 300]))
        node_names.append("Process Data")

    connections = build_connections(node_names)

    return {
        "name": name,
        "nodes": nodes,
        "connections": connections,
        "description": description,
        "ready_to_create": True,
    }


# ==================== Newsletter Conversion ====================

def convert_newsletter_to_webhook(
    workflow_id: str,
    site_name: str,
    webhook_path: str = "",
    crm_api_key: str = "",
    crm_url: str = "https://crm.groundrushlabs.com",
) -> dict:
    """Duplicate a newsletter workflow and convert form triggers to webhook triggers.

    This takes an existing newsletter workflow (form trigger → save → email → mark sent)
    and creates a new copy where:
    1. Form triggers are replaced with webhook triggers (for WordPress/Elementor forms)
    2. A CRM integration node is added to push subscribers to Twenty CRM
    3. Field references are updated to match webhook JSON format
    4. Everything else (welcome email, data table, etc.) stays intact

    Args:
        workflow_id: Source workflow to duplicate and convert
        site_name: Site identifier (e.g., 'rucktalk', 'loovacast', 'lumabot')
        webhook_path: Custom webhook path (defaults to '{site_name}-subscribe')
        crm_api_key: Twenty CRM API key (uses settings if empty)
        crm_url: Twenty CRM base URL

    Returns:
        Dict with new workflow details and webhook URL
    """
    if not webhook_path:
        webhook_path = f"{site_name.lower().replace(' ', '-')}-subscribe"

    if not crm_api_key:
        from config.settings import settings
        crm_api_key = settings.base_crm_api_key

    # Fetch the source workflow
    source = get_workflow(workflow_id)
    source_nodes = source.get("nodes", [])
    source_connections = source.get("connections", {})

    new_nodes = []
    form_node_name = None
    form_node_new_name = f"Webhook {site_name}"
    crm_node_name = f"Push to CRM ({site_name})"

    # Track the node that the form trigger connects to (first downstream node)
    form_downstream_node = None

    for node in source_nodes:
        node_type = node.get("type", "")
        node_name = node.get("name", "")

        if "formTrigger" in node_type:
            # Replace form trigger with webhook trigger
            form_node_name = node_name
            webhook_node = {
                "name": form_node_new_name,
                "type": "n8n-nodes-base.webhook",
                "typeVersion": 2,
                "position": node.get("position", [250, 300]),
                "parameters": {
                    "path": webhook_path,
                    "httpMethod": "POST",
                    "responseMode": "onReceived",
                    "responseData": "allEntries",
                    "options": {},
                },
            }
            new_nodes.append(webhook_node)

            # Find what the form trigger connected to
            if node_name in source_connections:
                conns = source_connections[node_name].get("main", [[]])
                if conns and conns[0]:
                    form_downstream_node = conns[0][0].get("node")
        else:
            # Keep the node, but update field references from form format to webhook JSON
            node_copy = _update_field_references(node, form_node_name or "Newsletter Signup Form")
            # Strip node ID so n8n assigns new ones
            node_copy = {k: v for k, v in node_copy.items() if k != "id"}
            new_nodes.append(node_copy)

    # Add CRM integration node (HTTP Request to Twenty CRM GraphQL API)
    crm_node_position = new_nodes[0].get("position", [250, 300]).copy()
    crm_node_position[0] += 200  # Offset right
    crm_node_position[1] -= 200  # Offset up (parallel branch)

    crm_node = {
        "name": crm_node_name,
        "type": "n8n-nodes-base.httpRequest",
        "typeVersion": 4.2,
        "position": crm_node_position,
        "parameters": {
            "method": "POST",
            "url": f"{crm_url}/api",
            "sendHeaders": True,
            "headerParameters": {
                "parameters": [
                    {"name": "Authorization", "value": f"Bearer {crm_api_key}"},
                    {"name": "Content-Type", "value": "application/json"},
                ]
            },
            "sendBody": True,
            "specifyBody": "json",
            "jsonBody": '={\n'
                '  "query": "mutation { createPerson(data: { '
                'name: { firstName: \\"{{$json.first_name || $json[\'First Name\']}}\\",'
                ' lastName: \\"{{$json.last_name || $json[\'Last Name\']}}\\" },'
                ' emails: { primaryEmail: \\"{{$json.email || $json.Email}}\\" },'
                ' phones: { primaryPhone: \\"{{$json.phone || $json[\'Phone Number\'] || \'\'}}\\" },'
                ' city: \\"' + site_name + '\\"'
                ' }) { id } }"\n}',
            "options": {},
        },
    }
    new_nodes.append(crm_node)

    # Rebuild connections
    new_connections = {}

    # Webhook connects to both the original downstream node AND the CRM node
    webhook_targets = []
    if form_downstream_node:
        webhook_targets.append({"node": form_downstream_node, "type": "main", "index": 0})
    new_connections[form_node_new_name] = {
        "main": [webhook_targets]
    }

    # CRM node connects from the webhook too (parallel branch)
    # Add it as a second output from the webhook
    if webhook_targets:
        new_connections[form_node_new_name]["main"].append(
            [{"node": crm_node_name, "type": "main", "index": 0}]
        )
    else:
        new_connections[form_node_new_name] = {
            "main": [[{"node": crm_node_name, "type": "main", "index": 0}]]
        }

    # Copy all other connections, updating the form trigger name
    for src_name, conn_data in source_connections.items():
        if src_name == form_node_name:
            continue  # Already handled above
        # Update any references to the old form node name in connections
        new_conn = json.loads(json.dumps(conn_data).replace(
            f'"{form_node_name}"', f'"{form_node_new_name}"'
        )) if form_node_name else conn_data
        new_connections[src_name] = new_conn

    # Create the new workflow
    new_name = f"{source.get('name', 'Newsletter')} - {site_name} Webhook"
    body = {
        "name": new_name,
        "nodes": new_nodes,
        "connections": new_connections,
        "settings": source.get("settings", {}),
    }
    result = _post("/workflows", body)

    return {
        **_format_workflow(result),
        "webhook_url": f"{BASE_URL}/webhook/{webhook_path}",
        "webhook_test_url": f"{BASE_URL}/webhook-test/{webhook_path}",
        "site": site_name,
        "crm_integrated": True,
        "payload_format": {
            "first_name": "string",
            "last_name": "string",
            "email": "string (required)",
            "phone": "string (optional)",
            "birthday": "string (optional, YYYY-MM-DD)",
            "gender": "string (optional)",
            "zipcode": "string (optional)",
        },
    }


def _update_field_references(node: dict, form_node_name: str) -> dict:
    """Update n8n expression references from form trigger format to webhook JSON format.

    Form triggers output: $json['First Name'], $json.Email, etc.
    Webhooks output: $json.body.first_name, $json.body.email, etc.

    Also updates references like $('Newsletter Signup Form').item.json.Email
    to use the webhook node's output format.
    """
    node_str = json.dumps(node)

    # Replace form-specific field references with webhook-compatible ones
    # The webhook will receive JSON body fields directly
    replacements = [
        (f"$(\'{form_node_name}\').item.json[\'First Name\']", "$json['first_name']"),
        (f"$(\'{form_node_name}\').item.json[\'Last Name\']", "$json['last_name']"),
        (f"$(\'{form_node_name}\').item.json.Email", "$json['email']"),
        (f"$(\'{form_node_name}\').item.json[\'Phone Number\']", "$json['phone']"),
        (f"$(\'{form_node_name}\').item.json.Birthday", "$json['birthday']"),
        (f"$(\'{form_node_name}\').item.json.Zipcode", "$json['zipcode']"),
        (f"$(\'{form_node_name}\').item.json.Gender", "$json['gender']"),
        (f"$(\'{form_node_name}\').item.json.formMode", "'webhook'"),
        # Also handle the form1 variant
        (f"$(\'{form_node_name}1\').item.json[\'First Name\']", "$json['first_name']"),
        (f"$(\'{form_node_name}1\').item.json[\'Last Name\']", "$json['last_name']"),
        (f"$(\'{form_node_name}1\').item.json.Email", "$json['email']"),
        (f"$(\'{form_node_name}1\').item.json[\'Phone Number\']", "$json['phone']"),
        (f"$(\'{form_node_name}1\').item.json.Birthday", "$json['birthday']"),
        (f"$(\'{form_node_name}1\').item.json.Zipcode", "$json['zipcode']"),
        (f"$(\'{form_node_name}1\').item.json.Gender", "$json['gender']"),
        (f"$(\'{form_node_name}1\').item.json.formMode", "'webhook'"),
    ]

    for old, new in replacements:
        node_str = node_str.replace(old, new)

    return json.loads(node_str)
