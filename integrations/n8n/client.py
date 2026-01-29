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
    """Create a new workflow with nodes and connections."""
    body = {
        "name": name,
        "nodes": nodes,
        "connections": connections,
        "active": active,
        "settings": {
            "executionOrder": "v1"
        },
    }
    data = _post("/workflows", body)
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
    """Execute a workflow manually with optional input data."""
    body = {"data": data} if data else {}
    result = _post(f"/workflows/{workflow_id}/run", body)
    return {
        "execution_id": result.get("executionId"),
        "success": result.get("finished", False),
        "data": result.get("data"),
    }


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
