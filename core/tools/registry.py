"""Tool registry - defines tools the LLM can call to interact with integrations."""

import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Tool registry
_tools: dict[str, dict] = {}


def tool(name: str, description: str, parameters: dict | None = None):
    """Decorator to register a function as an LLM-callable tool."""
    def decorator(func: Callable):
        _tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters or {},
            "function": func,
        }
        return func
    return decorator


def get_tools() -> list[dict]:
    """Get tool definitions for LLM context (without function references)."""
    return [
        {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
        for t in _tools.values()
    ]


# Tool categories for smart filtering
TOOL_CATEGORIES = {
    "email": ["send_email", "search_emails", "get_email"],
    "calendar": ["list_events", "create_event", "update_event", "delete_event"],
    "crm": ["crm_list_people", "crm_get_person", "crm_create_person", "crm_update_person",
            "crm_list_companies", "crm_get_company", "crm_create_company",
            "crm_list_opportunities", "crm_create_opportunity", "crm_update_opportunity"],
    "memory": ["remember", "recall", "store_knowledge", "recall_knowledge", "search_knowledge",
               "ask_knowledge", "store_to_knowledge", "list_knowledge_documents"],
    "server": ["list_servers", "server_status", "server_action", "add_server"],
    "stripe": ["stripe_balance", "stripe_list_customers", "stripe_create_customer",
               "stripe_list_payments", "stripe_refund", "stripe_list_subscriptions",
               "stripe_create_subscription", "stripe_cancel_subscription",
               "stripe_list_invoices", "stripe_create_invoice", "stripe_list_products",
               "stripe_create_payment_link", "stripe_revenue_summary"],
    "nextcloud": ["nc_list_files", "nc_upload_file", "nc_download_file", "nc_create_folder",
                  "nc_list_notes", "nc_create_note", "nc_send_talk_message",
                  "nc_list_talk_conversations", "nc_list_users", "nc_create_user"],
    "n8n": ["n8n_list_workflows", "n8n_execute_workflow", "n8n_create_workflow",
            "n8n_get_executions", "n8n_toggle_workflow"],
    "drive": ["drive_list_files", "drive_search", "drive_create_folder", "drive_upload",
              "drive_download", "drive_share", "drive_delete"],
    "docs": ["docs_create", "docs_read", "docs_append", "docs_replace", "docs_list"],
    "sheets": ["sheets_create", "sheets_read", "sheets_write", "sheets_append_row",
               "sheets_list", "sheets_get"],
    "slides": ["slides_create", "slides_get", "slides_add_slide", "slides_add_text", "slides_list"],
    "core": ["get_current_time", "generate_image"],
}

# Keywords that trigger each category
CATEGORY_KEYWORDS = {
    "email": ["email", "mail", "send", "inbox", "message"],
    "calendar": ["calendar", "schedule", "meeting", "event", "appointment", "book"],
    "crm": ["crm", "contact", "person", "company", "opportunity", "deal", "lead", "customer", "client"],
    "memory": ["remember", "recall", "knowledge", "what did", "what was", "forget",
                "wife", "husband", "birthday", "favorite", "prefer", "my name", "who is", "who am"],
    "server": ["server", "status", "reboot", "restart", "ssh"],
    "stripe": ["stripe", "payment", "invoice", "subscription", "charge", "refund", "billing", "revenue"],
    "nextcloud": ["nextcloud", "cloud storage", "nextcloud file"],
    "n8n": ["n8n", "workflow", "automation", "automate"],
    "drive": ["drive", "google drive", "upload file", "download file", "my files", "share file"],
    "docs": ["doc", "document", "google doc", "write document", "create document"],
    "sheets": ["sheet", "spreadsheet", "google sheet", "excel", "csv", "row", "cell", "column"],
    "slides": ["slide", "presentation", "google slide", "powerpoint", "deck"],
    "core": [],  # Always included
}


def get_relevant_tools(query: str) -> list[str]:
    """Get tool names relevant to the query based on keywords."""
    query_lower = query.lower()
    relevant = set(TOOL_CATEGORIES.get("core", []))  # Always include core tools

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                relevant.update(TOOL_CATEGORIES.get(category, []))
                break

    # If no specific tools matched, include common ones
    if len(relevant) <= 2:
        relevant.update(TOOL_CATEGORIES.get("email", []))
        relevant.update(TOOL_CATEGORIES.get("calendar", []))
        relevant.update(TOOL_CATEGORIES.get("memory", []))

    return list(relevant)


def get_tools_prompt(query: str = None) -> str:
    """Generate a tool description string for the LLM system prompt.

    If query is provided, only includes relevant tools to reduce context size.
    """
    if not _tools:
        return ""

    # Filter tools if query provided
    if query:
        relevant_names = get_relevant_tools(query)
        tools_to_include = {k: v for k, v in _tools.items() if k in relevant_names}
    else:
        tools_to_include = _tools

    if not tools_to_include:
        return ""

    lines = ["You have access to tools. To use one, respond with ONLY a JSON block:",
             '{"tool": "tool_name", "args": {"param1": "value1"}}',
             "",
             "Tools:"]

    for t in tools_to_include.values():
        params = ", ".join(f"{k}" for k in t["parameters"].keys()) if t["parameters"] else ""
        # Shorter description format
        lines.append(f"- {t['name']}: {t['description'][:80]}{'...' if len(t['description']) > 80 else ''}" +
                    (f" ({params})" if params else ""))

    return "\n".join(lines)


async def execute_tool(name: str, args: dict) -> Any:
    """Execute a registered tool by name."""
    import asyncio

    if name not in _tools:
        return {"error": f"Unknown tool: {name}"}

    func = _tools[name]["function"]
    logger.info(f"Executing tool: {name} with args: {args}")

    try:
        result = func(**args)
        # Handle async tools
        if asyncio.iscoroutine(result):
            result = await result
        return result
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return {"error": str(e)}


def parse_tool_call(response: str) -> tuple[str, dict] | None:
    """Try to extract a tool call from LLM response text."""
    # Look for JSON blocks in the response
    import re
    json_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
    matches = re.findall(json_pattern, response, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            if "tool" in data:
                return data["tool"], data.get("args", {})
        except json.JSONDecodeError:
            continue

    # Also try inline JSON
    try:
        if '{"tool"' in response:
            start = response.index('{"tool"')
            # Find matching closing brace
            depth = 0
            for i, c in enumerate(response[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        data = json.loads(response[start:i+1])
                        if "tool" in data:
                            return data["tool"], data.get("args", {})
                        break
    except (json.JSONDecodeError, ValueError):
        pass

    return None
