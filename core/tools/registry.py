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
    "email": ["check_email", "read_email", "send_email", "unread_email_count",
              "email_list_accounts", "email_inbox", "email_read", "email_send", "email_search", "email_unread",
              "email_trash", "email_mark_read", "email_mark_unread"],
    "calendar": ["check_calendar", "create_event", "find_free_time", "today_schedule"],
    "crm": ["crm_add_note_to_company", "crm_add_note_to_deal", "crm_add_note_to_person",
            "crm_add_task_to_company", "crm_add_task_to_deal", "crm_add_task_to_person",
            "crm_create_company", "crm_create_opportunity", "crm_create_person", "crm_create_task",
            "crm_delete_company", "crm_delete_opportunity", "crm_delete_person", "crm_delete_task",
            "crm_get_company", "crm_get_opportunity", "crm_get_person",
            "crm_list_companies", "crm_list_opportunities", "crm_list_people", "crm_list_tasks",
            "crm_pipeline_summary", "crm_search_companies", "crm_search_opportunities", "crm_search_people",
            "crm_update_company", "crm_update_deal_stage", "crm_update_person", "crm_update_task"],
    "memory": ["remember", "recall", "search_knowledge", "ask_knowledge",
               "store_to_knowledge", "list_knowledge_documents"],
    "server": ["list_servers", "server_status", "server_command",
               "docker_containers", "docker_restart"],
    "stripe": ["stripe_add_invoice_item", "stripe_cancel_subscription", "stripe_create_customer",
               "stripe_create_invoice", "stripe_create_payment_link", "stripe_create_price",
               "stripe_create_product", "stripe_create_subscription", "stripe_deactivate_payment_link",
               "stripe_delete_customer", "stripe_finalize_invoice", "stripe_get_balance",
               "stripe_get_customer", "stripe_get_invoice", "stripe_get_payment", "stripe_get_product",
               "stripe_get_subscription", "stripe_list_customers", "stripe_list_invoices",
               "stripe_list_payment_links", "stripe_list_payments", "stripe_list_payouts",
               "stripe_list_prices", "stripe_list_products", "stripe_list_refunds",
               "stripe_list_subscriptions", "stripe_refund_payment", "stripe_resume_subscription",
               "stripe_revenue_summary", "stripe_search_customers", "stripe_search_payments",
               "stripe_send_invoice", "stripe_update_customer", "stripe_void_invoice"],
    "nextcloud": ["nextcloud_add_participant", "nextcloud_add_user_to_group", "nextcloud_create_conversation",
                  "nextcloud_create_folder", "nextcloud_create_note", "nextcloud_create_user",
                  "nextcloud_delete_file", "nextcloud_delete_note", "nextcloud_delete_user",
                  "nextcloud_disable_user", "nextcloud_enable_user", "nextcloud_get_calendar_events",
                  "nextcloud_get_messages", "nextcloud_get_note", "nextcloud_get_tasks", "nextcloud_get_user",
                  "nextcloud_list_calendars", "nextcloud_list_conversations", "nextcloud_list_files",
                  "nextcloud_list_groups", "nextcloud_list_notes", "nextcloud_list_users",
                  "nextcloud_move_file", "nextcloud_read_file", "nextcloud_search_files",
                  "nextcloud_send_message", "nextcloud_storage_info", "nextcloud_update_note",
                  "nextcloud_upload_file", "nextcloud_upload_attached_file"],
    "n8n": ["n8n_activate_workflow", "n8n_create_scheduled_workflow", "n8n_create_webhook_slack_workflow",
            "n8n_create_workflow", "n8n_deactivate_workflow", "n8n_delete_workflow",
            "n8n_execute_workflow", "n8n_get_executions", "n8n_get_workflow", "n8n_list_workflows"],
    "radio": [
        # Basic info
        "radio_list_stations", "radio_now_playing", "radio_song_history", "radio_queue", "radio_listeners",
        # Playback control
        "radio_skip_song", "radio_clear_queue", "radio_request_song", "radio_restart",
        # Playlists
        "radio_playlists", "radio_toggle_playlist", "radio_create_playlist", "radio_delete_playlist", "radio_reshuffle_playlist",
        # Media
        "radio_search_media", "radio_upload_song", "radio_update_song", "radio_delete_song", "radio_create_folder",
        # DJs/Streamers
        "radio_list_djs", "radio_create_dj", "radio_update_dj", "radio_delete_dj",
        # Mount points
        "radio_list_mounts", "radio_create_mount", "radio_delete_mount",
        # Webhooks
        "radio_list_webhooks", "radio_toggle_webhook",
        # Station admin
        "radio_create_station", "radio_update_station", "radio_delete_station", "radio_clone_station",
        # User admin
        "radio_list_users", "radio_create_user", "radio_update_user", "radio_delete_user", "radio_list_roles",
        # Storage
        "radio_storage_locations", "radio_update_storage_quota", "radio_station_quota",
        # System
        "radio_system_status", "radio_services_status",
    ],
    "drive": ["drive_list_files", "drive_search", "drive_create_folder", "drive_upload",
              "drive_download", "drive_share", "drive_delete"],
    "docs": ["docs_create", "docs_read", "docs_append", "docs_replace", "docs_list",
             "analyze_document", "create_document"],
    "sheets": ["sheets_create", "sheets_read", "sheets_write", "sheets_append_row",
               "sheets_list", "sheets_get"],
    "slides": ["slides_create", "slides_get", "slides_add_slide", "slides_add_text", "slides_list"],
    "core": ["generate_image", "toggle_auto_speak", "toggle_hands_free"],
    "ui_control": ["toggle_auto_speak", "toggle_hands_free"],
    "meta_ads": [
        "meta_ads_account", "meta_ads_summary", "meta_ads_performance",
        "meta_ads_campaigns", "meta_ads_campaign_insights",
        "meta_ads_ad_sets", "meta_ads_ad_set_insights",
        "meta_ads_ads", "meta_ads_ad_insights", "meta_ads_insights",
        "meta_ads_audience", "meta_ads_placements",
        "meta_ads_issues", "meta_ads_daily_spend",
        # Write operations
        "meta_ads_pause_ad", "meta_ads_enable_ad",
        "meta_ads_pause_ad_set", "meta_ads_enable_ad_set",
        "meta_ads_pause_campaign", "meta_ads_enable_campaign",
        "meta_ads_update_ad_set_budget", "meta_ads_update_campaign_budget",
    ],
    "analytics": [
        "ga_list_properties", "ga_traffic_summary", "ga_realtime",
        "ga_traffic_sources", "ga_top_pages", "ga_devices",
        "ga_countries", "ga_daily_traffic", "ga_all_properties",
    ],
    "agents": [
        "spawn_agent", "check_agent_task", "list_agent_tasks", "cancel_agent_task",
    ],
    "briefing": [
        "daily_briefing", "quick_briefing",
    ],
}

# Keywords that trigger each category
CATEGORY_KEYWORDS = {
    "email": ["email", "mail", "send", "inbox", "message", "groundrush", "rucktalk", "loovacast",
              "lumabot", "luma bot", "support@loovacast", "mjohnson", "info@rucktalk", "info@loovacast", "info@groundrush",
              "trash", "delete email", "mark read", "mark unread"],
    "calendar": ["calendar", "schedule", "meeting", "event", "appointment", "book"],
    "crm": ["crm", "contact", "person", "company", "opportunity", "deal", "lead", "customer", "client"],
    "memory": ["remember", "recall", "knowledge", "what did", "what was", "forget",
                "wife", "husband", "birthday", "favorite", "prefer", "my name", "who is", "who am"],
    "server": ["server", "status", "reboot", "restart", "ssh"],
    "stripe": ["stripe", "payment", "invoice", "subscription", "charge", "refund", "billing", "revenue"],
    "nextcloud": ["nextcloud", "next cloud", "cloud storage", "nextcloud file",
                  "talk conversation", "talk chat", "my notes", "eswar", "donjuan"],
    "n8n": ["n8n", "workflow", "automation", "automate"],
    "radio": ["radio", "station", "playing", "song", "music", "playlist", "listener", "stream", "dj", "broadcast",
              "azuracast", "streamer", "mount", "webhook", "skip", "queue", "request", "upload song", "media library"],
    "drive": ["drive", "google drive", "upload file", "download file", "my files", "share file"],
    "docs": ["doc", "document", "google doc", "write document", "create document"],
    "sheets": ["sheet", "spreadsheet", "google sheet", "excel", "csv", "row", "cell", "column"],
    "slides": ["slide", "presentation", "google slide", "powerpoint", "deck"],
    "ui_control": ["auto speak", "auto-speak", "autospeak", "hands free", "hands-free", "handsfree",
                   "mute", "unmute", "stop speaking", "start speaking", "stop listening", "start listening"],
    "meta_ads": ["meta", "facebook", "instagram", "meta ads", "facebook ads", "instagram ads", "fb ads",
                 "campaign", "ad set", "ad performance", "ctr", "cpc", "cpm", "roas", "impressions",
                 "reach", "conversions", "ad spend", "advertising", "ads manager", "audience insights",
                 "pause ad", "enable ad", "unpause", "budget", "daily budget", "underperform"],
    "analytics": ["analytics", "google analytics", "ga4", "traffic", "visitors", "page views", "pageviews",
                  "bounce rate", "sessions", "active users", "realtime", "real-time", "top pages",
                  "traffic sources", "where visitors", "website stats", "site traffic", "daily traffic",
                  "lenssniper", "loovacast", "lumabot", "luma bot", "myhands", "car wash", "nightlife",
                  "rodwave", "rod wave", "rucktalk", "ag entertainment"],
    "agents": ["spawn agent", "agent", "delegate", "specialist", "coder agent", "research agent",
               "analyst agent", "writer agent", "parallel", "background task", "multi-step"],
    "briefing": ["briefing", "morning", "daily summary", "what's on my", "my day", "today's schedule",
                 "good morning", "start my day", "catch me up", "status update", "overview"],
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
