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


def get_tools_prompt() -> str:
    """Generate a tool description string for the LLM system prompt."""
    if not _tools:
        return ""

    lines = ["You have access to the following tools. To use a tool, respond with a JSON block like:",
             '```json',
             '{"tool": "tool_name", "args": {"param1": "value1"}}',
             '```',
             "",
             "Available tools:"]

    for t in _tools.values():
        params = ", ".join(f"{k}: {v}" for k, v in t["parameters"].items()) if t["parameters"] else "none"
        lines.append(f"- **{t['name']}**: {t['description']} (params: {params})")

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
