"""Smoke test: server module loads, tools.json parses, ScriptTool factory works."""
from pathlib import Path


def test_server_loads_tools_json():
    from mcp_claw_tools.server import load_tools
    tools = load_tools()
    assert len(tools) >= 50, f"expected ≥50 tools, got {len(tools)}"


def test_no_insert_tool_for_lightrag():
    """Critical safety: lightrag_client.py insert MUST NOT be exposed."""
    from mcp_claw_tools.server import load_tools
    tools = load_tools()
    bad = [t for t in tools if t.name.startswith("memory.") and "insert" in t.command]
    assert bad == [], f"forbidden insert tools registered: {[t.name for t in bad]}"


def test_tool_names_are_unique():
    from mcp_claw_tools.server import load_tools
    tools = load_tools()
    names = [t.name for t in tools]
    assert len(names) == len(set(names)), "duplicate tool names in tools.json"
