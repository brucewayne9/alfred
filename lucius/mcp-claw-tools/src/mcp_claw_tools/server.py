"""MCP server (stdio) for the Lucius claw-tools toolkit.

Loads tools.json (packaged), instantiates one ScriptTool per entry, and serves
them via stdio transport. Reads LUCIUS_ENV_FILE on startup to source the
secret subset that the wrapped integration scripts need.
"""
import json
import os
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from mcp_claw_tools.script_tool import ScriptTool


def _load_dotenv(path: Path) -> int:
    """Source a KEY=VALUE .env file into os.environ (best-effort, no expansions).

    Returns count of keys loaded. Skips comments and blank lines. Does not
    overwrite existing env vars (so explicit env from systemd wins).
    """
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


def load_tools() -> list[ScriptTool]:
    """Read tools.json (packaged) and return a list of ScriptTool instances."""
    raw = files("mcp_claw_tools").joinpath("tools.json").read_text()
    cfg = json.loads(raw)
    base = cfg["scripts_dir"]
    out: list[ScriptTool] = []
    for entry in cfg["tools"]:
        out.append(ScriptTool(
            name=entry["name"],
            script_path=os.path.join(base, entry["script"]),
            command=entry["command"],
            timeout=entry.get("timeout", 120),
        ))
    return out


def _tool_descriptors() -> list[Tool]:
    """Re-parse tools.json descriptions for MCP `Tool` registration."""
    raw = files("mcp_claw_tools").joinpath("tools.json").read_text()
    cfg = json.loads(raw)
    return [
        Tool(
            name=e["name"],
            description=e.get("description", e["name"]),
            inputSchema={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Positional CLI arguments passed to the underlying script command",
                    },
                },
                "required": [],
            },
        )
        for e in cfg["tools"]
    ]


async def main_async() -> None:
    # Source the Lucius env subset before instantiating tools so subprocess
    # invocations inherit it via {**os.environ}.
    env_file = os.environ.get("LUCIUS_ENV_FILE")
    if env_file:
        _load_dotenv(Path(env_file))

    server = Server("claw-tools")
    tools = load_tools()
    by_name = {t.name: t for t in tools}

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return _tool_descriptors()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        tool = by_name.get(name)
        if tool is None:
            return [TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]
        result = tool.invoke(arguments or {})
        return [TextContent(type="text", text=json.dumps(result))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
