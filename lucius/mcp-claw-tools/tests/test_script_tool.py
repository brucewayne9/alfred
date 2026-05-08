"""Test the generic ScriptTool wrapper that shells out to integration scripts."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


def test_script_tool_invokes_subcommand_and_parses_json(tmp_path):
    """ScriptTool calls `python3 <script> <command> [args]`, captures stdout as JSON."""
    fake_script = tmp_path / "fake_tool.py"
    fake_script.write_text(
        "import json, sys\n"
        "cmd = sys.argv[1]\n"
        "args = sys.argv[2:]\n"
        "print(json.dumps({'cmd': cmd, 'args': args}))\n"
    )
    from mcp_claw_tools.script_tool import ScriptTool

    tool = ScriptTool(
        name="fake_tool.search-people",
        script_path=str(fake_script),
        command="search-people",
        timeout=30,
    )
    result = tool.invoke({"args": ["alice"]})
    assert result == {"cmd": "search-people", "args": ["alice"]}


def test_script_tool_returns_error_on_nonzero_exit(tmp_path):
    fake_script = tmp_path / "fail.py"
    fake_script.write_text("import sys; sys.exit(2)\n")
    from mcp_claw_tools.script_tool import ScriptTool

    tool = ScriptTool(name="fail.bad", script_path=str(fake_script), command="bad", timeout=10)
    result = tool.invoke({"args": []})
    assert result["error"] == "non-zero exit"
    assert result["exit_code"] == 2


def test_script_tool_handles_non_json_stdout(tmp_path):
    fake_script = tmp_path / "plain.py"
    fake_script.write_text("print('hello world')\n")
    from mcp_claw_tools.script_tool import ScriptTool

    tool = ScriptTool(name="plain.x", script_path=str(fake_script), command="x", timeout=10)
    result = tool.invoke({"args": []})
    # Non-JSON falls back to {"output": "<raw>"}
    assert result == {"output": "hello world"}


def test_script_tool_respects_timeout(tmp_path):
    fake_script = tmp_path / "slow.py"
    fake_script.write_text("import time; time.sleep(60)\n")
    from mcp_claw_tools.script_tool import ScriptTool

    tool = ScriptTool(name="slow.x", script_path=str(fake_script), command="x", timeout=1)
    result = tool.invoke({"args": []})
    assert result["error"] == "timeout"
