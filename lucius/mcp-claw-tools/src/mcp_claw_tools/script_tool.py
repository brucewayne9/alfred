"""Generic shell-out wrapper for ~/.lucius/workspace/scripts/integrations/*.py.

Each Python script in the integrations dir exposes a CLI of the form
    python3 <script> <command> [args...]
and prints JSON to stdout. This wrapper makes one MCP tool per (script, command).
"""
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class ScriptTool:
    name: str
    script_path: str
    command: str
    timeout: int = 120

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run `python3 <script> <command> <args>` and return parsed result."""
        args = payload.get("args", [])
        if not isinstance(args, list):
            return {"error": "args must be a list of strings", "got": str(type(args))}

        cmd = [sys.executable, self.script_path, self.command, *map(str, args)]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ},
            )
        except subprocess.TimeoutExpired:
            return {"error": "timeout", "timeout_s": self.timeout, "command": self.command}

        if proc.returncode != 0:
            return {
                "error": "non-zero exit",
                "exit_code": proc.returncode,
                "stderr": proc.stderr.strip()[:2000],
                "command": self.command,
            }

        out = proc.stdout.strip()
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {"output": out}
