"""The server instructions name only tools this server has.

Clients show the instructions before any tool call, so a renamed or removed tool
left in them sends the model to a tool that does not exist.

Run: uv run python tests/test_instructions.py
"""

import asyncio
import os
import re
import sys

os.environ.setdefault("LENS_BASE_URL", "http://localhost")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lens_mcp.server import mcp

TOOLS = {t.name for t in asyncio.run(mcp.list_tools())}


def test_every_named_tool_exists():
    instructions = mcp.instructions or ""
    named = set(re.findall(r"`([a-z_]+)`", instructions))
    assert named, "the instructions name no tools"
    assert named <= TOOLS, f"unknown tools: {sorted(named - TOOLS)}"


if __name__ == "__main__":
    test_every_named_tool_exists()
    print("ok")
