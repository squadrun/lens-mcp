"""Every tool must ship annotations, and only the download_* three may be non-read-only.

Without annotations a client has to assume a tool can write, so every lens call
stops for a permission prompt. This is the guard against that regressing.

Run: uv run python tests/test_annotations.py
"""

import asyncio
import os
import sys

os.environ.setdefault("LENS_BASE_URL", "http://localhost")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lens_mcp.server import mcp  # noqa: E402

WRITERS = {"download_entity_logs", "download_trace_logs", "download_prompts"}

TOOLS = asyncio.run(mcp.list_tools())


def test_every_tool_is_annotated():
    missing = [t.name for t in TOOLS if t.annotations is None]
    assert not missing, f"tools with no annotations: {missing}"


def test_only_the_download_tools_are_not_read_only():
    writers = {t.name for t in TOOLS if not t.annotations.readOnlyHint}
    assert writers == WRITERS, f"expected {WRITERS}, got {writers}"


def test_nothing_is_destructive_and_everything_is_open_world():
    for t in TOOLS:
        a = t.annotations
        assert a.destructiveHint is False, f"{t.name} claims destructive"
        assert a.idempotentHint is True, f"{t.name} is not marked idempotent"
        assert a.openWorldHint is True, f"{t.name} is not marked open-world"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"ok   {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
