"""Prints how many characters a tool response costs before and after the compact-text and
fields/event_name changes. Not a test. Offline: replays cached span payloads (when the
first-sentence study cache is on disk, else synthetic ones) through the real tool functions.

Run: uv run python tests/measure_tokens.py
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fakes import FakeLens, main, patched, run, server, span
from mcp import types
from pydantic_core import to_json

STUDY_CACHE = os.environ.get("LENS_TURN_STUDY_DIR", "")
HANDLER = server.mcp._mcp_server.request_handlers[types.CallToolRequest]


def call_tool(name, **args):
    req = types.CallToolRequest(
        method="tools/call", params=types.CallToolRequestParams(name=name, arguments=args)
    )
    return run(HANDLER(req)).root


def old_render(payload):
    return len(to_json(payload, indent=2).decode())  # what FastMCP 1.28.1 emitted before


def cached_calls(n=3):
    out = []
    for p in sorted(glob.glob(os.path.join(STUDY_CACHE, "cache", "*.json"))):
        with open(p) as f:
            c = json.load(f)
        rows = [
            {**{k: v for k, v in r.items() if k != "md"}, "metadata": json.dumps(r.get("md", {}))}
            for kind in ("llm", "tts", "react", "tool")
            for r in c[kind]
        ]
        out.append(rows)
    out.sort(key=len, reverse=True)
    if out:
        return out[:n]
    return [  # synthetic stand-ins when the study cache is absent
        [span("tts.cartesia", "tts.first_audio", 1e12 + i, i % 9, "ttfb", 50.0 + i,
              cache_hit=bool(i % 2), characters=40 + i) for i in range(60 + 20 * k)]
        for k in range(n)
    ]  # fmt: skip


def measure():
    table = []
    for i, rows in enumerate(cached_calls(), 1):
        fake = FakeLens(spans={"c1": rows})
        with patched(fake):
            r = call_tool("get_call_spans", call_sid="c1")
            old, new = old_render(r.structuredContent), len(r.content[0].text)
            table.append(
                (
                    f"call {i}: get_call_spans, first {min(len(rows), 500)} of {len(rows)} spans",
                    old,
                    new,
                )
            )
            ev = rows[len(rows) // 2]["event_name"]
            r = call_tool("get_call_spans", call_sid="c1", event_name=ev,
                          fields="turn_number,value_ms,metadata.cache_hit")  # fmt: skip
            table.append((f"call {i}: + event_name={ev}, fields", old, len(r.content[0].text)))
    listing = [
        {"call_id": f"{i:032x}", "duration_ms": 60000 + i, "total_turns": 12, "error_count": 0,
         "providers": {"stt": "deepgram", "llm": "gemma4", "tts": "cartesia"},
         "start_time": "2026-10-20T10:00:00", "status": "completed", "entity_status": "success",
         "warmup_ms": 900, "flags": {"has_tool": True}}
        for i in range(100)
    ]  # fmt: skip
    with patched(FakeLens(calls=listing)):
        r = call_tool("list_calls", limit=100)
        old = old_render(r.structuredContent)
        table.append(("list_calls, 100 rows", old, len(r.content[0].text)))
        r = call_tool("list_calls", limit=100, fields="call_id,duration_ms,total_turns")
        table.append(("list_calls, 100 rows + fields", old, len(r.content[0].text)))
    return table


def test_prints_before_after_table():
    print(f"{'case':58s} {'before':>8s} {'after':>8s} {'ratio':>6s}")
    for name, old, new in measure():
        assert new < old
        print(f"{name:58s} {old:8d} {new:8d} {new / old:6.2f}")


if __name__ == "__main__":
    main(globals())
