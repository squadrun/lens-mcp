"""JSON tools must publish an object outputSchema and send the payload as structuredContent.

A tool annotated `-> str` is published as {"result": string}, so clients that read
structuredContent get the whole response as one escaped JSON string. The text block must stay
the same JSON either way, so clients that only read text see no change.

Run: uv run python tests/test_output_schema.py
"""

import asyncio
import json
import os
import sys

os.environ.setdefault("LENS_BASE_URL", "http://localhost")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp import types

import lens_mcp.server as server

TEXT_TOOLS = {"get_schema", "download_entity_logs", "download_trace_logs", "download_prompts"}
STRING_WRAPPER = {"result": {"title": "Result", "type": "string"}}

TOOLS = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
HANDLER = server.mcp._mcp_server.request_handlers[types.CallToolRequest]


def _call(name, args, backend_payload):
    async def fake_get(ctx, path, params=None):
        return backend_payload

    real, server._get = server._get, fake_get
    try:
        req = types.CallToolRequest(
            method="tools/call", params=types.CallToolRequestParams(name=name, arguments=args)
        )
        return asyncio.run(HANDLER(req)).root
    finally:
        server._get = real


def test_json_tools_publish_an_object_schema_and_text_tools_keep_the_string():
    for name, tool in TOOLS.items():
        props = (tool.outputSchema or {}).get("properties")
        if name in TEXT_TOOLS:
            assert props == STRING_WRAPPER, f"{name} should stay a plain-text result"
        else:
            assert tool.outputSchema.get("type") == "object", f"{name} has no object schema"
            assert props != STRING_WRAPPER, f"{name} still wraps its JSON in a string"


def test_structured_content_is_the_object_and_text_is_the_same_json():
    payload = {"call_id": "abc", "turns": [{"role": "bot", "text": "namaste — hi"}]}
    r = _call("get_call_transcript", {"call_sid": "abc"}, payload)

    assert r.isError is False
    assert r.structuredContent == payload
    # Text-only clients read the same JSON value, written compact with non-ASCII unescaped.
    assert r.content[0].text == json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def test_full_series_bypasses_the_overflow_summary():
    rows = [
        {"ts": f"t{i}", "event_name": f"e{i % 50}", "count": 1, "calls": 1} for i in range(3000)
    ]

    summarized = _call("event_counts_over_time", {}, {"series": rows}).structuredContent
    assert "TRUNCATED" in summarized and len(summarized["series"]) < len(rows)

    full = _call(
        "event_counts_over_time", {"full_series": True}, {"series": rows}
    ).structuredContent
    assert len(full["series"]) == 3000 and "TRUNCATED" not in full


def test_named_filters_are_bounded_too():
    rows = [{"ts": f"t{i}", "event_name": "a", "count": 1, "calls": 1} for i in range(3000)]
    out = _call("event_counts_over_time", {"event_types": "a"}, {"series": rows}).structuredContent
    assert "TRUNCATED" in out


def test_overflow_percentile_ranges_pass_output_validation():
    rows = [{"ts": f"t{i}", "node": f"n{i % 30}", "count": 1, "p90_ms": 1.0} for i in range(3000)]
    r = _call("latency_over_time", {}, {"series": rows})
    assert r.isError is False
    # (min, max) tuples in structuredContent reach text-only clients as the same JSON arrays.
    assert json.loads(r.content[0].text) == json.loads(json.dumps(r.structuredContent))


def test_rollup_without_a_series_list_errors_instead_of_mislabelling_rows():
    rows = [{"ts": "2026-01-01T03:00:00", "event_name": "a", "count": 1, "calls": 1}]
    r = _call("event_counts_over_time", {"granularity": "10min"}, {"rows": rows})
    assert r.isError is True
    assert "granularity=5min" in r.content[0].text


def test_extraction_stats_enums_match_what_client_and_backend_accept():
    props = TOOLS["get_extraction_stats"].inputSchema["properties"]
    assert props["view"]["enum"] == ["outcomes", "latency", "models", "filters"]
    assert props["granularity"]["enum"] == ["5min", "15min", "1hour"]  # backend 422s anything else


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
