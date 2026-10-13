"""Offline regressions for time-input precedence and the existing query-window limits.

Run: uv run python tests/test_windows.py
"""

import asyncio
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, patch

os.environ.setdefault("LENS_BASE_URL", "https://lens.invalid")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp import types

import lens_mcp.server as server

START = "2026-01-01T10:00:00+00:00"
END = "2026-01-01T10:30:00+00:00"
HANDLER = server.mcp._mcp_server.request_handlers[types.CallToolRequest]
TOOLS = {
    "slowest_calls": {"node": "llm"},
    "search_trace_logs": {"query": "example"},
}


def _call(name, args):
    backend = AsyncMock(return_value={"marker": "offline"})
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments={**TOOLS[name], **args}),
    )
    with patch.object(server, "_get", backend):
        result = asyncio.run(HANDLER(request)).root
    return result, backend


def _assert_explicit_request(name, args):
    result, backend = _call(name, args)
    assert result.isError is False, (name, result.content)
    assert backend.await_count == 1, name
    sent = backend.await_args.kwargs["params"]
    assert sent["time_from"] == args["time_from"], (name, sent)
    assert sent["time_to"] == args["time_to"], (name, sent)
    assert "time_range_minutes" not in sent, (name, sent)
    assert result.structuredContent["marker"] == "offline", name


def test_explicit_window_overrides_a_positive_relative_window():
    assert server._window_minutes(60, START, END) == 30.0


def test_explicit_window_overrides_the_cohort_default():
    assert server._window_minutes(360, START, END) == 30.0


def test_explicit_window_without_relative_minutes():
    assert server._window_minutes(0, START, END) == 30.0


def test_offset_timestamps_measure_the_same_explicit_window():
    start = "2026-01-01T12:00:00+02:00"
    assert server._window_minutes(60, start, END) == 30.0


def test_open_ended_explicit_window_uses_the_start_timezone():
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return cls(2026, 1, 1, 10, 30)
            return cls(2026, 1, 1, 10, 30, tzinfo=tz)

    with patch.object(server, "datetime", FixedDatetime):
        for start in (START, "2026-01-01T10:00:00"):
            assert server._window_minutes(60, start, "") == 30.0, start


def test_invalid_explicit_timestamps_do_not_fall_back_to_relative_minutes():
    for start, end in (("not-a-date", END), (START, "not-a-date")):
        assert server._window_minutes(60, start, end) is None, (start, end)


def test_mixed_aware_and_naive_explicit_timestamps_are_not_a_valid_window():
    assert server._window_minutes(60, START, "2026-01-01T10:30:00") is None


def test_relative_window_is_preserved_without_an_explicit_start():
    for minutes in (30, 60, 120):
        assert server._window_minutes(minutes, "", "") == float(minutes)
        assert server._window_minutes(minutes, "", END) == float(minutes)


def test_missing_lower_bound_remains_unknown():
    for minutes in (0, -1):
        assert server._window_minutes(minutes, "", "") is None
        assert server._window_minutes(minutes, "", END) is None


def test_registered_tools_accept_a_short_explicit_window_with_their_defaults():
    # The cohort tool's default is 360 minutes; the supplied interval is 30.
    # Exercise the published MCP handler and inspect only the mocked request.
    for name in TOOLS:
        _assert_explicit_request(name, {"time_from": START, "time_to": END})


def test_registered_tools_accept_the_explicit_limit_boundary():
    for name in TOOLS:
        _assert_explicit_request(name, {"time_from": START, "time_to": "2026-01-01T12:00:00+00:00"})


def test_registered_tools_accept_relative_windows_at_the_limit():
    for name in TOOLS:
        result, backend = _call(name, {"time_range_minutes": 120})
        assert result.isError is False, (name, result.content)
        assert backend.await_count == 1, name
        sent = backend.await_args.kwargs["params"]
        assert sent["time_range_minutes"] == 120, (name, sent)
        assert "time_from" not in sent and "time_to" not in sent, (name, sent)


def test_registered_tools_keep_the_existing_relative_limit():
    for name in TOOLS:
        result, backend = _call(name, {"time_range_minutes": 121})
        assert result.isError is True, name
        assert backend.await_count == 0, name


def test_registered_tools_keep_the_existing_explicit_limit_without_a_relative_window():
    # A control for the existing boundary: one minute past the limit.
    args = {
        "time_range_minutes": 0,
        "time_from": START,
        "time_to": "2026-01-01T12:01:00+00:00",
    }
    for name in TOOLS:
        result, backend = _call(name, args)
        assert result.isError is True, name
        assert backend.await_count == 0, name


def test_documented_scoped_windows_are_preserved():
    # Scoped requests may exceed the unscoped limit. All responses stay offline.
    scopes = {
        "slowest_calls": ("campaign_id", "agent_config_id", "prompt_ref"),
        "search_trace_logs": ("campaign_id",),
    }
    for name, filters in scopes.items():
        for field in filters:
            args = {
                "time_range_minutes": 0,
                "time_from": START,
                "time_to": "2026-01-01T12:01:00+00:00",
                field: "test-scope",
            }
            _assert_explicit_request(name, args)


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
