"""Checks for the *_over_time tools: the granularity enum, the overflow digest, the rollups.

_check_granularity only bounds the bucket count, never the second (node/event_name)
dimension, so an unfiltered call can fan out across every distinct value fleet-wide. Past
_MAX_ROWS_UNFILTERED that series is swapped for per-dimension totals rather than returned.

Run: uv run python tests/test_over_time.py
"""

import asyncio
import os
import sys

os.environ.setdefault("LENS_BASE_URL", "http://localhost")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lens_mcp.server import (
    _check_granularity,
    _rollup,
    _summarize_overflow,
    mcp,
)

TOOLS = {t.name: t for t in asyncio.run(mcp.list_tools())}


def test_granularity_is_published_as_an_enum_per_tool():
    for name in ("latency_over_time", "event_counts_over_time", "error_counts_over_time"):
        schema = TOOLS[name].inputSchema["properties"]["granularity"]
        assert schema["type"] == "string"

    # latency_over_time can't roll percentiles into coarser buckets — only counts are additive.
    assert TOOLS["latency_over_time"].inputSchema["properties"]["granularity"]["enum"] == [
        "5min",
        "15min",
        "1hour",
    ]
    for name in ("event_counts_over_time", "error_counts_over_time"):
        assert TOOLS[name].inputSchema["properties"]["granularity"]["enum"] == [
            "5min",
            "10min",
            "15min",
            "30min",
            "1hour",
            "1day",
        ]


def _oversized(rows_per_dim=1000):
    series = []
    for i in range(rows_per_dim):
        series.append({"ts": f"t{i}", "event_name": "big", "count": 3, "calls": 2})
        series.append({"ts": f"t{i}", "event_name": "small", "count": 1, "calls": 1})
    series.append({"ts": "t0", "event_name": "rare", "count": 1, "calls": 1})
    return {"series": series, "other": "kept"}


def test_unfiltered_overflow_becomes_ranked_totals():
    out = _summarize_overflow(_oversized(), "", "event_types", "event_name", ("count", "calls"))

    assert "series" not in out and "TRUNCATED" in out and "event_types" in out["TRUNCATED"]
    assert out["other"] == "kept"  # non-series keys survive
    assert [t["event_name"] for t in out["totals"]] == ["big", "small", "rare"]  # largest first
    assert out["totals"][0] == {"event_name": "big", "buckets": 1000, "count": 3000, "calls": 2000}


def test_overflow_never_combines_percentiles():
    series = [{"node": "llm", "count": 10, "p90_ms": float(v)} for v in (500, 100, 900)] * 700
    out = _summarize_overflow({"series": series}, "", "nodes", "node", ("count",), ("p90_ms",))

    assert out["totals"][0]["count"] == 21000
    assert out["totals"][0]["p90_ms_range"] == (100.0, 900.0)  # a range, not a fake mean
    assert "p90_ms" not in out["totals"][0]


def test_overflow_digest_skipped_when_filtered_small_or_unrecognized():
    big = _oversized()
    assert _summarize_overflow(big, "big,small", "event_types", "event_name", ("count",)) is big
    small = {"series": [{"event_name": "a", "count": 1}]}
    assert _summarize_overflow(small, "", "event_types", "event_name", ("count",)) is small
    for odd in ({}, {"series": "not-a-list"}):
        assert _summarize_overflow(odd, "", "event_types", "event_name", ("count",)) is odd


def test_rollup_1day_sums_exact_and_approx_across_hour_boundaries():
    series = [
        {"ts": "2026-01-01T00:00:00", "event_name": "a", "count": 5, "calls": 5},
        {"ts": "2026-01-01T03:00:00", "event_name": "a", "count": 7, "calls": 6},
        {"ts": "2026-01-01T00:00:00", "event_name": "b", "count": 1, "calls": 1},
        {"ts": "2026-01-02T00:00:00", "event_name": "a", "count": 2, "calls": 2},
    ]
    out = _rollup(series, "1day", "event_name", "count", "calls")
    by_key = {(r["ts"], r["event_name"]): r for r in out}

    assert len(out) == 3  # (day, dimension) pairs, not raw hourly rows
    assert by_key[("2026-01-01", "a")]["count"] == 12  # exact: every event belongs to one hour
    assert by_key[("2026-01-01", "a")]["calls"] == 11  # upper bound, not a true distinct count
    assert by_key[("2026-01-01", "b")]["count"] == 1
    assert by_key[("2026-01-02", "a")]["count"] == 2


def test_rollup_10min_and_30min_align_to_the_clock():
    five = [
        {"ts": f"2026-01-01T03:{m:02d}:00", "node": "llm", "errors": 1, "affected_calls": 1}
        for m in range(0, 60, 5)
    ]
    out = _rollup(five, "10min", "node", "errors", "affected_calls")
    assert [r["ts"][11:16] for r in out] == ["03:00", "03:10", "03:20", "03:30", "03:40", "03:50"]
    assert all(r["errors"] == 2 for r in out)

    # A window starting at :45 leaves a half-filled first bucket rather than shifting the grid.
    fifteen = [
        {"ts": ts, "node": "llm", "errors": 3, "affected_calls": 2}
        for ts in ("2026-01-01T02:45:00", "2026-01-01T03:00:00", "2026-01-01T03:15:00")
    ]
    out = _rollup(fifteen, "30min", "node", "errors", "affected_calls")
    assert [(r["ts"], r["errors"]) for r in out] == [
        ("2026-01-01T02:30:00", 3),
        ("2026-01-01T03:00:00", 6),
    ]


def test_rollup_rejects_a_row_with_no_parseable_ts():
    for bad_row in ({"event_name": "a", "count": 1}, {"ts": "garbage", "event_name": "a"}):
        try:
            _rollup([bad_row], "30min", "event_name", "count", "calls")
            raise AssertionError(f"expected {bad_row} to be rejected, not silently mis-grouped")
        except RuntimeError as e:
            assert "granularity=15min" in str(e)  # names the backend size that still works


def test_client_rollups_are_sized_on_the_rows_actually_fetched():
    assert _check_granularity("30min", 360, "", "") == "15min"
    assert _check_granularity("1hour", 360, "", "") == "1hour"
    # 5 days at 10min is 720 buckets, but it's fetched as 1440 5min buckets — past the cap.
    try:
        _check_granularity("10min", 7200, "", "")
        raise AssertionError("expected the 5min fetch behind 10min to be refused")
    except ValueError as e:
        assert "fetched as 5min" in str(e)


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
