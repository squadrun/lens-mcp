"""Checks for the *_over_time tools: the granularity enum, the overflow digest, the rollups.

_check_granularity only bounds the bucket count, never the second (node/event_name)
dimension, so a call can fan out across every distinct value fleet-wide. Past
_RESPONSE_BUDGET_CHARS the series is cut to per-dimension totals plus the largest series that fit.

Run: uv run python tests/test_over_time.py
"""

import asyncio
import os
import sys

os.environ.setdefault("LENS_BASE_URL", "http://localhost")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lens_mcp.server import (
    _CLIENT_ROLLUPS,
    _GRANULARITY_MINUTES,
    _RESPONSE_BUDGET_CHARS,
    _check_granularity,
    _json_chars,
    _rollup,
    _summarize_overflow,
    mcp,
)

TOOLS = {t.name: t for t in asyncio.run(mcp.list_tools())}


def test_granularity_enums_match_the_sizes_the_server_can_serve():
    def enum(name):
        return TOOLS[name].inputSchema["properties"]["granularity"]["enum"]

    # latency_over_time can't roll percentiles into coarser buckets — only counts are additive.
    assert set(enum("latency_over_time")) == set(_GRANULARITY_MINUTES)
    for name in ("event_counts_over_time", "error_counts_over_time"):
        assert set(enum(name)) == set(_GRANULARITY_MINUTES) | set(_CLIENT_ROLLUPS), name


def _oversized(rows_per_dim=250):
    # Each dimension's rows are ~13 KB of JSON: two fit in the budget, three don't.
    series = []
    for i in range(rows_per_dim):
        for name, count in (("big", 3), ("mid", 2), ("small", 1)):
            series.append({"ts": f"t{i}", "event_name": name, "count": count, "calls": 1})
    return {"series": series, "other": "kept"}


def test_overflow_keeps_totals_and_the_largest_series_that_fit():
    data = _oversized()
    assert _json_chars(data["series"]) > _RESPONSE_BUDGET_CHARS
    out = _summarize_overflow(data, "event_types", "event_name", ("count", "calls"))

    assert "TRUNCATED" in out and "event_types" in out["TRUNCATED"]
    assert out["other"] == "kept"  # non-series keys survive
    assert [t["event_name"] for t in out["totals"]] == ["big", "mid", "small"]  # largest first
    assert out["totals"][0] == {"event_name": "big", "buckets": 250, "count": 750, "calls": 250}
    assert {r["event_name"] for r in out["series"]} == {"big", "mid"}  # a top-K: small doesn't fit
    assert len(out["series"]) == 500  # every bucket of each kept dimension, none trimmed
    assert _json_chars({k: v for k, v in out.items() if k != "TRUNCATED"}) <= _RESPONSE_BUDGET_CHARS


def test_overflow_never_combines_percentiles():
    series = [{"node": "llm", "count": 10, "p90_ms": float(v)} for v in (500, 100, 900)] * 700
    out = _summarize_overflow({"series": series}, "nodes", "node", ("count",), ("p90_ms",))

    assert out["totals"][0]["count"] == 21000
    assert out["totals"][0]["p90_ms_range"] == (100.0, 900.0)  # a range, not a fake mean
    assert "p90_ms" not in out["totals"][0]
    # One node's 2100 rows alone are past the budget, so no series survives — and it says so.
    assert out["series"] == []
    assert "is empty" in out["TRUNCATED"]


def test_overflow_digest_skipped_when_small_or_unrecognized():
    small = {"series": [{"event_name": "a", "count": 1}]}
    assert _summarize_overflow(small, "event_types", "event_name", ("count",)) is small
    for odd in ({}, {"series": "not-a-list"}):
        assert _summarize_overflow(odd, "event_types", "event_name", ("count",)) is odd


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


def test_rollup_survives_a_null_dimension_beside_a_named_one():
    series = [
        {"ts": "2026-01-01T03:00:00", "node": None, "errors": 1, "affected_calls": 1},
        {"ts": "2026-01-01T03:05:00", "node": "llm", "errors": 2, "affected_calls": 1},
    ]
    out = _rollup(series, "10min", "node", "errors", "affected_calls")
    assert {r["node"]: r["errors"] for r in out} == {None: 1, "llm": 2}


def test_rollup_days_are_utc_even_when_ts_carries_an_offset():
    # 01:30 IST on 2 Jan is 20:00 UTC on 1 Jan.
    series = [{"ts": "2026-01-02T01:30:00+05:30", "event_name": "a", "count": 4, "calls": 1}]
    out = _rollup(series, "1day", "event_name", "count", "calls")
    assert [(r["ts"], r["count"]) for r in out] == [("2026-01-01", 4)]


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
    # 40 days at 1day is 960 hourly fetches; nothing is coarser, so only a shorter window helps.
    try:
        _check_granularity("1day", 57600, "", "")
        raise AssertionError("expected the 1hour fetch behind 1day to be refused")
    except ValueError as e:
        assert "coarser" not in str(e) and "shorter window" in str(e)


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
