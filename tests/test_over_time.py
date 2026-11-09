"""Checks for the *_over_time tools: the granularity enum, the overflow digest, the rollups.

_check_granularity only bounds the bucket count, never the second (node/event_name)
dimension, so an unfiltered call can fan out across every distinct value fleet-wide. Past
_MAX_ROWS_UNFILTERED that series is swapped for per-dimension totals rather than returned.

The backend matches `nodes` exactly; _get_series_for_nodes reads a name with no row of its
own as a prefix, so a stage or a partial name does not come back as an empty series.

Run: uv run python tests/test_over_time.py
"""

import asyncio
import os
import sys

os.environ.setdefault("LENS_BASE_URL", "http://localhost")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fakes import ctx, patched, run

import lens_mcp.server as server
from lens_mcp.server import (
    _CLIENT_ROLLUPS,
    _GRANULARITY_MINUTES,
    _check_granularity,
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


class NodeSeries:
    """A backend that, like the real one, returns only the nodes named exactly."""

    def __init__(self, nodes, buckets=1):
        self.rows = [
            {
                "ts": f"2026-01-01T{bucket % 24:02d}:00:00",
                "node": node,
                "count": 2,
                "p90_ms": 9.0,
                "errors": 1,
                "affected_calls": 1,
            }
            for bucket in range(buckets)
            for node in nodes
        ]
        self.requests = []

    async def get(self, ctx, path, params=None):
        self.requests.append(dict(params or {}))
        named = (params or {}).get("nodes")
        wanted = None if named is None else named.split(",")
        return {"series": [r for r in self.rows if wanted is None or r["node"] in wanted]}


FLEET = ("llm.arc", "llm.warmup", "tts.cartesia", "tts.cartesiacache#5", "stt.soniox_eu")


def _latency(fake, nodes, **kwargs):
    with patched(fake):
        return run(server.latency_over_time(ctx(), nodes=nodes, **kwargs)).structuredContent


def test_a_full_node_name_with_rows_costs_one_request_and_returns_only_that_node():
    fake = NodeSeries(FLEET)
    out = _latency(fake, "tts.cartesia,llm.arc")

    assert len(fake.requests) == 1
    assert {r["node"] for r in out["series"]} == {"tts.cartesia", "llm.arc"}
    assert "nodes_matched_by_prefix" not in out and "no_rows_for" not in out


def test_a_stage_is_matched_as_a_prefix_and_the_response_names_what_it_matched():
    fake = NodeSeries(FLEET)
    out = _latency(fake, "llm, tts.cartesia")

    assert [r.get("nodes") for r in fake.requests] == ["llm,tts.cartesia", None]
    assert {r["node"] for r in out["series"]} == {"llm.arc", "llm.warmup", "tts.cartesia"}
    assert out["nodes_matched_by_prefix"] == {"llm": ["llm.arc", "llm.warmup"]}
    assert "no_rows_for" not in out


def test_a_name_that_matches_nothing_lists_the_nodes_of_its_stage_that_have_rows():
    out = _latency(NodeSeries(FLEET), "stt.deepgram,llm.arc")

    assert {r["node"] for r in out["series"]} == {"llm.arc"}  # the name that did match stays
    assert out["no_rows_for"] == ["stt.deepgram"]
    assert out["nodes_with_rows"] == ["stt.soniox_eu"]


def test_a_prefix_widened_past_the_row_budget_becomes_totals_but_named_nodes_do_not():
    widened = _latency(NodeSeries(FLEET, buckets=700), "llm,tts", time_range_minutes=42000)
    assert "a nodes prefix" in widened["TRUNCATED"] and "series" not in widened
    assert {t["node"] for t in widened["totals"]} == set(FLEET) - {"stt.soniox_eu"}
    assert widened["nodes_matched_by_prefix"]["llm"] == ["llm.arc", "llm.warmup"]

    named = _latency(NodeSeries(FLEET, buckets=700), ",".join(FLEET), time_range_minutes=42000)
    assert len(named["series"]) == 3500 and "TRUNCATED" not in named


def test_error_counts_match_a_prefix_before_rolling_up():
    fake = NodeSeries(FLEET, buckets=3)
    with patched(fake):
        out = run(
            server.error_counts_over_time(ctx(), nodes="llm.", granularity="1day")
        ).structuredContent

    assert fake.requests[0]["granularity"] == "1hour"  # fetched at the source size
    assert {(r["ts"], r["node"], r["errors"]) for r in out["series"]} == {
        ("2026-01-01", "llm.arc", 3),
        ("2026-01-01", "llm.warmup", 3),
    }


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
