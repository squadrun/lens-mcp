"""Checks for the statistics helpers behind aggregate_spans / aggregate_calls.

These are the only place lens-mcp produces a number rather than relaying one, so
a silent error here is exactly the failure mode the tools exist to prevent.

Run: uv run python tests/test_stats.py
"""

import os
import sys

os.environ.setdefault("LENS_BASE_URL", "http://localhost")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lens_mcp.server import (  # noqa: E402
    _aggregate,
    _filters_dropped_by_server,
    _grouper,
    _metric_value,
    _parse_metric,
    _parse_percentiles,
    _parse_range,
    _percentile,
    _stats,
)


def test_percentile_matches_linear_interpolation():
    vals = [float(v) for v in range(1, 101)]  # 1..100
    assert _percentile(vals, 0) == 1.0
    assert _percentile(vals, 100) == 100.0
    assert _percentile(vals, 50) == 50.5
    assert round(_percentile(vals, 90), 4) == 90.1
    assert _percentile([42.0], 90) == 42.0
    # Interpolates between neighbours rather than snapping to one of them.
    assert _percentile([0.0, 10.0], 25) == 2.5


def test_stats_shape_and_values():
    out = _stats([10.0, 20.0, 30.0, 40.0], [50, 90])
    assert out["count"] == 4
    assert out["min_ms"] == 10.0
    assert out["max_ms"] == 40.0
    assert out["mean_ms"] == 25.0
    assert out["sum_ms"] == 100.0
    assert out["p50_ms"] == 25.0
    assert out["p90_ms"] == 37.0
    # Unsorted input must give the same answer as sorted input.
    assert _stats([40.0, 10.0, 30.0, 20.0], [50, 90]) == out


def test_parse_percentiles():
    assert _parse_percentiles("50,90,99") == [50.0, 90.0, 99.0]
    assert _parse_percentiles("p99, p50 ,50") == [50.0, 99.0]  # deduped and sorted
    assert _parse_percentiles("99.9") == [99.9]
    for bad in ("", "101", "-1", "abc"):
        try:
            _parse_percentiles(bad)
            raise AssertionError(f"expected {bad!r} to be rejected")
        except ValueError:
            pass


def test_parse_range():
    assert _parse_range("40-54", 100) == (39, 54)
    assert _parse_range("40-", 100) == (39, 100)
    assert _parse_range("-20", 100) == (0, 20)
    for bad in ("40", "0-5", "10-5"):
        try:
            _parse_range(bad, 100)
            raise AssertionError(f"expected {bad!r} to be rejected")
        except ValueError:
            pass


def test_turn_bucket_labels_and_order():
    key_fn, label_fn = _grouper("turn_bucket:10")
    assert key_fn({"turn_number": 0}) == 0
    assert key_fn({"turn_number": 9}) == 0
    assert key_fn({"turn_number": 10}) == 10
    assert label_fn(10) == "turns 10-19"
    key_fn, label_fn = _grouper("turn")
    assert label_fn(key_fn({"turn_number": 7})) == "turn 7"
    assert _grouper("") == (None, None)
    try:
        _grouper("nope")
        raise AssertionError("expected unknown group_by to be rejected")
    except ValueError:
        pass


def test_aggregate_excludes_events_and_guards_mixed_node_trend():
    spans = [
        {"node": "llm.openai", "turn_number": t, "value_ms": float(100 + t)}
        for t in range(9)
    ]
    spans.append({"node": "llm.openai", "turn_number": 9, "value_ms": None})  # event

    out = _aggregate(spans, [50], "turn_bucket:5")
    assert out["spans_matched"] == 10
    assert out["spans_measured"] == 9
    assert out["spans_without_metric"] == 1
    assert out["overall"]["count"] == 9  # the event never reaches the statistics
    assert [g["group"] for g in out["groups"]] == ["turns 0-4", "turns 5-9"]
    assert isinstance(out["trend"], dict)
    assert out["trend"]["delta_pct"] > 0  # 100..108 is rising

    mixed = spans + [{"node": "stt.deepgram", "turn_number": 0, "value_ms": 5.0}]
    assert isinstance(_aggregate(mixed, [50], "")["trend"], str)  # refused, not computed

    events_only = [{"node": "pipeline.call", "turn_number": 0, "value_ms": None}]
    out = _aggregate(events_only, [50], "")
    assert "overall" not in out and "note" in out


def test_parse_metric_and_extraction():
    assert _parse_metric("value_ms") == ("", "_ms")
    assert _parse_metric("") == ("", "_ms")
    assert _parse_metric("metadata:prompt_tokens") == ("prompt_tokens", "")
    for bad in ("metadata:", "tokens", "metadata"):
        try:
            _parse_metric(bad)
            raise AssertionError(f"expected {bad!r} to be rejected")
        except ValueError:
            pass

    span = {"value_ms": 12.5, "metadata": '{"prompt_tokens": 26430, "model": "gemma4", "cache_hit": true}'}
    assert _metric_value(span, "") == 12.5
    assert _metric_value(span, "prompt_tokens") == 26430.0
    assert _metric_value(span, "model") is None  # strings are not measurements
    assert _metric_value(span, "cache_hit") is None  # bools are flags, not measurements
    assert _metric_value(span, "absent") is None
    assert _metric_value({"metadata": "not json"}, "prompt_tokens") is None
    assert _metric_value({}, "") is None


def test_aggregate_on_metadata_metric_drops_the_ms_suffix():
    # llm.usage events carry tokens but no duration — they must still be measurable.
    spans = [
        {
            "node": "llm.simplismart",
            "turn_number": t,
            "value_ms": None,
            "metadata": f'{{"prompt_tokens": {26000 + t * 100}}}',
        }
        for t in range(9)
    ]
    out = _aggregate(spans, [50, 90], "turn", metric="metadata:prompt_tokens")
    assert out["metric"] == "metadata:prompt_tokens"
    assert out["spans_measured"] == 9
    assert out["overall"]["p50"] == 26400.0
    assert "p50_ms" not in out["overall"]  # tokens are not milliseconds
    assert out["trend"]["delta_pct"] > 0  # context is growing
    assert [g["group"] for g in out["groups"]][:2] == ["turn 0", "turn 1"]


def _resp(*matches):
    return {"calls": [{"matches": list(matches)}]}


def test_detects_a_filter_the_backend_dropped():
    # A lens older than #55 returns ttfb rows regardless of what you asked for.
    resp = _resp({"phase": "ttfb", "event_name": "llm.request"})

    assert _filters_dropped_by_server(resp, phase="complete") == ["phase='complete'"]
    assert _filters_dropped_by_server(resp, event_name="llm.usage") == ["event_name='llm.usage'"]
    assert _filters_dropped_by_server(resp, phase="complete", event_name="llm.usage") == [
        "phase='complete'",
        "event_name='llm.usage'",
    ]


def test_stays_quiet_when_the_filter_held():
    resp = _resp({"phase": "ttfb"}, {"phase": "ttfb"})

    assert _filters_dropped_by_server(resp, phase="ttfb") == []
    assert _filters_dropped_by_server(resp) == []  # nothing filtered, nothing to check


def test_a_single_matching_row_is_enough_to_stay_quiet():
    # One-sided by design: a mixed page means the filter ran on at least that value,
    # so we do not cry wolf. Only a total absence of matches is evidence.
    resp = _resp({"phase": "ttfb"}, {"phase": "complete"})

    assert _filters_dropped_by_server(resp, phase="complete") == []


def test_empty_result_is_not_evidence_either_way():
    # Zero rows is a legitimate "no such span" — never flag it as a dropped filter.
    assert _filters_dropped_by_server({"calls": []}, phase="ttfb") == []
    assert _filters_dropped_by_server({}, phase="ttfb") == []
    assert _filters_dropped_by_server(_resp(), phase="ttfb") == []


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
