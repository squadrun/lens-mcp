"""ID filters that match nothing are named in a success body, and only an empty result pays for it.

A comma list reaches the backend as written (it splits it, squadrun/lens sql_filters). When the
main result is empty, each ID filter value is counted alone in the same window, and those with no
spans are listed in `unmatched_filters`. The caller's MCP breaker counts error payloads, so the
result must stay isError=False, and a failed check must add nothing rather than raise.

Run: uv run python tests/test_unmatched_filters.py
"""

from fakes import ctx, main, patched, run

import lens_mcp.server as server


class Backend:
    """Serves the breakdown/percentile/count endpoints; /search/count totals come from `spans`."""

    def __init__(self, breakdown=(), calls=(), spans=None, count_fails=False):
        self.breakdown, self.calls = list(breakdown), list(calls)
        self.spans = spans or {}
        self.count_fails = count_fails
        self.requests = []

    async def get(self, ctx, path, params=None):
        params = dict(params or {})
        self.requests.append((path, params))
        if path == "/observability/breakdown":
            return {"breakdown": self.breakdown}
        if path == "/calls":
            return {"calls": []}
        if path == "/observability/call-percentiles":
            return {"cohort": None, "calls": self.calls, "calls_returned": len(self.calls)}
        if path == "/search/count":
            if self.count_fails:
                raise RuntimeError("count unavailable")
            ((_, value),) = [
                (k, v)
                for k, v in params.items()
                if k in ("campaign_id", "agent_config_id", "prompt_ref")
            ]
            total = self.spans.get(value, 0)
            return {"total_spans": total, "affected_calls": min(total, 1)}
        raise AssertionError(f"unexpected path {path}")

    def counts(self):
        return [p for path, p in self.requests if path == "/search/count"]


def test_two_configs_go_to_the_backend_as_one_list_with_no_extra_query():
    fake = Backend(calls=[{"call_id": "a", "p90_ms": 900.0}, {"call_id": "b", "p90_ms": 800.0}])
    with patched(fake):
        result = run(server.slowest_calls(ctx(), node="llm", agent_config_id="7412,6046"))
    assert fake.requests[0][1]["agent_config_id"] == "7412,6046"
    assert not fake.counts(), "a non-empty result must not pay for the unmatched check"
    assert "unmatched_filters" not in result.structuredContent


def test_unmatched_config_is_named_and_the_matched_one_is_not():
    fake = Backend(spans={"7412": 40})
    with patched(fake):
        result = run(server.latency_breakdown(ctx(), agent_config_id="7412,9999"))
    assert result.isError is False
    assert result.structuredContent["breakdown"] == []
    assert result.structuredContent["unmatched_filters"] == [
        'agent_config_id "9999" matched no spans in this window'
    ]
    assert fake.counts() == [
        {"agent_config_id": "7412", "time_range_minutes": 360},
        {"agent_config_id": "9999", "time_range_minutes": 360},
    ]


def test_list_calls_checks_prompt_reference_id_under_the_count_endpoints_name():
    fake = Backend(spans={"c-1": 5})
    with patched(fake):
        result = run(server.list_calls(ctx(), campaign_id="c-1", prompt_reference_id="P1"))
    assert result.structuredContent["unmatched_filters"] == [
        'prompt_reference_id "P1" matched no spans in this window'
    ]
    assert {"prompt_ref": "P1", "time_range_minutes": 60} in fake.counts()


def test_a_window_past_the_count_endpoints_cap_is_sent_as_time_from():
    fake = Backend()
    with patched(fake):
        run(server.latency_breakdown(ctx(), agent_config_id="9999", time_range_minutes=43200))
    (check,) = fake.counts()
    assert "time_range_minutes" not in check and check["time_from"]


def test_a_failed_check_adds_no_note_and_no_error():
    fake = Backend(count_fails=True)
    with patched(fake):
        result = run(server.latency_breakdown(ctx(), agent_config_id="9999"))
    assert result.isError is False
    assert "unmatched_filters" not in result.structuredContent


def test_no_id_filter_means_no_check():
    fake = Backend()
    with patched(fake):
        run(server.latency_breakdown(ctx()))
    assert not fake.counts()


if __name__ == "__main__":
    main(globals())
