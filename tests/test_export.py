"""Token-efficient reads and bulk exports: fields projection, list_calls cohort paging,
export_spans, and the standalone CLI context. All offline against tests/fakes.py.

Run: uv run python tests/test_export.py
"""

import csv
import json
import os
import stat
import sys
import tempfile

import httpx

sys.path.insert(0, os.path.dirname(__file__))
from fakes import FakeLens, ctx, main, patched, run, server, span
from mcp import types

HANDLER = server.mcp._mcp_server.request_handlers[types.CallToolRequest]
SCRATCH = tempfile.mkdtemp()
os.environ["CLAUDE_SCRATCHPAD_DIR"] = SCRATCH


def tool(name, **args):
    req = types.CallToolRequest(
        method="tools/call", params=types.CallToolRequestParams(name=name, arguments=args)
    )
    r = run(HANDLER(req)).root
    assert r.isError is False, r.content[0].text
    return r.structuredContent


def usage_spans(n, start=0):
    return [
        span(
            "llm.x",
            "llm.usage",
            1000 + i,
            i,
            "complete",
            5.0 + i,
            prompt_tokens=100 + i,
            text=f"hello {i}",
            model="m",
        )
        for i in range(start, start + n)
    ]


def read_text(path):
    with open(path) as f:
        return f.read()


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_fields_projects_top_level_and_metadata_keys():
    fake = FakeLens(spans={"c1": usage_spans(3)})
    with patched(fake):
        out = tool("get_call_spans", call_sid="c1", fields="turn_number,metadata.prompt_tokens")
    assert out["spans"][1] == {"turn_number": 1, "metadata.prompt_tokens": 101}
    assert out["total"] == 3 and out["has_more"] is False  # envelope kept


def test_fields_empty_leaves_rows_unchanged_and_event_name_reaches_backend():
    rows = [*usage_spans(2), span("tts.a", "tts.first_audio", 5, 1, "ttfb", 1.0)]
    fake = FakeLens(spans={"c1": rows})
    with patched(fake):
        out = tool("get_call_spans", call_sid="c1", event_name="llm.usage")
    assert fake.requests[0][1]["event_name"] == "llm.usage"
    assert out["spans"] == usage_spans(2) and out["total"] == 2


def test_fields_keeps_truncated_notice():
    fake = FakeLens(spans={"c1": usage_spans(600)})
    with patched(fake):
        out = tool("get_call_spans", call_sid="c1", fields="turn_number")
    assert "TRUNCATED" in out and len(out["spans"]) == 500
    assert out["spans"][0] == {"turn_number": 0}


def test_search_spans_fields_projects_matches_and_keeps_call_envelope():
    payload = {"calls": [{"call_id": "c1", "matches": usage_spans(2)}], "total": 1}

    class One(FakeLens):
        async def get(self, ctx, path, params=None):
            return payload

    with patched(One()):
        out = tool("search_spans", fields="event_name,metadata.model")
    assert out["calls"][0]["call_id"] == "c1"
    assert out["calls"][0]["matches"][0] == {"event_name": "llm.usage", "metadata.model": "m"}


def calls(n):
    return [
        {"call_id": f"c{i}", "duration_ms": i * 1000, "total_turns": i % 10, "status": "ok"}
        for i in range(n)
    ]


def test_list_calls_passes_new_filters_and_projects():
    fake = FakeLens(calls=calls(30))
    with patched(fake):
        out = tool(
            "list_calls",
            offset=5,
            agent_id="a1",
            pipecat_version="0.0.63",
            limit=2,
            fields="call_id",
        )
    params = fake.requests[0][1]
    assert params["offset"] == 5 and params["agent_id"] == "a1"
    assert params["pipecat_version"] == "0.0.63"
    assert out["calls"] == [{"call_id": "c5"}, {"call_id": "c6"}]


def test_list_calls_cohort_pages_until_max_scan():
    fake = FakeLens(calls=calls(450))
    with patched(fake):
        out = tool("list_calls", min_turns=9, limit=100, max_scan=250)
    assert [p["offset"] for _, p in fake.requests] == [0, 200]
    assert [p["limit"] for _, p in fake.requests] == [200, 50]  # last page clipped to cap
    assert out["scanned"] == 250 and out["matched"] == 25 and out["scan_complete"] is False
    assert out["next_offset"] == 250
    assert all(c["total_turns"] >= 9 for c in out["calls"]) and len(out["calls"]) == 25


def test_list_calls_cohort_reports_complete_when_listing_ends():
    fake = FakeLens(calls=calls(130))
    with patched(fake):
        out = tool("list_calls", min_duration_ms=100_000, max_scan=1000, limit=100)
    assert out["scanned"] == 130 and out["scan_complete"] is True
    assert out["matched"] == 30 and out["next_offset"] is None


def test_list_calls_cohort_stops_once_limit_is_met_and_respects_fields():
    fake = FakeLens(calls=calls(450))
    with patched(fake):
        out = tool("list_calls", min_turns=1, limit=5, max_scan=1000, fields="call_id,total_turns")
    assert len(fake.requests) == 1 and len(out["calls"]) == 5
    assert out["scan_complete"] is False and out["next_offset"] == 200
    assert set(out["calls"][0]) == {"call_id", "total_turns"}


def test_list_calls_cohort_skips_a_call_repeated_on_the_next_page():
    page1 = [
        {"call_id": f"c{i}", "total_turns": int(i == 199), "duration_ms": 1} for i in range(200)
    ]
    page2 = [{"call_id": "c199", "total_turns": 5, "duration_ms": 1}] + [
        {"call_id": f"d{i}", "total_turns": 5, "duration_ms": 1} for i in range(3)
    ]
    fake = FakeLens(calls_pages=[page1, page2])
    with patched(fake):
        out = tool("list_calls", min_turns=1, limit=100, max_scan=1000)
    ids = [c["call_id"] for c in out["calls"]]
    assert ids == ["c199", "d0", "d1", "d2"] and out["matched"] == 4 and out["scanned"] == 204


def test_list_calls_scan_cap_is_1000():
    fake = FakeLens(calls=calls(1500))
    with patched(fake):
        out = tool("list_calls", min_turns=99, max_scan=5000)
    assert out["scanned"] == 1000 and out["matched"] == 0


def export(**kw):
    with patched(kw.pop("fake")):
        return run(server._export_spans_core(ctx(), **kw))


def test_export_spans_reads_each_call_in_one_request_and_writes_csv_0600():
    fake = FakeLens(spans={"c1": usage_spans(1200), "c2": usage_spans(3)})
    out = export(fake=fake, call_sids="c1,c2", event_name="llm.usage", out_name="a")
    assert [p["limit"] for _, p in fake.requests] == [5000, 5000]
    assert out["rows"] == 1203 and out["calls_ok"] == 2 and out["calls_failed"] == []
    assert out["calls_truncated"] == [] and out["path"] == os.path.join(SCRATCH, "a.csv")
    assert stat.S_IMODE(os.stat(out["path"]).st_mode) == 0o600
    rows = read_csv(out["path"])
    assert len(rows) == 1203 and rows[0]["call_id"] == "c1"
    assert out["columns"][0] == "call_id" and "metadata.prompt_tokens" in out["columns"]
    assert "metadata" not in out["columns"]


def test_export_spans_drops_text_keys_unless_include_text():
    fake = FakeLens(spans={"c1": usage_spans(2)})
    out = export(fake=fake, call_sids="c1", out_name="b")
    assert "metadata.text" not in out["columns"]
    assert "hello" not in read_text(out["path"])
    out = export(fake=fake, call_sids="c1", out_name="b2", include_text=True)
    assert "metadata.text" in out["columns"] and "hello 1" in read_text(out["path"])


def test_export_spans_fields_and_dropped_text_column_is_reported():
    fake = FakeLens(spans={"c1": usage_spans(2)})
    out = export(
        fake=fake, call_sids="c1", fields="turn_number,metadata.text,metadata.model", out_name="c"
    )
    assert out["columns"] == ["call_id", "turn_number", "metadata.model"]
    assert out["dropped_text_columns"] == ["metadata.text"]
    assert read_csv(out["path"])[1] == {"call_id": "c1", "turn_number": "1", "metadata.model": "m"}


def test_export_spans_sends_event_name_to_the_server():
    spans = [*usage_spans(2), span("tts.x", "tts.ttfb", 5000, 9, "ttfb", 3.0, model="t")]
    fake = FakeLens(spans={"c1": spans})
    out = export(fake=fake, call_sids="c1", event_name="llm.usage", out_name="ev")
    assert out["rows"] == 2
    assert all(p.get("event_name") == "llm.usage" for _, p in fake.requests)
    fake = FakeLens(spans={"c1": spans})
    export(fake=fake, call_sids="c1", out_name="ev2")
    assert all("event_name" not in p for _, p in fake.requests)


def test_export_spans_withholds_long_strings_and_containers_unless_include_text():
    md = {
        "model": "x",
        "char_count": 12,
        "long_note": "n" * 70,
        "messages": [{"role": "user", "content": "hi"}],
    }
    fake = FakeLens(spans={"c1": [span("llm.x", "llm.usage", 1, 0, "complete", 1.0, **md)]})
    out = export(fake=fake, call_sids="c1", out_name="w")
    assert "metadata.model" in out["columns"] and "metadata.char_count" in out["columns"]
    assert "metadata.long_note" not in out["columns"] and "metadata.messages" not in out["columns"]
    assert out["dropped_text_columns"] == ["metadata.long_note", "metadata.messages"]
    out = export(
        fake=fake, call_sids="c1", out_name="w2", fields="metadata.model,metadata.messages"
    )
    assert out["columns"] == ["call_id", "metadata.model"]
    assert out["dropped_text_columns"] == ["metadata.long_note", "metadata.messages"]
    out = export(fake=fake, call_sids="c1", out_name="w3", include_text=True)
    for c in ("metadata.model", "metadata.char_count", "metadata.long_note", "metadata.messages"):
        assert c in out["columns"]
    assert "dropped_text_columns" not in out


def test_export_spans_reports_failed_and_truncated_calls():
    fake = FakeLens(spans={"c1": usage_spans(2), "c3": usage_spans(1)}, failing={"c2"})
    out = export(fake=fake, call_sids="c1,c2,c3", out_name="d")
    assert out["calls_ok"] == 2 and out["rows"] == 3
    assert out["calls_failed"] == [{"call_id": "c2", "error": "RuntimeError: boom"}]
    # Paging that never ends is flagged, not silently returned as complete.
    real = server._MAX_SPANS_PER_CALL, dict(server._page_caps)
    server._MAX_SPANS_PER_CALL, server._page_caps["spans"] = 500, 500
    try:
        out = export(fake=FakeLens(spans={"c1": usage_spans(600)}), call_sids="c1", out_name="e")
    finally:
        server._MAX_SPANS_PER_CALL = real[0]
        server._page_caps.update(real[1])
    assert out["calls_truncated"] == ["c1"] and out["rows"] == 500


def test_export_spans_jsonl_and_file_input():
    listing = os.path.join(SCRATCH, "sids.txt")
    with open(listing, "w") as f:
        f.write("c1\n\nc2\nc1\n")
    fake = FakeLens(spans={"c1": usage_spans(1), "c2": usage_spans(1)})
    out = export(fake=fake, call_sids=f"@{listing}", format="jsonl", out_name="f")
    assert out["path"].endswith("f.jsonl") and out["calls_ok"] == 2  # duplicate id collapsed
    lines = [json.loads(line) for line in read_text(out["path"]).splitlines()]
    assert [r["call_id"] for r in lines] == ["c1", "c2"] and "metadata.text" not in lines[0]


def test_export_spans_drops_a_key_withheld_on_any_row_from_every_row():
    rows = [
        span("llm.x", "llm.usage", 1, 0, "complete", 1.0, note="short"),
        span("llm.x", "llm.usage", 2, 1, "complete", 1.0, note="n" * 70),
    ]
    out = export(fake=FakeLens(spans={"c1": rows}), call_sids="c1", out_name="mixed")
    assert "metadata.note" not in out["columns"]
    assert out["dropped_text_columns"] == ["metadata.note"]
    assert "short" not in read_text(out["path"])
    out = export(fake=FakeLens(spans={"c1": rows}), call_sids="c1", out_name="mj", format="jsonl")
    assert all("metadata.note" not in json.loads(x) for x in read_text(out["path"]).splitlines())


def test_export_spans_withholds_a_long_error_message_unless_include_text():
    short = {**span("llm.x", "llm.error", 1, 0, "error"), "error_message": "TimeoutError"}
    long = {**span("llm.x", "llm.error", 2, 1, "error"), "error_message": "user said " + "x" * 70}
    fake = FakeLens(spans={"c1": [short], "c2": [long]})
    out = export(fake=fake, call_sids="c1", out_name="err1")
    assert "error_message" in out["columns"] and "dropped_text_columns" not in out
    out = export(fake=fake, call_sids="c2", out_name="err2")
    assert "error_message" not in out["columns"]
    assert out["dropped_text_columns"] == ["error_message"]
    out = export(fake=fake, call_sids="c2", out_name="err3", include_text=True)
    assert "user said" in read_text(out["path"])


def test_parallel_exports_get_distinct_files_and_leave_no_temp_files():
    import asyncio

    fake = FakeLens(spans={"c1": usage_spans(2)})

    async def both():
        return await asyncio.gather(
            server._export_spans_core(ctx(), "c1"), server._export_spans_core(ctx(), "c1")
        )

    with patched(fake):
        a, b = run(both())
    assert a["path"] != b["path"] and os.path.exists(a["path"]) and os.path.exists(b["path"])
    assert not [n for n in os.listdir(SCRATCH) if n.endswith((".tmp", ".jsonl")) and n[0] == "."]


def test_export_spans_rejects_bad_input():
    for kw, msg in (
        ({"call_sids": ",".join(f"c{i}" for i in range(1001))}, "1000-call cap"),
        ({"call_sids": "c1", "format": "xml"}, "format"),
        ({"call_sids": "c1", "out_name": "../x"}, "out_name"),
        ({"call_sids": "@relative.txt"}, "absolute"),
        ({"call_sids": " "}, "empty"),
    ):
        try:
            export(fake=FakeLens(), **kw)
        except ValueError as e:
            assert msg in str(e), (kw, e)
        else:
            raise AssertionError(f"no error for {kw}")


def test_export_tools_are_registered_as_local_writers():
    tools = {t.name: t for t in run(server.mcp.list_tools())}
    for name in ("export_spans", "export_turn_latency"):
        assert tools[name].annotations.readOnlyHint is False
        assert tools[name].outputSchema["type"] == "object"


def test_cli_runs_export_spans_through_a_standalone_context():
    import io
    from contextlib import redirect_stdout

    fake = FakeLens(spans={"c1": usage_spans(2)})
    buf = io.StringIO()
    with patched(fake), redirect_stdout(buf):
        code = server._cli(
            [
                "export-spans",
                "c1",
                "--event-name",
                "llm.usage",
                "--out-name",
                "cli",
                "--fields",
                "turn_number",
            ]
        )
    assert code == 0
    out = json.loads(buf.getvalue())  # one compact JSON line
    assert (
        out["rows"] == 2 and out["calls_ok"] == 1 and out["columns"] == ["call_id", "turn_number"]
    )
    assert "\n" not in buf.getvalue().strip()
    # The context the CLI hands _get is the same shape the MCP lifespan provides.
    c = server._CliContext(None).request_context.lifespan_context
    assert c == {"client": None, "authenticated": False}


class OldBackend(FakeLens):
    """A backend that predates squadrun/lens#98: 422 above 500 spans / 100 calls per request."""

    async def get(self, ctx, path, params=None):
        params = params or {}
        cap = 100 if path == "/calls" else 500
        if params.get("limit", 0) > cap:
            self.requests.append((path, dict(params)))
            req = httpx.Request("GET", "http://localhost" + path)
            raise httpx.HTTPStatusError(
                "422", request=req, response=httpx.Response(422, request=req)
            )
        return await super().get(ctx, path, params)


def caps_reset():
    server._page_caps.update(spans=server._SPAN_FETCH_SIZE, calls=server._CALLS_PAGE)


def test_old_backend_spans_fall_back_to_500_and_page():
    caps_reset()
    fake = OldBackend(spans={"c1": usage_spans(1200)})
    try:
        out = export(fake=fake, call_sids="c1", out_name="old_spans")
        limits = [p["limit"] for path, p in fake.requests if "/spans" in path]
        assert limits[0] == 5000 and set(limits[1:]) == {500}  # one 422, then 500-row pages
        assert out["rows"] == 1200 and out["calls_truncated"] == [] and out["calls_ok"] == 1
        assert server._page_caps["spans"] == 500  # remembered for the rest of the process
    finally:
        caps_reset()


def test_old_backend_calls_scan_falls_back_to_100():
    caps_reset()
    fake = OldBackend(calls=calls(250))
    try:
        with patched(fake):
            out = tool("list_calls", min_turns=9, limit=100, max_scan=250)
        limits = [p["limit"] for _, p in fake.requests]
        assert limits[0] == 200 and limits[1:] == [100, 100, 50]
        assert out["scanned"] == 250 and out["scan_complete"] is False
        assert out["next_offset"] == 250
    finally:
        caps_reset()


def test_a_422_at_the_fallback_size_is_raised_not_retried():
    caps_reset()
    server._page_caps["spans"] = 500

    class Always422(FakeLens):
        async def get(self, ctx, path, params=None):
            req = httpx.Request("GET", "http://localhost" + path)
            raise httpx.HTTPStatusError(
                "422", request=req, response=httpx.Response(422, request=req)
            )

    try:
        out = export(fake=Always422(), call_sids="c1", out_name="bad")
        assert out["calls_ok"] == 0 and "422" in out["calls_failed"][0]["error"]
    finally:
        caps_reset()


def test_list_calls_scan_pins_a_relative_window():
    caps_reset()
    fake = FakeLens(calls=calls(450))
    with patched(fake):
        out = tool("list_calls", min_turns=9, limit=100, max_scan=400, time_range_minutes=90)
    to, frm = out["start_time_to_pinned"], out["start_time_from_pinned"]
    assert to and frm and frm < to
    assert all(p["start_time_to"] == to and p["start_time_from"] == frm for _, p in fake.requests)
    assert all("time_range_minutes" not in p for _, p in fake.requests)
    with patched(FakeLens(calls=calls(10))) as f2:
        out = tool("list_calls", min_turns=1, start_time_to="2026-01-01T00:00:00+00:00")
    assert out["start_time_to_pinned"] is None and out["start_time_from_pinned"] is None
    assert f2.requests[0][1]["start_time_to"] == "2026-01-01T00:00:00+00:00"


def test_list_calls_scan_uses_server_side_filters_when_the_backend_reports_them():
    caps_reset()

    class NewBackend(FakeLens):
        async def get(self, ctx, path, params=None):
            params = params or {}
            self.requests.append((path, dict(params)))
            rows = [c for c in self.calls if c["total_turns"] >= params.get("min_turns", 0)]
            off, lim = params.get("offset", 0), params.get("limit", 20)
            return {"calls": rows[off : off + lim], "filters_applied": ["min_turns"]}

    fake = NewBackend(calls=calls(450))
    with patched(fake):
        out = tool("list_calls", min_turns=9, limit=20, max_scan=1000)
    assert fake.requests[0][1]["min_turns"] == 9 and out["server_filtered"] is True
    assert out["matched"] >= 20 and all(c["total_turns"] >= 9 for c in out["calls"])
    with patched(FakeLens(calls=calls(450))):  # old backend: params ignored, client filters
        out = tool("list_calls", min_turns=9, limit=20, max_scan=1000)
    assert out["server_filtered"] is False and all(c["total_turns"] >= 9 for c in out["calls"])


if __name__ == "__main__":
    main(globals())
