"""export_turn_latency: the per-turn join, ported from the first-sentence study's tested
extractor (synthetic cases + a cross-check against its output when that data is on disk).

Run: uv run python tests/test_turn_latency.py
"""

import csv
import glob
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from fakes import FakeLens, ctx, main, patched, run, server, span

SCRATCH = tempfile.mkdtemp()
os.environ["CLAUDE_SCRATCHPAD_DIR"] = SCRATCH

BASE = 1_700_000_000_000.0
ISO = server._iso_ms


def llm(t, turn, model="gemma4", node="llm.simplismart", v=200):
    return span(node, "llm.request", BASE + t, turn, "ttfb", v, model=model)


def react(t, turn, v=500):
    return span(
        "pipeline.bot_reaction_time", "pipeline.bot_reaction_time", BASE + t, turn, "ttfb", v
    )


def cartesia(turn, t0, hit, chars, sent_chars, sent_dt=40, seqs=(1,)):
    out = [
        span("tts.cartesia", "tts.first_audio", BASE + t0 + 100, turn, "ttfb", 60,
             cache_hit=hit, characters=chars),
        span("tts.cartesiacachettsservice#1", "tts.first_audio", BASE + t0 + 101, turn, "ttfa", 60,
             leading_silence_ms=80),
        span("tts.cartesia", "tts.text_sent", BASE + t0 + sent_dt, turn, None, None,
             char_count=sent_chars),
    ]  # fmt: skip
    for i, s in enumerate(seqs):
        out.append(span("tts.cartesia", "tts.chunk_cache", BASE + t0 + 50 + i, turn, None, None,
                        synthesis_seq=s, characters=10, cached=hit))  # fmt: skip
    return out


def rows_for(llms, ttss, reacts, tools=(), max_turn=10):
    return server._turn_rows("x", list(llms), list(ttss), list(reacts), list(tools), max_turn)


def test_cartesia_hit_miss_tool_two_requests_and_missing_text_sent():
    llms = [llm(0, 1), llm(1000, 2), llm(2000, 3), llm(3000, 4), llm(3100, 4), llm(4000, 5)]
    tts = (cartesia(1, 0, True, 56, 56) + cartesia(2, 1000, False, 30, 72)
           + cartesia(3, 2000, True, 20, 20) + cartesia(4, 3000, True, 15, 15))  # fmt: skip
    tts += [r for r in cartesia(5, 4000, True, 33, 33) if r["event_name"] != "tts.text_sent"]
    reacts = [react(300, 1), react(1300, 2), react(2300, 3), react(3300, 4), react(4300, 5)]
    tools = [span("tool.x", "tool.execution", BASE + 2100, 3, "complete")]
    by = {r["turn"]: r for r in rows_for(llms, tts, reacts, tools)}

    r = by[1]
    assert r["cache_hit"] == "true" and r["sent_chars"] == "56" and r["lead_silence_ms"] == "80"
    assert r["n_pieces"] == "1" and r["tool_in_turn"] == 0 and r["tts_provider"] == "cartesia"
    assert r["ttfb_ms"] == "200" and r["reaction_ms"] == "500" and r["fa_ttfb_ms"] == "60"
    assert r["llm_node"] == "llm.simplismart" and r["llm_model"] == "gemma4"
    assert r["fa_ts"] == ISO(BASE + 100) and r["reaction_ts"] == ISO(BASE + 300)
    assert r["sent_ts"] == ISO(BASE + 40)
    assert by[2]["cache_hit"] == "false" and by[2]["sent_chars"] == "72"
    assert by[3]["tool_in_turn"] == 1
    assert by[4]["n_llm_requests"] == 2 and by[4]["ttfb_ts"] == ISO(BASE + 3000)  # earliest
    assert by[5]["sent_ts"] == "" and by[5]["sent_chars"] == ""
    assert list(by) == [1, 2, 3, 4, 5]


def test_elevenlabs_three_piece_sentence():
    def chunk(t, seq, chars, cached=True):
        return span("tts.elevenlabs", "tts.chunk_cache", BASE + t, 1, None, None,
                    synthesis_seq=seq, characters=chars, cached=cached, dynamic=False)  # fmt: skip

    tts = [
        span("tts.elevenlabs", "tts.first_audio", BASE + 190, 1, "ttfb", 30, cache_hit=True),
        chunk(120, 4, 20), chunk(125, 4, 12, False), chunk(130, 4, 9), chunk(150, 5, 40),
        span("tts.elevenlabs", "tts.segment_sent", BASE + 126, 1, None, None, chars=12),
        chunk(-500, 3, 99),  # before the LLM request: not part of this turn's sentence
    ]  # fmt: skip
    (r,) = rows_for([llm(100, 1)], tts, [react(400, 1)])
    assert r["tts_provider"] == "elevenlabs" and r["n_pieces"] == "3"
    assert r["sent_chars"] == "41" and r["sent_ts"] == ISO(BASE + 120), r


def test_elevenlabs_without_cache_uses_segment_sent_and_unknown_tts_leaves_sent_empty():
    tts = [
        span("tts.elevenlabs", "tts.first_audio", BASE + 190, 1, "ttfb", 30),
        span("tts.elevenlabs", "tts.segment_sent", BASE + 126, 1, None, None, chars=12),
    ]
    (r,) = rows_for([llm(100, 1)], tts, [react(400, 1)])
    assert r["sent_chars"] == "12" and r["n_pieces"] == "1" and r["cache_hit"] == ""
    other = [span("tts.deepgram", "tts.first_audio", BASE + 190, 1, "ttfb", 30)]
    (r,) = rows_for([llm(100, 1)], other, [react(400, 1)])
    assert r["tts_provider"] == "deepgram" and r["sent_chars"] == "" and r["sent_ts"] == ""


def test_turns_missing_a_leg_or_out_of_order_are_skipped_and_max_turn_applies():
    tts = cartesia(1, 0, True, 5, 5) + cartesia(2, 1000, True, 5, 5) + cartesia(3, 2000, True, 5, 5)
    llms = [llm(0, 1), llm(1000, 2), llm(2000, 3)]
    reacts = [react(300, 1), react(7000, 2), react(2300, 3)]  # turn 2: reaction 6s after the LLM
    assert [r["turn"] for r in rows_for(llms, tts, reacts)] == [1, 3]
    assert [r["turn"] for r in rows_for(llms, tts, reacts, max_turn=1)] == [1]
    assert [r["turn"] for r in rows_for(llms, tts, [react(300, 1)])] == [1]  # no reaction span


def test_sent_chars_falls_back_to_text_length_without_storing_text():
    tts = cartesia(1, 0, True, 5, 5)
    tts[2] = span("tts.cartesia", "tts.text_sent", BASE + 40, 1, None, None, text="namaste")
    (r,) = rows_for([llm(0, 1)], tts, [react(300, 1)])
    assert r["sent_chars"] == "7"
    assert "namaste" not in json.dumps(r)


def test_export_turn_latency_end_to_end_with_served_llm_summary():
    spans = {
        "c1": [llm(0, 1), *cartesia(1, 0, True, 56, 56), react(300, 1)],
        "c2": [llm(0, 1, model="gpt", node="llm.openai"), *cartesia(1, 0, True, 5, 5), react(300, 1)],
        "c3": [llm(0, 1), *cartesia(1, 0, True, 56, 56), react(300, 1)],
    }  # fmt: skip
    with patched(FakeLens(spans=spans, failing={"c4"})):
        out = run(server._export_turns_core(ctx(), "c1,c2,c3,c4", out_name="t"))
    assert out["rows"] == 3 and out["calls_ok"] == 3 and len(out["calls_failed"]) == 1
    assert out["served_llm"] == {"llm.openai/gpt": 1, "llm.simplismart/gemma4": 2}
    assert out["columns"][:3] == ["call_id", "turn", "llm_node"] and len(out["columns"]) == 18
    with open(out["path"]) as f:
        rows = list(csv.DictReader(f))
    assert [r["call_id"] for r in rows] == ["c1", "c2", "c3"]  # input order, one row per turn
    assert rows[1]["llm_model"] == "gpt" and rows[0]["sent_chars"] == "56"
    assert oct(os.stat(out["path"]).st_mode & 0o777) == "0o600"
    assert not any("text" in c for c in out["columns"])


# The study's extractor produced a CSV from these cached (numeric-only) spans; replaying the
# same spans through the library tool must reproduce its shared columns exactly. Skipped when
# LENS_TURN_STUDY_DIR is unset (the study output is local data, not part of this repo).
STUDY = os.environ.get("LENS_TURN_STUDY_DIR", "")
SHARED = {
    "turn": "turn", "llm_node": "llm_node", "llm_model": "llm_model_span",
    "n_llm_requests": "n_llm_requests", "ttfb_ms": "ttfb_ms", "ttfb_ts": "ttfb_ts",
    "sent_ts": "sent_ts", "sent_chars": "sent_chars", "n_pieces": "n_pieces", "fa_ts": "fa_ts",
    "fa_ttfb_ms": "fa_ttfb_ms", "cache_hit": "cache_hit", "lead_silence_ms": "lead_silence_ms",
    "reaction_ms": "reaction_ms", "reaction_ts": "reaction_ts", "tool_in_turn": "tool_in_turn",
}  # fmt: skip


def test_cached_calls_reproduce_the_study_csv_columns():
    expected = {}
    for p in glob.glob(os.path.join(STUDY, "S?_script.csv")):
        with open(p) as f:
            for r in csv.DictReader(f):
                expected.setdefault(r["call_id"], {})[r["turn"]] = r
    sids = [
        os.path.basename(p)[:-5]
        for p in sorted(glob.glob(os.path.join(STUDY, "cache", "*.json")))
        if os.path.basename(p)[:-5] in expected
    ][:5]
    if not sids:
        print("     (study data not on disk — cross-check skipped)")
        return
    spans = {}
    for sid in sids:
        with open(os.path.join(STUDY, "cache", sid + ".json")) as f:
            cached = json.load(f)
        spans[sid] = [
            {**{k: v for k, v in r.items() if k != "md"}, "metadata": json.dumps(r.get("md", {}))}
            for kind in ("llm", "tts", "react", "tool")
            for r in cached[kind]
        ]
    with patched(FakeLens(spans=spans)):
        out = run(server._export_turns_core(ctx(), ",".join(sids), out_name="study"))
    with open(out["path"]) as f:
        mine = {(r["call_id"], r["turn"]): r for r in csv.DictReader(f)}
    compared = 0
    for sid in sids:
        for turn, want in expected[sid].items():
            got = mine.get((sid, turn))
            assert got is not None, (
                f"turn missing from tool output: call #{sids.index(sid)} turn {turn}"
            )
            for ours, theirs in SHARED.items():
                assert got[ours] == want[theirs], f"{ours}: call #{sids.index(sid)} turn {turn}"
            compared += 1
    assert compared >= len(sids)
    print(f"     cross-checked {compared} turn rows from {len(sids)} cached calls")


if __name__ == "__main__":
    main(globals())
