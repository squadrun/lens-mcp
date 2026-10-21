# lens-mcp

MCP server exposing the Lens ext API — call investigation, latency aggregation,
and cross-call comparison as tools for Claude Code.

## Tools

Full argument docs live in each tool's docstring — that is what an agent reads. This is the
map for humans.

**Start here.** `get_schema` (tables, semantics, retention) · `list_filter_values` (the
campaigns / models / providers you can actually filter on — check before guessing)

**Fleet: what is happening across calls.** `latency_over_time` · `latency_breakdown` ·
`event_counts_over_time` · `error_counts_over_time` · `tool_outcomes` · `slowest_calls` ·
`get_extraction_stats` · `list_calls` · `search_spans` · `count_spans`

**One call: what happened in it.** `get_call_details` · `get_call_spans` · `aggregate_spans` ·
`get_call_transcript` · `get_call_entities` · `get_call_context` · `get_lead_details` ·
`get_call_config` · `get_call_prompt` · `get_entity_prompt`

**Logs.** `search_trace_logs` (find which calls contain a line) → `download_trace_logs` /
`download_entity_logs` (pull them local, then grep) · `get_call_trace_logs` (targeted
server-side)

**Compare.** `compare_calls` · `compare_prompts` · `aggregate_calls` · `download_prompts`

**Bulk analysis: export, don't read.** `export_spans` · `export_turn_latency` — page every call
to one local file (0600, in `CLAUDE_SCRATCHPAD_DIR`) and return only the path, row counts and
which calls failed or were truncated, so thousands of spans cost no context. Analyse the file
with Bash. Conversation-text metadata keys (`text`, `transcript`, `message`, `prompt`, ...), any string
over 64 characters and any dict or list value are dropped unless `include_text=True`; the result
lists every withheld key in `dropped_text_columns`.

`list_calls` cohort filters (`min_turns`, `min_duration_ms`): the newest calls are often still running (`total_turns` 0, no `duration_ms`), so set `start_time_to` an hour or so in the past. To page, always pass back the returned `next_offset` with `start_time_from_pinned` / `start_time_to_pinned` (the scan pins its window at the first request); on an older backend `offset` is a scan position in the unfiltered listing, not a match count.

```text
export_spans(call_sids="@/tmp/sids.txt", node="llm", event_name="llm.usage",
             fields="turn_number,metadata.prompt_tokens,metadata.model")
export_turn_latency(call_sids="sid1,sid2,sid3", max_turn=10)
```

The same code runs without an MCP client (needs `LENS_BASE_URL` and a cached login; prints the
result as one JSON line):

```bash
uv --directory ~/.cache/lens-mcp/repo run lens-mcp export-spans sid1,sid2 --event-name llm.usage \
    --fields turn_number,metadata.prompt_tokens --format jsonl
uv --directory ~/.cache/lens-mcp/repo run lens-mcp export-turns @/tmp/sids.txt --max-turn 10
```

To read a little rather than export: `get_call_spans`, `search_spans` and `list_calls` take
`fields="a,b,metadata.key"`; `get_call_spans` takes `event_name`; `list_calls` takes `min_turns`,
`min_duration_ms` and `max_scan` (filtered by the backend where it supports it, else client-side; reports `server_filtered`, `scanned` / `scan_complete`).

Page sizes follow the backend: whole-call span fetches ask for 5000 rows and cohort scans 200 calls per request (squadrun/lens#98). An older backend rejects those with 422, and the client drops to 500 / 100 for the rest of the process.
Tool results are compact JSON, not indented.

Two rules the tools enforce rather than merely document:

- **Never compute a percentile from rows.** `get_call_spans` caps at 500 per page and
  `search_spans` at 100, so a statistic from one response is wrong on anything busy. Use
  `aggregate_spans` (one call), `slowest_calls` (a cohort) or `latency_breakdown` (the fleet).
  Row responses carry a loud `TRUNCATED` notice when they are a partial view.
- **Retention differs by source, and responses say which.** Aggregate views 90 days ·
  spans 30 · prompts/configs 30 · extraction logs 7 · trace logs 4. An empty result on an old
  call usually means the evidence aged out, not that nothing happened.

## Install / update

There is no package registry and no auto-update: each machine runs whatever commit
it last pulled. **After any change to this repo, every user must update by hand.**

```bash
git -C ~/.cache/lens-mcp/repo pull
uv --directory ~/.cache/lens-mcp/repo run lens-mcp auth   # first time only
```

First-time setup:

```bash
git clone https://github.com/squadrun/lens-mcp ~/.cache/lens-mcp/repo
uv --directory ~/.cache/lens-mcp/repo run lens-mcp auth
```

Then register it (`~/.claude.json`, per project or globally):

```json
{
  "mcpServers": {
    "lens": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "/Users/<you>/.cache/lens-mcp/repo", "run", "lens-mcp"],
      "env": {
        "LENS_BASE_URL": "https://<your-lens-host>",
        "LENS_GOOGLE_CLIENT_ID": "<google oauth client id>"
      }
    }
  }
}
```

## Checking which version you're on

`get_schema` reports the running version in its client-notes header, so any session
can tell you. Compare it against `version` in `pyproject.toml` on `main`.

This matters more than it sounds. Skills and docs that reference a tool by name fail
opaquely when the install predates it — the tool is simply absent from the session,
with no error explaining why. If a documented tool is missing, update before
debugging anything else.

**Bump `version` in `pyproject.toml` in any PR that adds or changes a tool.** It is
the only signal a user has.

## Env vars

| var | required | purpose |
|---|---|---|
| `LENS_BASE_URL` | yes | Lens API base URL |
| `LENS_GOOGLE_CLIENT_ID` | for OAuth | Google device-flow client ID |
| `LENS_API_KEY` | no | static key fallback, instead of OAuth |
| `CLAUDE_SCRATCHPAD_DIR` | no | where `download_*` tools write (default: tempdir) |

## Backend coupling

Some tools depend on fixes in [`squadrun/lens`](https://github.com/squadrun/lens)
and degrade visibly, not silently, when the backend is older:

- `search_spans(event_name=…, phase=…)` needs lens#55. Without it the backend drops
  both filters and returns a `200` of unfiltered rows; `search_spans` inspects the
  result and prefixes `FILTER_IGNORED` when it can prove that happened.

**Deploy the backend before the client** when a change spans both.

## Tests

```bash
for t in tests/test_*.py; do uv run python "$t" || exit 1; done
uv run python tests/measure_tokens.py   # before/after size of tool responses (not a test)
```

Covers the statistics helpers and the backend-skew detection — the only code here
that produces a number or a judgement rather than relaying one.
