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
        "LENS_BASE_URL": "https://lens.agents.squadstack.com",
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
uv run python tests/test_stats.py
```

Covers the statistics helpers and the backend-skew detection — the only code here
that produces a number or a judgement rather than relaying one.
