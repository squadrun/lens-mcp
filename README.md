# lens-mcp

MCP server exposing the Lens ext API — call investigation, latency aggregation,
and cross-call comparison as tools for Claude Code.

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
