"""
Lens MCP Server — call analysis tools for Claude Code.

Wraps the Lens ext API to provide call investigation, comparison, and search
capabilities as MCP tools. Connect via Claude Code settings or any MCP client.

Auth (in priority order):
    1. Cached JWT from prior Google OAuth device flow
    2. Google OAuth device flow — user approves on accounts.google.com
    3. LENS_API_KEY env var — static key fallback

Env vars:
    LENS_BASE_URL              — (required) Lens API base URL
    LENS_GOOGLE_CLIENT_ID      — Google OAuth client ID for device flow
    LENS_API_KEY               — (optional) static API key fallback
"""

import argparse
import asyncio
import base64
import csv
import json
import os
import re
import secrets
import sys
import tempfile
from collections.abc import Iterable
from contextlib import asynccontextmanager, contextmanager, suppress
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import httpx
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations

BASE_URL = os.environ.get("LENS_BASE_URL", "")
if not BASE_URL:
    raise RuntimeError("LENS_BASE_URL is required. Set it in your MCP server config.")
API_KEY = os.environ.get("LENS_API_KEY", "")
GOOGLE_CLIENT_ID = os.environ.get("LENS_GOOGLE_CLIENT_ID", "")
API_PREFIX = "/api/ext/v1"
TIMEOUT = 30.0

try:
    _VERSION = version("lens-mcp")
except Exception:
    _VERSION = "unknown"

# /calls page for the cohort scan; each page re-runs the full dedup aggregation, so ask for the
# most rows the server allows. The ext API allows 200 once squadrun/lens#98 is deployed and 100
# before it: a 422 drops to the fallback for the rest of the process (see _get_capped).
_CALLS_PAGE = 200
_CALLS_PAGE_FALLBACK = 100
_MAX_CALLS_SCAN = 1000
_SPAN_PAGE_SIZE = 500  # get_call_spans page: the backend's default, sized for token cost
# Whole-call fetches ask for the backend's per-call maximum. Paging /call/{id}/spans re-sorts the
# call on every page and is not stable across equal timestamps; the backend measured no call above
# 5000 spans, so one request per call is the normal case.
# Older ext backends cap /call/{id}/spans at 500 (raised by squadrun/lens#98): same 422 fallback.
_SPAN_FETCH_SIZE = 5000
_SPAN_FETCH_FALLBACK = 500
_MAX_SPANS_PER_CALL = 10_000  # beyond any real call, bounds runaway paging
# Current per-request cap by kind; lowered once on a 422 from a backend that predates lens#98.
_page_caps = {"spans": _SPAN_FETCH_SIZE, "calls": _CALLS_PAGE}
_COHORT_CONCURRENCY = 8
# A fleet-wide trace-log scan and a sort on a non-indexed span column both read/sort the whole
# matched set. Safe over a short window; unbounded they pressure the box. Cap the window to 2h
# unless a high-selectivity filter (campaign_id/agent_config_id/prompt_ref) narrows first.
_UNSCOPED_WINDOW_CAP_MIN = 120
_HIGH_SELECTIVITY = ("campaign_id", "agent_config_id", "prompt_ref")


def _window_minutes(time_range_minutes: int, time_from: str, time_to: str) -> float | None:
    """Effective look-back window in minutes; None if an explicit window can't be parsed
    (treated as unbounded so an unparseable time_from cannot slip past a cap)."""
    if time_range_minutes and time_range_minutes > 0:
        return float(time_range_minutes)
    if time_from:
        try:
            start = datetime.fromisoformat(time_from)
            end = datetime.fromisoformat(time_to) if time_to else datetime.now(start.tzinfo)
            return (end - start).total_seconds() / 60
        except (ValueError, TypeError):
            return None
    return None  # no explicit window → fail the cap closed (the request carries no time bound, so the guard must not treat it as a safe 60 min)


_CALL_SID_PATTERN = re.compile(r"^[a-zA-Z0-9\-]+$")


def _sanitize_sid(sid: str) -> str:
    sid = sid.strip()
    if not sid or not _CALL_SID_PATTERN.match(sid):
        raise ValueError(f"Invalid call SID: {sid!r}")
    return sid


_TOKEN_CACHE_PATH = Path.home() / ".cache" / "lens-mcp" / "token.json"


def _load_cached_token() -> str | None:
    try:
        if not _TOKEN_CACHE_PATH.exists():
            return None
        data = json.loads(_TOKEN_CACHE_PATH.read_text())
        return data.get("token")
    except Exception:
        return None


def _clear_cached_token() -> None:
    try:
        if _TOKEN_CACHE_PATH.exists():
            _TOKEN_CACHE_PATH.unlink()
    except Exception:
        pass


def _save_cached_token(token: str, username: str, expires_at: str) -> None:
    try:
        _TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _TOKEN_CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "token": token,
                    "username": username,
                    "expires_at": expires_at,
                }
            )
        )
        tmp.chmod(0o600)
        tmp.rename(_TOKEN_CACHE_PATH)
    except Exception:
        pass


def _is_token_valid(token: str) -> bool:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return False
        payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        exp = claims.get("exp", 0)
        return datetime.now(UTC).timestamp() < exp
    except Exception:
        return False


async def _device_flow_auth(client: httpx.AsyncClient) -> dict | None:
    """Run Google OAuth device flow — user approves on accounts.google.com."""
    if not GOOGLE_CLIENT_ID:
        return None

    async with httpx.AsyncClient(timeout=10) as google:
        code_resp = await google.post(
            "https://oauth2.googleapis.com/device/code",
            data={"client_id": GOOGLE_CLIENT_ID, "scope": "email profile"},
        )
        if code_resp.status_code != 200:
            return None
        code_data = code_resp.json()

        user_code = code_data.get("user_code", "")
        verification_url = code_data.get("verification_url", "https://www.google.com/device")
        device_code = code_data.get("device_code", "")
        interval = code_data.get("interval", 5)
        expires_in = code_data.get("expires_in", 1800)
        max_polls = expires_in // interval

        print("\n  Lens MCP: Google authentication required")
        print(f"  Go to: {verification_url}")
        print(f"  Enter code: {user_code}\n", flush=True)

        for _ in range(max_polls):
            await asyncio.sleep(interval)
            resp = await client.post(
                f"{API_PREFIX}/auth/google/device",
                json={"device_code": device_code},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "pending":
                    continue
                if data.get("status") == "slow_down":
                    interval = data.get("interval", interval + 5)
                    continue
                if "token" in data:
                    _save_cached_token(data["token"], data.get("username", ""), data["expires_at"])
                    print(
                        f"  Authenticated as: {data.get('email', data.get('username', ''))}\n",
                        flush=True,
                    )
                    return {"Authorization": f"Bearer {data['token']}"}
            else:
                detail = ""
                with suppress(Exception):
                    detail = resp.json().get("detail", "")
                if detail:
                    print(f"  Auth failed: {resp.status_code} {detail}", flush=True)
                break

    return None


def _set_auth(client: httpx.AsyncClient, headers: dict) -> None:
    client.headers.pop("Authorization", None)
    client.headers.pop("X-Lens-Api-Key", None)
    client.headers.update(headers)


async def _authenticate(client: httpx.AsyncClient) -> dict:
    """Resolve auth headers. Uses cached JWT or static API key. Run 'lens-mcp auth' to authenticate."""
    cached = _load_cached_token()
    if cached and _is_token_valid(cached):
        return {"Authorization": f"Bearer {cached}"}

    if API_KEY:
        return {"X-Lens-Api-Key": API_KEY}

    raise RuntimeError(
        "Not authenticated. Run this command first:\n"
        "  uv --directory ~/.cache/lens-mcp/repo run lens-mcp auth"
    )


@asynccontextmanager
async def lifespan(server: FastMCP):
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=TIMEOUT,
    ) as client:
        yield {"client": client, "authenticated": False}


mcp = FastMCP("lens", lifespan=lifespan)

# openWorldHint is True throughout: every tool hits the remote Lens API.
READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
)
# The download_* tools write into the scratchpad dir, so they are not read-only.
# They overwrite the same per-SID paths on a repeat call, hence idempotent.
WRITES_LOCAL_FILES = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True
)
# The export_* tools write a new file on every call (the default name is unique), so a repeat
# is not a no-op.
EXPORTS_LOCAL_FILES = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
)


def _client(ctx: Context) -> httpx.AsyncClient:
    return ctx.request_context.lifespan_context["client"]


def _out(data: dict | list) -> dict[str, Any]:
    """A tool result as an object, so FastMCP sends it as structuredContent plus the same JSON
    as text. Returning a str instead publishes it as one escaped string under "result".

    The text is compact JSON: FastMCP's own rendering indents with two spaces, which costs
    roughly a third more tokens on row-heavy payloads for no information. Returning a
    CallToolResult makes FastMCP pass it through untouched; the tools keep their
    `-> dict[str, Any]` annotation, so the published outputSchema is unchanged."""
    payload = data if isinstance(data, dict) else {"result": data}
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)
    return CallToolResult(
        content=[TextContent(type="text", text=text)], structuredContent=payload, isError=False
    )


async def _ensure_auth(ctx: Context) -> None:
    lc = ctx.request_context.lifespan_context
    if lc["authenticated"]:
        return
    client = lc["client"]
    auth_headers = await _authenticate(client)
    _set_auth(client, auth_headers)
    lc["authenticated"] = True


# 429 (per-principal rate limit) and 503 (heavy-query concurrency shed) are load-shedding, not
# broken queries — the server asks you to back off, so retry rather than surface an error. 502/504
# are gateway blips. All are transient.
_RETRY_STATUS = {429, 502, 503, 504}
_LOAD_SHED_STATUS = {429, 503}
_MAX_RETRIES = 2
_MAX_RETRY_SLEEP = 15.0


def _retry_sleep(resp: httpx.Response | None, attempt: int) -> float:
    """Honour a Retry-After header when the server sends one (rate limit / capacity), else
    exponential backoff. Capped so a hostile Retry-After can't wedge the tool."""
    if resp is not None:
        header = resp.headers.get("Retry-After")
        if header:
            try:
                return min(float(header), _MAX_RETRY_SLEEP)
            except ValueError:
                pass
    return min(0.5 * 2**attempt, 8.0)


async def _get(ctx: Context, path: str, params: dict | None = None) -> dict:
    await _ensure_auth(ctx)
    client = _client(ctx)
    url = f"{API_PREFIX}{path}"

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = await client.get(url, params=params)
        except (httpx.TimeoutException, httpx.TransportError):
            # Gateway blips on wide time ranges are transient; retrying here is far
            # cheaper than surfacing a 504 and making the caller redo the analysis.
            if attempt == _MAX_RETRIES:
                raise
            await asyncio.sleep(_retry_sleep(None, attempt))
            continue

        if resp.status_code == 401:
            _clear_cached_token()
            _set_auth(client, await _authenticate(client))
            resp = await client.get(url, params=params)

        if resp.status_code in _RETRY_STATUS:
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_retry_sleep(resp, attempt))
                continue
            if resp.status_code in _LOAD_SHED_STATUS:
                # Exhausted retries on a deliberate shed — the query is fine, the server is busy
                # or you are rate-limited. Say so, so the caller slows down instead of hammering.
                raise RuntimeError(
                    f"lens is shedding load (HTTP {resp.status_code}) after {_MAX_RETRIES + 1} "
                    f"attempts — your query is fine, the server is at capacity or you are "
                    f"rate-limited. Wait a moment and retry, or narrow the query."
                )

        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError as e:
            raise RuntimeError(
                f"GET {path} returned {resp.status_code} with a non-JSON body "
                f"(gateway or auth-proxy page?): {resp.text[:120]!r}"
            ) from e

    raise RuntimeError(f"GET {path} exhausted retries")


# ── Schema reference ───────────────────────────────────────────────────


# Semantics the server-side schema does not carry, each of which has silently
# produced a wrong answer: see squadrun/lens-mcp#1.
_CLIENT_NOTES = f"""

## Client notes (lens-mcp {_VERSION}) — read before computing any number

Installs are per-machine `git pull`s, so this version is the only reliable
statement of which tools you have. If a tool named below is missing from your
session, the install is stale — re-run the install command in the lens-mcp README.

### turn_number is a BOT-UTTERANCE counter, not a user-turn counter
It counts bot utterances — incremented each time the bot stops speaking, so
fillers, idle nudges and any non-LLM TTS advance it.
Two calls at the same turn_number are NOT at equal conversational depth — do not
use it to align calls when comparing them.

### metadata_filter needs a node scope
metadata is unindexed, so filtering it across a whole time window would scan every
span; the server requires a narrowing filter alongside it and search_spans checks
for one before making the request. Before lens#55 the unscoped form returned 0 hits
instead of an error, which is why older notes claim it "fails for transport".
Values match exactly and case-sensitively against the raw JSON value —
the full model id hits; a prefix of it, or a different case, does not. Numbers and bools
match as written in the JSON (`prompt_tokens:26430`, `speech_final:true`).

### event_name / phase on /search need lens#55
Before that fix the backend accepted both and filtered on neither, returning
unfiltered rows with a 200. search_spans now checks the rows it gets back and
prefixes the response with FILTER_IGNORED if the backend is still dropping them —
if you see that, the filter did not apply and the rows are not what you asked for.
count_spans has always honoured both.

### node accepts a prefix
`node=llm.`, `node=llm` and `node=llm.<provider>` all work, on both search_spans and
get_call_spans. Verified as genuinely filtering, along with level, campaign_id
and value_ms_min/max.

### Row caps — never percentile a single page
get_call_spans returns at most 500 rows per request, search_spans at most 100.
A p90 computed from one truncated page is wrong with no warning. Use
aggregate_spans (single call) or aggregate_calls (cohort): both page to
exhaustion and report `complete: false` if they could not.

### Bulk analysis: export, don't read
For more than a couple of calls, do not read spans into context. `export_spans` and
`export_turn_latency` page every call to a local 0600 file and return only its path; analyse
it with Bash. To read a little: `get_call_spans` / `search_spans` / `list_calls` take
`fields="a,b,metadata.k"` and `get_call_spans` takes `event_name` (cheapest single-event read).

### Retention differs per table
call_metadata 6 months · call_spans 30 days · call_prompts / call_configs 30 days
· entity-extraction logs 7 days · call_trace_logs 4 days. A call stays listable
for 6 months but its log-level evidence is gone after 4 days.

### Per-turn token counts exist in spans, but not for every provider
The llm.usage event carries per-turn prompt_tokens, cache_read_input_tokens,
completion_tokens and total_tokens, keyed by turn_number — so prompt-context
growth is answerable from spans (30 days) rather than trace logs (4 days):

    aggregate_spans(call_sid, node="llm", event_name="llm.usage",
                    metric="metadata:prompt_tokens", group_by="turn")

Coverage is provider-dependent: some LLM providers emit llm.usage, others
do not (their llm.request spans carry only model and transport). Call-level
totals are on pipeline.cost_summary regardless. Where llm.usage is absent, per-turn
growth still needs trace logs within their 4-day window: download_trace_logs then
grep "prompt cache:".
"""


@mcp.tool(annotations=READ_ONLY)
async def get_schema(ctx: Context) -> str:
    """Get the ClickHouse table schemas, column types, indexes, and query strategy.

    READ THIS FIRST before searching or investigating. It tells you:
    - What columns exist and which are indexed (fast to filter on)
    - The difference between call_spans (structured, fast) vs call_trace_logs (raw text, slow)
    - The correct query strategy to avoid expensive operations
    - Semantics that silently produce wrong numbers if you assume them
      (turn_number, metadata_filter scoping, row caps, retention)

    Key rules:
    - Use search_spans with structured filters (node, event_name, level, campaign_id) — these are indexed
    - NEVER broad-search trace logs — only drill into specific call_sids
    - Time bounds are mandatory for any search
    - Never compute a percentile from raw span rows — use aggregate_spans/aggregate_calls
    """
    data = await _get(ctx, "/schema")
    return data.get("schema", "") + _CLIENT_NOTES


# ── Per-call investigation (fast, indexed by call_id) ──────────────────


_DETAIL_SECTIONS = ("metadata", "transcript", "entities", "recording", "config")


def _parse_range(spec: str, total: int) -> tuple[int, int]:
    """Parse a 1-based inclusive "N-M" range (either end optional) to slice bounds."""
    spec = spec.strip()
    if "-" not in spec:
        raise ValueError(f"Invalid range {spec!r} — expected 'N-M', 'N-' or '-M'")
    lo_s, hi_s = spec.split("-", 1)
    lo = int(lo_s) if lo_s.strip() else 1
    hi = int(hi_s) if hi_s.strip() else total
    if lo < 1 or hi < lo:
        raise ValueError(f"Invalid range {spec!r} — start must be >= 1 and <= end")
    return lo - 1, hi


@mcp.tool(annotations=READ_ONLY)
async def get_call_details(
    call_sid: str,
    ctx: Context,
    include_config: bool = False,
    sections: str = "",
    transcript_range: str = "",
) -> dict[str, Any]:
    """Get details for a call — metadata, transcript, entities, and config.

    This is the primary 'what happened in this call' tool. Use it first when
    investigating a call.

    ALREADY INCLUDES the full transcript — do NOT also call get_call_transcript,
    that would be redundant and waste tokens.

    Long calls return 100KB+ payloads that overflow the tool output limit. Use
    `sections` and `transcript_range` to fetch only what you need instead of
    spilling the whole thing to disk:
        sections="metadata"                    → just the call-level facts
        sections="metadata,entities"           → facts + extraction outcome
        sections="config"                      → the agent config alone
        transcript_range="40-54"               → only utterances 40-54
        sections="transcript", transcript_range="-20"  → first 20 utterances

    The config (agent_config_json) is large (~10K tokens). Set include_config=True
    only when you need the agent config. Default is False to save tokens.
    Note the config holds the prompt TEMPLATE — for the rendered prompt actually
    sent to the LLM, use get_call_prompt instead.

    For deep investigation after this, ALWAYS use download_trace_logs to download
    raw logs locally, then grep with Bash. Never use get_call_trace_logs.

    Args:
        call_sid: The call SID (e.g. "abc123def456").
        include_config: Include the full agent config JSON (default False). Set True only when investigating config. Implied when sections includes "config".
        sections: Comma-separated subset of metadata,transcript,entities,recording,config. Default (empty) returns all.
        transcript_range: 1-based inclusive utterance range, e.g. "40-54", "40-" (from 40 on) or "-20" (first 20). Applies to the transcript section only.
    """
    call_sid = _sanitize_sid(call_sid)
    wanted = [s.strip() for s in sections.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in _DETAIL_SECTIONS]
    if unknown:
        raise ValueError(f"Unknown section(s) {unknown}. Valid: {', '.join(_DETAIL_SECTIONS)}")

    params = {}
    if not (include_config or "config" in wanted):
        params["include_config"] = "false"
    data = await _get(ctx, f"/call/{call_sid}/details", params=params or None)
    if "recording" in data:
        data["recording"] = {"exists": (data["recording"] or {}).get("exists", False)}

    transcript = data.get("transcript")
    if transcript_range and isinstance(transcript, dict):
        turns = transcript.get("turns") or []
        lo, hi = _parse_range(transcript_range, len(turns))
        sliced = turns[lo:hi]
        data["transcript"] = {
            "turns": sliced,
            "count": len(turns),
            "returned_range": (
                f"{lo + 1}-{lo + len(sliced)}"
                if sliced
                else f"empty — the transcript has only {len(turns)} utterances"
            ),
        }

    if wanted:
        data = {k: v for k, v in data.items() if k in wanted}
    return _out(data)


@mcp.tool(annotations=READ_ONLY)
async def get_call_transcript(call_sid: str, ctx: Context) -> dict[str, Any]:
    """Get the full conversation transcript for a call.

    NOTE: get_call_details already includes the transcript. Only use this
    if you specifically need the transcript alone without metadata/entities.
    Do NOT call both get_call_details and get_call_transcript — that
    duplicates the transcript and wastes tokens.

    Args:
        call_sid: The call SID.
    """
    call_sid = _sanitize_sid(call_sid)
    return _out(await _get(ctx, f"/call/{call_sid}/transcript"))


@mcp.tool(annotations=READ_ONLY)
async def get_call_prompt(call_sids: str, ctx: Context) -> dict[str, Any]:
    """Get the rendered (interpolated) system prompt used for one or more calls.

    This is the actual final prompt sent to the LLM — lead details and custom
    variables already substituted in. Not the same as the prompt template in
    get_call_details(include_config=True), which is the uninterpolated config.

    Returns prompt_reference_id, prompt_hash, and prompt_text per call.
    Prompts are retained for 30 days.

    Prompts are large (often several thousand tokens each). Request only the
    SIDs you actually need — do not pass a long list to browse.
    To see what changed between two calls, use compare_prompts instead.

    Args:
        call_sids: One or more call SIDs, comma-separated (e.g. "abc123" or "abc123,def456").
    """
    sids = [_sanitize_sid(s) for s in call_sids.split(",") if s.strip()]
    if not sids:
        raise ValueError("No call SIDs provided")

    async def fetch(sid: str) -> dict:
        try:
            return await _get(ctx, f"/call/{sid}/prompt")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"call_id": sid, "error": "Prompt not found (older than 30d retention?)"}
            return {"call_id": sid, "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            # Isolate per SID: a timeout on one call must not fail the whole batch.
            return {"call_id": sid, "error": f"{type(e).__name__}: {e}"}

    results = await asyncio.gather(*(fetch(s) for s in sids))
    return _out(results[0] if len(results) == 1 else {"prompts": list(results)})


@mcp.tool(annotations=READ_ONLY)
async def get_entity_prompt(call_sids: str, ctx: Context) -> dict[str, Any]:
    """Get the rendered entity-extraction prompt (system + user) for one or more calls.

    This is the extraction-side counterpart to get_call_prompt:
    - get_call_prompt   → the live-call agent prompt (what the bot said on the call)
    - get_entity_prompt → the post-call extraction prompt (how outcomes/entities
      were derived from the transcript)

    Returns the processor variant, campaign_id, voice_mission_id,
    entity_config_ref_id, lead_details, system_prompt and user_prompt.

    Retention is 7 days — shorter than the 30 days for get_call_prompt. Older
    calls return an error entry, not a prompt.

    Prompts are large (~10K chars each). Request only the SIDs you need.

    Args:
        call_sids: One or more call SIDs, comma-separated (e.g. "abc123" or "abc123,def456").
    """
    sids = [_sanitize_sid(s) for s in call_sids.split(",") if s.strip()]
    if not sids:
        raise ValueError("No call SIDs provided")

    async def fetch(sid: str) -> dict:
        try:
            return await _get(ctx, f"/call/{sid}/entity-prompt")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {
                    "call_id": sid,
                    "error": "Entity prompt not found (no extraction run, or older than 7d retention)",
                }
            return {"call_id": sid, "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            # Isolate per SID: a timeout on one call must not fail the whole batch.
            return {"call_id": sid, "error": f"{type(e).__name__}: {e}"}

    results = await asyncio.gather(*(fetch(s) for s in sids))
    return _out(results[0] if len(results) == 1 else {"prompts": list(results)})


@mcp.tool(annotations=READ_ONLY)
async def get_call_spans(
    call_sid: str,
    ctx: Context,
    node: str = "",
    phase: str = "",
    limit: int = 500,
    offset: int = 0,
    event_name: str = "",
    fields: str = "",
) -> dict[str, Any]:
    """Get structured instrumentation spans and events for a call.

    Returns the latency waterfall: LLM/STT/TTS spans with timing,
    tool call durations, pipeline events, and error spans. Indexed by
    call_id — fast even for calls with hundreds of spans.

    DO NOT compute percentiles, means or trends from this tool's output. The
    server caps a response at 500 rows, so a busy call returns a truncated
    prefix and any statistic derived from it is wrong. Use aggregate_spans —
    it pages to exhaustion and reports whether it saw everything. Use this tool
    to READ individual spans (errors, metadata, ordering), not to measure.

    Each span has: node (e.g. "llm.<provider>"), phase (complete/error/ttfb),
    value_ms (duration), level, error_message, and metadata.

    Note turn_number is a bot-utterance counter (bot_stopped_count), not a
    user-turn counter — fillers and idle nudges advance it.

    Args:
        call_sid: The call SID.
        node: Filter by node PREFIX (e.g. "llm" matches all LLM spans; "stt" matches all STT spans). Always use this for targeted lookups.
        phase: Filter by phase (e.g. "error", "ttfb", "complete"). Useful for quick error checks.
        limit: Rows per page (default 500, which is also the server maximum).
        offset: Row offset, for paging past the first 500.
        event_name: Exact event name (e.g. "llm.usage"). The cheapest way to read one event type — the server filters, so only those rows cost tokens.
        fields: Comma-separated keys to keep per row (e.g. "turn_number,value_ms,metadata.prompt_tokens"). "metadata.<key>" picks one metadata key, flattened. Empty = whole rows. For many calls use export_spans instead.
    """
    call_sid = _sanitize_sid(call_sid)
    params: dict = {"limit": min(max(limit, 1), _SPAN_PAGE_SIZE), "offset": max(offset, 0)}
    if node:
        params["node"] = node
    if phase:
        params["phase"] = phase
    if event_name:
        params["event_name"] = event_name
    data = await _get(ctx, f"/call/{call_sid}/spans", params=params)
    if fields and "spans" in data:
        data = {**data, "spans": _project_rows(data["spans"], _parse_fields(fields))}
    if data.get("has_more"):
        returned = len(data.get("spans", []))
        data = {
            "TRUNCATED": (
                f"Showing {returned} of {data.get('total')} matching spans "
                f"(offset {params['offset']}). Any percentile, mean or trend computed "
                f"from these rows is WRONG. Use aggregate_spans for statistics, or "
                f"page with offset={params['offset'] + returned} to read the rest."
            ),
            **data,
        }
    return _out(data)


# ── Aggregation (pages the row cap away, returns statistics not rows) ──


async def _fetch_all_spans(
    ctx: Context, call_sid: str, node: str = "", phase: str = "", event_name: str = ""
) -> tuple[list[dict], bool]:
    """Fetch a call's spans to exhaustion. Returns (spans, complete).

    Asks for the backend's per-call maximum, so a call is one request. A further page is read
    only past that; pages are ordered by timestamp_ms alone, so rows sharing a timestamp at a
    page boundary can repeat or drop.
    """
    base: dict = {}
    if node:
        base["node"] = node
    if phase:
        base["phase"] = phase
    if event_name:
        base["event_name"] = event_name

    spans: list[dict] = []
    offset = 0
    while len(spans) < _MAX_SPANS_PER_CALL:
        params = {**base, "limit": _page_caps["spans"], "offset": offset}
        data = await _get_capped(
            ctx, f"/call/{call_sid}/spans", params, "spans", _SPAN_FETCH_FALLBACK
        )
        page = data.get("spans", [])
        spans.extend(page)
        if not page or not data.get("has_more"):
            return spans, True
        offset += len(page)
    return spans, False


async def _get_capped(ctx: Context, path: str, params: dict, kind: str, fallback: int) -> dict:
    """GET with params["limit"] at the current cap for `kind`. A backend that predates
    squadrun/lens#98 rejects the larger limit with 422: drop to `fallback` for the rest of the
    process and retry once, so the tools work before and after that deploy."""
    try:
        return await _get(ctx, path, params=params)
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 422 or params.get("limit", 0) <= fallback:
            raise
        _page_caps[kind] = fallback
        return await _get(ctx, path, params={**params, "limit": fallback})


def _percentile(ordered: list[float], p: float) -> float:
    """Linear-interpolated percentile over an already-sorted list."""
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _p_label(p: float) -> str:
    return str(int(p)) if p == int(p) else str(p)


def _parse_percentiles(spec: str) -> list[float]:
    pcts = []
    for part in spec.split(","):
        part = part.strip().lstrip("pP")
        if not part:
            continue
        try:
            p = float(part)
        except ValueError:
            raise ValueError(f"Not a percentile: {part!r} — use e.g. '50,90,99'") from None
        if not 0 <= p <= 100:
            raise ValueError(f"Percentile out of range: {p}")
        pcts.append(p)
    if not pcts:
        raise ValueError("No percentiles given — use e.g. '50,90,99'")
    return sorted(set(pcts))


def _stats(values: list[float], pcts: list[float], suffix: str = "_ms") -> dict:
    ordered = sorted(values)
    out: dict = {
        "count": len(ordered),
        f"min{suffix}": round(ordered[0], 1),
        f"max{suffix}": round(ordered[-1], 1),
        f"mean{suffix}": round(sum(ordered) / len(ordered), 1),
        f"sum{suffix}": round(sum(ordered), 1),
    }
    for p in pcts:
        out[f"p{_p_label(p)}{suffix}"] = round(_percentile(ordered, p), 1)
    return out


def _trend(values: list[float], suffix: str = "_ms") -> dict | None:
    """Median of the first vs last third, in call order — 'is it growing?'.

    Thirds rather than a regression: robust to the outliers that dominate voice
    latency, and readable without a stats background.
    """
    if len(values) < 6:
        return None
    third = len(values) // 3
    first = sorted(values[:third])
    last = sorted(values[-third:])
    a, b = _percentile(first, 50), _percentile(last, 50)
    return {
        f"first_third_p50{suffix}": round(a, 1),
        f"last_third_p50{suffix}": round(b, 1),
        "delta_pct": round((b - a) / a * 100, 1) if a else None,
        "n_per_third": third,
    }


def _parse_metric(metric: str) -> tuple[str, str]:
    """Validate a metric spec, returning (metadata_key or "", key suffix).

    "value_ms" measures span durations; "metadata:<key>" measures a numeric field
    inside the span's metadata JSON — which is where per-turn token counts live
    (llm.usage carries prompt_tokens / cache_read_input_tokens / completion_tokens).
    """
    if metric in ("", "value_ms"):
        return "", "_ms"
    if metric.startswith("metadata:"):
        key = metric.split(":", 1)[1].strip()
        if not key:
            raise ValueError('metric "metadata:" needs a key, e.g. "metadata:prompt_tokens"')
        return key, ""
    raise ValueError(
        f'Unknown metric {metric!r} — use "value_ms" or "metadata:<key>" '
        f'(e.g. "metadata:prompt_tokens")'
    )


def _metric_value(span: dict, key: str) -> float | None:
    """Read the measured number off a span, or None if it carries no usable value."""
    if not key:
        val = span.get("value_ms")
    else:
        try:
            val = json.loads(span.get("metadata") or "{}").get(key)
        except (TypeError, ValueError):
            return None
    # bool is an int subclass — a flag is not a measurement.
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return float(val)


def _grouper(group_by: str):
    """Return (key_fn, label_fn) for a group_by spec, or (None, None) for no grouping.

    Keys sort the buckets, labels render them — they differ for turn buckets,
    where the key is the numeric floor but the label is the turn range.
    """
    if not group_by:
        return None, None
    if group_by in ("node", "phase", "event_name"):
        return (lambda s: s.get(group_by) or "(none)"), str
    if group_by == "turn":
        return (lambda s: int(s.get("turn_number") or 0)), (lambda k: f"turn {k}")
    if group_by.startswith("turn_bucket:"):
        raw = group_by.split(":", 1)[1].strip()
        try:
            size = int(raw) if raw else 10
        except ValueError:
            raise ValueError(f"turn_bucket size must be an integer, got {raw!r}") from None
        if size < 1:
            raise ValueError("turn_bucket size must be >= 1")
        return (
            lambda s: (int(s.get("turn_number") or 0) // size) * size,
            lambda k: f"turns {k}-{k + size - 1}",
        )
    raise ValueError(
        f"Unknown group_by {group_by!r}. Valid: node, phase, event_name, turn, turn_bucket:N"
    )


def _aggregate(
    spans: list[dict], pcts: list[float], group_by: str, metric: str = "value_ms"
) -> dict:
    """Reduce spans to statistics. Spans carrying no value for the metric are counted, never measured."""
    key, suffix = _parse_metric(metric)
    measured = [(s, v) for s in spans if (v := _metric_value(s, key)) is not None]
    result: dict = {
        "metric": metric or "value_ms",
        "spans_matched": len(spans),
        "spans_measured": len(measured),
        "spans_without_metric": len(spans) - len(measured),
    }
    if not measured:
        result["note"] = (
            f"No span carried a numeric {'metadata.' + key if key else 'value_ms'} — "
            f"nothing to measure. Point-in-time events carry no duration, and not "
            f"every node emits every metadata key."
        )
        return result

    values = [v for _, v in measured]
    result["overall"] = _stats(values, pcts, suffix)

    # A trend over mixed node types measures the pipeline's shape, not degradation —
    # only report it once the caller has narrowed to one kind of operation.
    nodes = {s.get("node") for s, _ in measured}
    if len(nodes) > 1:
        result["trend"] = (
            f"not computed — spans span {len(nodes)} node types, so a first-vs-last "
            f"comparison would compare different operations. Re-run with node=<one node>."
        )
    else:
        trend = _trend(values, suffix)
        if trend:
            result["trend"] = trend

    key_fn, label_fn = _grouper(group_by)
    if key_fn:
        buckets: dict = {}
        for span, value in measured:
            buckets.setdefault(key_fn(span), []).append(value)
        result["group_by"] = group_by
        result["groups"] = [
            {"group": label_fn(k), **_stats(buckets[k], pcts, suffix)} for k in sorted(buckets)
        ]
    return result


@mcp.tool(annotations=READ_ONLY)
async def aggregate_spans(
    call_sid: str,
    ctx: Context,
    node: str = "",
    phase: str = "",
    event_name: str = "",
    group_by: str = "",
    percentiles: str = "50,90,99",
    metric: str = "value_ms",
) -> dict[str, Any]:
    """Compute statistics over a call's spans — server-paged, never truncated.

    USE THIS INSTEAD OF get_call_spans WHENEVER YOU WANT A NUMBER. It pages past
    the 500-row cap, so percentiles are computed over every matching span, and it
    returns ~20 lines instead of hundreds of raw rows.

    Answers directly:
    - "how slow is LLM TTFB on this call?"   → node="llm", phase="ttfb"
    - "is latency growing across the call?"  → the `trend` block, or
                                               group_by="turn_bucket:10"
    - "which stage dominates?"               → group_by="node"
    - "is the prompt context growing?"       → event_name="llm.usage",
                                               metric="metadata:prompt_tokens",
                                               group_by="turn"

    That last one is the context-growth curve straight from spans (30-day
    retention) rather than from trace logs (4 days). llm.usage carries per-turn
    prompt_tokens, cache_read_input_tokens, completion_tokens and total_tokens —
    but only on providers that emit it (not all do); where it is absent, fall
    back to trace logs.

    Returns overall count/min/mean/max/percentiles, a `trend` block comparing the
    median of the first third of the call against the last third, and per-group
    statistics when group_by is set. `complete: false` means the call exceeded the
    paging ceiling and the numbers cover only what was read — it never lies about
    coverage.

    Spans carrying no value for the chosen metric are counted under
    spans_without_metric and excluded from the statistics.

    Args:
        call_sid: The call SID.
        node: Node prefix filter (e.g. "llm", "llm.<provider>", "tts."). Strongly recommended — mixing STT and LLM durations makes the percentiles meaningless.
        phase: Phase filter ("ttfb", "complete", "error", "connect").
        event_name: Exact event_name filter (e.g. "llm.usage"), applied client-side after fetching.
        group_by: One of "node", "phase", "event_name", "turn", or "turn_bucket:N" (e.g. "turn_bucket:10" bins every 10 bot utterances). Empty for a single overall result.
        percentiles: Comma-separated percentiles (default "50,90,99").
        metric: What to measure — "value_ms" (span duration, default) or "metadata:<key>" to measure a numeric metadata field such as "metadata:prompt_tokens".
    """
    call_sid = _sanitize_sid(call_sid)
    pcts = _parse_percentiles(percentiles)
    _grouper(group_by)  # validate before spending requests
    _parse_metric(metric)

    spans, complete = await _fetch_all_spans(ctx, call_sid, node=node, phase=phase)
    if event_name:
        spans = [s for s in spans if s.get("event_name") == event_name]

    out = {
        "call_id": call_sid,
        "filters": {"node": node or None, "phase": phase or None, "event_name": event_name or None},
        "complete": complete,
        **_aggregate(spans, pcts, group_by, metric),
    }
    if not complete:
        out["WARNING"] = (
            f"Stopped after {_MAX_SPANS_PER_CALL} spans — statistics "
            f"cover only that prefix of the call. Narrow with node/phase and re-run."
        )
    return _out(out)


@mcp.tool(annotations=READ_ONLY)
async def aggregate_calls(
    ctx: Context,
    node: str,
    phase: str = "",
    event_name: str = "",
    metric: str = "value_ms",
    campaign_id: str = "",
    voice_mission_id: str = "",
    agent_config_id: str = "",
    customer: str = "",
    llm_model: str = "",
    status: str = "",
    time_range_minutes: int = 60,
    start_time_from: str = "",
    start_time_to: str = "",
    percentiles: str = "50,90",
    max_calls: int = 20,
    sort_by: str = "p90",
) -> dict[str, Any]:
    """Compare one latency metric across a COHORT of calls — "is it this call or the fleet?".

    The follow-up question after aggregate_spans finds a slow call. Selects calls
    with list_calls filters, then computes the same statistic per call and pooled
    across the cohort, so you can see whether a slow p90 is one outlier or the
    whole campaign.

    Example — a campaign over the last 24h, ranked by LLM time-to-first-byte:
        aggregate_calls(node="llm", phase="ttfb", campaign_id="<campaign_id>",
                        time_range_minutes=1440, sort_by="p90")

    Example — which calls in the campaign carry the biggest prompt context:
        aggregate_calls(node="llm", event_name="llm.usage",
                        metric="metadata:prompt_tokens", campaign_id="<campaign_id>",
                        sort_by="max")

    `cohort` pools every span from every scanned call (the true fleet percentile,
    not an average of averages); `calls` lists per-call statistics sorted worst
    first. Calls with no matching spans are reported as a count, not silently
    dropped.

    COST: one paged span fetch per call, so keep max_calls modest. Scanning 20
    calls is a few seconds; 50 is the ceiling. Narrow the time window rather than
    raising max_calls — this samples the most recent N calls matching the filters,
    it is not a full-population query.

    For a whole-population ranking in a single query, use slowest_calls instead.
    Stay here when you want per-call metadata (llm_model, duration) beside the
    numbers, or when the backend predates the call-percentiles endpoint.

    Args:
        node: Node prefix to measure (e.g. "llm", "stt.<provider>", "tts"). Required — an unfiltered cohort percentile mixes unrelated operations and means nothing.
        phase: Phase filter ("ttfb", "complete", "error").
        event_name: Exact event_name filter (e.g. "llm.usage"), applied client-side after fetching.
        metric: What to measure — "value_ms" (span duration, default) or "metadata:<key>" (e.g. "metadata:prompt_tokens").
        campaign_id: Restrict to a campaign.
        voice_mission_id: Restrict to a voice mission.
        agent_config_id: Restrict to an agent config.
        customer: Restrict to a customer/tenant.
        llm_model: Restrict to an LLM model.
        status: Call status filter (e.g. "completed") — worth setting, since in-progress calls have partial spans.
        time_range_minutes: Look back N minutes (default 60, max 10080 = 7 days).
        start_time_from: ISO8601 start (alternative to time_range_minutes, no cap).
        start_time_to: ISO8601 end.
        percentiles: Comma-separated percentiles (default "50,90").
        max_calls: How many calls to scan (default 20, max 50).
        sort_by: Rank calls by this stat — a percentile like "p90", or "mean", "max", "count".
    """
    if not node:
        raise ValueError("node is required — an unfiltered cohort percentile is meaningless")
    pcts = _parse_percentiles(percentiles)
    metric_key, suffix = _parse_metric(metric)

    if sort_by == "count":
        sort_key = "count"
    elif sort_by in ("mean", "max", "min", "sum"):
        sort_key = f"{sort_by}{suffix}"
    else:
        try:
            p = float(sort_by.lstrip("pP"))
        except ValueError:
            raise ValueError(
                f"Unknown sort_by {sort_by!r} — use a percentile like 'p90', or mean/max/min/sum/count"
            ) from None
        pcts = sorted(set(pcts) | {p})
        sort_key = f"p{_p_label(p)}{suffix}"

    params: dict = {"limit": min(max(max_calls, 1), 50)}
    for key, val in (
        ("campaign_id", campaign_id),
        ("voice_mission_id", voice_mission_id),
        ("agent_config_id", agent_config_id),
        ("customer", customer),
        ("llm_model", llm_model),
        ("status", status),
        ("start_time_to", start_time_to),
    ):
        if val:
            params[key] = val
    if start_time_from:
        params["start_time_from"] = start_time_from
    elif time_range_minutes > 0:
        params["time_range_minutes"] = min(time_range_minutes, 10080)

    listing = await _get(ctx, "/calls", params=params)
    calls = listing.get("calls", [])
    if not calls:
        return _out(
            {"calls_matched": 0, "note": "No calls matched the filters — widen the time window."}
        )

    sem = asyncio.Semaphore(_COHORT_CONCURRENCY)

    async def measure(call: dict) -> dict:
        sid = call.get("call_id")
        if not sid:
            return {"call_id": None, "error": "listing row had no call_id"}
        async with sem:
            try:
                spans, complete = await _fetch_all_spans(ctx, sid, node=node, phase=phase)
            except Exception as e:
                # Isolate per call: one failure must not void the whole cohort.
                return {"call_id": sid, "error": f"{type(e).__name__}: {e}"}
        if event_name:
            spans = [s for s in spans if s.get("event_name") == event_name]
        values = [v for s in spans if (v := _metric_value(s, metric_key)) is not None]
        row = {
            "call_id": sid,
            "start_time": call.get("start_time"),
            "duration_ms": call.get("duration_ms"),
            "llm_model": call.get("llm_model"),
        }
        if not values:
            row["count"] = 0
            return row
        row.update(_stats(values, pcts, suffix))
        if not complete:
            row["incomplete"] = True
        row["_values"] = values
        return row

    rows = await asyncio.gather(*(measure(c) for c in calls))

    pooled: list[float] = []
    measured, empty, failed = [], 0, []
    for row in rows:
        if row.get("error"):
            failed.append(row)
        elif row.get("count"):
            pooled.extend(row.pop("_values"))
            measured.append(row)
        else:
            empty += 1

    measured.sort(key=lambda r: r.get(sort_key, 0), reverse=True)

    out: dict = {
        "metric": metric,
        "filters": {
            "node": node,
            "phase": phase or None,
            "event_name": event_name or None,
            **{k: v for k, v in params.items() if k != "limit"},
        },
        "calls_scanned": len(calls),
        "calls_with_spans": len(measured),
        "calls_without_matching_spans": empty,
        "sorted_by": sort_key,
        "cohort": _stats(pooled, pcts, suffix) if pooled else None,
        "calls": measured,
    }
    if failed:
        out["failed"] = failed
    if len(calls) == params["limit"]:
        out["NOTE"] = (
            f"Scanned the {params['limit']}-call ceiling — this is a sample of the most "
            f"recent matching calls, not the full population. Narrow the time window to "
            f"make the sample representative."
        )
    return _out(out)


@mcp.tool(annotations=READ_ONLY)
async def get_call_trace_logs(
    call_sid: str,
    ctx: Context,
    search: str = "",
    logger_name: str = "",
    level: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """Query trace logs via ClickHouse — before calling this, check:

    1. Have you already called download_trace_logs for this call_sid?
       If yes, grep the local file instead — it's faster and free.
    2. Is there a *_traces.log file in your temp/scratchpad directory for this call?
       If yes, use that — all logs are already there.
    3. Only use this tool if download_trace_logs failed or you need a very
       targeted server-side query on a call you haven't downloaded yet.

    Warning: individual log lines can be 36KB+ (config dumps at call start),
    so even limit=5 can return 200K+ chars. Always set search or logger_name.

    Common search patterns:
    - logger_name="lead_details_api" — lead data source, name resolution, Redis/HTTP path
    - logger_name="prompts", search="lead_details" — resolved lead fields used in prompt rendering
    - search="reconnect" — STT WebSocket reconnection events
    - search="rate_limit" — API rate limiting from any provider
    - search="fallback" — LLM/STT/TTS fallback triggers
    - search="timeout" — timeout events across providers
    - search="capacity" — provider capacity gating
    - search="prompt_tokens" — LLM token usage details

    Args:
        call_sid: The call SID (required — never search without one).
        search: Text filter (substring match on log message). ALWAYS provide this.
        logger_name: Filter by Python logger name (e.g. "lead_details_api", "prompts", "calling_bot"). Much more precise than search.
        level: Level filter (mostly useless — trace logs are all info).
        limit: Max log lines (default 50, max 5000).
    """
    call_sid = _sanitize_sid(call_sid)
    params: dict = {"limit": min(limit, 5000)}
    if search:
        params["search"] = search
    if logger_name:
        params["logger_name"] = logger_name
    if level:
        params["level"] = level
    return _out(await _get(ctx, f"/call/{call_sid}/traces", params=params))


@mcp.tool(annotations=READ_ONLY)
async def get_call_entities(call_sid: str, ctx: Context) -> dict[str, Any]:
    """Get entity extraction results for a call.

    Returns extracted entities (outcome, answers like call_outcome,
    rescheduled_to, etc.) and extraction metadata (status, timing).

    Args:
        call_sid: The call SID.
    """
    call_sid = _sanitize_sid(call_sid)
    return _out(await _get(ctx, f"/call/{call_sid}/entities"))


@mcp.tool(annotations=READ_ONLY)
async def get_call_context(call_sid: str, ctx: Context) -> dict[str, Any]:
    """Get tool call details and pipeline summaries for a call.

    Returns structured tool call records (with args, results, durations)
    and pipeline summary events (call_summary, cost_summary, router_decision).

    Args:
        call_sid: The call SID.
    """
    call_sid = _sanitize_sid(call_sid)
    return _out(await _get(ctx, f"/call/{call_sid}/context"))


@mcp.tool(annotations=READ_ONLY)
async def get_lead_details(call_sid: str, ctx: Context) -> dict[str, Any]:
    """Get lead details for a call — resolved lead data and custom variables side by side.

    Returns the lead name/identity as resolved at call start (from Redis
    or HTTP API) alongside the custom_variables from the agent config. Flags
    mismatches between the two (e.g. greeting used a different name than the
    prompt config).

    Use this when investigating:
    - Name mismatches (greeting says one name, prompt uses another)
    - Lead data source (was it from Redis, custom_variables, or HTTP?)
    - What identity fields the bot had at call start

    Args:
        call_sid: The call SID.
    """
    call_sid = _sanitize_sid(call_sid)
    return _out(await _get(ctx, f"/call/{call_sid}/lead-details"))


# ── Trace log download (for multi-grep in Claude Code) ─────────────────


async def _download_logs(
    ctx: Context, call_sids: str, endpoint: str, file_suffix: str, label: str
) -> str:
    """Page a trace-log endpoint to one local file per call, then report paths.

    Shared by download_trace_logs (live-call pipeline) and download_entity_logs
    (post-call extraction worker); they differ only in endpoint and filename.
    Writes via a 0600 temp file + rename so a partial download is never left
    behind under the final path.
    """
    scratchpad = os.environ.get("CLAUDE_SCRATCHPAD_DIR", tempfile.gettempdir())
    sids = [_sanitize_sid(s) for s in call_sids.split(",") if s.strip()]

    await _ensure_auth(ctx)

    sem = asyncio.Semaphore(10)

    async def _download_one(sid: str) -> str:
        filepath = os.path.join(scratchpad, f"{sid}_{file_suffix}.log")

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            with open(filepath) as f:
                line_count = sum(1 for _ in f)
            return f"{sid}: already downloaded ({line_count} lines) -> {filepath}"

        total_lines = 0
        batch_size = 5000
        max_batches = 10

        try:
            async with sem:
                fd = os.open(filepath + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as f:
                    offset = 0
                    for _ in range(max_batches):
                        data = await _get(
                            ctx,
                            f"/call/{sid}/{endpoint}",
                            params={"limit": batch_size, "offset": offset},
                        )
                        traces = data.get("traces", [])
                        if not traces:
                            break
                        for row in traces:
                            ts = row.get("timestamp_ms", "")
                            lvl = row.get("level", "info")
                            logger = row.get("logger_name", "")
                            msg = row.get("message", "").replace("\n", "\\n")
                            f.write(f"{ts} {lvl.upper():7s} [{logger}] {msg}\n")
                            total_lines += 1
                        if len(traces) < batch_size:
                            break
                        offset += batch_size
                os.rename(filepath + ".tmp", filepath)
            return f"{sid}: {total_lines} lines -> {filepath}"
        except Exception as e:
            for p in (filepath + ".tmp", filepath):
                if os.path.exists(p):
                    os.remove(p)
            return f"{sid}: FAILED ({e})"

    results = await asyncio.gather(*[_download_one(sid) for sid in sids])

    summary = "\n".join(results)
    return (
        f"Downloaded {label}:\n{summary}\n\n"
        f"All logs are now local. Use Bash grep for any further searching — "
        f"no need to call the {label} API again for these calls."
    )


@mcp.tool(annotations=WRITES_LOCAL_FILES)
async def download_entity_logs(call_sids: str, ctx: Context) -> str:
    """Download entity-extraction worker logs for one or more calls, then grep locally.

    The extraction-side counterpart to download_trace_logs. These are DIFFERENT
    log streams:
    - download_trace_logs  → the live-call pipeline (STT/LLM/TTS/VAD). 4-day retention.
    - download_entity_logs → the post-call async worker (transcript fetch, lead
      details, extraction, posting results). 7-day retention.

    Use this one for anything about entity extraction, call outcomes, or the
    extraction prompt. Calls with no extraction run produce an empty file (0 lines).

    Same workflow as download_trace_logs: call ONCE with all SIDs, then grep the
    local files with Bash — each grep is free.

    Example greps after download:
        grep -i "entity extraction prompt" /path/to/file.log   # the rendered prompt
        grep -i "model" /path/to/file.log                      # which model/provider ran
        grep -i "fallback" /path/to/file.log                   # model routing / fallbacks
        grep -i "outcome\\|answers" /path/to/file.log            # extracted entities
        grep -i "error\\|timeout\\|failed" /path/to/file.log      # failures

    For just the prompt, prefer get_entity_prompt — it returns it parsed.

    Args:
        call_sids: One or more call SIDs, comma-separated (e.g. "abc123" or "abc123,def456").
    """
    return await _download_logs(ctx, call_sids, "ee-traces", "ee_traces", "entity extraction logs")


@mcp.tool(annotations=WRITES_LOCAL_FILES)
async def download_trace_logs(call_sids: str, ctx: Context) -> str:
    """Download trace logs for one or more calls to local files, then grep locally.

    This is the ONLY correct way to search trace logs. Downloads once per call,
    then grep locally with Bash — unlimited searches, zero extra tokens,
    case-insensitive, no ClickHouse round-trips.

    Covers the LIVE-CALL pipeline only (STT/LLM/TTS/VAD). For the post-call
    entity-extraction worker, use download_entity_logs instead.

    When investigating multiple calls, pass ALL call SIDs at once to download
    them all upfront. Do not download one and query the API for the rest.

    Workflow:
    1. Call this tool ONCE with all call_sids you need (comma-separated)
    2. grep the local files with Bash — file paths are in the response
    3. Run as many greps as needed — each is free (no API calls, no tokens)

    Example grep commands after download:
        grep -i "vad" /path/to/file.log              # VAD events
        grep -i "stt" /path/to/file.log                 # STT events
        grep -i "aggregation" /path/to/file.log       # turn aggregation
        grep "03:49:3" /path/to/file.log              # filter by timestamp
        grep -i "error\\|timeout" /path/to/file.log    # errors
        grep -c "reconnect" /path/to/file.log            # count matches

    Args:
        call_sids: One or more call SIDs, comma-separated (e.g. "abc123" or "abc123,def456,ghi789").
    """
    return await _download_logs(ctx, call_sids, "traces", "traces", "trace logs")


@contextmanager
def _private_file(path: str):
    """A text handle on a 0600 temp file beside path, renamed onto it on success, so a partial
    write is never visible under the final path. The temp name is unique, so concurrent writers
    of one path never share it; the last to finish wins."""
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(path) or ".", prefix=f".{os.path.basename(path)}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            yield f
        os.replace(tmp, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _write_private(path: str, text: str) -> int:
    """Write text to path through _private_file. Returns the character count."""
    with _private_file(path) as f:
        f.write(text)
    return len(text)


# ── Bulk export (write to a file, return the path — never the rows) ──

_EXPORT_MAX_CALLS = 1000
_EXPORT_CONCURRENCY = 4
# Free-text metadata keys. Spans carry the live conversation here, so exports drop them unless
# the caller opts in with include_text=True.
_TEXT_KEYS = frozenset(
    {
        "text",
        "transcript",
        "message",
        "content",
        "last_user_message",
        "prompt",
        "user_text",
        "llm_reply",
    }
)


def _span_metadata(span: dict) -> dict:
    """A span's metadata as a dict. The backend sends it as a JSON string; tolerate a dict."""
    md = span.get("metadata")
    if isinstance(md, dict):
        return md
    try:
        parsed = json.loads(md or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_fields(spec: str) -> list[str]:
    return [f.strip() for f in spec.split(",") if f.strip()]


def _project_row(row: dict, fields: list[str]) -> dict:
    """Keep only the requested keys of one row. "metadata.<key>" is read out of the metadata
    JSON and emitted flattened under that same name; a key the row lacks is omitted."""
    out: dict = {}
    md: dict | None = None
    for f in fields:
        if f.startswith("metadata."):
            if md is None:
                md = _span_metadata(row)
            key = f[len("metadata.") :]
            if key in md:
                out[f] = md[key]
        elif f in row:
            out[f] = row[f]
    return out


def _project_rows(rows: list, fields: list[str]) -> list:
    return [_project_row(r, fields) if isinstance(r, dict) else r for r in rows]


def _project_search(data: dict, fields: list[str]) -> dict:
    """/search nests span rows under calls[].matches[]; project those, keep each call's envelope."""
    calls = [
        {**c, "matches": _project_rows(c["matches"], fields)} if "matches" in c else c
        for c in data.get("calls", [])
    ]
    return {**data, "calls": calls}


def _parse_sids(spec: str) -> list[str]:
    """call_sids as a comma list, or "@/abs/path" naming a file with one SID per line."""
    spec = spec.strip()
    if spec.startswith("@"):
        path = Path(spec[1:]).expanduser()
        if not path.is_absolute():
            raise ValueError(f"call_sids file must be an absolute path, got {spec[1:]!r}")
        spec = path.read_text()
    sids = list(dict.fromkeys(_sanitize_sid(s) for s in re.split(r"[,\s]+", spec) if s))
    if not sids:
        raise ValueError("call_sids is empty — pass a comma list or @/abs/path to a file")
    if len(sids) > _EXPORT_MAX_CALLS:
        raise ValueError(
            f"{len(sids)} calls exceeds the {_EXPORT_MAX_CALLS}-call cap per export — "
            f"split the list and export in batches."
        )
    return sids


def _export_path(out_name: str, stem: str, ext: str) -> str:
    scratchpad = os.environ.get("CLAUDE_SCRATCHPAD_DIR", tempfile.gettempdir())
    if out_name:
        if not re.fullmatch(r"[\w.\-]+", out_name):
            raise ValueError(f"out_name must be a plain file name, got {out_name!r}")
        name = out_name if out_name.endswith(f".{ext}") else f"{out_name}.{ext}"
    else:
        # Parallel tool calls land in the same second; the suffix keeps their files apart.
        name = f"{stem}_{datetime.now(UTC):%Y%m%dT%H%M%S}_{secrets.token_hex(3)}.{ext}"
    return os.path.join(scratchpad, name)


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, dict | list | bool):
        return json.dumps(v, separators=(",", ":"), ensure_ascii=False)
    return v


def _write_csv(path: str, columns: list[str], rows: Iterable[dict]) -> None:
    with _private_file(path) as f:
        w = csv.DictWriter(f, columns, extrasaction="ignore", restval="", lineterminator="\n")
        w.writeheader()
        w.writerows({k: _cell(v) for k, v in r.items()} for r in rows)


def _write_jsonl(path: str, columns: list[str], rows: Iterable[dict]) -> None:
    with _private_file(path) as f:
        for r in rows:
            f.write(_jsonl_line({c: r[c] for c in columns if c in r}))


def _jsonl_line(row: dict) -> str:
    return json.dumps(row, separators=(",", ":"), ensure_ascii=False, default=str) + "\n"


async def _per_call(sids: list[str], work) -> list[tuple[str, Any, str | None]]:
    """Run work(sid) for every SID, four at a time. One failing call is reported, not raised."""
    sem = asyncio.Semaphore(_EXPORT_CONCURRENCY)

    async def one(sid: str):
        async with sem:
            try:
                return sid, await work(sid), None
            except Exception as e:
                return sid, None, f"{type(e).__name__}: {e}"

    return await asyncio.gather(*[one(s) for s in sids])


_MAX_SHORT_TEXT = 64
# Top-level span columns that are free text. error_message can quote what the caller said.
_TEXT_COLUMNS = frozenset({"error_message"})


def _withheld(key: str, value: Any) -> bool:
    """Privacy default for include_text=False: the denylist, plus anything that could carry text.

    Long strings and containers (messages arrays nest conversation text) are dropped; numbers,
    bools, None and short strings (model ids, providers, voice ids) are kept.
    """
    if key.lower() in _TEXT_KEYS:
        return True
    if isinstance(value, dict | list):
        return True
    return isinstance(value, str) and len(value) > _MAX_SHORT_TEXT


def _flatten_span(span: dict, include_text: bool, withheld: set[str] | None = None) -> dict:
    """Flatten a span; with include_text=False, withheld metadata keys are added to `withheld`."""
    flat = {}
    for k, v in span.items():
        if k == "metadata":
            continue
        if not include_text and k in _TEXT_COLUMNS and _withheld(k, v):
            if withheld is not None:
                withheld.add(k)
            continue
        flat[k] = v
    for k, v in _span_metadata(span).items():
        if include_text or not _withheld(k, v):
            flat[f"metadata.{k}"] = v
        elif withheld is not None:
            withheld.add(f"metadata.{k}")
    return flat


async def _export_spans_core(
    ctx: Context,
    call_sids: str,
    node: str = "",
    phase: str = "",
    event_name: str = "",
    fields: str = "",
    format: str = "csv",
    include_text: bool = False,
    out_name: str = "",
) -> dict:
    if format not in ("csv", "jsonl"):
        raise ValueError(f'format must be "csv" or "jsonl", got {format!r}')
    sids = _parse_sids(call_sids)
    wanted = _parse_fields(fields)
    path = _export_path(out_name, "spans", format)
    await _ensure_auth(ctx)

    withheld: set[str] = set()
    seen: dict[str, None] = {}  # every column written, in first-seen order

    # Rows stream to a private staging file as each call finishes, so memory holds at most
    # _EXPORT_CONCURRENCY calls. Which keys are withheld is only known at the end, so the final
    # file is written from the staging file in a second pass.
    fd, stage_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".spans_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w+") as stage:

            async def work(sid: str):
                spans, complete = await _fetch_all_spans(
                    ctx, sid, node=node, phase=phase, event_name=event_name
                )
                # No await from here on, so concurrent calls never interleave their rows.
                for s in spans:
                    flat = {"call_id": sid, **_flatten_span(s, include_text, withheld)}
                    if wanted:
                        flat = {k: flat[k] for k in ("call_id", *wanted) if k in flat}
                    seen.update(dict.fromkeys(flat))
                    stage.write(_jsonl_line(flat))
                return len(spans), complete

            n_rows, ok, failed, truncated = 0, 0, [], []
            for sid, res, err in await _per_call(sids, work):
                if err:
                    failed.append({"call_id": sid, "error": err})
                    continue
                ok += 1
                n, complete = res
                n_rows += n
                if not complete:
                    truncated.append(sid)

            # A key withheld on any row is dropped from every row, so a column is never
            # partly blank.
            dropped: set[str] = set(withheld)
            if wanted:
                if not include_text:
                    dropped |= {
                        f
                        for f in wanted
                        if f.startswith("metadata.") and f[9:].lower() in _TEXT_KEYS
                    }
                columns = ["call_id"] + [f for f in wanted if f not in dropped]
            else:
                columns = ["call_id"] + [c for c in seen if c != "call_id" and c not in dropped]

            out = {"path": None, "rows": n_rows, "columns": columns}
            if ok:
                stage.seek(0)
                rows = (json.loads(line) for line in stage)
                (_write_csv if format == "csv" else _write_jsonl)(path, columns, rows)
                out["path"] = path
    finally:
        with suppress(FileNotFoundError):
            os.unlink(stage_path)
    out.update(calls_ok=ok, calls_failed=failed, calls_truncated=truncated)
    if dropped:
        out["dropped_text_columns"] = sorted(dropped)
    return out


@mcp.tool(annotations=EXPORTS_LOCAL_FILES)
async def export_spans(
    call_sids: str,
    ctx: Context,
    node: str = "",
    phase: str = "",
    event_name: str = "",
    fields: str = "",
    format: str = "csv",
    include_text: bool = False,
    out_name: str = "",
) -> dict[str, Any]:
    """Export spans from many calls to ONE local file; returns the path, never the rows.

    Use this instead of get_call_spans / search_spans whenever you want to analyse spans
    across more than a couple of calls: the rows go to disk, so none of them cost context
    tokens. Then work on the file with Bash (python, duckdb, awk). Each call is read whole,
    so the file is complete unless the call appears in calls_truncated. Rows are grouped by
    call, in the order the calls finish.

    Privacy: unless include_text=True, metadata keys that hold conversation text (text,
    transcript, message, content, last_user_message, prompt, user_text, llm_reply) are dropped,
    and so is any metadata value that is a string over 64 characters or a dict/list (they can
    nest conversation text). A long error_message is withheld the same way. Numbers, bools,
    None and short strings are kept. A key withheld on any row is dropped from every row, and
    every withheld key is listed in dropped_text_columns, including keys requested via fields.
    The file is written 0600 into CLAUDE_SCRATCHPAD_DIR (default: the system temp dir).

    Args:
        call_sids: Comma-separated call SIDs, or "@/abs/path" naming a file with one SID per
            line. At most 1000 calls per invocation.
        node: Node PREFIX filter (e.g. "llm", "tts.cartesia"). Narrow it — it is the cheapest filter.
        phase: Phase filter (e.g. "ttfb", "error").
        event_name: Exact event name (e.g. "llm.usage"). Filtered by the server.
        fields: Comma-separated columns; "metadata.<key>" selects a metadata key. Empty = every
            top-level key plus every metadata key, flattened as metadata.<key>.
        format: "csv" (default) or "jsonl".
        include_text: Keep the metadata the privacy rule above would withhold.
        out_name: File name inside the scratchpad dir (default spans_<UTC timestamp>_<random>.<format>).
    """
    return _out(
        await _export_spans_core(
            ctx, call_sids, node, phase, event_name, fields, format, include_text, out_name
        )
    )


# ── Per-turn latency export (turn logic ported from the first-sentence study) ──


def _tsms(v: Any) -> float:
    """Epoch ms from an ISO string (naive = UTC) or a numeric ms."""
    if isinstance(v, int | float):
        return float(v)
    d = datetime.fromisoformat(str(v).strip().replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    return d.timestamp() * 1000.0


def _iso_ms(v: Any) -> str:
    ms = _tsms(v)
    d = datetime.fromtimestamp(ms / 1000.0, tz=UTC)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{round(ms) % 1000:03d}"


def _numstr(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return str(x)


def _num_or_none(v: Any):
    return v if isinstance(v, int | float) and not isinstance(v, bool) else None


def _prep_spans(spans: list[dict]) -> list[dict]:
    """Normalise raw span rows and order them by time. Rows without a timestamp or a
    turn_number cannot be placed in a turn and are skipped."""
    out = []
    for i, s in enumerate(spans):
        if s.get("timestamp_ms") is None or s.get("turn_number") is None:
            continue
        try:
            ts = _tsms(s["timestamp_ms"])
        except ValueError:
            continue
        out.append(
            {
                "node": s.get("node") or "",
                "phase": s.get("phase"),
                "event_name": s.get("event_name"),
                "value_ms": s.get("value_ms"),
                "ts": ts,
                "turn": int(s["turn_number"]),
                "idx": i,
                "md": _span_metadata(s),
            }
        )
    out.sort(key=lambda r: (r["ts"], r["idx"]))
    return out


def _tts_family(node: str) -> str | None:
    n = (node or "").lower()
    if "cartesia" in n:
        return "cartesia"
    if "elevenlabs" in n:
        return "elevenlabs"
    return None


def _turn_rows(
    call_id: str,
    llm: list[dict],
    tts: list[dict],
    react: list[dict],
    tool: list[dict],
    max_turn: int = 10,
) -> list[dict]:
    """One row per turn that has an LLM ttfb, a TTS first-audio and a reaction-time span.

    Rules (each came out of a case that gave a wrong number):
    - The turn's LLM request is the EARLIEST llm ttfb span; n_llm_requests counts them all.
    - First audio is the first tts.first_audio ttfb span at or after that request.
    - A turn is dropped unless the reaction span lands within 6s of the LLM request.
    - Sentence fields (sent_*, n_pieces) are read only from TTS events at or after the request,
      per provider: Cartesia emits tts.text_sent plus tts.chunk_cache pieces sharing a
      synthesis_seq; ElevenLabs emits tts.chunk_cache pieces (one sentence = all pieces of the
      first synthesis_seq) or, without a cache, tts.segment_sent.
    - lead_silence_ms comes from the ttfa-phase first_audio of the turn.
    - tool_in_turn is 1 if any tool span carries that turn_number.
    """
    llm = [r for r in _prep_spans(llm) if r["phase"] == "ttfb" and r["node"].startswith("llm")]
    tts = _prep_spans(tts)
    react = _prep_spans(react)
    tools = _prep_spans(tool)
    rows = []
    for T in range(1, max_turn + 1):
        A = [r for r in llm if r["turn"] == T]
        if not A:
            continue
        a = A[0]
        fas = [
            r
            for r in tts
            if r["turn"] == T
            and r["event_name"] == "tts.first_audio"
            and r["phase"] == "ttfb"
            and r["ts"] >= a["ts"]
        ]
        if not fas:
            continue
        fa = fas[0]
        fam = _tts_family(fa["node"])
        C = [r for r in react if r["turn"] == T and r["node"] == "pipeline.bot_reaction_time"]
        if not C:
            continue
        c = C[0]
        if c["ts"] - a["ts"] >= 6000:
            continue
        ttfa = [
            r
            for r in tts
            if r["turn"] == T
            and r["event_name"] == "tts.first_audio"
            and r["phase"] == "ttfa"
            and r["ts"] >= a["ts"]
        ]
        lead = _num_or_none(ttfa[0]["md"].get("leading_silence_ms")) if ttfa else None
        sent_ts = sent_chars = n_pieces = None
        evs = [r for r in tts if r["turn"] == T and r["ts"] >= a["ts"]]
        cc = [
            r for r in evs if r["event_name"] == "tts.chunk_cache" and _tts_family(r["node"]) == fam
        ]
        if fam == "cartesia":
            sent = [
                r
                for r in evs
                if r["event_name"] == "tts.text_sent" and _tts_family(r["node"]) == "cartesia"
            ]
            if sent:
                m = sent[0]["md"]
                chars = _num_or_none(m.get("char_count"))
                if chars is None and isinstance(m.get("text"), str):
                    chars = len(m["text"])
                sent_ts, sent_chars = _iso_ms(sent[0]["ts"]), chars
            if cc:
                seq = cc[0]["md"].get("synthesis_seq")
                if seq is not None:
                    n_pieces = sum(1 for r in cc if r["md"].get("synthesis_seq") == seq)
        elif fam == "elevenlabs":
            seg = [
                r
                for r in evs
                if r["event_name"] == "tts.segment_sent" and _tts_family(r["node"]) == fam
            ]

            def chn(r):
                m = r["md"]
                v = _num_or_none(m.get("characters"))
                return v if v is not None else _num_or_none(m.get("chars"))

            if cc:
                seq = cc[0]["md"].get("synthesis_seq")
                if seq is not None:
                    piece = [r for r in cc if r["md"].get("synthesis_seq") == seq]
                    vals = [chn(r) for r in piece]
                    if all(v is not None for v in vals):
                        sent_chars = sum(vals)
                    sent_ts, n_pieces = _iso_ms(min(r["ts"] for r in piece)), len(piece)
                else:
                    sent_chars, sent_ts, n_pieces = chn(cc[0]), _iso_ms(cc[0]["ts"]), 1
            elif seg:
                sent_chars, sent_ts, n_pieces = chn(seg[0]), _iso_ms(seg[0]["ts"]), 1
        ch = fa["md"].get("cache_hit")
        rows.append(
            {
                "call_id": call_id,
                "turn": T,
                "llm_node": a["node"],
                "llm_model": a["md"].get("model") or "",
                "n_llm_requests": len(A),
                "ttfb_ms": _numstr(a["value_ms"]),
                "ttfb_ts": _iso_ms(a["ts"]),
                "sent_ts": sent_ts or "",
                "sent_chars": _numstr(sent_chars),
                "n_pieces": _numstr(n_pieces),
                "tts_provider": fam or fa["node"].removeprefix("tts."),
                "fa_ts": _iso_ms(fa["ts"]),
                "fa_ttfb_ms": _numstr(fa["value_ms"]),
                "cache_hit": "" if ch is None else ("true" if ch else "false"),
                "lead_silence_ms": _numstr(lead),
                "reaction_ms": _numstr(c["value_ms"]),
                "reaction_ts": _iso_ms(c["ts"]),
                "tool_in_turn": 1 if any(r["turn"] == T for r in tools) else 0,
            }
        )
    return rows


_TURN_COLUMNS = [
    "call_id", "turn", "llm_node", "llm_model", "n_llm_requests", "ttfb_ms", "ttfb_ts",
    "sent_ts", "sent_chars", "n_pieces", "tts_provider", "fa_ts", "fa_ttfb_ms", "cache_hit",
    "lead_silence_ms", "reaction_ms", "reaction_ts", "tool_in_turn",
]  # fmt: skip


async def _export_turns_core(
    ctx: Context, call_sids: str, max_turn: int = 10, out_name: str = ""
) -> dict:
    sids = _parse_sids(call_sids)
    max_turn = max(max_turn, 1)
    path = _export_path(out_name, "turn_latency", "csv")
    await _ensure_auth(ctx)

    async def work(sid: str):
        parts, complete = [], True
        for node, phase in (
            ("llm", "ttfb"),
            ("tts", ""),
            ("pipeline.bot_reaction_time", ""),
            ("tool", ""),
        ):
            spans, ok = await _fetch_all_spans(ctx, sid, node=node, phase=phase)
            parts.append(spans)
            complete = complete and ok
        llm, tts, react, tool = parts
        return _turn_rows(sid, llm, tts, react, tool, max_turn), llm, complete

    rows: list[dict] = []
    ok, failed, truncated = 0, [], []
    served: dict[str, set[str]] = {}
    for sid, res, err in await _per_call(sids, work):
        if err:
            failed.append({"call_id": sid, "error": err})
            continue
        ok += 1
        turn_rows, llm, complete = res
        if not complete:
            truncated.append(sid)
        rows.extend(turn_rows)
        for s in llm:
            if s.get("phase") == "ttfb" and (s.get("node") or "").startswith("llm"):
                model = _span_metadata(s).get("model") or "unknown"
                served.setdefault(f"{s['node']}/{model}", set()).add(sid)

    out = {"path": None, "rows": len(rows), "columns": _TURN_COLUMNS}
    if ok:
        _write_csv(path, _TURN_COLUMNS, rows)
        out["path"] = path
    out.update(calls_ok=ok, calls_failed=failed, calls_truncated=truncated)
    out["served_llm"] = {k: len(v) for k, v in sorted(served.items())}
    return out


@mcp.tool(annotations=EXPORTS_LOCAL_FILES)
async def export_turn_latency(
    call_sids: str, ctx: Context, max_turn: int = 10, out_name: str = ""
) -> dict[str, Any]:
    """Export one CSV row per conversational turn with LLM, TTS and reaction timings.

    Per call it pages the LLM ttfb, TTS, pipeline.bot_reaction_time and tool spans and joins
    them by turn_number, so the file answers "how long until the caller heard the first
    sentence, and what served it" without reading a span. Returns the path, never the rows.
    Turns missing an LLM ttfb, a TTS first-audio or a reaction span are skipped.

    Columns: call_id, turn, llm_node, llm_model, n_llm_requests, ttfb_ms, ttfb_ts, sent_ts,
    sent_chars, n_pieces (first sentence as sent to TTS), tts_provider, fa_ts, fa_ttfb_ms,
    cache_hit, lead_silence_ms, reaction_ms, reaction_ts, tool_in_turn. sent_* are filled for
    Cartesia and ElevenLabs only. The result also has served_llm: llm node/model -> number of
    calls that used it. Note turn_number counts utterances, not user turns.

    Args:
        call_sids: Comma-separated call SIDs, or "@/abs/path" naming a file with one SID per
            line. At most 1000 calls per invocation.
        max_turn: Highest turn_number to export (default 10).
        out_name: File name inside the scratchpad dir (default turn_latency_<UTC timestamp>_<random>.csv).
    """
    return _out(await _export_turns_core(ctx, call_sids, max_turn, out_name))


async def _get_or_none(ctx: Context, path: str) -> dict | None:
    """GET that treats 404 as absence rather than failure — a call can have one
    prompt without the other (e.g. extraction never ran, or the 7-day
    ee_trace_logs window has passed while the 30-day agent prompt survives)."""
    try:
        return await _get(ctx, path)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise


@mcp.tool(annotations=WRITES_LOCAL_FILES)
async def download_prompts(call_sids: str, ctx: Context) -> str:
    """Download the agent prompt and entity prompt for one or more calls to local files.

    PREFER THIS over get_call_prompt / get_entity_prompt when you want the prompt
    text itself. Prompts run 30–60 KB each, which overflows an inline tool
    response — the content then has to be paged back out of a spill file as
    double-encoded JSON. This writes plain .txt instead: readable, greppable, and
    directly shareable with someone else.

    Same idea as download_trace_logs, applied to prompts.

    Writes up to three files per call:
        <sid>_agent_prompt.txt          the live-call agent prompt (30-day retention)
        <sid>_entity_prompt_system.txt  the extraction instructions (7-day retention)
        <sid>_entity_prompt_user.txt    the transcript the extractor was given

    A call missing either prompt is reported, not failed — the two have different
    retention windows and extraction may never have run.

    Still use get_entity_prompt when you want the parsed metadata inline
    (variant, campaign_id, entity_config_ref_id, lead_details) rather than the
    prompt bodies; that response is small enough to return directly.

    Args:
        call_sids: One or more call SIDs, comma-separated (e.g. "abc123" or "abc123,def456").
    """
    scratchpad = os.environ.get("CLAUDE_SCRATCHPAD_DIR", tempfile.gettempdir())
    sids = [_sanitize_sid(s) for s in call_sids.split(",") if s.strip()]
    if not sids:
        raise ValueError("No call SIDs provided")

    results: list[str] = []
    for sid in sids:
        try:
            agent = await _get_or_none(ctx, f"/call/{sid}/prompt")
            entity = await _get_or_none(ctx, f"/call/{sid}/entity-prompt")
        except Exception as e:
            results.append(f"{sid}: FAILED ({type(e).__name__}: {e})")
            continue

        written: list[str] = []
        if agent and agent.get("prompt_text"):
            p = os.path.join(scratchpad, f"{sid}_agent_prompt.txt")
            n = _write_private(p, agent["prompt_text"])
            ref = agent.get("prompt_reference_id") or "?"
            written.append(f"    agent prompt  ({n:,} chars, ref {ref}) -> {p}")
        else:
            written.append("    agent prompt  NOT FOUND (older than 30d retention?)")

        if entity and entity.get("system_prompt"):
            p = os.path.join(scratchpad, f"{sid}_entity_prompt_system.txt")
            n = _write_private(p, entity["system_prompt"])
            variant = entity.get("variant") or "base"
            written.append(f"    entity prompt ({n:,} chars, variant {variant}) -> {p}")
            up = os.path.join(scratchpad, f"{sid}_entity_prompt_user.txt")
            un = _write_private(up, entity.get("user_prompt") or "")
            written.append(f"    extractor input ({un:,} chars) -> {up}")
        else:
            written.append("    entity prompt NOT FOUND (no extraction run, or older than 7d)")

        results.append(f"{sid}:\n" + "\n".join(written))

    return (
        "Downloaded prompts:\n" + "\n".join(results) + "\n\n"
        "Files are plain text. Read or grep them directly — no need to re-fetch."
    )


# ── Structured search (fast, uses indexed columns) ────────────────────


def _filters_dropped_by_server(data: dict, **filters: str) -> list[str]:
    """Name any filter the server accepted but did not apply.

    /search ignored event_name and phase until squadrun/lens#55, returning
    unfiltered rows with a 200. Client and backend deploy separately, so checking
    the rows we got back is the only way to tell a fixed backend from an old one.

    One-sided by construction: no row matching the filter proves it was dropped,
    but every row matching proves nothing — an ignored filter still looks correct
    when the unfiltered page happens to be uniform. Silence here is not a pass.
    """
    rows = [m for call in data.get("calls", []) for m in call.get("matches", [])]
    if not rows:
        return []
    return [
        f"{field}={value!r}"
        for field, value in filters.items()
        if value and all(row.get(field) != value for row in rows)
    ]


@mcp.tool(annotations=READ_ONLY)
async def search_spans(
    ctx: Context,
    query: str = "",
    node: str = "",
    event_name: str = "",
    phase: str = "",
    level: str = "",
    value_ms_min: float = 0,
    value_ms_max: float = 0,
    campaign_id: str = "",
    agent_config_id: str = "",
    prompt_ref: str = "",
    customer: str = "",
    customer_exclude: str = "",
    include_metadata: bool = False,
    metadata_filter: str = "",
    time_range_minutes: int = 60,
    time_from: str = "",
    time_to: str = "",
    limit: int = 50,
    fields: str = "",
) -> dict[str, Any]:
    """Search structured spans across all calls using indexed columns.

    THIS IS THE PRIMARY SEARCH TOOL. Prefer structured filters (node, level,
    campaign_id) over the query param — they use indexed columns and avoid
    expensive LIKE scans.

    This searches SPAN COLUMNS, never log text. The query param LIKE-matches
    event_name, node, campaign_id, prompt_reference_id, call_id and error_message
    only — a phrase from a log line will not match here and returns an empty result
    that looks like "no such calls". For log bodies use search_trace_logs.

    The query param is OPTIONAL. When you have exact filter values, omit it.
    When provided, query text is LIKE-matched against event_name, node,
    campaign_id, prompt_reference_id, call_id, error_message — this is slower.

    metadata_filter requires a node scope. Unscoped it cannot be answered without
    scanning every span, so the server rejects it — this tool checks first and
    fails locally with the same guidance, saving the round trip. Values match
    exactly and case-sensitively: the full model id hits, a prefix of it
    or a different case does not. Numbers and bools match as written
    in the JSON ("prompt_tokens:26430", "speech_final:true").

    Returns at most 100 spans per page (use offset via repeated calls). This is
    a row sampler, not a measurement tool — for percentiles across calls use
    aggregate_calls, which pages properly.

    If a response comes back with FILTER_IGNORED, the lens backend predates
    squadrun/lens#55 and is dropping event_name/phase server-side: the rows are
    real spans but not the ones you asked for. Do not draw conclusions from them.

    For known node/event values, check the bot repo for the latest instrumentation.

    Args:
        query: Optional free-text search (LIKE match — expensive). Prefer structured filters instead.
        node: Node prefix filter (e.g. "llm.", "stt.", "tool."). Uses indexed SET column. "llm.", "llm" and "llm.<provider>" all work.
        event_name: Exact event name filter (e.g. "llm.usage", "pipeline.cost_summary"). Indexed.
        phase: Phase filter ("ttfb", "complete", "error", "timeout", "cancelled", "connect").
        level: Level filter — "info" or "error". Indexed.
        value_ms_min: Min duration in ms (find slow operations).
        value_ms_max: Max duration in ms.
        campaign_id: Filter by campaign. Indexed LowCardinality column.
        agent_config_id: Filter by agent config.
        prompt_ref: Filter by prompt reference ID.
        customer: Filter by customer/tenant name.
        customer_exclude: Exclude a specific customer.
        include_metadata: If true, also search inside metadata JSON (slower).
        metadata_filter: Key:value filter on metadata JSON (e.g. "outcome:error", "transport:sse"). REQUIRES node to also be set.
        time_range_minutes: Look back N minutes from now (default 60, max 10080 = 7 days).
        time_from: ISO8601 start time (alternative to time_range_minutes).
        time_to: ISO8601 end time.
        limit: Max results (default 50, max 100).
        fields: Comma-separated keys to keep per matched span; "metadata.<key>" picks one metadata key, flattened. Empty = whole rows. Call-level keys (call_id, totals) are always kept.
    """
    if metadata_filter and not node:
        raise ValueError(
            f"metadata_filter={metadata_filter!r} needs a node scope — metadata is "
            f"unindexed, so filtering it across a whole time window would scan every "
            f"span. The server rejects this too; failing here saves the round trip. "
            f"Retry with e.g. node='llm.<provider>' or node='llm.'."
        )

    params: dict = {"limit": min(limit, 100)}
    if query:
        params["q"] = query
    if node:
        params["node"] = node
    if event_name:
        params["event_name"] = event_name
    if phase:
        params["phase"] = phase
    if level:
        params["level"] = level
    if value_ms_min > 0:
        params["value_ms_min"] = value_ms_min
    if value_ms_max > 0:
        params["value_ms_max"] = value_ms_max
    if campaign_id:
        params["campaign_id"] = campaign_id
    if agent_config_id:
        params["agent_config_id"] = agent_config_id
    if prompt_ref:
        params["prompt_ref"] = prompt_ref
    if customer:
        params["customer"] = customer
    if customer_exclude:
        params["customer_exclude"] = customer_exclude
    if include_metadata:
        params["include_metadata"] = "true"
    if metadata_filter:
        params["metadata_filter"] = metadata_filter
    if time_from:
        params["time_from"] = time_from
    elif time_range_minutes > 0:
        params["time_range_minutes"] = time_range_minutes
    if time_to:
        params["time_to"] = time_to
    data = await _get(ctx, "/search", params=params)
    ignored = _filters_dropped_by_server(data, event_name=event_name, phase=phase)
    if ignored:
        data = {
            "FILTER_IGNORED": (
                f"This lens backend dropped {', '.join(ignored)} server-side — the rows below are "
                f"real spans but NOT the ones you filtered for. Do not draw conclusions from them. "
                f"Fixed by squadrun/lens#55; until it is deployed, use count_spans (which honours "
                f"both) or filter within a call via aggregate_spans."
            ),
            **data,
        }
    if fields:
        data = _project_search(data, _parse_fields(fields))
    return _out(data)


@mcp.tool(annotations=READ_ONLY)
async def list_calls(
    ctx: Context,
    campaign_id: str = "",
    voice_mission_id: str = "",
    agent_config_id: str = "",
    customer: str = "",
    status: str = "",
    ee_status: str = "",
    stt_provider: str = "",
    llm_model: str = "",
    tts_provider: str = "",
    prompt_reference_id: str = "",
    time_range_minutes: int = 60,
    start_time_from: str = "",
    start_time_to: str = "",
    limit: int = 20,
    offset: int = 0,
    agent_id: str = "",
    pipecat_version: str = "",
    fields: str = "",
    min_turns: int = 0,
    min_duration_ms: int = 0,
    max_scan: int = 300,
) -> dict[str, Any]:
    """List calls from call_metadata with structured filters.

    Queries the call_metadata table (one row per call, ReplacingMergeTree).
    Use this to find calls by provider, duration, status, campaign, etc.

    Returns: call_id, duration_ms, total_turns, error_count, providers,
    start_time, status, entity_status, warmup_ms, and flags.

    Args:
        campaign_id: Filter by campaign.
        voice_mission_id: Filter by voice mission (a campaign contains many).
        agent_config_id: Filter by agent config ID.
        customer: Filter by customer/tenant.
        status: Filter by call status (e.g. "completed").
        ee_status: Filter by entity-extraction status (e.g. "success"). Use this
            when you need a call that actually ran extraction — e.g. before
            calling get_entity_prompt, which has nothing to return otherwise.
        stt_provider: Filter by STT provider.
        llm_model: Filter by LLM model.
        tts_provider: Filter by TTS provider.
        prompt_reference_id: Filter by prompt reference.
        time_range_minutes: Look back N minutes (default 60, max 10080 = 7 days). For older calls, use start_time_from/to instead.
        start_time_from: ISO8601 start time (alternative to time_range_minutes). No cap — use for calls older than 7 days.
        start_time_to: ISO8601 end time.
        limit: Max results (default 20, max 100). With min_turns / min_duration_ms, the max number of matching calls returned.
        offset: Row offset into the listing. With min_turns / min_duration_ms, always continue
            with the returned next_offset plus start_time_from_pinned / start_time_to_pinned
            (the scan pins its window at the first request): on a backend that filters
            server-side (server_filtered=true) offset counts matching calls, otherwise it is a
            scan position in the unfiltered listing, so offset=<number of matches> re-reads rows.
        agent_id: Filter by agent ID.
        pipecat_version: Filter by the pipeline framework version the call ran on.
        fields: Comma-separated keys to keep per call (e.g. "call_id,duration_ms,total_turns"). Empty = whole rows.
        min_turns: Keep only calls with at least this many total_turns. Sent to the backend, which filters server-side when it supports it (squadrun/lens#98; the result says server_filtered); otherwise the listing is paged and filtered here.
        min_duration_ms: Keep only calls at least this long. Same as min_turns.
            Warning: the newest calls are often still running and show total_turns 0 and no
            duration_ms, so min_turns / min_duration_ms can match nothing on a window ending "now".
            Set start_time_to an hour or so in the past to skip them.
        max_scan: With a min_* filter, the most listing rows to scan (default 300, cap 1000). The result reports scanned, matched and scan_complete — if scan_complete is false the list may be missing matches; narrow the time window or raise max_scan.
    """
    params: dict = {"limit": min(limit, 100)}
    if offset > 0:
        params["offset"] = offset
    if agent_id:
        params["agent_id"] = agent_id
    if pipecat_version:
        params["pipecat_version"] = pipecat_version
    if campaign_id:
        params["campaign_id"] = campaign_id
    if voice_mission_id:
        params["voice_mission_id"] = voice_mission_id
    if agent_config_id:
        params["agent_config_id"] = agent_config_id
    if customer:
        params["customer"] = customer
    if status:
        params["status"] = status
    if ee_status:
        params["ee_status"] = ee_status
    if stt_provider:
        params["stt_provider"] = stt_provider
    if llm_model:
        params["llm_model"] = llm_model
    if tts_provider:
        params["tts_provider"] = tts_provider
    if prompt_reference_id:
        params["prompt_reference_id"] = prompt_reference_id
    if start_time_from:
        params["start_time_from"] = start_time_from
    elif time_range_minutes > 0:
        params["time_range_minutes"] = min(time_range_minutes, 10080)
    if start_time_to:
        params["start_time_to"] = start_time_to
    wanted = _parse_fields(fields)
    if min_turns <= 0 and min_duration_ms <= 0:
        data = await _get(ctx, "/calls", params=params)
        if wanted:
            data = {**data, "calls": _project_rows(data.get("calls", []), wanted)}
        return _out(data)

    cap = min(max(max_scan, 1), _MAX_CALLS_SCAN)
    keep = min(max(limit, 1), 100)
    # Pin the window at the first request, so calls arriving mid-scan cannot push rows down
    # between pages and a relative window cannot slide. Pass the reported
    # start_time_from_pinned / start_time_to_pinned back with next_offset to continue.
    pinned_to = pinned_from = ""
    if not start_time_to:
        now = datetime.now(UTC)
        pinned_to = params["start_time_to"] = now.isoformat(timespec="seconds")
        if "time_range_minutes" in params:
            minutes = params.pop("time_range_minutes")
            pinned_from = (now - timedelta(minutes=minutes)).isoformat(timespec="seconds")
            params["start_time_from"] = pinned_from
    # Backends with squadrun/lens#98 filter these server-side and say so in filters_applied;
    # older ones ignore the params, and the client-side check below does the work.
    if min_turns > 0:
        params["min_turns"] = min_turns
    if min_duration_ms > 0:
        params["min_duration_ms"] = min_duration_ms
    server_filtered = False
    matched: list[dict] = []
    seen: set[str] = set()
    scanned, pos, complete = 0, max(offset, 0), False
    while scanned < cap:
        page_size = min(_page_caps["calls"], cap - scanned)
        data = await _get_capped(
            ctx,
            "/calls",
            {**params, "limit": page_size, "offset": pos},
            "calls",
            _CALLS_PAGE_FALLBACK,
        )
        page_size = min(page_size, _page_caps["calls"])  # what was actually asked for
        server_filtered = bool(data.get("filters_applied"))
        rows = data.get("calls", [])
        scanned += len(rows)
        pos += len(rows)
        for r in rows:
            # Calls starting mid-scan shift the newest-first listing down, so a row can come
            # back on the next page.
            if r.get("call_id") in seen:
                continue
            seen.add(r.get("call_id"))
            if (r.get("total_turns") or 0) >= min_turns and (
                r.get("duration_ms") or 0
            ) >= min_duration_ms:
                matched.append(r)
        if len(rows) < page_size:
            complete = True
            break
        if len(matched) >= keep:
            break
    shown = matched[:keep]
    return _out(
        {
            "calls": _project_rows(shown, wanted) if wanted else shown,
            "scanned": scanned,
            "matched": len(matched),
            "scan_complete": complete,
            "next_offset": None if complete else pos,
            "server_filtered": server_filtered,
            "start_time_from_pinned": pinned_from or None,
            "start_time_to_pinned": pinned_to or None,
        }
    )


@mcp.tool(annotations=READ_ONLY)
async def count_spans(
    ctx: Context,
    node: str = "",
    level: str = "",
    event_name: str = "",
    phase: str = "",
    campaign_id: str = "",
    agent_config_id: str = "",
    prompt_ref: str = "",
    time_range_minutes: int = 60,
    time_from: str = "",
    time_to: str = "",
) -> dict[str, Any]:
    """Count matching spans without returning row data.

    Use this when someone asks "how many?" — returns the exact count and
    number of affected calls. Fast because it's just a COUNT query on
    indexed columns, no row data transferred.

    Args:
        node: Node prefix filter (e.g. "llm.", "tool.").
        level: Level filter — "error" to count failures.
        event_name: Exact event name filter (e.g. "pipeline.call_start").
        phase: Phase filter (e.g. "error", "timeout").
        campaign_id: Campaign filter.
        agent_config_id: Agent config filter.
        prompt_ref: Prompt reference filter.
        time_range_minutes: Look back N minutes (default 60).
        time_from: ISO8601 start time (alternative to time_range_minutes).
        time_to: ISO8601 end time.
    """
    params: dict = {}
    if node:
        params["node"] = node
    if level:
        params["level"] = level
    if event_name:
        params["event_name"] = event_name
    if phase:
        params["phase"] = phase
    if campaign_id:
        params["campaign_id"] = campaign_id
    if agent_config_id:
        params["agent_config_id"] = agent_config_id
    if prompt_ref:
        params["prompt_ref"] = prompt_ref
    if time_from:
        params["time_from"] = time_from
    elif time_range_minutes > 0:
        params["time_range_minutes"] = time_range_minutes
    if time_to:
        params["time_to"] = time_to
    return _out(await _get(ctx, "/search/count", params=params))


# ── Fleet aggregation (server-side, one hop instead of a sweep) ────────


def _window(params: dict, time_range_minutes: int, time_from: str, time_to: str) -> dict:
    """Apply the time window every fleet endpoint shares. Mutates and returns params."""
    if time_from:
        params["time_from"] = time_from
    elif time_range_minutes > 0:
        params["time_range_minutes"] = time_range_minutes
    if time_to:
        params["time_to"] = time_to
    return params


async def _get_new_endpoint(ctx: Context, path: str, params: dict, needs: str) -> dict:
    """GET an endpoint that may not be deployed yet, and say so if it isn't.

    The backend and this client ship separately, so a 404 here means "your lens is
    older than this tool", not "no data". Left as a raw HTTPStatusError that reads
    like a broken tool rather than a version gap.
    """
    try:
        return await _get(ctx, path, params=params)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                f"This lens backend does not have {path} yet — it needs {needs}. "
                f"Nothing is wrong with your query. Use the alternative named in this "
                f"tool's description until the backend is deployed."
            ) from None
        raise


_GRANULARITY_MINUTES = {"5min": 5, "15min": 15, "1hour": 60}  # all the backend accepts
# Sizes the backend lacks, built by summing its buckets: name -> (source size, minutes).
# Each divides an hour or a day evenly, so rolled-up buckets start on :00/:10/:30 or midnight.
_CLIENT_ROLLUPS = {"10min": ("5min", 10), "30min": ("15min", 30), "1day": ("1hour", 1440)}
_MAX_BUCKETS = 800  # 30 days hourly (720) is a legitimate trend query; 5min over days is not


def _check_granularity(
    granularity: str, time_range_minutes: int, time_from: str, time_to: str
) -> str:
    """Refuse a window/granularity pair that would return tens of thousands of rows, and return
    the backend bucket size to request.

    A client-only size (_CLIENT_ROLLUPS) is fetched at its source size and summed afterwards,
    so it is sized on the finer rows actually fetched. Bounds buckets alone — these endpoints
    return one row per (bucket, node-or-event_name), so 30 days at 5min is ~8600 buckets times
    every distinct value, but this check cannot see that second factor. See _summarize_overflow
    for the guard on that fan-out when the caller has not named specific values.
    """
    backend = _CLIENT_ROLLUPS[granularity][0] if granularity in _CLIENT_ROLLUPS else granularity
    if backend not in _GRANULARITY_MINUTES:
        raise ValueError(
            f"granularity must be one of {sorted(_GRANULARITY_MINUTES)}, got {granularity!r}"
        )
    if time_from:
        minutes = _window_minutes(0, time_from, time_to)
        if minutes is None:
            return backend  # open-ended or unparseable explicit window — cannot size
    else:
        minutes = time_range_minutes
    buckets = minutes / _GRANULARITY_MINUTES[backend]
    if buckets > _MAX_BUCKETS:
        fetched_as = f" (fetched as {backend} buckets)" if backend != granularity else ""
        coarser = "" if granularity == "1day" else "a coarser granularity or "
        raise ValueError(
            f"{int(minutes)} minutes at {granularity} granularity{fetched_as} is "
            f"~{int(buckets)} buckets per node, which will overflow the tool response. Use "
            f"{coarser}a shorter window (max ~{_MAX_BUCKETS} buckets)."
        )
    return backend


# Observed live: a single unfiltered hourly bucket returned ~80 distinct event_names
# (2026-09-25 sample of event_counts_over_time) — buckets alone don't bound the response when
# the fan-out dimension (node / event_name) is wide open. Measured on the actual row count
# rather than a guessed fan-out factor, since cardinality varies by fleet and time.
_MAX_ROWS_UNFILTERED = 2000


def _summarize_overflow(
    data: dict,
    fan_out_filter: str,
    fan_out_param: str,
    dim_key: str,
    sum_keys: tuple[str, ...],
    range_keys: tuple[str, ...] = (),
    caveat: str = "",
) -> dict:
    """Swap an oversized unfiltered series for one total per dim_key over the whole window.

    The rows are already fetched, so a digest answers "which ones matter" at no extra cost,
    where refusing would waste the fetch and force a blind retry. range_keys are non-additive
    (percentiles), so they are reported as the [min, max] seen across buckets, never combined.
    Skipped when fan_out_filter is set — a caller who named values chose that multiplier.
    """
    series = data.get("series")
    if fan_out_filter or not isinstance(series, list) or len(series) <= _MAX_ROWS_UNFILTERED:
        return data
    totals: dict = {}
    for row in series:
        t = totals.setdefault(row.get(dim_key), {dim_key: row.get(dim_key), "buckets": 0})
        t["buckets"] += 1
        for k in sum_keys:
            t[k] = t.get(k, 0) + (row.get(k) or 0)
        for k in range_keys:
            v = row.get(k)
            if isinstance(v, (int, float)):
                lo, hi = t.get(f"{k}_range", (v, v))
                t[f"{k}_range"] = (round(min(lo, v), 1), round(max(hi, v), 1))
    ranked = sorted(totals.values(), key=lambda t: t.get(sum_keys[0], 0), reverse=True)
    return {
        "TRUNCATED": (
            f"{len(series)} rows with no {fan_out_param} filter is past the "
            f"{_MAX_ROWS_UNFILTERED}-row budget, so the per-bucket series was replaced by one "
            f"total per {dim_key} over the whole window, largest first.{caveat} For the time "
            f"series, re-run with {fan_out_param}=<the {dim_key}s you care about>, narrow the "
            f"window, or pass full_series=true for every row regardless of size."
        ),
        **{k: v for k, v in data.items() if k != "series"},
        "totals": ranked,
    }


def _rollup(series: list, granularity: str, dim_key: str, exact_key: str, approx_key: str) -> list:
    """Sum backend buckets into a client-only granularity from _CLIENT_ROLLUPS.

    exact_key (a plain occurrence count) sums correctly — an event belongs to exactly one
    bucket. approx_key (a distinct/affected-call count) is summed too but is only an upper
    bound: a call whose activity spans a bucket boundary is counted again in each bucket.
    Buckets are floored in UTC — the backend's naive ts is UTC — so a "1day" bucket is a UTC day.
    """
    source, size = _CLIENT_ROLLUPS[granularity]
    out: dict = {}
    for row in series:
        try:
            ts = datetime.fromisoformat(row["ts"])
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(
                f"Unexpected bucket shape from the backend (no parseable 'ts'): {row!r} — "
                f"granularity={granularity} rollup cannot proceed. Retry with "
                f"granularity={source}."
            ) from None
        if ts.tzinfo is not None:
            ts = ts.astimezone(UTC).replace(tzinfo=None)
        minute_of_day = ts.hour * 60 + ts.minute
        start = minute_of_day - minute_of_day % size
        start_ts = ts.replace(hour=start // 60, minute=start % 60, second=0, microsecond=0)
        label = start_ts.date().isoformat() if granularity == "1day" else start_ts.isoformat()
        bucket = out.setdefault(
            (label, row.get(dim_key)),
            {"ts": label, dim_key: row.get(dim_key), exact_key: 0, approx_key: 0},
        )
        bucket[exact_key] += row.get(exact_key) or 0
        bucket[approx_key] += row.get(approx_key) or 0
    # str() because a null dimension next to a named one would make the tuples incomparable.
    return [out[k] for k in sorted(out, key=lambda k: (k[0], str(k[1])))]


def _apply_rollup(
    data: dict, granularity: str, dim_key: str, exact_key: str, approx_key: str
) -> dict:
    """data with its series replaced by _rollup and a NOTE added; unchanged for backend sizes."""
    if granularity not in _CLIENT_ROLLUPS:
        return data
    source = _CLIENT_ROLLUPS[granularity][0]
    if not isinstance(data.get("series"), list):
        # Passing the rows through would label source-size buckets as the size asked for.
        raise RuntimeError(
            f"Unexpected response from the backend (no 'series' list) — granularity="
            f"{granularity} rollup cannot proceed. Retry with granularity={source}."
        )
    return {
        **data,
        "series": _rollup(data["series"], granularity, dim_key, exact_key, approx_key),
        "NOTE": (
            f"granularity={granularity} sums the underlying {source} buckets client-side, in UTC: "
            f"`{exact_key}` is exact (each one belongs to one {source} bucket), `{approx_key}` is "
            f"an upper bound (a call whose activity spans a {source} boundary is counted again in "
            f"each bucket it touched). The first and last buckets may cover only part of their span."
        ),
    }


def _upper_bound_caveat(approx_key: str, granularity: str) -> str:
    """The _summarize_overflow caveat for the count tools, whose approx_key sums to an upper bound."""
    day_hint = "" if granularity == "1day" else ' granularity="1day" gives one row per day.'
    return (
        f" `{approx_key}` totals are upper bounds (a call active in several buckets counts in "
        f"each).{day_hint}"
    )


_RETENTION = {
    "mv": "aggregates retain 90 days — longer than the 30-day raw spans they summarise",
    "spans": "raw spans retain 30 days",
    "traces": "trace logs retain only 4 days",
}


@mcp.tool(annotations=READ_ONLY)
async def latency_over_time(
    ctx: Context,
    granularity: Literal["5min", "15min", "1hour"] = "1hour",
    nodes: str = "",
    campaign_id: str = "",
    agent_config_id: str = "",
    prompt_ref: str = "",
    time_range_minutes: int = 360,
    time_from: str = "",
    time_to: str = "",
    full_series: bool = False,
) -> dict[str, Any]:
    """Latency percentiles per node per time bucket — the fleet trend in one call.

    USE THIS INSTEAD OF sweeping search_spans across time windows. Returns
    p50/p90/p95/p99 + count for every node in every bucket, computed server-side
    from a materialized view, so a week-long window costs the same as an hour.

    Answers:
    - "is latency getting worse?"            → compare early vs late buckets
    - "which stage regressed, and when?"     → node column across buckets
    - "was this worse last week?"            → time_from/time_to on that week

    Reads a 5-minute materialized view retained for 90 DAYS — three times the
    30-day raw span window. Fleet trends stay answerable long after the spans
    that produced them have aged out.

    Without nodes, a bucket fans out across every distinct node seen fleet-wide. Past a
    row budget the response carries TRUNCATED and one `totals` entry per node for the
    whole window instead of `series` — then re-run with nodes= for the time series.

    Args:
        granularity: Bucket size — "5min", "15min" or "1hour" (default). Coarser is smaller; prefer 1hour for windows over a day. No "10min" / "30min" / "1day": the backend has only these three, and percentiles can't be summed into coarser buckets — event_counts_over_time / error_counts_over_time offer them, since counts can.
        nodes: Comma-separated node names to restrict to (e.g. "llm.<provider>,tts.<provider>"). Empty returns every node, summarized to totals past the row budget above.
        campaign_id: Restrict to a campaign (comma-separated for several).
        agent_config_id: Restrict to an agent config.
        prompt_ref: Restrict to a prompt reference ID.
        time_range_minutes: Look back N minutes (default 360, max 43200 = 30 days).
        time_from: ISO8601 start (alternative to time_range_minutes).
        time_to: ISO8601 end.
        full_series: Return every per-bucket row even past the row budget, instead of the totals summary.
    """
    _check_granularity(granularity, time_range_minutes, time_from, time_to)
    params: dict = {"granularity": granularity}
    if nodes:
        params["nodes"] = nodes
    if campaign_id:
        params["campaign_id"] = campaign_id
    if agent_config_id:
        params["agent_id"] = agent_config_id
    if prompt_ref:
        params["prompt_ref"] = prompt_ref
    _window(params, time_range_minutes, time_from, time_to)
    data = await _get(ctx, "/observability/timeseries", params=params)
    if not full_series:
        data = _summarize_overflow(
            data,
            nodes,
            "nodes",
            "node",
            ("count",),
            ("p50_ms", "p90_ms", "p95_ms", "p99_ms"),
            caveat=" Percentiles can't be combined across buckets, so each p*_ms_range is the "
            "[min, max] seen across them.",
        )
    return _out({"retention": _RETENTION["mv"], **data})


@mcp.tool(annotations=READ_ONLY)
async def latency_breakdown(
    ctx: Context,
    campaign_id: str = "",
    agent_config_id: str = "",
    prompt_ref: str = "",
    time_range_minutes: int = 360,
    time_from: str = "",
    time_to: str = "",
) -> dict[str, Any]:
    """Percentiles per node and phase across the whole fleet — the stage attribution table.

    One call replaces walking search_spans through llm. / stt. / tts. /
    pipeline.* one prefix at a time. Returns count + p25/p50/p80/p90/p95/p99 per
    (node, phase) over every matching span — hundreds of thousands of spans, not
    a 100-row sample.

    Use this to answer "which stage dominates latency" before drilling into any
    single call. For one call's breakdown use aggregate_spans instead.

    Reads raw spans (30-day window), so it is heavier than latency_over_time.
    Narrow the window if it times out.

    Args:
        campaign_id: Restrict to a campaign (comma-separated for several).
        agent_config_id: Restrict to an agent config.
        prompt_ref: Restrict to a prompt reference ID.
        time_range_minutes: Look back N minutes (default 360, max 43200 = 30 days).
        time_from: ISO8601 start (alternative to time_range_minutes).
        time_to: ISO8601 end.
    """
    params: dict = {}
    if campaign_id:
        params["campaign_id"] = campaign_id
    if agent_config_id:
        params["agent_id"] = agent_config_id
    if prompt_ref:
        params["prompt_ref"] = prompt_ref
    _window(params, time_range_minutes, time_from, time_to)
    data = await _get(ctx, "/observability/breakdown", params=params)
    return _out({"retention": _RETENTION["spans"], **data})


@mcp.tool(annotations=READ_ONLY)
async def event_counts_over_time(
    ctx: Context,
    event_types: str = "",
    granularity: Literal["5min", "10min", "15min", "30min", "1hour", "1day"] = "1hour",
    campaign_id: str = "",
    agent_config_id: str = "",
    time_range_minutes: int = 360,
    time_from: str = "",
    time_to: str = "",
    full_series: bool = False,
) -> dict[str, Any]:
    """Event counts per event_name per time bucket, with affected-call counts.

    USE THIS INSTEAD OF calling count_spans once per event name or once per
    provider. Returns every event_name broken out per bucket in one request.

    Answers "how often did X fire, and is that rising?" — voicemail detections,
    interruptions, capacity gates, provider selections, hotswaps.

    Reads a materialized view retained for 90 DAYS.

    Without event_types, a bucket fans out across every distinct event_name seen
    fleet-wide (commonly 50-100+). Past a row budget the response carries TRUNCATED
    and one `totals` entry per event_name for the whole window instead of `series` —
    then re-run with event_types= for the time series.

    Args:
        event_types: Comma-separated event names to restrict to (e.g. "pipeline.voicemail,tool.completed"). Empty returns all, summarized to totals past the row budget above.
        granularity: "5min", "15min", "1hour" (default), or "10min" / "30min" / "1day" (UTC days), which sum 5min / 15min / 1hour buckets client-side — exact for count, an upper bound for calls; see the response's NOTE.
        campaign_id: Restrict to a campaign.
        agent_config_id: Restrict to an agent config.
        time_range_minutes: Look back N minutes (default 360, max 43200 = 30 days).
        time_from: ISO8601 start (alternative to time_range_minutes).
        time_to: ISO8601 end.
        full_series: Return every per-bucket row even past the row budget, instead of the totals summary.
    """
    backend_granularity = _check_granularity(granularity, time_range_minutes, time_from, time_to)
    params: dict = {"granularity": backend_granularity}
    if event_types:
        params["event_types"] = event_types
    if campaign_id:
        params["campaign_id"] = campaign_id
    if agent_config_id:
        params["agent_id"] = agent_config_id
    _window(params, time_range_minutes, time_from, time_to)
    data = await _get(ctx, "/observability/events", params=params)
    data = _apply_rollup(data, granularity, "event_name", "count", "calls")
    if not full_series:
        data = _summarize_overflow(
            data,
            event_types,
            "event_types",
            "event_name",
            ("count", "calls"),
            caveat=_upper_bound_caveat("calls", granularity),
        )
    return _out({"retention": _RETENTION["mv"], **data})


@mcp.tool(annotations=READ_ONLY)
async def error_counts_over_time(
    ctx: Context,
    granularity: Literal["5min", "10min", "15min", "30min", "1hour", "1day"] = "1hour",
    nodes: str = "",
    campaign_id: str = "",
    prompt_ref: str = "",
    time_range_minutes: int = 360,
    time_from: str = "",
    time_to: str = "",
    full_series: bool = False,
) -> dict[str, Any]:
    """Error span counts per node per time bucket, with affected-call counts.

    The error counterpart to latency_over_time. Use it to find WHEN a provider
    started failing, rather than counting errors one node at a time with
    count_spans(level="error").

    Reads a materialized view retained for 90 DAYS.

    Without nodes, a bucket fans out across every distinct node seen fleet-wide. Past a
    row budget the response carries TRUNCATED and one `totals` entry per node for the
    whole window instead of `series` — then re-run with nodes= for the time series.

    Args:
        granularity: "5min", "15min", "1hour" (default), or "10min" / "30min" / "1day" (UTC days), which sum 5min / 15min / 1hour buckets client-side — exact for errors, an upper bound for affected_calls; see the response's NOTE.
        nodes: Comma-separated node names to restrict to. Empty returns all, summarized to totals past the row budget above.
        campaign_id: Restrict to a campaign.
        prompt_ref: Restrict to a prompt reference ID.
        time_range_minutes: Look back N minutes (default 360, max 43200 = 30 days).
        time_from: ISO8601 start (alternative to time_range_minutes).
        time_to: ISO8601 end.
        full_series: Return every per-bucket row even past the row budget, instead of the totals summary.
    """
    backend_granularity = _check_granularity(granularity, time_range_minutes, time_from, time_to)
    params: dict = {"granularity": backend_granularity}
    if nodes:
        params["nodes"] = nodes
    if campaign_id:
        params["campaign_id"] = campaign_id
    if prompt_ref:
        params["prompt_ref"] = prompt_ref
    _window(params, time_range_minutes, time_from, time_to)
    data = await _get(ctx, "/observability/error-timeseries", params=params)
    data = _apply_rollup(data, granularity, "node", "errors", "affected_calls")
    if not full_series:
        data = _summarize_overflow(
            data,
            nodes,
            "nodes",
            "node",
            ("errors", "affected_calls"),
            caveat=_upper_bound_caveat("affected_calls", granularity),
        )
    return _out({"retention": _RETENTION["mv"], **data})


@mcp.tool(annotations=READ_ONLY)
async def tool_outcomes(
    ctx: Context,
    campaign_id: str = "",
    prompt_ref: str = "",
    time_range_minutes: int = 360,
    time_from: str = "",
    time_to: str = "",
) -> dict[str, Any]:
    """Success/failure/timeout counts and durations per bot tool function.

    Answers "which tool is failing or slow" across the fleet in one call —
    per-tool outcome counts and latency, rather than searching node="tool.*"
    and tallying by hand.

    Args:
        campaign_id: Restrict to a campaign.
        prompt_ref: Restrict to a prompt reference ID.
        time_range_minutes: Look back N minutes (default 360, max 43200 = 30 days).
        time_from: ISO8601 start (alternative to time_range_minutes).
        time_to: ISO8601 end.
    """
    params: dict = {}
    if campaign_id:
        params["campaign_id"] = campaign_id
    if prompt_ref:
        params["prompt_ref"] = prompt_ref
    _window(params, time_range_minutes, time_from, time_to)
    return _out(await _get(ctx, "/observability/tools", params=params))


@mcp.tool(annotations=READ_ONLY)
async def slowest_calls(
    ctx: Context,
    node: str,
    phase: str = "",
    event_name: str = "",
    metric: str = "value_ms",
    campaign_id: str = "",
    agent_config_id: str = "",
    prompt_ref: str = "",
    percentiles: str = "50,90",
    order_by: str = "p90",
    min_spans: int = 1,
    time_range_minutes: int = 360,
    time_from: str = "",
    time_to: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """Rank every call in a window by a latency percentile — one query, whole population.

    The fleet-wide answer to "which calls are worst, and is this call unusual".
    Computed with a single GROUP BY in ClickHouse, so it covers every matching
    call rather than a sample.

    Prefer this over aggregate_calls for fleet questions: aggregate_calls scans the
    most recent N calls one request at a time, whereas this ranks the whole
    population in one hop. Use aggregate_calls when you want per-call metadata
    (model, duration) alongside the numbers.

    `cohort` is pooled over the entire window, not just the returned page — so it
    stays a true population percentile even though `calls` is a top-N slice.

    Takes the same metric= as aggregate_spans, so it also ranks calls by prompt
    size: metric="metadata:prompt_tokens", event_name="llm.usage", order_by="max".

    Args:
        node: Node prefix to measure (e.g. "llm", "pipeline.bot_reaction_time"). Required — an unfiltered percentile mixes unrelated operations.
        phase: Phase filter ("ttfb", "complete", "error").
        event_name: Exact event_name filter (e.g. "llm.usage").
        metric: "value_ms" (default) or "metadata:<key>" for a numeric metadata field.
        campaign_id: Restrict to a campaign (comma-separated for several).
        agent_config_id: Restrict to an agent config.
        prompt_ref: Restrict to a prompt reference ID.
        percentiles: Comma-separated percentiles (default "50,90").
        order_by: Rank by this — a percentile like "p90", or avg/max/min/sum/count.
        min_spans: Skip calls with fewer than this many matching spans — raise it to stop one-span calls topping the ranking.
        time_range_minutes: Look back N minutes (default 360, max 43200 = 30 days).
        time_from: ISO8601 start (alternative to time_range_minutes).
        time_to: ISO8601 end.
        limit: Max calls returned (default 50, max 500).
    """
    if not node:
        raise ValueError("node is required — an unfiltered cohort percentile is meaningless")
    if not any((campaign_id, agent_config_id, prompt_ref)):
        window = _window_minutes(time_range_minutes, time_from, time_to)
        if window is None or window > _UNSCOPED_WINDOW_CAP_MIN:
            raise ValueError(
                f"Ranking a whole cohort scans and groups every span by call_id. Without a "
                f"high-selectivity filter ({', '.join(_HIGH_SELECTIVITY)}), the window is "
                f"capped at {_UNSCOPED_WINDOW_CAP_MIN} minutes (~2h) — node alone is too loose. "
                f"Add campaign_id (or agent_config_id / prompt_ref) to rank over a wider window."
            )
    params: dict = {
        "node": node,
        "percentiles": percentiles,
        "order_by": order_by,
        "min_spans": max(min_spans, 1),
        "limit": min(max(limit, 1), 500),
    }
    for key, val in (
        ("phase", phase),
        ("event_name", event_name),
        ("campaign_id", campaign_id),
        ("agent_config_id", agent_config_id),
        ("prompt_ref", prompt_ref),
    ):
        if val:
            params[key] = val
    if metric and metric != "value_ms":
        params["metric"] = metric
    _window(params, time_range_minutes, time_from, time_to)
    data = await _get_new_endpoint(
        ctx,
        "/observability/call-percentiles",
        params,
        needs="squadrun/lens#56 (use aggregate_calls meanwhile)",
    )
    return _out({"retention": _RETENTION["spans"], **data})


@mcp.tool(annotations=READ_ONLY)
async def list_filter_values(ctx: Context) -> dict[str, Any]:
    """List the values you can actually filter on — campaigns, models, providers, agents.

    CALL THIS BEFORE GUESSING a campaign_id, llm_model, stt_provider or
    prompt_reference_id. Returns the distinct values seen in the last 7 days, so a
    filter built from it cannot silently match nothing.

    Returns campaigns, prompt_refs, agents, agent_config_ids, llm_models,
    llm_providers, stt_providers, tts_providers, customers, voice_mission_ids.

    A filter value absent from this list will return zero rows — which is
    indistinguishable from "the thing you asked about did not happen".

    Feeds the filters on list_calls, search_spans, slowest_calls, latency_over_time
    and latency_breakdown.
    """
    return _out(await _get(ctx, "/observability/filters"))


@mcp.tool(annotations=READ_ONLY)
async def search_trace_logs(
    ctx: Context,
    query: str,
    time_range_minutes: int = 60,
    time_from: str = "",
    time_to: str = "",
    level: str = "",
    campaign_id: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Free-text search across raw log bodies for ALL calls — not one call at a time.

    This is the fleet-wide log grep. Use it to find which calls contain a log
    line, then drill in with download_trace_logs.

    Not to be confused with search_spans, which matches span columns and cannot see
    log text at all. If you are looking for a phrase a service logged, it is this
    tool. Previously this meant calling
    get_call_trace_logs once per call_sid, which needs the call list up front —
    exactly what you do not have when hunting an unknown failure.

    Examples:
        search_trace_logs("rate limit exceeded")
        search_trace_logs("connection error, will retry", time_range_minutes=180)
        search_trace_logs("prompt cache", campaign_id="<campaign_id>")

    Multi-word queries are AND-ed as substrings; quote a phrase to keep it
    together. This is a text scan over call_trace_logs, so keep the window tight
    — and note trace logs retain only 4 DAYS, far less than spans' 30. An empty
    result on an older call means the evidence aged out, not that it never
    happened.

    WINDOW LIMIT: unscoped (no campaign_id) this is bounded to 2 hours, because a
    fleet-wide text scan of call_trace_logs times out past a few hours at current
    log volume. You can walk a wider period by paging time_from/time_to in <=2h
    slices. To search a wide window in one shot, pass campaign_id: it narrows to
    that campaign's calls on the primary key first, which is ~100x faster and lifts
    the limit to the full 4-day retention.

    Args:
        query: Text to find in log messages (required).
        time_range_minutes: Look back N minutes (default 60). Max 120 (2h) unless campaign_id is set, then up to 5760 (4 days).
        time_from: ISO8601 start. Unscoped, the from/to span must be <=2h; with campaign_id, any window.
        time_to: ISO8601 end.
        level: Level filter (mostly useless — trace logs are nearly all info).
        campaign_id: Restrict to calls in a campaign. Strongly preferred — it makes the search fast and unlocks the full window.
        limit: Max log lines (default 50, max 500).
        offset: Row offset — the response carries has_more; page with offset=offset+limit.
    """
    if not query.strip():
        raise ValueError("query is required — this endpoint is a text search")
    if not campaign_id:
        window = _window_minutes(time_range_minutes, time_from, time_to)
        if window is None or window > _UNSCOPED_WINDOW_CAP_MIN:
            raise ValueError(
                f"A fleet-wide trace search (no campaign_id) is bounded to "
                f"{_UNSCOPED_WINDOW_CAP_MIN} minutes (~2h) — an unbounded text scan over "
                f"call_trace_logs times out at current scale. Either page the period in "
                f"<=2h time_from/time_to slices, or pass campaign_id to search a wider window "
                f"in one shot (it narrows on the primary key first, ~100x faster)."
            )
    params: dict = {"q": query, "limit": min(max(limit, 1), 500), "offset": max(offset, 0)}
    if level:
        params["level"] = level
    if campaign_id:
        params["campaign_id"] = campaign_id
    _window(params, time_range_minutes, time_from, time_to)
    data = await _get(ctx, "/search/traces", params=params)
    return _out({"retention": _RETENTION["traces"], **data})


@mcp.tool(annotations=READ_ONLY)
async def get_call_config(call_sid: str, ctx: Context) -> dict[str, Any]:
    """Get the agent config JSON for a call, on its own.

    Same data as get_call_details(sections="config"), but without fetching the
    transcript server-side. Use when the question is purely about configuration.

    This is the config TEMPLATE. For the rendered prompt actually sent to the
    LLM, use get_call_prompt. Configs retain 30 days.

    Args:
        call_sid: The call SID.
    """
    call_sid = _sanitize_sid(call_sid)
    return _out(
        await _get_new_endpoint(
            ctx,
            f"/call/{call_sid}/config",
            {},
            needs='squadrun/lens#56 (use get_call_details(sections="config") meanwhile)',
        )
    )


@mcp.tool(annotations=READ_ONLY)
async def get_extraction_stats(
    ctx: Context,
    view: Literal["outcomes", "latency", "models", "filters"] = "outcomes",
    customer: str = "",
    campaign_id: str = "",
    voice_mission_id: str = "",
    granularity: Literal["5min", "15min", "1hour"] = "1hour",
    time_range_minutes: int = 1440,
    time_from: str = "",
    time_to: str = "",
) -> dict[str, Any]:
    """Entity-extraction (post-call) stats across the fleet — outcomes, latency, models.

    The extraction-side counterpart to latency_over_time. Answers "what fraction
    of extractions are failing, and on which model" without opening calls
    one by one.

    Args:
        view: Which slice — "outcomes" (success/failure over time), "latency" (stage durations), "models" (per-model breakdown), or "filters" (valid filter values).
        customer: Restrict to a customer/tenant.
        campaign_id: Restrict to a campaign.
        voice_mission_id: Restrict to a voice mission.
        granularity: "5min", "15min" or "1hour" (default). Ignored by "models" and "filters".
        time_range_minutes: Look back N minutes (default 1440 = 24h, max 43200).
        time_from: ISO8601 start (alternative to time_range_minutes).
        time_to: ISO8601 end.
    """
    endpoints = {
        "outcomes": "/observability/ee-outcomes",
        "latency": "/observability/ee-latency",
        "models": "/observability/ee-models",
        "filters": "/observability/ee-filters",
    }
    params: dict = {}
    if view in ("outcomes", "latency"):
        params["granularity"] = granularity
    if view != "filters":
        for key, val in (
            ("customer", customer),
            ("campaign_id", campaign_id),
            ("voice_mission_id", voice_mission_id),
        ):
            if val:
                params[key] = val
    _window(params, time_range_minutes, time_from, time_to)
    return _out(await _get(ctx, endpoints[view], params=params))


# ── Comparison tools ───────────────────────────────────────────────────


@mcp.tool(annotations=READ_ONLY)
async def compare_calls(call_sids: str, ctx: Context) -> dict[str, Any]:
    """Compare 2-10 calls side by side — metadata and timing spans.

    Returns metadata (duration, outcome, providers, errors) and spans
    for each call. Capped at 500 spans per call, fairly distributed via
    per-call partitioning. Returns span_counts per call to show truncation.

    Args:
        call_sids: Comma-separated call SIDs (e.g. "abc123,def456,ghi789").
    """
    sids = [_sanitize_sid(s) for s in call_sids.split(",") if s.strip()]
    return _out(await _get(ctx, "/compare", params={"call_ids": ",".join(sids)}))


@mcp.tool(annotations=READ_ONLY)
async def compare_prompts(call_sid_a: str, call_sid_b: str, ctx: Context) -> dict[str, Any]:
    """Diff the system prompts used in two calls.

    Returns a unified diff showing exactly what changed between the prompts.
    Useful for investigating behavior differences between calls. To read a
    prompt in full rather than diff it, use get_call_prompt.

    Args:
        call_sid_a: First call SID.
        call_sid_b: Second call SID.
    """
    call_sid_a = _sanitize_sid(call_sid_a)
    call_sid_b = _sanitize_sid(call_sid_b)
    return _out(
        await _get(
            ctx,
            "/compare/prompt-diff",
            params={"call_id_a": call_sid_a, "call_id_b": call_sid_b},
        )
    )


async def _run_auth():
    """Interactive auth — run once to authenticate with Google."""
    if not BASE_URL:
        print("Error: LENS_BASE_URL is required.")
        return False
    if not GOOGLE_CLIENT_ID:
        print("Error: LENS_GOOGLE_CLIENT_ID is required.")
        return False

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
        result = await _device_flow_auth(client)
        if result:
            print("Authentication successful. You can now use Lens MCP.")
            return True
        print("Authentication failed.")
        return False


async def _ensure_auth_before_serve():
    cached = _load_cached_token()
    if cached and _is_token_valid(cached):
        return True
    if API_KEY:
        return True
    print("No valid token found. Starting authentication...\n")
    return await _run_auth()


class _CliContext:
    """Just enough of Context for _get / _ensure_auth outside a running MCP server."""

    def __init__(self, client: httpx.AsyncClient):
        self.request_context = SimpleNamespace(
            lifespan_context={"client": client, "authenticated": False}
        )


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lens-mcp", description="Bulk exports without an MCP client.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("export-spans", help="export spans from many calls to one file")
    sp.add_argument("call_sids", help='comma list, or "@/abs/path" with one SID per line')
    sp.add_argument("--node", default="")
    sp.add_argument("--phase", default="")
    sp.add_argument("--event-name", default="")
    sp.add_argument("--fields", default="")
    sp.add_argument("--format", default="csv", choices=["csv", "jsonl"])
    sp.add_argument("--include-text", action="store_true")
    sp.add_argument("--out-name", default="")
    tp = sub.add_parser("export-turns", help="export per-turn latency rows to one CSV")
    tp.add_argument("call_sids", help='comma list, or "@/abs/path" with one SID per line')
    tp.add_argument("--max-turn", type=int, default=10)
    tp.add_argument("--out-name", default="")
    return ap


async def _run_cli(args: argparse.Namespace) -> dict:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
        ctx = _CliContext(client)
        if args.cmd == "export-spans":
            return await _export_spans_core(
                ctx,
                args.call_sids,
                args.node,
                args.phase,
                args.event_name,
                args.fields,
                args.format,
                args.include_text,
                args.out_name,
            )
        return await _export_turns_core(ctx, args.call_sids, args.max_turn, args.out_name)


def _cli(argv: list[str]) -> int:
    args = _build_cli().parse_args(argv)
    if not BASE_URL:
        print("Error: LENS_BASE_URL is required.", file=sys.stderr)
        return 1
    try:
        result = asyncio.run(_run_cli(args))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=False, default=str))
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("export-spans", "export-turns"):
        sys.exit(_cli(sys.argv[1:]))
    if len(sys.argv) > 1 and sys.argv[1] == "auth":
        asyncio.run(_run_auth())
    elif len(sys.argv) > 1 and sys.argv[1] == "serve":
        if not asyncio.run(_ensure_auth_before_serve()):
            return
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        mcp.settings.port = port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
