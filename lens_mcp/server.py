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

import asyncio
import base64
import json
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from mcp.server.fastmcp import Context, FastMCP

BASE_URL = os.environ.get("LENS_BASE_URL", "")
if not BASE_URL:
    raise RuntimeError("LENS_BASE_URL is required. Set it in your MCP server config.")
API_KEY = os.environ.get("LENS_API_KEY", "")
GOOGLE_CLIENT_ID = os.environ.get("LENS_GOOGLE_CLIENT_ID", "")
API_PREFIX = "/api/ext/v1"
TIMEOUT = 30.0

_CALL_SID_PATTERN = __import__("re").compile(r"^[a-zA-Z0-9\-]+$")


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
        tmp.write_text(json.dumps({
            "token": token,
            "username": username,
            "expires_at": expires_at,
        }))
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
        return datetime.now(timezone.utc).timestamp() < exp
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

        print(f"\n  Lens MCP: Google authentication required")
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
                    print(f"  Authenticated as: {data.get('email', data.get('username', ''))}\n", flush=True)
                    return {"Authorization": f"Bearer {data['token']}"}
            else:
                detail = ""
                try:
                    detail = resp.json().get("detail", "")
                except Exception:
                    pass
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
        base_url=BASE_URL, timeout=TIMEOUT,
    ) as client:
        yield {"client": client, "authenticated": False}


mcp = FastMCP("lens", lifespan=lifespan)


def _client(ctx: Context) -> httpx.AsyncClient:
    return ctx.request_context.lifespan_context["client"]


def _fmt(data: dict | list) -> str:
    return json.dumps(data, indent=2, default=str)


async def _ensure_auth(ctx: Context) -> None:
    lc = ctx.request_context.lifespan_context
    if lc["authenticated"]:
        return
    client = lc["client"]
    auth_headers = await _authenticate(client)
    _set_auth(client, auth_headers)
    lc["authenticated"] = True


async def _get(ctx: Context, path: str, params: dict | None = None) -> dict:
    await _ensure_auth(ctx)
    client = _client(ctx)
    resp = await client.get(f"{API_PREFIX}{path}", params=params)
    if resp.status_code == 401:
        _clear_cached_token()
        auth_headers = await _authenticate(client)
        _set_auth(client, auth_headers)
        resp = await client.get(f"{API_PREFIX}{path}", params=params)
    resp.raise_for_status()
    return resp.json()


# ── Schema reference ───────────────────────────────────────────────────


@mcp.tool()
async def get_schema(ctx: Context) -> str:
    """Get the ClickHouse table schemas, column types, indexes, and query strategy.

    READ THIS FIRST before searching or investigating. It tells you:
    - What columns exist and which are indexed (fast to filter on)
    - The difference between call_spans (structured, fast) vs call_trace_logs (raw text, slow)
    - The correct query strategy to avoid expensive operations

    Key rules:
    - Use search_spans with structured filters (node, event_name, level, campaign_id) — these are indexed
    - NEVER broad-search trace logs — only drill into specific call_sids
    - Time bounds are mandatory for any search
    """
    data = await _get(ctx, "/schema")
    return data.get("schema", "")


# ── Per-call investigation (fast, indexed by call_id) ──────────────────


@mcp.tool()
async def get_call_details(call_sid: str, ctx: Context, include_config: bool = False) -> str:
    """Get complete details for a call — metadata, transcript, entities, and config.

    This is the primary 'what happened in this call' tool. Use it first when
    investigating a call. Returns everything in one request.

    ALREADY INCLUDES the full transcript — do NOT also call get_call_transcript,
    that would be redundant and waste tokens.

    The config (agent_config_json) is large (~10K tokens). Set include_config=True
    only when you need the agent config. Default is False to save tokens.

    For deep investigation after this, ALWAYS use download_trace_logs to download
    raw logs locally, then grep with Bash. Never use get_call_trace_logs.

    Args:
        call_sid: The call SID (e.g. "abc123def456").
        include_config: Include the full agent config JSON (default False). Set True only when investigating config.
    """
    call_sid = _sanitize_sid(call_sid)
    params = {}
    if not include_config:
        params["include_config"] = "false"
    data = await _get(ctx, f"/call/{call_sid}/details", params=params or None)
    if "recording" in data:
        data["recording"] = {"exists": data["recording"].get("exists", False)}
    return _fmt(data)


@mcp.tool()
async def get_call_transcript(call_sid: str, ctx: Context) -> str:
    """Get the full conversation transcript for a call.

    NOTE: get_call_details already includes the transcript. Only use this
    if you specifically need the transcript alone without metadata/entities.
    Do NOT call both get_call_details and get_call_transcript — that
    duplicates the transcript and wastes tokens.

    Args:
        call_sid: The call SID.
    """
    call_sid = _sanitize_sid(call_sid)
    return _fmt(await _get(ctx, f"/call/{call_sid}/transcript"))


@mcp.tool()
async def get_call_spans(call_sid: str, ctx: Context, node: str = "", phase: str = "") -> str:
    """Get structured instrumentation spans and events for a call.

    Returns the latency waterfall: LLM/STT/TTS spans with timing,
    tool call durations, pipeline events, and error spans. Indexed by
    call_id — fast even for calls with hundreds of spans. Capped at 500 rows.

    Each span has: node (e.g. "llm.openai"), phase (complete/error/ttfb),
    value_ms (duration), level, error_message, and metadata.

    Args:
        call_sid: The call SID.
        node: Filter by node PREFIX (e.g. "llm" matches all LLM spans; "stt" matches all STT spans). Always use this for targeted lookups.
        phase: Filter by phase (e.g. "error", "ttfb", "complete"). Useful for quick error checks.
    """
    call_sid = _sanitize_sid(call_sid)
    params: dict = {}
    if node:
        params["node"] = node
    if phase:
        params["phase"] = phase
    return _fmt(await _get(ctx, f"/call/{call_sid}/spans", params=params or None))


@mcp.tool()
async def get_call_trace_logs(
    call_sid: str,
    ctx: Context,
    search: str = "",
    logger_name: str = "",
    level: str = "",
    limit: int = 50,
) -> str:
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
    return _fmt(await _get(ctx, f"/call/{call_sid}/traces", params=params))


@mcp.tool()
async def get_call_entities(call_sid: str, ctx: Context) -> str:
    """Get entity extraction results for a call.

    Returns extracted entities (outcome, answers like call_outcome,
    rescheduled_to, etc.) and extraction metadata (status, timing).

    Args:
        call_sid: The call SID.
    """
    call_sid = _sanitize_sid(call_sid)
    return _fmt(await _get(ctx, f"/call/{call_sid}/entities"))


@mcp.tool()
async def get_call_context(call_sid: str, ctx: Context) -> str:
    """Get tool call details and pipeline summaries for a call.

    Returns structured tool call records (with args, results, durations)
    and pipeline summary events (call_summary, cost_summary, router_decision).

    Args:
        call_sid: The call SID.
    """
    call_sid = _sanitize_sid(call_sid)
    return _fmt(await _get(ctx, f"/call/{call_sid}/context"))


@mcp.tool()
async def get_lead_details(call_sid: str, ctx: Context) -> str:
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
    return _fmt(await _get(ctx, f"/call/{call_sid}/lead-details"))


# ── Trace log download (for multi-grep in Claude Code) ─────────────────


@mcp.tool()
async def download_trace_logs(call_sids: str, ctx: Context) -> str:
    """Download trace logs for one or more calls to local files, then grep locally.

    This is the ONLY correct way to search trace logs. Downloads once per call,
    then grep locally with Bash — unlimited searches, zero extra tokens,
    case-insensitive, no ClickHouse round-trips.

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
        grep -c "openai" /path/to/file.log            # count matches

    Args:
        call_sids: One or more call SIDs, comma-separated (e.g. "abc123" or "abc123,def456,ghi789").
    """
    scratchpad = os.environ.get("CLAUDE_SCRATCHPAD_DIR", tempfile.gettempdir())
    sids = [_sanitize_sid(s) for s in call_sids.split(",") if s.strip()]
    results = []

    for sid in sids:
        filepath = os.path.join(scratchpad, f"{sid}_traces.log")

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            line_count = sum(1 for _ in open(filepath))
            results.append(f"{sid}: already downloaded ({line_count} lines) -> {filepath}")
            continue

        total_lines = 0
        batch_size = 5000
        max_batches = 10

        try:
            fd = os.open(filepath + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                offset = 0
                for _ in range(max_batches):
                    data = await _get(ctx, f"/call/{sid}/traces", params={"limit": batch_size, "offset": offset})
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
            results.append(f"{sid}: {total_lines} lines -> {filepath}")
        except Exception as e:
            for p in (filepath + ".tmp", filepath):
                if os.path.exists(p):
                    os.remove(p)
            results.append(f"{sid}: FAILED ({e})")

    summary = "\n".join(results)
    return (
        f"Downloaded trace logs:\n{summary}\n\n"
        f"All logs are now local. Use Bash grep for any further searching — "
        f"no need to call the trace logs API again for these calls."
    )


# ── Structured search (fast, uses indexed columns) ────────────────────


@mcp.tool()
async def search_spans(
    ctx: Context,
    query: str = "",
    node: str = "",
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
) -> str:
    """Search structured spans across all calls using indexed columns.

    THIS IS THE PRIMARY SEARCH TOOL. Prefer structured filters (node, level,
    campaign_id) over the query param — they use indexed columns and avoid
    expensive LIKE scans.

    The query param is OPTIONAL. When you have exact filter values, omit it.
    When provided, query text is LIKE-matched against event_name, node,
    campaign_id, prompt_reference_id, call_id, error_message — this is slower.

    For known node/event values, check the bot repo for the latest instrumentation.

    Args:
        query: Optional free-text search (LIKE match — expensive). Prefer structured filters instead.
        node: Node prefix filter (e.g. "llm.", "stt.", "tool."). Uses indexed SET column.
        level: Level filter — "info" or "error". Indexed.
        value_ms_min: Min duration in ms (find slow operations).
        value_ms_max: Max duration in ms.
        campaign_id: Filter by campaign. Indexed LowCardinality column.
        agent_config_id: Filter by agent config.
        prompt_ref: Filter by prompt reference ID.
        customer: Filter by customer/tenant name.
        customer_exclude: Exclude a specific customer.
        include_metadata: If true, also search inside metadata JSON (slower).
        metadata_filter: Key:value filter on metadata JSON (e.g. "outcome:error", "model:gpt-4").
        time_range_minutes: Look back N minutes from now (default 60, max 10080 = 7 days).
        time_from: ISO8601 start time (alternative to time_range_minutes).
        time_to: ISO8601 end time.
        limit: Max results (default 50, max 100).
    """
    params: dict = {"limit": min(limit, 100)}
    if query:
        params["q"] = query
    if node:
        params["node"] = node
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
    return _fmt(await _get(ctx, "/search", params=params))


@mcp.tool()
async def list_calls(
    ctx: Context,
    campaign_id: str = "",
    agent_config_id: str = "",
    customer: str = "",
    status: str = "",
    stt_provider: str = "",
    llm_model: str = "",
    tts_provider: str = "",
    prompt_reference_id: str = "",
    time_range_minutes: int = 60,
    start_time_from: str = "",
    start_time_to: str = "",
    limit: int = 20,
) -> str:
    """List calls from call_metadata with structured filters.

    Queries the call_metadata table (one row per call, ReplacingMergeTree).
    Use this to find calls by provider, duration, status, campaign, etc.

    Returns: call_id, duration_ms, total_turns, error_count, providers,
    start_time, status, entity_status, warmup_ms, and flags.

    Args:
        campaign_id: Filter by campaign.
        agent_config_id: Filter by agent config ID.
        customer: Filter by customer/tenant.
        status: Filter by call status (e.g. "completed").
        stt_provider: Filter by STT provider.
        llm_model: Filter by LLM model.
        tts_provider: Filter by TTS provider.
        prompt_reference_id: Filter by prompt reference.
        time_range_minutes: Look back N minutes (default 60, max 10080 = 7 days). For older calls, use start_time_from/to instead.
        start_time_from: ISO8601 start time (alternative to time_range_minutes). No cap — use for calls older than 7 days.
        start_time_to: ISO8601 end time.
        limit: Max results (default 20, max 100).
    """
    params: dict = {"limit": min(limit, 100)}
    if campaign_id:
        params["campaign_id"] = campaign_id
    if agent_config_id:
        params["agent_config_id"] = agent_config_id
    if customer:
        params["customer"] = customer
    if status:
        params["status"] = status
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
    return _fmt(await _get(ctx, "/calls", params=params))


@mcp.tool()
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
) -> str:
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
    return _fmt(await _get(ctx, "/search/count", params=params))


# ── Comparison tools ───────────────────────────────────────────────────


@mcp.tool()
async def compare_calls(call_sids: str, ctx: Context) -> str:
    """Compare 2-10 calls side by side — metadata and timing spans.

    Returns metadata (duration, outcome, providers, errors) and spans
    for each call. Capped at 500 spans per call, fairly distributed via
    per-call partitioning. Returns span_counts per call to show truncation.

    Args:
        call_sids: Comma-separated call SIDs (e.g. "abc123,def456,ghi789").
    """
    sids = [_sanitize_sid(s) for s in call_sids.split(",") if s.strip()]
    return _fmt(await _get(ctx, "/compare", params={"call_ids": ",".join(sids)}))


@mcp.tool()
async def compare_prompts(
    call_sid_a: str, call_sid_b: str, ctx: Context
) -> str:
    """Diff the system prompts used in two calls.

    Returns a unified diff showing exactly what changed between the prompts.
    Useful for investigating behavior differences between calls.

    Args:
        call_sid_a: First call SID.
        call_sid_b: Second call SID.
    """
    call_sid_a = _sanitize_sid(call_sid_a)
    call_sid_b = _sanitize_sid(call_sid_b)
    return _fmt(await _get(
        ctx, "/compare/prompt-diff",
        params={"call_id_a": call_sid_a, "call_id_b": call_sid_b},
    ))


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
        else:
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


def main():
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "auth":
        asyncio.run(_run_auth())
    elif len(_sys.argv) > 1 and _sys.argv[1] == "serve":
        if not asyncio.run(_ensure_auth_before_serve()):
            return
        port = int(_sys.argv[2]) if len(_sys.argv) > 2 else 8000
        mcp.settings.port = port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
