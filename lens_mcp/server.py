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

_SPAN_PAGE_SIZE = 500  # server-side hard cap on /call/{id}/spans
_MAX_SPAN_PAGES = 20  # 10k spans — beyond any real call, bounds runaway paging
_COHORT_CONCURRENCY = 8

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


_RETRY_STATUS = {502, 503, 504}
_MAX_RETRIES = 2


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
            await asyncio.sleep(0.5 * 2**attempt)
            continue

        if resp.status_code == 401:
            _clear_cached_token()
            _set_auth(client, await _authenticate(client))
            resp = await client.get(url, params=params)

        if resp.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES:
            await asyncio.sleep(0.5 * 2**attempt)
            continue

        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"GET {path} exhausted retries")


# ── Schema reference ───────────────────────────────────────────────────


# Semantics the server-side schema does not carry, each of which has silently
# produced a wrong answer: see squadrun/lens-mcp#1.
_CLIENT_NOTES = """

## Client notes (lens-mcp) — read before computing any number

### turn_number is a BOT-UTTERANCE counter, not a user-turn counter
It is `session_state.bot_stopped_count` in squadstack-bot, incremented on every
BotStoppedSpeakingFrame. Fillers, idle nudges and any non-LLM TTS advance it.
Two calls at the same turn_number are NOT at equal conversational depth — do not
use it to align calls when comparing them.

### metadata_filter needs a node scope
Unscoped, it silently returns 0 hits: `metadata_filter=transport:sse` alone
matches nothing, while the same filter with `node=llm.openai` matches. search_spans
rejects the unscoped form rather than returning a misleading empty result.
Values match exactly and case-sensitively against the raw JSON value —
`model:gemma4` hits; `model:gemma` and `model:GEMMA4` do not. Numbers and bools
match as written in the JSON (`prompt_tokens:26430`, `speech_final:true`).

### /search ignores event_name and phase
It accepts both and filters on neither — `phase=ttfb` and `phase=complete` return
byte-identical rows. search_spans therefore does not expose them. count_spans DOES
honour both. To filter spans by event_name or phase, work within a call
(get_call_spans / aggregate_spans), where phase is applied server-side and
event_name client-side.

### node accepts a prefix
`node=llm.`, `node=llm` and `node=llm.openai` all work, on both search_spans and
get_call_spans. Verified as genuinely filtering, along with level, campaign_id
and value_ms_min/max.

### Row caps — never percentile a single page
get_call_spans returns at most 500 rows per request, search_spans at most 100.
A p90 computed from one truncated page is wrong with no warning. Use
aggregate_spans (single call) or aggregate_calls (cohort): both page to
exhaustion and report `complete: false` if they could not.

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

Coverage is provider-dependent: the simplismart path emits llm.usage, the OpenAI
path does not (its llm.request spans carry only model and transport). Call-level
totals are on pipeline.cost_summary regardless. Where llm.usage is absent, per-turn
growth still needs trace logs within their 4-day window: download_trace_logs then
grep "prompt cache:".
"""


@mcp.tool()
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


@mcp.tool()
async def get_call_details(
    call_sid: str,
    ctx: Context,
    include_config: bool = False,
    sections: str = "",
    transcript_range: str = "",
) -> str:
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
        raise ValueError(
            f"Unknown section(s) {unknown}. Valid: {', '.join(_DETAIL_SECTIONS)}"
        )

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
async def get_call_prompt(call_sids: str, ctx: Context) -> str:
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
    return _fmt(results[0] if len(results) == 1 else {"prompts": list(results)})


@mcp.tool()
async def get_entity_prompt(call_sids: str, ctx: Context) -> str:
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
    return _fmt(results[0] if len(results) == 1 else {"prompts": list(results)})


@mcp.tool()
async def get_call_spans(
    call_sid: str,
    ctx: Context,
    node: str = "",
    phase: str = "",
    limit: int = 500,
    offset: int = 0,
) -> str:
    """Get structured instrumentation spans and events for a call.

    Returns the latency waterfall: LLM/STT/TTS spans with timing,
    tool call durations, pipeline events, and error spans. Indexed by
    call_id — fast even for calls with hundreds of spans.

    DO NOT compute percentiles, means or trends from this tool's output. The
    server caps a response at 500 rows, so a busy call returns a truncated
    prefix and any statistic derived from it is wrong. Use aggregate_spans —
    it pages to exhaustion and reports whether it saw everything. Use this tool
    to READ individual spans (errors, metadata, ordering), not to measure.

    Each span has: node (e.g. "llm.openai"), phase (complete/error/ttfb),
    value_ms (duration), level, error_message, and metadata.

    Note turn_number is a bot-utterance counter (bot_stopped_count), not a
    user-turn counter — fillers and idle nudges advance it.

    Args:
        call_sid: The call SID.
        node: Filter by node PREFIX (e.g. "llm" matches all LLM spans; "stt" matches all STT spans). Always use this for targeted lookups.
        phase: Filter by phase (e.g. "error", "ttfb", "complete"). Useful for quick error checks.
        limit: Rows per page (default 500, which is also the server maximum).
        offset: Row offset, for paging past the first 500.
    """
    call_sid = _sanitize_sid(call_sid)
    params: dict = {"limit": min(max(limit, 1), _SPAN_PAGE_SIZE), "offset": max(offset, 0)}
    if node:
        params["node"] = node
    if phase:
        params["phase"] = phase
    data = await _get(ctx, f"/call/{call_sid}/spans", params=params)
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
    return _fmt(data)


# ── Aggregation (pages the row cap away, returns statistics not rows) ──


async def _fetch_all_spans(
    ctx: Context, call_sid: str, node: str = "", phase: str = ""
) -> tuple[list[dict], bool]:
    """Page /call/{id}/spans to exhaustion. Returns (spans, complete).

    A single response is capped at 500 rows and only advertises truncation via
    has_more, so statistics computed from one unpaged response are silently wrong
    on long calls. Paging is stable: rows come back in (turn_number, timestamp_ms)
    order with no overlap between pages.
    """
    base: dict = {"limit": _SPAN_PAGE_SIZE}
    if node:
        base["node"] = node
    if phase:
        base["phase"] = phase

    spans: list[dict] = []
    offset = 0
    for _ in range(_MAX_SPAN_PAGES):
        data = await _get(ctx, f"/call/{call_sid}/spans", params={**base, "offset": offset})
        page = data.get("spans", [])
        spans.extend(page)
        if not page or not data.get("has_more"):
            return spans, True
        offset += len(page)
    return spans, False


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
        f"Unknown metric {metric!r} — use \"value_ms\" or \"metadata:<key>\" "
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


def _aggregate(spans: list[dict], pcts: list[float], group_by: str, metric: str = "value_ms") -> dict:
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


@mcp.tool()
async def aggregate_spans(
    call_sid: str,
    ctx: Context,
    node: str = "",
    phase: str = "",
    event_name: str = "",
    group_by: str = "",
    percentiles: str = "50,90,99",
    metric: str = "value_ms",
) -> str:
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
    but only on providers that emit it (simplismart does; the OpenAI path does
    not, so there fall back to trace logs).

    Returns overall count/min/mean/max/percentiles, a `trend` block comparing the
    median of the first third of the call against the last third, and per-group
    statistics when group_by is set. `complete: false` means the call exceeded the
    paging ceiling and the numbers cover only what was read — it never lies about
    coverage.

    Spans carrying no value for the chosen metric are counted under
    spans_without_metric and excluded from the statistics.

    Args:
        call_sid: The call SID.
        node: Node prefix filter (e.g. "llm", "llm.openai", "tts."). Strongly recommended — mixing STT and LLM durations makes the percentiles meaningless.
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
            f"Stopped after {_MAX_SPAN_PAGES * _SPAN_PAGE_SIZE} spans — statistics "
            f"cover only that prefix of the call. Narrow with node/phase and re-run."
        )
    return _fmt(out)


@mcp.tool()
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
) -> str:
    """Compare one latency metric across a COHORT of calls — "is it this call or the fleet?".

    The follow-up question after aggregate_spans finds a slow call. Selects calls
    with list_calls filters, then computes the same statistic per call and pooled
    across the cohort, so you can see whether a slow p90 is one outlier or the
    whole campaign.

    Example — campaign 9407 over the last 24h, ranked by LLM time-to-first-byte:
        aggregate_calls(node="llm", phase="ttfb", campaign_id="9407",
                        time_range_minutes=1440, sort_by="p90")

    Example — which calls in the campaign carry the biggest prompt context:
        aggregate_calls(node="llm", event_name="llm.usage",
                        metric="metadata:prompt_tokens", campaign_id="9407",
                        sort_by="max")

    `cohort` pools every span from every scanned call (the true fleet percentile,
    not an average of averages); `calls` lists per-call statistics sorted worst
    first. Calls with no matching spans are reported as a count, not silently
    dropped.

    COST: one paged span fetch per call, so keep max_calls modest. Scanning 20
    calls is a few seconds; 50 is the ceiling. Narrow the time window rather than
    raising max_calls — this samples the most recent N calls matching the filters,
    it is not a full-population query.

    Args:
        node: Node prefix to measure (e.g. "llm", "stt.deepgram", "tts"). Required — an unfiltered cohort percentile mixes unrelated operations and means nothing.
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
        return _fmt({"calls_matched": 0, "note": "No calls matched the filters — widen the time window."})

    sem = asyncio.Semaphore(_COHORT_CONCURRENCY)

    async def measure(call: dict) -> dict:
        sid = call["call_id"]
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
    return _fmt(out)


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
            line_count = sum(1 for _ in open(filepath))
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
                            ctx, f"/call/{sid}/{endpoint}", params={"limit": batch_size, "offset": offset}
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


@mcp.tool()
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


@mcp.tool()
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
        grep -c "openai" /path/to/file.log            # count matches

    Args:
        call_sids: One or more call SIDs, comma-separated (e.g. "abc123" or "abc123,def456,ghi789").
    """
    return await _download_logs(ctx, call_sids, "traces", "traces", "trace logs")


def _write_private(path: str, text: str) -> int:
    """Write text via a 0600 temp file + rename, so a partial write is never
    visible under the final path. Returns the character count."""
    fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.rename(path + ".tmp", path)
    return len(text)


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


@mcp.tool()
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

    THE SERVER DOES NOT FILTER BY event_name OR phase HERE. It accepts both and
    ignores them: phase="ttfb" and phase="complete" return byte-identical rows.
    They are deliberately not exposed on this tool. Use count_spans (which does
    honour them) to count, or aggregate_spans/get_call_spans to filter within a
    known call.

    metadata_filter ONLY WORKS when scoped by node. Unscoped it returns zero hits
    against spans that plainly contain the key — the empty result is
    indistinguishable from "no data", so this tool rejects that combination
    instead of returning it. Values match exactly and case-sensitively:
    metadata_filter="model:gemma4" hits, "model:gemma" and "model:GEMMA4" do not.
    Numbers and bools match as written in the JSON ("prompt_tokens:26430",
    "speech_final:true").

    Verified as actually filtering: node, level, campaign_id, value_ms_min/max,
    and node-scoped metadata_filter.

    Returns at most 100 spans per page (use offset via repeated calls). This is
    a row sampler, not a measurement tool — for percentiles across calls use
    aggregate_calls, which pages properly.

    For known node/event values, check the bot repo for the latest instrumentation.

    Args:
        query: Optional free-text search (LIKE match — expensive). Prefer structured filters instead.
        node: Node prefix filter (e.g. "llm.", "stt.", "tool."). Uses indexed SET column. "llm.", "llm" and "llm.openai" all work.
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
    """
    if metadata_filter and not node:
        raise ValueError(
            f"metadata_filter={metadata_filter!r} needs a node scope — unscoped it "
            f"silently returns 0 hits even when spans carry that key. "
            f"Retry with e.g. node='llm.openai'."
        )

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
) -> str:
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
        limit: Max results (default 20, max 100).
    """
    params: dict = {"limit": min(limit, 100)}
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
    Useful for investigating behavior differences between calls. To read a
    prompt in full rather than diff it, use get_call_prompt.

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
