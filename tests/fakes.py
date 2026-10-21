"""A fake Lens backend shared by the offline tests: patches server._get and server._ensure_auth.

Serves /call/{id}/spans (node prefix, phase, event_name, limit, offset, has_more — the same
paging contract as the real endpoint) and /calls (limit, offset). Records every request.
"""

import asyncio
import json
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

os.environ.setdefault("LENS_BASE_URL", "http://localhost")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lens_mcp.server as server


def span(node, event, ts, turn, phase=None, value=None, **md):
    return {
        "node": node,
        "phase": phase,
        "event_name": event,
        "timestamp_ms": ts,
        "value_ms": value,
        "turn_number": turn,
        "level": "info",
        "metadata": json.dumps(md),
    }


class FakeLens:
    def __init__(self, spans=None, calls=None, failing=()):
        self.spans = spans or {}
        self.calls = calls or []
        self.failing = set(failing)
        self.requests = []

    async def get(self, ctx, path, params=None):
        params = params or {}
        self.requests.append((path, dict(params)))
        if path == "/calls":
            off, lim = params.get("offset", 0), params.get("limit", 20)
            return {"calls": self.calls[off : off + lim]}
        sid = path.split("/")[2]
        if sid in self.failing:
            raise RuntimeError("boom")
        rows = self.spans.get(sid, [])
        if params.get("node"):
            rows = [r for r in rows if r["node"].startswith(params["node"])]
        if params.get("phase"):
            rows = [r for r in rows if r["phase"] == params["phase"]]
        if params.get("event_name"):
            rows = [r for r in rows if r["event_name"] == params["event_name"]]
        off, lim = params.get("offset", 0), min(params.get("limit", 500), 500)
        page = rows[off : off + lim]
        return {"spans": page, "total": len(rows), "has_more": off + lim < len(rows)}


@contextmanager
def patched(fake):
    async def no_auth(ctx):
        return None

    real = server._get, server._ensure_auth
    server._get, server._ensure_auth = fake.get, no_auth
    try:
        yield fake
    finally:
        server._get, server._ensure_auth = real


def ctx():
    return SimpleNamespace(request_context=SimpleNamespace(lifespan_context={"client": None}))


def run(coro):
    return asyncio.run(coro)


def main(globals_):
    failures = 0
    for name, fn in sorted(globals_.items()):
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
