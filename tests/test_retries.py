"""Checks for _get's retry loop on transport failures.

A timeout gets one retry, not _MAX_RETRIES, and the final error names its type: str() of an
httpx timeout is "", so the bare exception reached the caller as "Error executing tool x: ".

Run: uv run python tests/test_retries.py
"""

import asyncio
import os
import sys
from types import SimpleNamespace

import httpx

os.environ.setdefault("LENS_BASE_URL", "http://localhost")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lens_mcp import server
from lens_mcp.server import _MAX_TIMEOUT_RETRIES, _get


async def _no_sleep(_seconds):
    return None


def _call(responses):
    """Run _get against a transport that raises or returns each of `responses` in turn."""
    sent = []

    def handler(request):
        sent.append(request)
        outcome = responses[len(sent) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def run():
        async with httpx.AsyncClient(
            base_url="http://localhost", transport=httpx.MockTransport(handler)
        ) as client:
            ctx = SimpleNamespace(
                request_context=SimpleNamespace(
                    lifespan_context={"client": client, "authenticated": True}
                )
            )
            return await _get(ctx, "/calls")

    server.asyncio.sleep, real_sleep = _no_sleep, server.asyncio.sleep
    try:
        return asyncio.run(run()), sent
    except RuntimeError as e:
        return e, sent
    finally:
        server.asyncio.sleep = real_sleep


def test_a_timeout_gets_one_retry_and_a_named_error():
    result, sent = _call([httpx.ReadTimeout("")] * 3)
    assert len(sent) == _MAX_TIMEOUT_RETRIES + 1, len(sent)
    assert isinstance(result, RuntimeError), result
    assert "GET /calls" in str(result) and "ReadTimeout" in str(result), str(result)


def test_a_timeout_then_an_answer_returns_the_answer():
    result, sent = _call([httpx.ReadTimeout(""), httpx.Response(200, json={"rows": []})])
    assert result == {"rows": []}, result
    assert len(sent) == 2, len(sent)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
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
