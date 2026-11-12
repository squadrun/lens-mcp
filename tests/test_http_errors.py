"""Permanent endpoint refusals retain their remedy; transient failures still retry.

Run: uv run python tests/test_http_errors.py
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

os.environ.setdefault("LENS_BASE_URL", "http://localhost")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lens_mcp.server as server


class HttpErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_trace_scan_refusal_preserves_remedy_without_retrying(self):
        detail = "Query exceeded a resource limit. Search a shorter time window."
        for field in ("detail", "error"):
            with self.subTest(field=field):
                requests = []

                def respond(request, requests=requests, field=field):
                    requests.append(request)
                    return httpx.Response(413, json={field: detail})

                async with httpx.AsyncClient(
                    base_url="http://localhost", transport=httpx.MockTransport(respond)
                ) as client:
                    ctx = SimpleNamespace(
                        request_context=SimpleNamespace(
                            lifespan_context={"client": client, "authenticated": True}
                        )
                    )
                    with (
                        patch.object(server.asyncio, "sleep", new_callable=AsyncMock) as sleep,
                        self.assertRaises(httpx.HTTPStatusError) as error,
                    ):
                        await server.search_trace_logs(
                            ctx, query="blank_turn", campaign_id="149", time_range_minutes=1440
                        )

                self.assertIn(detail, str(error.exception))
                self.assertEqual(error.exception.response.status_code, 413)
                self.assertEqual(len(requests), 1)
                sleep.assert_not_awaited()

    async def test_disabled_search_preserves_the_remedy_without_retrying(self):
        detail = (
            "Disabled — broad span search causes ClickHouse overload. "
            "Use list_calls to find calls, then get_call_spans for per-call details."
        )
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(503, json={"detail": detail})

        async with httpx.AsyncClient(
            base_url="http://localhost", transport=httpx.MockTransport(respond)
        ) as client:
            ctx = SimpleNamespace(
                request_context=SimpleNamespace(
                    lifespan_context={"client": client, "authenticated": True}
                )
            )
            with (
                patch.object(server.asyncio, "sleep", new_callable=AsyncMock) as sleep,
                self.assertRaises(RuntimeError) as error,
            ):
                await server.search_spans(ctx, node="llm", time_range_minutes=5, limit=1)

        self.assertIn(detail, str(error.exception))
        self.assertNotIn("shedding load", str(error.exception))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/api/ext/v1/search")
        sleep.assert_not_awaited()

    async def test_transient_errors_still_retry_and_return_recovered_data(self):
        for response in (
            httpx.Response(503, json={"detail": "Heavy-query capacity exhausted"}),
            httpx.Response(503, json={"detail": {"message": "busy"}}),
            httpx.Response(503, json=["unavailable"]),
            httpx.Response(503, text="Service unavailable"),
            httpx.Response(429, json={"detail": "Rate limited"}),
        ):
            with self.subTest(status=response.status_code, body=response.text):
                requests = []

                def respond(request, requests=requests, response=response):
                    requests.append(request)
                    return (
                        response if len(requests) == 1 else httpx.Response(200, json={"calls": []})
                    )

                async with httpx.AsyncClient(
                    base_url="http://localhost", transport=httpx.MockTransport(respond)
                ) as client:
                    ctx = SimpleNamespace(
                        request_context=SimpleNamespace(
                            lifespan_context={"client": client, "authenticated": True}
                        )
                    )
                    with patch.object(server.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                        result = await server._get(ctx, "/search")

                self.assertEqual(result, {"calls": []})
                self.assertEqual(len(requests), 2)
                sleep.assert_awaited_once_with(0.5)


if __name__ == "__main__":
    unittest.main()
