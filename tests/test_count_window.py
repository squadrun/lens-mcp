"""count_spans refuses an unscoped window the backend would reject with a bare 400.

Run: uv run python tests/test_count_window.py
"""

import asyncio
import os
import sys

os.environ.setdefault("LENS_BASE_URL", "http://localhost")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lens_mcp.server as server

sent = []


async def fake_get(ctx, path, params=None):
    sent.append(params)
    return {"total_spans": 0, "affected_calls": 0}


server._get = fake_get


def test_unscoped_window_over_a_day_is_refused():
    try:
        asyncio.run(server.count_spans(None, time_range_minutes=1500))
    except ValueError as error:
        assert "one day" in str(error)
    else:
        raise AssertionError("an unscoped 1500-minute count reached the backend")
    assert not sent


def test_a_day_or_a_scoped_window_reaches_the_backend():
    sent.clear()
    asyncio.run(server.count_spans(None, time_range_minutes=1440))
    asyncio.run(server.count_spans(None, campaign_id="1", time_range_minutes=2880))
    assert len(sent) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
