"""Integration tests — SLOW, require network access.

Run separately from unit tests:
    python -m pytest tests/test_integration.py -v

These hit the real OpenF1 API and validate the full pipeline.
"""

import asyncio
import pytest

# Mark all tests in this module as slow/integration
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        True,  # Skip by default; set to False for manual runs
        reason="Integration tests are slow and require network access"
    ),
]


@pytest.fixture
def client():
    from pitwall.data.openf1_client import OpenF1Client
    c = OpenF1Client()
    yield c
    asyncio.run(c.close())


class TestOpenF1Integration:
    def test_resolve_bahrain_2024(self, client):
        session = asyncio.run(client.resolve_session("bahrain-2024"))
        assert session is not None
        assert "session_key" in session

    def test_resolve_driver(self, client):
        session = asyncio.run(client.resolve_session("bahrain-2024"))
        assert session is not None
        driver_num = asyncio.run(client.resolve_driver(session["session_key"], "VER"))
        assert driver_num is not None

    def test_get_laps(self, client):
        session = asyncio.run(client.resolve_session("bahrain-2024"))
        assert session is not None
        laps = asyncio.run(client.get_laps(session["session_key"], driver_number=1))
        assert len(laps) > 0


class TestFullPipeline:
    def test_bahrain_2024_ver(self):
        """Full pipeline replay test. Takes ~30-60s with API calls."""
        from pitwall.graph import RaceRunner

        runner = RaceRunner("bahrain-2024", "VER", speed_multiplier=10000)
        asyncio.run(runner.run())
        # If it completes without crashing, that's a pass
