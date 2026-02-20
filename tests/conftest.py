"""Shared fixtures for MCP EU AI Act test suite."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import RateLimiter


@pytest.fixture(autouse=True)
def isolate_rate_limiter_persistence(tmp_path):
    """Ensure each test gets an isolated RateLimiter persistence file.

    Without this, all RateLimiter instances share data/mcp_rate_limits.json,
    causing cross-test pollution when one test's writes affect another test's reads.
    """
    original_path = RateLimiter._PERSIST_PATH
    RateLimiter._PERSIST_PATH = tmp_path / "mcp_rate_limits.json"
    yield
    RateLimiter._PERSIST_PATH = original_path
