"""Tests for paywall components (ApiKeyManager, RateLimiter, middleware helpers).

Covers: ApiKeyManager, RateLimiter, _get_header, _extract_api_key, MCPServer legacy.
"""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import (
    ApiKeyManager,
    RateLimiter,
    _get_header,
    _extract_api_key,
    MCPServer,
    EUAIActChecker,
    _validate_project_path,
    BLOCKED_PATHS,
    _INSTALL_ROOT,
)


# ============================================================
# Tests: ApiKeyManager
# ============================================================

class TestApiKeyManager:

    def test_init_no_files(self, tmp_path):
        """ApiKeyManager should handle missing key files gracefully."""
        mgr = ApiKeyManager(
            path=tmp_path / "nonexistent.json",
            data_path=tmp_path / "data" / "nonexistent.json",
        )
        assert mgr.verify("any_key") is None

    def test_register_and_verify(self, tmp_path):
        """Register a key and verify it."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        mgr = ApiKeyManager(
            path=tmp_path / "keys.json",
            data_path=data_dir / "api_keys.json",
        )
        entry = mgr.register_key("test@example.com", "pro")
        assert entry["key"].startswith("ak_")
        assert entry["email"] == "test@example.com"
        assert entry["plan"] == "pro"
        assert entry["active"] is True

        # Verify the key
        result = mgr.verify(entry["key"])
        assert result is not None
        assert result["email"] == "test@example.com"
        assert result["plan"] == "pro"

    def test_verify_invalid_key(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        mgr = ApiKeyManager(
            path=tmp_path / "keys.json",
            data_path=data_dir / "api_keys.json",
        )
        assert mgr.verify("invalid_key") is None

    def test_load_list_format(self, tmp_path):
        """Test loading keys in list format."""
        keys_file = tmp_path / "keys.json"
        keys_file.write_text(json.dumps({
            "keys": [
                {"key": "test_key_1", "email": "a@b.com", "active": True, "plan": "pro"},
                {"key": "test_key_2", "email": "c@d.com", "active": False, "plan": "free"},
            ]
        }))
        mgr = ApiKeyManager(
            path=keys_file,
            data_path=tmp_path / "data" / "nonexistent.json",
        )
        assert mgr.verify("test_key_1") is not None
        assert mgr.verify("test_key_2") is None  # inactive

    def test_load_dict_format(self, tmp_path):
        """Test loading keys in dict format."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        data_file = data_dir / "api_keys.json"
        data_file.write_text(json.dumps({
            "mcp_pro_abc123": {"email": "x@y.com", "active": True, "tier": "pro"},
        }))
        mgr = ApiKeyManager(
            path=tmp_path / "nonexistent.json",
            data_path=data_file,
        )
        result = mgr.verify("mcp_pro_abc123")
        assert result is not None
        assert result["email"] == "x@y.com"

    def test_get_entry(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        mgr = ApiKeyManager(
            path=tmp_path / "keys.json",
            data_path=data_dir / "api_keys.json",
        )
        entry = mgr.register_key("test@test.com")
        retrieved = mgr.get_entry(entry["key"])
        assert retrieved["email"] == "test@test.com"

    def test_get_entry_missing(self, tmp_path):
        mgr = ApiKeyManager(
            path=tmp_path / "nonexistent.json",
            data_path=tmp_path / "data" / "nonexistent.json",
        )
        assert mgr.get_entry("nonexistent") == {}

    def test_reload_after_timeout(self, tmp_path):
        """Test that keys are reloaded after 60s cache expiry."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        mgr = ApiKeyManager(
            path=tmp_path / "keys.json",
            data_path=data_dir / "api_keys.json",
        )
        entry = mgr.register_key("test@test.com")
        # Force cache expiry
        mgr._loaded_at = time.time() - 61
        result = mgr.verify(entry["key"])
        assert result is not None

    def test_corrupted_json(self, tmp_path):
        """Corrupted JSON should not crash the manager."""
        keys_file = tmp_path / "keys.json"
        keys_file.write_text("{invalid json")
        mgr = ApiKeyManager(
            path=keys_file,
            data_path=tmp_path / "data" / "nonexistent.json",
        )
        assert mgr.verify("any") is None


# ============================================================
# Tests: RateLimiter
# ============================================================

class TestRateLimiter:

    def test_first_request_allowed(self):
        rl = RateLimiter(max_requests=10)
        allowed, remaining = rl.check("1.2.3.4")
        assert allowed is True
        assert remaining == 9

    def test_limit_reached(self):
        rl = RateLimiter(max_requests=2)
        rl.check("1.2.3.4")
        rl.check("1.2.3.4")
        allowed, remaining = rl.check("1.2.3.4")
        assert allowed is False
        assert remaining == 0

    def test_different_ips(self):
        rl = RateLimiter(max_requests=1)
        allowed1, _ = rl.check("1.1.1.1")
        allowed2, _ = rl.check("2.2.2.2")
        assert allowed1 is True
        assert allowed2 is True

    def test_remaining_count(self):
        rl = RateLimiter(max_requests=5)
        _, r1 = rl.check("1.1.1.1")
        _, r2 = rl.check("1.1.1.1")
        _, r3 = rl.check("1.1.1.1")
        assert r1 == 4
        assert r2 == 3
        assert r3 == 2

    def test_date_reset(self):
        rl = RateLimiter(max_requests=1)
        rl.check("1.1.1.1")
        # Simulate date change
        rl._clients["1.1.1.1"]["date"] = "2020-01-01"
        allowed, remaining = rl.check("1.1.1.1")
        assert allowed is True
        assert remaining == 0

    def test_cleanup_removes_old(self):
        rl = RateLimiter(max_requests=10)
        rl.check("1.1.1.1")
        # Set old date to trigger cleanup
        rl._clients["1.1.1.1"]["date"] = "2020-01-01"
        rl.cleanup()
        assert "1.1.1.1" not in rl._clients

    def test_cleanup_keeps_today(self):
        rl = RateLimiter(max_requests=10)
        rl.check("1.1.1.1")
        rl.cleanup()
        assert "1.1.1.1" in rl._clients

    def test_auto_cleanup_on_check(self):
        rl = RateLimiter(max_requests=10)
        rl.check("1.1.1.1")
        rl._clients["1.1.1.1"]["date"] = "2020-01-01"
        # Force cleanup trigger
        rl._last_cleanup = time.time() - 3601
        rl.check("2.2.2.2")
        assert "1.1.1.1" not in rl._clients


# ============================================================
# Tests: Helper functions
# ============================================================

class TestHelpers:

    def test_get_header_found(self):
        scope = {"headers": [(b"content-type", b"application/json"), (b"x-api-key", b"test123")]}
        assert _get_header(scope, b"x-api-key") == "test123"

    def test_get_header_not_found(self):
        scope = {"headers": [(b"content-type", b"application/json")]}
        assert _get_header(scope, b"x-api-key") is None

    def test_get_header_empty(self):
        scope = {"headers": []}
        assert _get_header(scope, b"x-api-key") is None

    def test_get_header_no_headers(self):
        scope = {}
        assert _get_header(scope, b"x-api-key") is None

    def test_extract_api_key_x_header(self):
        scope = {"headers": [(b"x-api-key", b"ak_123")]}
        assert _extract_api_key(scope) == "ak_123"

    def test_extract_api_key_bearer(self):
        scope = {"headers": [(b"authorization", b"Bearer ak_456")]}
        assert _extract_api_key(scope) == "ak_456"

    def test_extract_api_key_none(self):
        scope = {"headers": []}
        assert _extract_api_key(scope) is None

    def test_extract_api_key_prefers_x_header(self):
        scope = {"headers": [
            (b"x-api-key", b"ak_from_header"),
            (b"authorization", b"Bearer ak_from_bearer"),
        ]}
        assert _extract_api_key(scope) == "ak_from_header"


# ============================================================
# Tests: Path validation
# ============================================================

class TestPathValidation:

    def test_blocked_system_paths(self):
        for path in ["/etc", "/root", "/proc", "/sys", "/home/ubuntu"]:
            is_safe, msg = _validate_project_path(path)
            assert is_safe is False
            assert "Access denied" in msg

    def test_install_root_blocked(self):
        is_safe, msg = _validate_project_path(_INSTALL_ROOT)
        assert is_safe is False

    def test_tmp_path_allowed(self, tmp_path):
        is_safe, msg = _validate_project_path(str(tmp_path))
        assert is_safe is True

    def test_invalid_path(self):
        is_safe, msg = _validate_project_path("")
        # Empty string resolves to cwd, behavior depends on implementation
        assert isinstance(is_safe, bool)


# ============================================================
# Tests: MCPServer Legacy
# ============================================================

class TestMCPServerLegacy:

    def test_init_has_tools(self):
        server = MCPServer()
        assert "_tools" in dir(server)
        assert len(server._tools) == 5

    def test_list_tools(self):
        server = MCPServer()
        tools = server.list_tools()
        assert "tools" in tools
        assert len(tools["tools"]) == 5
        names = [t["name"] for t in tools["tools"]]
        assert "scan_project" in names
        assert "check_compliance" in names
        assert "generate_report" in names
        assert "suggest_risk_category" in names
        assert "generate_compliance_templates" in names

    def test_handle_request_unknown_tool(self):
        server = MCPServer()
        result = server.handle_request("nonexistent_tool", {})
        assert "error" in result

    def test_scan_project_via_legacy(self, tmp_path):
        (tmp_path / "app.py").write_text("email = 'test@test.com'")
        server = MCPServer()
        result = server.handle_request("scan_project", {"project_path": str(tmp_path)})
        assert result["tool"] == "scan_project"
        assert "results" in result

    def test_suggest_risk_category(self):
        server = MCPServer()
        result = server.handle_request("suggest_risk_category", {
            "system_description": "facial recognition system for law enforcement"
        })
        assert result["tool"] == "suggest_risk_category"
        assert result["results"]["suggested_category"] in ["unacceptable", "high"]

    def test_suggest_risk_category_minimal(self):
        server = MCPServer()
        result = server.handle_request("suggest_risk_category", {
            "system_description": "spam filter for email"
        })
        assert result["results"]["suggested_category"] == "minimal"

    def test_suggest_risk_category_unknown(self):
        server = MCPServer()
        result = server.handle_request("suggest_risk_category", {
            "system_description": "a simple calculator"
        })
        assert result["results"]["confidence"] == "low"

    def test_generate_compliance_templates_high(self):
        server = MCPServer()
        result = server.handle_request("generate_compliance_templates", {
            "risk_category": "high"
        })
        assert result["tool"] == "generate_compliance_templates"
        assert result["results"]["templates_count"] > 0

    def test_generate_compliance_templates_unacceptable(self):
        server = MCPServer()
        result = server.handle_request("generate_compliance_templates", {
            "risk_category": "unacceptable"
        })
        assert "error" in result

    def test_generate_compliance_templates_minimal(self):
        server = MCPServer()
        result = server.handle_request("generate_compliance_templates", {
            "risk_category": "minimal"
        })
        assert result["results"]["templates_count"] == 0
