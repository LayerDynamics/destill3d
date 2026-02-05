"""Integration tests for Thangs platform adapter."""

import pytest
from unittest.mock import MagicMock

from destill3d.acquire.platforms.thangs import ThangsAdapter


def _mock_credentials(platform: str):
    """Create a mock CredentialManager."""
    creds = MagicMock()
    creds.retrieve.return_value = "test-api-key"
    return creds


class TestThangsAdapter:
    """Test Thangs platform adapter."""

    def test_platform_id(self):
        """Test platform identifier."""
        adapter = ThangsAdapter(credentials=_mock_credentials("thangs"))
        assert adapter.platform_id == "thangs"

    def test_rate_limit(self):
        """Test rate limit configuration."""
        adapter = ThangsAdapter(credentials=_mock_credentials("thangs"))
        assert adapter.rate_limit.requests == 500
        assert adapter.rate_limit.period_seconds == 86400

    def test_parse_url_valid(self):
        """Test URL parsing for valid Thangs model URLs."""
        adapter = ThangsAdapter(credentials=_mock_credentials("thangs"))
        result = adapter.parse_url("https://thangs.com/3d-model/cool-robot-arm-12345")
        assert result == "cool-robot-arm-12345"

    def test_parse_url_valid_with_designer(self):
        """Test URL parsing for Thangs designer URLs."""
        adapter = ThangsAdapter(credentials=_mock_credentials("thangs"))
        result = adapter.parse_url("https://thangs.com/designer/somebody/3d-model/my-model-789")
        assert result == "my-model-789"

    def test_parse_url_invalid(self):
        """Test URL parsing for non-Thangs URLs."""
        adapter = ThangsAdapter(credentials=_mock_credentials("thangs"))
        result = adapter.parse_url("https://example.com/model")
        assert result is None

    def test_parse_url_invalid_thangs_path(self):
        """Test URL parsing for Thangs non-model URLs."""
        adapter = ThangsAdapter(credentials=_mock_credentials("thangs"))
        result = adapter.parse_url("https://thangs.com/designer/someone")
        assert result is None
