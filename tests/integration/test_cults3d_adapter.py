"""Integration tests for Cults3D platform adapter."""

import pytest
from unittest.mock import MagicMock

from destill3d.acquire.platforms.cults3d import Cults3DAdapter


def _mock_credentials(platform: str):
    """Create a mock CredentialManager."""
    creds = MagicMock()
    creds.retrieve.return_value = "test-session"
    return creds


class TestCults3DAdapter:
    """Test Cults3D platform adapter."""

    def test_platform_id(self):
        """Test platform identifier."""
        adapter = Cults3DAdapter(credentials=_mock_credentials("cults3d"))
        assert adapter.platform_id == "cults3d"

    def test_rate_limit(self):
        """Test rate limit configuration."""
        adapter = Cults3DAdapter(credentials=_mock_credentials("cults3d"))
        assert adapter.rate_limit.requests == 50
        assert adapter.rate_limit.period_seconds == 3600

    def test_parse_url_valid(self):
        """Test URL parsing for valid Cults3D URLs."""
        adapter = Cults3DAdapter(credentials=_mock_credentials("cults3d"))
        result = adapter.parse_url("https://cults3d.com/en/3d-model/gadget/sample-model")
        assert result == "gadget/sample-model"

    def test_parse_url_valid_french(self):
        """Test URL parsing with French language prefix."""
        adapter = Cults3DAdapter(credentials=_mock_credentials("cults3d"))
        result = adapter.parse_url("https://cults3d.com/fr/3d-model/art/cool-sculpture")
        assert result == "art/cool-sculpture"

    def test_parse_url_invalid(self):
        """Test URL parsing for non-Cults3D URLs."""
        adapter = Cults3DAdapter(credentials=_mock_credentials("cults3d"))
        result = adapter.parse_url("https://example.com/model")
        assert result is None

    def test_parse_url_invalid_cults3d_path(self):
        """Test URL parsing for Cults3D non-model URLs."""
        adapter = Cults3DAdapter(credentials=_mock_credentials("cults3d"))
        result = adapter.parse_url("https://cults3d.com/en/users/someone")
        assert result is None
