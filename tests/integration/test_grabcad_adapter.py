"""Integration tests for GrabCAD platform adapter."""

import pytest
from unittest.mock import MagicMock

from destill3d.acquire.platforms.grabcad import GrabCADAdapter


def _mock_credentials(platform: str):
    """Create a mock CredentialManager."""
    creds = MagicMock()
    creds.retrieve.return_value = "test-session"
    return creds


class TestGrabCADAdapter:
    """Test GrabCAD platform adapter."""

    def test_platform_id(self):
        """Test platform identifier."""
        adapter = GrabCADAdapter(credentials=_mock_credentials("grabcad"))
        assert adapter.platform_id == "grabcad"

    def test_rate_limit(self):
        """Test rate limit configuration."""
        adapter = GrabCADAdapter(credentials=_mock_credentials("grabcad"))
        assert adapter.rate_limit.requests == 100
        assert adapter.rate_limit.period_seconds == 3600

    def test_parse_url_valid(self):
        """Test URL parsing for valid GrabCAD URLs."""
        adapter = GrabCADAdapter(credentials=_mock_credentials("grabcad"))
        result = adapter.parse_url("https://grabcad.com/library/sample-model-123")
        assert result == "sample-model-123"

    def test_parse_url_valid_with_trailing_slash(self):
        """Test URL parsing with trailing slash."""
        adapter = GrabCADAdapter(credentials=_mock_credentials("grabcad"))
        result = adapter.parse_url("https://grabcad.com/library/sample-model-123/")
        assert result == "sample-model-123"

    def test_parse_url_valid_with_params(self):
        """Test URL parsing with query parameters."""
        adapter = GrabCADAdapter(credentials=_mock_credentials("grabcad"))
        result = adapter.parse_url("https://grabcad.com/library/cool-robot?utm_source=test")
        assert result == "cool-robot"

    def test_parse_url_invalid(self):
        """Test URL parsing for non-GrabCAD URLs."""
        adapter = GrabCADAdapter(credentials=_mock_credentials("grabcad"))
        result = adapter.parse_url("https://example.com/model")
        assert result is None

    def test_parse_url_invalid_grabcad_path(self):
        """Test URL parsing for GrabCAD non-library URLs."""
        adapter = GrabCADAdapter(credentials=_mock_credentials("grabcad"))
        result = adapter.parse_url("https://grabcad.com/community/engineers")
        assert result is None
