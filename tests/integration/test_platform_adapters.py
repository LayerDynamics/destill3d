"""
Integration tests for platform adapters (with mocked APIs).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from destill3d.acquire.base import SearchFilters, SearchResults
from destill3d.acquire.credentials import CredentialManager
from destill3d.acquire.platforms.thingiverse import ThingiverseAdapter
from destill3d.acquire.platforms.sketchfab import SketchfabAdapter
from destill3d.acquire.platforms.local import LocalFilesystemAdapter


def _mock_credentials(platform: str = "thingiverse", api_key: str = "test-key") -> CredentialManager:
    """Create a mock CredentialManager that returns a test key."""
    creds = MagicMock(spec=CredentialManager)
    creds.retrieve.return_value = api_key
    return creds


class TestThingiverseAdapter:
    """Test Thingiverse adapter with mocked HTTP."""

    def test_platform_name(self):
        adapter = ThingiverseAdapter(credentials=_mock_credentials("thingiverse"))
        assert adapter.platform_id == "thingiverse"

    def test_rate_limit(self):
        adapter = ThingiverseAdapter(credentials=_mock_credentials("thingiverse"))
        rl = adapter.rate_limit
        assert rl.requests == 300
        assert rl.period_seconds == 300

    def test_parse_url_valid(self):
        adapter = ThingiverseAdapter(credentials=_mock_credentials("thingiverse"))
        result = adapter.parse_url("https://www.thingiverse.com/thing:12345")
        assert result is not None

    def test_parse_url_invalid(self):
        adapter = ThingiverseAdapter(credentials=_mock_credentials("thingiverse"))
        result = adapter.parse_url("https://example.com/not-a-thing")
        assert result is None


class TestSketchfabAdapter:
    """Test Sketchfab adapter."""

    def test_platform_name(self):
        adapter = SketchfabAdapter(credentials=_mock_credentials("sketchfab"))
        assert adapter.platform_id == "sketchfab"

    def test_rate_limit(self):
        adapter = SketchfabAdapter(credentials=_mock_credentials("sketchfab"))
        rl = adapter.rate_limit
        assert rl.requests == 1000
        assert rl.period_seconds == 86400


class TestLocalFilesystemAdapter:
    """Test local filesystem adapter."""

    def test_platform_name(self):
        adapter = LocalFilesystemAdapter()
        assert adapter.platform_id == "local"

    @pytest.mark.asyncio
    async def test_search_local_directory(self, temp_dir, sample_stl_file):
        adapter = LocalFilesystemAdapter()
        filters = SearchFilters()
        results = await adapter.search(str(sample_stl_file.parent), filters=filters)
        assert isinstance(results, SearchResults)
