"""Integration tests for GitHub platform adapter."""

import pytest
from unittest.mock import MagicMock

from destill3d.acquire.platforms.github import GitHubAdapter


def _mock_credentials(platform: str):
    """Create a mock CredentialManager."""
    creds = MagicMock()
    creds.retrieve.return_value = "ghp_test_token_123"
    return creds


class TestGitHubAdapter:
    """Test GitHub platform adapter."""

    def test_platform_id(self):
        """Test platform identifier."""
        adapter = GitHubAdapter(credentials=_mock_credentials("github"))
        assert adapter.platform_id == "github"

    def test_rate_limit(self):
        """Test rate limit configuration."""
        adapter = GitHubAdapter(credentials=_mock_credentials("github"))
        assert adapter.rate_limit.requests == 5000
        assert adapter.rate_limit.period_seconds == 3600

    def test_parse_url_valid(self):
        """Test URL parsing for valid GitHub repo URLs."""
        adapter = GitHubAdapter(credentials=_mock_credentials("github"))
        result = adapter.parse_url("https://github.com/openscad/openscad")
        assert result == "openscad/openscad"

    def test_parse_url_valid_with_path(self):
        """Test URL parsing for GitHub URLs with additional path segments."""
        adapter = GitHubAdapter(credentials=_mock_credentials("github"))
        result = adapter.parse_url("https://github.com/user/repo/tree/main/models")
        assert result == "user/repo"

    def test_parse_url_invalid(self):
        """Test URL parsing for non-GitHub URLs."""
        adapter = GitHubAdapter(credentials=_mock_credentials("github"))
        result = adapter.parse_url("https://example.com/repo")
        assert result is None

    def test_parse_url_invalid_github_path(self):
        """Test URL parsing for GitHub non-repo URLs."""
        adapter = GitHubAdapter(credentials=_mock_credentials("github"))
        result = adapter.parse_url("https://github.com/settings/notifications")
        assert result is None

    def test_parse_url_marketplace(self):
        """Test URL parsing filters out marketplace."""
        adapter = GitHubAdapter(credentials=_mock_credentials("github"))
        result = adapter.parse_url("https://github.com/user/marketplace")
        assert result is None
