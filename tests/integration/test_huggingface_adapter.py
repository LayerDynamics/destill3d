"""Integration tests for HuggingFace platform adapter."""

import pytest
from unittest.mock import MagicMock

from destill3d.acquire.platforms.huggingface import HuggingFaceAdapter


def _mock_credentials(platform: str):
    """Create a mock CredentialManager."""
    creds = MagicMock()
    creds.retrieve.return_value = "hf_test_token"
    return creds


class TestHuggingFaceAdapter:
    """Test HuggingFace platform adapter."""

    def test_platform_id(self):
        """Test platform identifier."""
        adapter = HuggingFaceAdapter(credentials=_mock_credentials("huggingface"))
        assert adapter.platform_id == "huggingface"

    def test_rate_limit(self):
        """Test rate limit configuration."""
        adapter = HuggingFaceAdapter(credentials=_mock_credentials("huggingface"))
        assert adapter.rate_limit.requests == 1000
        assert adapter.rate_limit.period_seconds == 3600

    def test_parse_url_dataset(self):
        """Test URL parsing for HuggingFace dataset URLs."""
        adapter = HuggingFaceAdapter(credentials=_mock_credentials("huggingface"))
        result = adapter.parse_url("https://huggingface.co/datasets/ShapeNet/ShapeNetCore")
        assert result == "datasets/ShapeNet/ShapeNetCore"

    def test_parse_url_model(self):
        """Test URL parsing for HuggingFace model URLs."""
        adapter = HuggingFaceAdapter(credentials=_mock_credentials("huggingface"))
        result = adapter.parse_url("https://huggingface.co/openai/point-e")
        assert result == "openai/point-e"

    def test_parse_url_invalid(self):
        """Test URL parsing for non-HuggingFace URLs."""
        adapter = HuggingFaceAdapter(credentials=_mock_credentials("huggingface"))
        result = adapter.parse_url("https://example.com/models/something")
        assert result is None

    def test_parse_url_invalid_hf_path(self):
        """Test URL parsing for HuggingFace non-repo URLs."""
        adapter = HuggingFaceAdapter(credentials=_mock_credentials("huggingface"))
        result = adapter.parse_url("https://huggingface.co/docs/transformers")
        assert result is None

    def test_parse_url_spaces_ignored(self):
        """Test URL parsing ignores Spaces paths."""
        adapter = HuggingFaceAdapter(credentials=_mock_credentials("huggingface"))
        result = adapter.parse_url("https://huggingface.co/spaces/gradio/demo")
        assert result is None
