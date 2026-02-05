"""Unit tests for CredentialManager."""

import os
from unittest.mock import MagicMock, patch

import pytest

from destill3d.acquire.credentials import CredentialManager


class TestCredentialManager:
    @patch("destill3d.acquire.credentials.keyring")
    def test_store_and_retrieve(self, mock_keyring):
        store = {}

        def set_pw(service, user, pw):
            store[(service, user)] = pw

        def get_pw(service, user):
            return store.get((service, user))

        mock_keyring.set_password = set_pw
        mock_keyring.get_password = get_pw

        cm = CredentialManager()
        cm.store("thingiverse", "my-api-key")
        result = cm.retrieve("thingiverse")
        assert result == "my-api-key"

    @patch("destill3d.acquire.credentials.keyring")
    def test_delete(self, mock_keyring):
        mock_keyring.delete_password = MagicMock()
        cm = CredentialManager()
        cm.delete("thingiverse")
        mock_keyring.delete_password.assert_called_once()

    @patch("destill3d.acquire.credentials.keyring")
    def test_retrieve_not_found(self, mock_keyring):
        mock_keyring.get_password = MagicMock(return_value=None)
        cm = CredentialManager()
        # Also no env var
        with patch.dict(os.environ, {}, clear=False):
            result = cm.retrieve("unknown_platform")
        assert result is None

    @patch("destill3d.acquire.credentials.keyring")
    def test_env_var_fallback(self, mock_keyring):
        mock_keyring.get_password = MagicMock(return_value=None)
        cm = CredentialManager()
        with patch.dict(os.environ, {"THINGIVERSE_API_KEY": "env-key"}):
            result = cm.retrieve("thingiverse")
        assert result == "env-key"
