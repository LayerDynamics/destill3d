"""
Secure credential management for platform API keys.

Uses system keyring for secure storage with environment
variable fallback.
"""

import logging
import os
from typing import List, Optional

import keyring

logger = logging.getLogger(__name__)


class CredentialManager:
    """Secure storage and retrieval of API credentials."""

    SERVICE_NAME = "destill3d"

    KNOWN_PLATFORMS = [
        "thingiverse",
        "sketchfab",
        "grabcad",
        "cults3d",
        "myminifactory",
        "thangs",
        "github",
        "huggingface",
    ]

    def __init__(self, use_keyring: bool = True):
        self.use_keyring = use_keyring

    def store(self, platform: str, api_key: str) -> None:
        """Store API key securely in system keyring."""
        if self.use_keyring:
            keyring.set_password(self.SERVICE_NAME, platform, api_key)
            logger.info(f"Stored credential for {platform}")
        else:
            raise ValueError("Keyring disabled; use environment variables")

    def retrieve(self, platform: str) -> Optional[str]:
        """
        Retrieve API key from environment or keyring.

        Priority:
        1. Environment variable: {PLATFORM}_API_KEY
        2. System keyring
        """
        # Try environment variable first
        env_var = f"{platform.upper()}_API_KEY"
        if env_key := os.environ.get(env_var):
            return env_key

        # Fall back to keyring
        if self.use_keyring:
            try:
                return keyring.get_password(self.SERVICE_NAME, platform)
            except Exception:
                return None

        return None

    def delete(self, platform: str) -> None:
        """Remove stored credential."""
        if self.use_keyring:
            try:
                keyring.delete_password(self.SERVICE_NAME, platform)
                logger.info(f"Deleted credential for {platform}")
            except keyring.errors.PasswordDeleteError:
                pass  # Already deleted

    def list_platforms(self) -> List[str]:
        """List platforms with stored credentials."""
        platforms = []
        for platform in self.KNOWN_PLATFORMS:
            if self.retrieve(platform):
                platforms.append(platform)
        return platforms
