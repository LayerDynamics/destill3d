"""
Additional data models for the acquire module.

Provides platform registry and URL resolution utilities.
"""

import logging
from typing import Dict, List, Optional

from destill3d.acquire.base import (
    DownloadResult,
    FileInfo,
    ModelMetadata,
    PlatformAdapter,
    RateLimit,
    SearchFilters,
    SearchResult,
    SearchResults,
)

logger = logging.getLogger(__name__)


class PlatformRegistry:
    """Registry for managing platform adapters."""

    def __init__(self, auto_register: bool = True):
        """
        Initialize the platform registry.

        Args:
            auto_register: If True, automatically register default adapters.
        """
        self._adapters: Dict[str, PlatformAdapter] = {}

        if auto_register:
            self._register_default_adapters()

    def register(self, adapter: PlatformAdapter) -> None:
        """Register a platform adapter."""
        self._adapters[adapter.platform_id] = adapter
        logger.info(f"Registered platform adapter: {adapter.platform_id}")

    def get(self, platform_id: str) -> Optional[PlatformAdapter]:
        """Get adapter by platform ID."""
        return self._adapters.get(platform_id)

    def list_platforms(self) -> List[str]:
        """List all registered platform IDs."""
        return list(self._adapters.keys())

    def resolve_url(self, url: str) -> Optional[tuple]:
        """
        Resolve a URL to (platform_id, model_id).

        Tries each registered adapter's parse_url method.

        Returns:
            Tuple of (platform_id, model_id) or None if no adapter matches.
        """
        for platform_id, adapter in self._adapters.items():
            model_id = adapter.parse_url(url)
            if model_id is not None:
                return (platform_id, model_id)
        return None

    def __contains__(self, platform_id: str) -> bool:
        return platform_id in self._adapters

    def __len__(self) -> int:
        return len(self._adapters)

    def _register_default_adapters(self) -> None:
        """Register the default platform adapters."""
        from destill3d.acquire.credentials import CredentialManager
        from destill3d.acquire.platforms.cults3d import Cults3DAdapter
        from destill3d.acquire.platforms.github import GitHubAdapter
        from destill3d.acquire.platforms.grabcad import GrabCADAdapter
        from destill3d.acquire.platforms.huggingface import HuggingFaceAdapter
        from destill3d.acquire.platforms.local import LocalFilesystemAdapter
        from destill3d.acquire.platforms.sketchfab import SketchfabAdapter
        from destill3d.acquire.platforms.thangs import ThangsAdapter
        from destill3d.acquire.platforms.thingiverse import ThingiverseAdapter

        creds = CredentialManager()

        # Register adapters - local doesn't need credentials
        self.register(LocalFilesystemAdapter())
        self.register(ThingiverseAdapter(creds))
        self.register(SketchfabAdapter(creds))
        self.register(GrabCADAdapter(credentials=creds))
        self.register(Cults3DAdapter(credentials=creds))
        self.register(ThangsAdapter(credentials=creds))
        self.register(GitHubAdapter(credentials=creds))
        self.register(HuggingFaceAdapter(credentials=creds))
