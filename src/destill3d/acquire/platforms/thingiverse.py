"""
Thingiverse platform adapter.

Implements the PlatformAdapter protocol for Thingiverse API v1.
Requires an API key stored via CredentialManager or
THINGIVERSE_API_KEY environment variable.
"""

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import httpx

from destill3d.acquire.base import (
    DownloadResult,
    FileInfo,
    ModelMetadata,
    RateLimit,
    SearchFilters,
    SearchResult,
    SearchResults,
)
from destill3d.acquire.credentials import CredentialManager
from destill3d.core.exceptions import AcquisitionError, AuthenticationError

logger = logging.getLogger(__name__)


class ThingiverseAdapter:
    """Thingiverse platform adapter."""

    BASE_URL = "https://api.thingiverse.com"

    def __init__(self, credentials: CredentialManager):
        self._credentials = credentials
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def platform_id(self) -> str:
        return "thingiverse"

    @property
    def rate_limit(self) -> RateLimit:
        return RateLimit(requests=300, period_seconds=300)  # 300/5min

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            api_key = self._credentials.retrieve(self.platform_id)
            if not api_key:
                raise AuthenticationError(
                    self.platform_id,
                    "Thingiverse API key not configured. "
                    "Set THINGIVERSE_API_KEY env var or use credentials manager.",
                )

            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )
        return self._client

    async def search(
        self,
        query: str,
        filters: SearchFilters,
        page: int = 1,
    ) -> SearchResults:
        """Search Thingiverse for models."""
        client = await self._get_client()

        params = {
            "term": query,
            "page": page,
            "per_page": 20,
            "sort": "relevant",
        }

        if filters.category:
            params["category"] = filters.category

        response = await client.get("/search/", params=params)
        response.raise_for_status()
        data = response.json()

        results = [
            SearchResult(
                platform=self.platform_id,
                model_id=str(item["id"]),
                title=item["name"],
                author=item.get("creator", {}).get("name", "Unknown"),
                url=item["public_url"],
                thumbnail_url=item.get("thumbnail"),
                download_count=item.get("download_count"),
                license=item.get("license"),
                created_at=self._parse_date(item.get("added")),
            )
            for item in data.get("hits", [])
        ]

        total = data.get("total", len(results))

        return SearchResults(
            results=results,
            total_count=total,
            page=page,
            has_more=page * 20 < total,
        )

    async def get_metadata(self, model_id: str) -> ModelMetadata:
        """Get detailed metadata for a thing."""
        client = await self._get_client()

        # Get thing details
        response = await client.get(f"/things/{model_id}")
        response.raise_for_status()
        thing = response.json()

        # Get files
        files_response = await client.get(f"/things/{model_id}/files")
        files_response.raise_for_status()
        files_data = files_response.json()

        files = [
            FileInfo(
                filename=f["name"],
                url=f["direct_url"],
                size_bytes=f.get("size"),
                format=Path(f["name"]).suffix.lower(),
            )
            for f in files_data
        ]

        return ModelMetadata(
            platform=self.platform_id,
            model_id=model_id,
            title=thing["name"],
            description=thing.get("description", ""),
            author=thing.get("creator", {}).get("name", "Unknown"),
            license=thing.get("license", "Unknown"),
            tags=[t["name"] for t in thing.get("tags", [])],
            files=files,
            created_at=self._parse_date(thing.get("added")),
            modified_at=self._parse_date(thing.get("modified")),
            download_count=thing.get("download_count"),
            like_count=thing.get("like_count"),
        )

    async def download(
        self,
        model_id: str,
        target_dir: Path,
    ) -> DownloadResult:
        """Download all files for a thing."""
        start = time.monotonic()

        metadata = await self.get_metadata(model_id)
        client = await self._get_client()

        downloaded_files: List[Path] = []
        target_dir.mkdir(parents=True, exist_ok=True)

        for file_info in metadata.files:
            target_path = target_dir / file_info.filename

            async with client.stream("GET", file_info.url) as response:
                response.raise_for_status()
                with open(target_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

            downloaded_files.append(target_path)

        elapsed = (time.monotonic() - start) * 1000

        return DownloadResult(
            platform=self.platform_id,
            model_id=model_id,
            files=downloaded_files,
            metadata=metadata,
            download_time_ms=elapsed,
        )

    def parse_url(self, url: str) -> Optional[str]:
        """Extract thing ID from Thingiverse URL."""
        patterns = [
            r"thingiverse\.com/thing:(\d+)",
            r"thingiverse\.com/things/(\d+)",
        ]

        for pattern in patterns:
            if match := re.search(pattern, url):
                return match.group(1)

        return None

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            return None

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
