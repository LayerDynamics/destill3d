"""
Thangs platform adapter for Destill3D.

Implements REST API integration for Thangs 3D model search.
Rate limited to 500 requests per day.
"""

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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


class ThangsAdapter:
    """Adapter for Thangs platform (REST API)."""

    API_URL = "https://api.thangs.com/v1"
    BASE_URL = "https://thangs.com"
    URL_PATTERN = re.compile(r"thangs\.com/(?:designer/[^/]+/)?3d-model/([^/?#]+)")

    def __init__(self, credentials: Optional[CredentialManager] = None):
        self._credentials = credentials or CredentialManager()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def platform_id(self) -> str:
        return "thangs"

    @property
    def rate_limit(self) -> RateLimit:
        return RateLimit(requests=500, period_seconds=86400)

    def parse_url(self, url: str) -> Optional[str]:
        """Extract model slug from Thangs URL."""
        match = self.URL_PATTERN.search(url)
        return match.group(1) if match else None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client with authentication."""
        if self._client is None:
            api_key = self._credentials.retrieve(self.platform_id)
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.API_URL,
                headers=headers,
                follow_redirects=True,
                timeout=30.0,
            )
        return self._client

    async def search(
        self,
        query: str,
        filters: SearchFilters,
        page: int = 1,
    ) -> SearchResults:
        """Search Thangs for 3D models."""
        client = await self._get_client()

        params = {
            "q": query,
            "page": page,
            "per_page": 20,
            "scope": "thangs",
        }

        if filters.category:
            params["category"] = filters.category

        response = await client.get("/search", params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                platform=self.platform_id,
                model_id=str(item.get("id", "")),
                title=item.get("name", ""),
                author=item.get("owner", {}).get("username", "Unknown"),
                url=f"{self.BASE_URL}/3d-model/{item.get('slug', item.get('id', ''))}",
                thumbnail_url=item.get("thumbnailUrl"),
                download_count=item.get("downloadCount"),
                license=item.get("license"),
                created_at=self._parse_date(item.get("createdAt")),
            ))

        total = data.get("totalCount", len(results))

        return SearchResults(
            results=results,
            total_count=total,
            page=page,
            has_more=page * 20 < total,
        )

    async def get_metadata(self, model_id: str) -> ModelMetadata:
        """Fetch detailed metadata for a specific model."""
        client = await self._get_client()

        response = await client.get(f"/models/{model_id}")
        response.raise_for_status()
        data = response.json()

        files = []
        for f in data.get("files", []):
            files.append(FileInfo(
                filename=f.get("filename", f.get("name", "")),
                url=f.get("downloadUrl", f.get("url", "")),
                size_bytes=f.get("size"),
                format=Path(f.get("filename", "")).suffix.lower() or None,
            ))

        return ModelMetadata(
            platform=self.platform_id,
            model_id=model_id,
            title=data.get("name", model_id),
            description=data.get("description", ""),
            author=data.get("owner", {}).get("username", "Unknown"),
            license=data.get("license", "Unknown"),
            tags=data.get("tags", []),
            files=files,
            created_at=self._parse_date(data.get("createdAt")) or datetime.utcnow(),
            modified_at=self._parse_date(data.get("updatedAt")),
            download_count=data.get("downloadCount"),
            like_count=data.get("likeCount"),
        )

    async def download(
        self,
        model_id: str,
        target_dir: Path,
    ) -> DownloadResult:
        """Download model files to target directory."""
        start = time.monotonic()

        metadata = await self.get_metadata(model_id)
        client = await self._get_client()

        downloaded_files: List[Path] = []
        target_dir.mkdir(parents=True, exist_ok=True)

        for file_info in metadata.files:
            try:
                async with client.stream("GET", file_info.url) as response:
                    response.raise_for_status()
                    target_path = target_dir / file_info.filename
                    with open(target_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                    downloaded_files.append(target_path)
            except httpx.HTTPError as e:
                logger.warning(f"Failed to download {file_info.filename}: {e}")

        elapsed = (time.monotonic() - start) * 1000

        return DownloadResult(
            platform=self.platform_id,
            model_id=model_id,
            files=downloaded_files,
            metadata=metadata,
            download_time_ms=elapsed,
        )

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO date string."""
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
