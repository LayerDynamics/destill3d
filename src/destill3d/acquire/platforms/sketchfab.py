"""
Sketchfab platform adapter.

Implements the PlatformAdapter protocol for Sketchfab API v3.
Supports OAuth2 authentication. GLTF is the primary download format.
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


class SketchfabAdapter:
    """Sketchfab platform adapter."""

    BASE_URL = "https://api.sketchfab.com/v3"

    def __init__(self, credentials: CredentialManager):
        self._credentials = credentials
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def platform_id(self) -> str:
        return "sketchfab"

    @property
    def rate_limit(self) -> RateLimit:
        return RateLimit(requests=1000, period_seconds=86400)  # 1000/day

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            api_key = self._credentials.retrieve(self.platform_id)
            if not api_key:
                raise AuthenticationError(
                    self.platform_id,
                    "Sketchfab API key not configured. "
                    "Set SKETCHFAB_API_KEY env var or use credentials manager.",
                )

            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={"Authorization": f"Token {api_key}"},
                timeout=30.0,
            )
        return self._client

    async def search(
        self,
        query: str,
        filters: SearchFilters,
        page: int = 1,
    ) -> SearchResults:
        """Search Sketchfab for downloadable models."""
        client = await self._get_client()

        params = {
            "q": query,
            "type": "models",
            "downloadable": "true",
            "count": 24,
            "offset": (page - 1) * 24,
            "sort_by": "-relevance",
        }

        if filters.category:
            params["categories"] = filters.category

        if filters.license:
            params["license"] = ",".join(filters.license)

        response = await client.get("/search", params=params)
        response.raise_for_status()
        data = response.json()

        results = [
            SearchResult(
                platform=self.platform_id,
                model_id=item["uid"],
                title=item["name"],
                author=item.get("user", {}).get("displayName", "Unknown"),
                url=item["viewerUrl"],
                thumbnail_url=self._get_thumbnail(item),
                download_count=item.get("downloadCount"),
                license=item.get("license", {}).get("slug") if isinstance(item.get("license"), dict) else item.get("license"),
                created_at=self._parse_date(item.get("publishedAt")),
            )
            for item in data.get("results", [])
        ]

        total = data.get("cursors", {}).get("totalCount", len(results))

        return SearchResults(
            results=results,
            total_count=total,
            page=page,
            has_more=page * 24 < total,
        )

    async def get_metadata(self, model_id: str) -> ModelMetadata:
        """Get detailed metadata for a Sketchfab model."""
        client = await self._get_client()

        response = await client.get(f"/models/{model_id}")
        response.raise_for_status()
        model = response.json()

        # Build file info from available archives
        files = []
        archives = model.get("archives", {})
        if archives:
            for fmt, archive in archives.items():
                if archive and isinstance(archive, dict):
                    files.append(
                        FileInfo(
                            filename=f"{model_id}.{fmt}",
                            url=archive.get("url", ""),
                            size_bytes=archive.get("size"),
                            format=fmt,
                        )
                    )

        # Get download URL for GLTF
        try:
            dl_response = await client.get(f"/models/{model_id}/download")
            if dl_response.status_code == 200:
                dl_data = dl_response.json()
                if "gltf" in dl_data:
                    files.append(
                        FileInfo(
                            filename=f"{model_id}.gltf.zip",
                            url=dl_data["gltf"]["url"],
                            size_bytes=dl_data["gltf"].get("size"),
                            format="gltf",
                        )
                    )
        except Exception:
            pass  # Download info may not be available for all models

        license_info = model.get("license", {})
        license_slug = license_info.get("slug", "Unknown") if isinstance(license_info, dict) else str(license_info)

        return ModelMetadata(
            platform=self.platform_id,
            model_id=model_id,
            title=model["name"],
            description=model.get("description", ""),
            author=model.get("user", {}).get("displayName", "Unknown"),
            license=license_slug,
            tags=[t.get("name", t) if isinstance(t, dict) else t for t in model.get("tags", [])],
            files=files,
            created_at=self._parse_date(model.get("publishedAt")),
            modified_at=self._parse_date(model.get("updatedAt")),
            download_count=model.get("downloadCount"),
            like_count=model.get("likeCount"),
        )

    async def download(
        self,
        model_id: str,
        target_dir: Path,
    ) -> DownloadResult:
        """Download model files (GLTF preferred)."""
        start = time.monotonic()

        metadata = await self.get_metadata(model_id)
        client = await self._get_client()

        downloaded_files: List[Path] = []
        target_dir.mkdir(parents=True, exist_ok=True)

        # Try to get download URL
        dl_response = await client.get(f"/models/{model_id}/download")
        dl_response.raise_for_status()
        dl_data = dl_response.json()

        # Prefer GLTF format
        for fmt in ["gltf", "source"]:
            if fmt in dl_data:
                url = dl_data[fmt]["url"]
                filename = f"{model_id}.{fmt}.zip"
                target_path = target_dir / filename

                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with open(target_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)

                downloaded_files.append(target_path)
                break

        elapsed = (time.monotonic() - start) * 1000

        return DownloadResult(
            platform=self.platform_id,
            model_id=model_id,
            files=downloaded_files,
            metadata=metadata,
            download_time_ms=elapsed,
        )

    def parse_url(self, url: str) -> Optional[str]:
        """Extract model UID from Sketchfab URL."""
        patterns = [
            r"sketchfab\.com/3d-models/[^/]+-([a-f0-9]{32})",
            r"sketchfab\.com/models/([a-f0-9]{32})",
        ]

        for pattern in patterns:
            if match := re.search(pattern, url):
                return match.group(1)

        return None

    def _get_thumbnail(self, item: dict) -> Optional[str]:
        """Extract thumbnail URL from Sketchfab model data."""
        thumbnails = item.get("thumbnails", {})
        if isinstance(thumbnails, dict):
            images = thumbnails.get("images", [])
            if images:
                # Get the largest thumbnail
                return max(images, key=lambda i: i.get("width", 0)).get("url")
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
