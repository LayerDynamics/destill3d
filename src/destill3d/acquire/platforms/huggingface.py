"""
HuggingFace Hub platform adapter for Destill3D.

Implements REST API integration for HuggingFace datasets and models
containing 3D files. Rate limited to 1000 requests per hour.
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
from destill3d.core.exceptions import AcquisitionError

logger = logging.getLogger(__name__)

# 3D file extensions to look for in repositories
_3D_EXTENSIONS = {".stl", ".obj", ".ply", ".step", ".stp", ".gltf", ".glb", ".3mf", ".pcd"}


class HuggingFaceAdapter:
    """Adapter for HuggingFace Hub (datasets and models with 3D files)."""

    API_URL = "https://huggingface.co/api"
    BASE_URL = "https://huggingface.co"
    DATASET_URL_PATTERN = re.compile(r"huggingface\.co/(datasets/[^/]+/[^/?#]+)")
    MODEL_URL_PATTERN = re.compile(r"huggingface\.co/([^/]+/[^/?#]+)")

    def __init__(self, credentials: Optional[CredentialManager] = None):
        self._credentials = credentials or CredentialManager()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def platform_id(self) -> str:
        return "huggingface"

    @property
    def rate_limit(self) -> RateLimit:
        return RateLimit(requests=1000, period_seconds=3600)

    def parse_url(self, url: str) -> Optional[str]:
        """Extract repo identifier from HuggingFace URL.

        Returns:
            For datasets: "datasets/org/name"
            For models: "org/name"
            None if URL doesn't match
        """
        # Try dataset URL first (more specific)
        match = self.DATASET_URL_PATTERN.search(url)
        if match:
            return match.group(1)

        # Try model URL
        match = self.MODEL_URL_PATTERN.search(url)
        if match:
            repo_id = match.group(1)
            # Filter out non-repo paths
            first_part = repo_id.split("/")[0]
            if first_part in {"datasets", "spaces", "docs", "blog", "settings", "api"}:
                return None
            return repo_id

        return None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client with authentication."""
        if self._client is None:
            headers = {
                "User-Agent": "Destill3D/0.1.0",
            }
            token = self._credentials.retrieve(self.platform_id)
            if token:
                headers["Authorization"] = f"Bearer {token}"

            self._client = httpx.AsyncClient(
                headers=headers,
                follow_redirects=True,
                timeout=60.0,
            )
        return self._client

    async def search(
        self,
        query: str,
        filters: SearchFilters,
        page: int = 1,
    ) -> SearchResults:
        """Search HuggingFace for datasets with 3D data."""
        client = await self._get_client()

        # Search datasets with 3D-related tags
        params = {
            "search": query,
            "limit": 20,
            "offset": (page - 1) * 20,
        }

        response = await client.get(f"{self.API_URL}/datasets", params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data:
            card_data = item.get("cardData") or {}
            results.append(SearchResult(
                platform=self.platform_id,
                model_id=f"datasets/{item['id']}",
                title=card_data.get("pretty_name", item["id"]),
                author=item["id"].split("/")[0],
                url=f"{self.BASE_URL}/datasets/{item['id']}",
                thumbnail_url=None,
                download_count=item.get("downloads", 0),
                license=card_data.get("license"),
            ))

        return SearchResults(
            results=results,
            total_count=len(results),
            page=page,
            has_more=len(results) >= 20,
        )

    async def get_metadata(self, model_id: str) -> ModelMetadata:
        """Fetch detailed metadata for a HuggingFace repo."""
        client = await self._get_client()

        # Handle both datasets and models
        if model_id.startswith("datasets/"):
            repo_id = model_id.replace("datasets/", "", 1)
            endpoint = f"{self.API_URL}/datasets/{repo_id}"
        else:
            repo_id = model_id
            endpoint = f"{self.API_URL}/models/{repo_id}"

        response = await client.get(endpoint)
        response.raise_for_status()
        data = response.json()

        # Get file tree
        tree_endpoint = f"{endpoint}/tree/main"
        files_response = await client.get(tree_endpoint)
        files_data = files_response.json() if files_response.status_code == 200 else []

        files = []
        for f in files_data:
            if f.get("type") != "file":
                continue
            ext = Path(f["path"]).suffix.lower()
            if ext in _3D_EXTENSIONS:
                files.append(FileInfo(
                    filename=f["path"],
                    url=f"{self.BASE_URL}/{repo_id}/resolve/main/{f['path']}",
                    size_bytes=f.get("size", 0),
                    format=ext,
                ))

        card_data = data.get("cardData") or {}
        created_at_str = data.get("createdAt", "")

        try:
            created_at = datetime.fromisoformat(
                created_at_str.replace("Z", "+00:00")
            ) if created_at_str else datetime.utcnow()
        except ValueError:
            created_at = datetime.utcnow()

        return ModelMetadata(
            platform=self.platform_id,
            model_id=model_id,
            title=card_data.get("pretty_name", repo_id),
            description=data.get("description", ""),
            author=repo_id.split("/")[0],
            license=card_data.get("license", "Unknown"),
            tags=data.get("tags", []),
            files=files,
            created_at=created_at,
            download_count=data.get("downloads"),
        )

    async def download(
        self,
        model_id: str,
        target_dir: Path,
    ) -> DownloadResult:
        """Download 3D files from a HuggingFace repo."""
        start = time.monotonic()

        metadata = await self.get_metadata(model_id)
        client = await self._get_client()

        downloaded_files: List[Path] = []
        target_dir.mkdir(parents=True, exist_ok=True)

        for file_info in metadata.files:
            try:
                response = await client.get(file_info.url)
                response.raise_for_status()

                file_path = target_dir / Path(file_info.filename).name
                file_path.write_bytes(response.content)
                downloaded_files.append(file_path)
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

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
