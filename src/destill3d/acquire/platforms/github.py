"""
GitHub platform adapter for Destill3D.

Implements REST API integration for searching and downloading
3D model files from GitHub repositories.
Rate limited to 5000 requests per hour (authenticated).
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

# 3D file extensions to search for
_3D_EXTENSIONS = {".stl", ".obj", ".step", ".stp", ".ply", ".gltf", ".glb", ".3mf", ".amf"}


class GitHubAdapter:
    """Adapter for GitHub platform (REST API)."""

    API_URL = "https://api.github.com"
    BASE_URL = "https://github.com"
    RAW_URL = "https://raw.githubusercontent.com"
    URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)")

    def __init__(self, credentials: Optional[CredentialManager] = None):
        self._credentials = credentials or CredentialManager()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def platform_id(self) -> str:
        return "github"

    @property
    def rate_limit(self) -> RateLimit:
        return RateLimit(requests=5000, period_seconds=3600)

    def parse_url(self, url: str) -> Optional[str]:
        """Extract owner/repo from GitHub URL."""
        match = self.URL_PATTERN.search(url)
        if not match:
            return None
        repo = match.group(1)
        # Filter out non-repo paths
        if repo.split("/")[1] in {"settings", "notifications", "issues", "pulls", "marketplace"}:
            return None
        return repo

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client with authentication."""
        if self._client is None:
            token = self._credentials.retrieve(self.platform_id)
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Destill3D/0.1.0",
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"

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
        """Search GitHub for repositories containing 3D model files."""
        client = await self._get_client()

        # Build search query for 3D files
        ext_query = " OR ".join(f"extension:{ext.lstrip('.')}" for ext in _3D_EXTENSIONS)
        search_query = f"{query} ({ext_query})"

        params = {
            "q": search_query,
            "page": page,
            "per_page": 20,
            "sort": "stars",
            "order": "desc",
        }

        response = await client.get("/search/code", params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        seen_repos = set()

        for item in data.get("items", []):
            repo = item.get("repository", {})
            repo_full_name = repo.get("full_name", "")

            # Deduplicate by repository
            if repo_full_name in seen_repos:
                continue
            seen_repos.add(repo_full_name)

            results.append(SearchResult(
                platform=self.platform_id,
                model_id=repo_full_name,
                title=repo.get("name", repo_full_name),
                author=repo.get("owner", {}).get("login", "Unknown"),
                url=repo.get("html_url", f"{self.BASE_URL}/{repo_full_name}"),
                thumbnail_url=repo.get("owner", {}).get("avatar_url"),
                license=None,
            ))

        total = data.get("total_count", len(results))

        return SearchResults(
            results=results,
            total_count=total,
            page=page,
            has_more=page * 20 < total,
        )

    async def get_metadata(self, model_id: str) -> ModelMetadata:
        """Fetch repository metadata and list 3D files."""
        client = await self._get_client()

        # Get repo info
        response = await client.get(f"/repos/{model_id}")
        response.raise_for_status()
        repo = response.json()

        # Get file tree to find 3D files
        tree_response = await client.get(
            f"/repos/{model_id}/git/trees/HEAD",
            params={"recursive": "1"},
        )

        files = []
        if tree_response.status_code == 200:
            tree_data = tree_response.json()
            for item in tree_data.get("tree", []):
                if item.get("type") != "blob":
                    continue
                ext = Path(item["path"]).suffix.lower()
                if ext in _3D_EXTENSIONS:
                    files.append(FileInfo(
                        filename=item["path"],
                        url=f"{self.RAW_URL}/{model_id}/HEAD/{item['path']}",
                        size_bytes=item.get("size"),
                        format=ext,
                    ))

        # Extract license
        license_info = repo.get("license")
        license_str = "Unknown"
        if license_info and isinstance(license_info, dict):
            license_str = license_info.get("spdx_id", license_info.get("name", "Unknown"))

        return ModelMetadata(
            platform=self.platform_id,
            model_id=model_id,
            title=repo.get("name", model_id),
            description=repo.get("description") or "",
            author=repo.get("owner", {}).get("login", "Unknown"),
            license=license_str,
            tags=repo.get("topics", []),
            files=files,
            created_at=self._parse_date(repo.get("created_at")) or datetime.utcnow(),
            modified_at=self._parse_date(repo.get("updated_at")),
            download_count=repo.get("stargazers_count"),
        )

    async def download(
        self,
        model_id: str,
        target_dir: Path,
    ) -> DownloadResult:
        """Download 3D model files from a GitHub repository."""
        start = time.monotonic()

        metadata = await self.get_metadata(model_id)
        client = await self._get_client()

        downloaded_files: List[Path] = []
        target_dir.mkdir(parents=True, exist_ok=True)

        for file_info in metadata.files:
            try:
                response = await client.get(
                    file_info.url,
                    headers={"Accept": "application/octet-stream"},
                )
                response.raise_for_status()

                # Preserve directory structure
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
