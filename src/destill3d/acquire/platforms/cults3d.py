"""
Cults3D platform adapter for Destill3D.

Implements web scraping approach for Cults3D marketplace.
Rate limited to 50 requests per hour to be respectful.
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


class Cults3DAdapter:
    """Adapter for Cults3D platform (web scraping)."""

    BASE_URL = "https://cults3d.com"
    URL_PATTERN = re.compile(
        r"cults3d\.com/[a-z]{2}/3d-model/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)"
    )

    def __init__(self, credentials: Optional[CredentialManager] = None):
        self._credentials = credentials or CredentialManager()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def platform_id(self) -> str:
        return "cults3d"

    @property
    def rate_limit(self) -> RateLimit:
        return RateLimit(requests=50, period_seconds=3600)

    def parse_url(self, url: str) -> Optional[str]:
        """Extract category/slug from Cults3D URL."""
        match = self.URL_PATTERN.search(url)
        return match.group(1) if match else None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                follow_redirects=True,
                timeout=30.0,
                headers={
                    "User-Agent": "Destill3D/0.1.0 (3D Model Research Tool)",
                    "Accept-Language": "en",
                },
            )
        return self._client

    async def search(
        self,
        query: str,
        filters: SearchFilters,
        page: int = 1,
    ) -> SearchResults:
        """Search Cults3D for models."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise AcquisitionError(
                self.platform_id,
                "beautifulsoup4",
                "beautifulsoup4 not installed. Install with: pip install destill3d[scraping]",
            )

        client = await self._get_client()
        params = {"q": query, "page": page}

        response = await client.get("/en/search", params=params)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for item in soup.select(".cults-creation, .creation-card, [data-creation-id]"):
            title_elem = item.select_one("a[href*='/3d-model/'], .creation-title a, h3 a")
            if not title_elem:
                continue

            href = title_elem.get("href", "")
            model_id = self.parse_url(f"{self.BASE_URL}{href}")
            if not model_id:
                continue

            # Extract author
            author_elem = item.select_one(".creator, .username, .author")
            author = author_elem.text.strip() if author_elem else "Unknown"

            # Extract thumbnail
            img_elem = item.select_one("img")
            thumbnail_url = img_elem.get("src") if img_elem else None

            results.append(SearchResult(
                platform=self.platform_id,
                model_id=model_id,
                title=title_elem.text.strip(),
                author=author,
                url=f"{self.BASE_URL}{href}",
                thumbnail_url=thumbnail_url,
            ))

        return SearchResults(
            results=results,
            total_count=len(results),
            page=page,
            has_more=len(results) >= 20,
        )

    async def get_metadata(self, model_id: str) -> ModelMetadata:
        """Fetch detailed metadata for a specific model."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise AcquisitionError(
                self.platform_id,
                "beautifulsoup4",
                "beautifulsoup4 not installed. Install with: pip install destill3d[scraping]",
            )

        client = await self._get_client()
        response = await client.get(f"/en/3d-model/{model_id}")
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title_elem = soup.select_one("h1, .creation-title, [itemprop='name']")
        title = title_elem.text.strip() if title_elem else model_id

        # Extract description
        desc_elem = soup.select_one(
            ".creation-description, [itemprop='description'], .description"
        )
        description = desc_elem.text.strip() if desc_elem else ""

        # Extract author
        author_elem = soup.select_one(".creator-name, .author-name, [itemprop='author']")
        author = author_elem.text.strip() if author_elem else "Unknown"

        # Extract license info
        license_elem = soup.select_one(".license, [itemprop='license']")
        license_str = license_elem.text.strip() if license_elem else "Unknown"

        # Extract tags
        tags = [tag.text.strip() for tag in soup.select(".tags a, .tag, [rel='tag']")]

        # Extract file information
        files = []
        for file_link in soup.select(".download-button, a[href*='download']"):
            filename = file_link.get("data-filename", "") or file_link.text.strip()
            if not filename:
                continue
            files.append(FileInfo(
                filename=filename,
                url=f"{self.BASE_URL}/en/3d-model/{model_id}/download",
                size_bytes=0,
            ))

        if not files:
            files.append(FileInfo(
                filename=f"{model_id.split('/')[-1]}.zip",
                url=f"{self.BASE_URL}/en/3d-model/{model_id}/download",
                size_bytes=0,
            ))

        return ModelMetadata(
            platform=self.platform_id,
            model_id=model_id,
            title=title,
            description=description,
            author=author,
            license=license_str,
            tags=tags,
            files=files,
            created_at=datetime.utcnow(),
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

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
