"""
Base protocol and data models for platform adapters.

Defines the PlatformAdapter protocol that all platform-specific
adapters must implement, along with shared data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Protocol

from destill3d.core.exceptions import AcquisitionError


class PlatformAdapter(Protocol):
    """Protocol for platform-specific acquisition logic."""

    @property
    def platform_id(self) -> str:
        """Unique identifier (e.g., 'thingiverse', 'sketchfab')."""
        ...

    @property
    def rate_limit(self) -> "RateLimit":
        """Platform-specific rate limiting configuration."""
        ...

    async def search(
        self,
        query: str,
        filters: "SearchFilters",
        page: int = 1,
    ) -> "SearchResults":
        """Search for models matching criteria."""
        ...

    async def get_metadata(self, model_id: str) -> "ModelMetadata":
        """Fetch detailed metadata for a specific model."""
        ...

    async def download(
        self,
        model_id: str,
        target_dir: Path,
    ) -> "DownloadResult":
        """Download model files to target directory."""
        ...

    def parse_url(self, url: str) -> Optional[str]:
        """Extract model_id from platform URL, or None if not matching."""
        ...


@dataclass
class RateLimit:
    """Rate limit specification."""

    requests: int
    period_seconds: int

    @classmethod
    def from_string(cls, spec: str) -> "RateLimit":
        """
        Parse rate limit string like '300/5min', '1000/day', '10/second'.

        Supported suffixes: second, sec, s, minute, min, m, hour, hr, h, day, d.
        """
        import re

        match = re.match(r"(\d+)\s*/\s*(\d*)\s*(second|sec|s|minute|min|m|hour|hr|h|day|d)", spec.lower())
        if not match:
            raise ValueError(f"Invalid rate limit format: {spec}")

        requests = int(match.group(1))
        multiplier = int(match.group(2)) if match.group(2) else 1

        unit = match.group(3)
        unit_seconds = {
            "second": 1, "sec": 1, "s": 1,
            "minute": 60, "min": 60, "m": 60,
            "hour": 3600, "hr": 3600, "h": 3600,
            "day": 86400, "d": 86400,
        }

        period = multiplier * unit_seconds[unit]
        return cls(requests=requests, period_seconds=period)


@dataclass
class SearchFilters:
    """Filters for platform search."""

    license: Optional[List[str]] = None
    format: Optional[List[str]] = None
    category: Optional[str] = None
    min_downloads: Optional[int] = None
    date_after: Optional[datetime] = None
    date_before: Optional[datetime] = None


@dataclass
class SearchResult:
    """Single search result."""

    platform: str
    model_id: str
    title: str
    author: str
    url: str
    thumbnail_url: Optional[str] = None
    download_count: Optional[int] = None
    license: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class SearchResults:
    """Paginated search results."""

    results: List[SearchResult]
    total_count: int
    page: int
    has_more: bool

    @property
    def items(self) -> List[SearchResult]:
        """Alias for results for API compatibility."""
        return self.results


@dataclass
class ModelMetadata:
    """Detailed model metadata from platform."""

    platform: str
    model_id: str
    title: str
    description: str
    author: str
    license: str
    tags: List[str]
    files: List["FileInfo"]
    created_at: datetime
    modified_at: Optional[datetime] = None
    download_count: Optional[int] = None
    like_count: Optional[int] = None


@dataclass
class FileInfo:
    """Information about a downloadable file."""

    filename: str
    url: str
    size_bytes: Optional[int] = None
    format: Optional[str] = None


@dataclass
class DownloadResult:
    """Result of download operation."""

    platform: str
    model_id: str
    files: List[Path]
    metadata: ModelMetadata
    download_time_ms: float
