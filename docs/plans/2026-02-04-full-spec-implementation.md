# Destill3D: Full Specification Implementation Plan

**Version**: 1.0.0
**Date**: 2026-02-04
**Target**: Complete implementation of SPEC.md (v0.1.0 → v1.0.0)
**Estimated Scope**: ~15,000 lines of code across 50+ files

---

## Executive Summary

This plan brings Destill3D from v0.1.0 MVP to full v1.0.0 spec compliance. The implementation is organized into 8 phases, each building on the previous. Total implementation covers:

- **Acquire Module**: Platform adapters, queue management, rate limiting
- **Pipeline System**: Stages, checkpointing, resumability
- **Zero-Shot Classification**: OpenShape/CLIP integration
- **Advanced Storage**: PostgreSQL, FAISS similarity search
- **Multi-View Rendering**: 12-view depth maps, MVCNN
- **Security & Compliance**: Input validation, credentials, licensing
- **Extended CLI**: Full command tree per spec
- **Comprehensive Testing**: E2E, benchmarks, 80%+ coverage

---

## Phase 1: Core Infrastructure Upgrades

**Goal**: Establish foundation for all subsequent phases
**Files**: 8 new, 4 modified
**Priority**: Critical (blocks all other phases)

### 1.1 Unified API Class (`src/destill3d/api.py`)

Create the `Destill3D` unified interface specified in Section 10.1:

```python
# New file: src/destill3d/api.py
class Destill3D:
    """Unified API for all Destill3D operations."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        temp_dir: Optional[Path] = None,
        config: Optional[Destill3DConfig] = None,
    ):
        self.config = config or Destill3DConfig()
        self._db = None
        self._extractor = None
        self._classifier = None

    @property
    def acquire(self) -> "AcquireAPI":
        """Acquisition operations."""
        ...

    @property
    def extract(self) -> "ExtractAPI":
        """Extraction operations."""
        ...

    @property
    def classify(self) -> "ClassifyAPI":
        """Classification operations."""
        ...

    @property
    def db(self) -> "DatabaseAPI":
        """Database operations."""
        ...

    def export(self, ...) -> ExportResult:
        """Export to ML formats."""
        ...
```

### 1.2 Async Support Infrastructure

Add async variants for I/O-bound operations:

```python
# New file: src/destill3d/core/async_utils.py
import asyncio
from typing import TypeVar, Callable, Awaitable

T = TypeVar('T')

def run_sync(coro: Awaitable[T]) -> T:
    """Run async function synchronously."""
    try:
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

class AsyncContextManager:
    """Base for async resource management."""
    ...
```

### 1.3 Enhanced Configuration (`src/destill3d/core/config.py`)

Extend config to support all spec sections:

```python
# Additions to existing config.py

class AcquisitionConfig(BaseSettings):
    """Acquisition module configuration."""
    temp_directory: Path = Path("/tmp/destill3d")
    max_concurrent_downloads: int = 4
    retry_attempts: int = 3
    retry_delay_seconds: int = 60

    class PlatformConfig(BaseSettings):
        api_key: Optional[str] = None
        rate_limit: str = "100/hour"
        enabled: bool = True

    platforms: Dict[str, PlatformConfig] = {}

class SecurityConfig(BaseSettings):
    """Security and validation settings."""
    max_file_size: int = 500 * 1024 * 1024  # 500 MB
    allowed_extensions: Set[str] = {...}
    allowed_domains: Set[str] = {...}
    validate_urls: bool = True
    validate_files: bool = True

class TelemetryConfig(BaseSettings):
    """Optional telemetry/monitoring."""
    enabled: bool = False
    endpoint: Optional[str] = None

class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = "INFO"
    file: Optional[Path] = None
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 1.4 Retry Infrastructure (`src/destill3d/core/retry.py`)

```python
# New file: src/destill3d/core/retry.py
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetryConfig:
    """Retry configuration for various operations."""
    download_attempts: int = 3
    download_wait_min: int = 1
    download_wait_max: int = 60
    extraction_attempts: int = 2
    classification_attempts: int = 2

def with_retry(
    attempts: int = 3,
    wait_min: int = 1,
    wait_max: int = 60,
    exceptions: tuple = (Exception,),
):
    """Decorator factory for retry logic."""
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
```

### 1.5 Dependencies Update (`pyproject.toml`)

```toml
# Add to dependencies
dependencies = [
    # ... existing ...
    "tenacity>=8.0",
    "aiofiles>=23.0",
    "keyring>=24.0",
]

[project.optional-dependencies]
# Add new groups
hdf5 = ["h5py>=3.0"]
tfrecord = ["tensorflow>=2.0"]  # Or tensorflow-io for just tfrecord
acquire = ["aiohttp>=3.9", "aiolimiter>=1.1"]
zero-shot = ["transformers>=4.30", "sentence-transformers>=2.0"]
```

---

## Phase 2: Acquire Module

**Goal**: Platform adapters for automated model acquisition
**Files**: 15 new
**Spec Sections**: 4.1-4.4, 9.1 (acquire commands)

### 2.1 Directory Structure

```
src/destill3d/acquire/
├── __init__.py           # Public API
├── base.py               # PlatformAdapter protocol, base classes
├── queue.py              # Download queue management
├── rate_limiter.py       # Per-platform rate limiting
├── downloader.py         # Async download with retry
├── credentials.py        # Secure credential management
├── platforms/
│   ├── __init__.py
│   ├── thingiverse.py    # Thingiverse adapter (P0)
│   ├── sketchfab.py      # Sketchfab adapter (P0)
│   ├── grabcad.py        # GrabCAD adapter (P1)
│   ├── cults3d.py        # Cults3D adapter (P1)
│   ├── myminifactory.py  # MyMiniFactory adapter (P2)
│   ├── thangs.py         # Thangs adapter (P2)
│   ├── github.py         # GitHub adapter (P2)
│   └── local.py          # Local filesystem adapter (P0)
└── models.py             # Data models (SearchResult, DownloadResult, etc.)
```

### 2.2 Core Protocol (`src/destill3d/acquire/base.py`)

```python
from typing import Protocol, Optional, List, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from enum import Enum

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
        """Parse '300/5min', '1000/day', etc."""
        ...

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
```

### 2.3 Queue Management (`src/destill3d/acquire/queue.py`)

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, AsyncIterator
from pathlib import Path
import asyncio

class QueueStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class QueueEntry:
    """Entry in the download queue."""
    queue_id: int
    platform: str
    model_id: str
    source_url: str
    status: QueueStatus
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    temp_path: Optional[Path] = None
    created_at: datetime = None
    updated_at: datetime = None
    scheduled_at: Optional[datetime] = None  # For delayed retry

class DownloadQueue:
    """Manages the download queue with persistence."""

    def __init__(self, db: "Database"):
        self.db = db
        self._workers: List[asyncio.Task] = []
        self._running = False

    async def add(
        self,
        url: str,
        priority: int = 0,
        platform: Optional[str] = None,
    ) -> QueueEntry:
        """Add URL to download queue."""
        ...

    async def add_batch(
        self,
        urls: List[str],
        priority: int = 0,
    ) -> List[QueueEntry]:
        """Add multiple URLs to queue."""
        ...

    async def get_pending(
        self,
        limit: int = 100,
    ) -> List[QueueEntry]:
        """Get pending entries ordered by priority."""
        ...

    async def update_status(
        self,
        queue_id: int,
        status: QueueStatus,
        error: Optional[str] = None,
    ):
        """Update entry status."""
        ...

    async def process(
        self,
        concurrency: int = 4,
        callback: Optional[Callable] = None,
    ) -> AsyncIterator["ProcessingProgress"]:
        """Process queue with progress updates."""
        ...

    async def retry_failed(
        self,
        max_age_hours: int = 24,
    ) -> int:
        """Retry failed entries within age limit."""
        ...

@dataclass
class ProcessingProgress:
    """Progress update during queue processing."""
    completed: int
    total: int
    current: Optional[QueueEntry] = None
    rate: float = 0.0  # items per minute
    eta_seconds: Optional[float] = None
```

### 2.4 Rate Limiter (`src/destill3d/acquire/rate_limiter.py`)

```python
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional
import time

class RateLimiter:
    """Token bucket rate limiter with per-platform limits."""

    def __init__(self):
        self._buckets: Dict[str, "TokenBucket"] = {}
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def configure(self, platform: str, rate_limit: RateLimit):
        """Configure rate limit for a platform."""
        self._buckets[platform] = TokenBucket(
            capacity=rate_limit.requests,
            refill_period=rate_limit.period_seconds,
        )

    async def acquire(self, platform: str, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.
        Returns wait time in seconds.
        """
        async with self._locks[platform]:
            bucket = self._buckets.get(platform)
            if not bucket:
                return 0.0

            wait_time = bucket.consume(tokens)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            return wait_time

    def available(self, platform: str) -> int:
        """Get available tokens for platform."""
        bucket = self._buckets.get(platform)
        return bucket.available if bucket else float('inf')

class TokenBucket:
    """Token bucket implementation."""

    def __init__(self, capacity: int, refill_period: float):
        self.capacity = capacity
        self.refill_period = refill_period
        self.tokens = capacity
        self.last_refill = time.monotonic()

    def consume(self, tokens: int = 1) -> float:
        """Consume tokens, return wait time if insufficient."""
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return 0.0

        # Calculate wait time for enough tokens
        needed = tokens - self.tokens
        wait_time = (needed / self.capacity) * self.refill_period
        return wait_time

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        refill_amount = (elapsed / self.refill_period) * self.capacity
        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now

    @property
    def available(self) -> int:
        self._refill()
        return int(self.tokens)
```

### 2.5 Credential Manager (`src/destill3d/acquire/credentials.py`)

```python
import os
from typing import Optional
import keyring

class CredentialManager:
    """Secure storage and retrieval of API credentials."""

    SERVICE_NAME = "destill3d"

    def __init__(self, use_keyring: bool = True):
        self.use_keyring = use_keyring

    def store(self, platform: str, api_key: str) -> None:
        """Store API key securely."""
        if self.use_keyring:
            keyring.set_password(self.SERVICE_NAME, platform, api_key)
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
            except keyring.errors.PasswordDeleteError:
                pass  # Already deleted

    def list_platforms(self) -> List[str]:
        """List platforms with stored credentials."""
        # This is limited by keyring capabilities
        # Return known platforms that have credentials
        platforms = []
        for platform in ["thingiverse", "sketchfab", "grabcad",
                        "cults3d", "myminifactory", "thangs", "github"]:
            if self.retrieve(platform):
                platforms.append(platform)
        return platforms
```

### 2.6 Thingiverse Adapter (`src/destill3d/acquire/platforms/thingiverse.py`)

```python
import httpx
import re
from typing import Optional, List
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

from ..base import (
    PlatformAdapter, RateLimit, SearchFilters, SearchResults,
    SearchResult, ModelMetadata, FileInfo, DownloadResult
)
from ..credentials import CredentialManager

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
                raise ValueError("Thingiverse API key not configured")

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
        import time
        start = time.monotonic()

        metadata = await self.get_metadata(model_id)
        client = await self._get_client()

        downloaded_files = []
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
```

### 2.7 Sketchfab Adapter (`src/destill3d/acquire/platforms/sketchfab.py`)

Similar structure to Thingiverse, with Sketchfab-specific API handling:

```python
class SketchfabAdapter:
    """Sketchfab platform adapter."""

    BASE_URL = "https://api.sketchfab.com/v3"

    @property
    def platform_id(self) -> str:
        return "sketchfab"

    @property
    def rate_limit(self) -> RateLimit:
        return RateLimit(requests=1000, period_seconds=86400)  # 1000/day

    # ... similar implementation with Sketchfab API specifics
    # Key differences:
    # - OAuth2 authentication
    # - GLTF as primary format
    # - Different API structure for search/download
```

### 2.8 CLI Commands (`src/destill3d/cli/commands/acquire.py`)

```python
import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

app = typer.Typer(help="Acquire models from platforms")
console = Console()

@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    platform: str = typer.Option("thingiverse", "-p", "--platform"),
    limit: int = typer.Option(20, "-l", "--limit"),
    license: Optional[str] = typer.Option(None, "--license"),
    queue: bool = typer.Option(False, "-q", "--queue", help="Add results to queue"),
):
    """Search platforms and optionally queue results."""
    ...

@app.command("url")
def add_url_cmd(
    url: str = typer.Argument(..., help="URL to add"),
    priority: int = typer.Option(0, "-p", "--priority"),
):
    """Add specific URL to download queue."""
    ...

@app.command("list")
def list_queue_cmd(
    status: Optional[str] = typer.Option(None, "-s", "--status"),
    limit: int = typer.Option(50, "-l", "--limit"),
):
    """List entries in the download queue."""
    ...

@app.command("run")
def run_queue_cmd(
    concurrency: int = typer.Option(4, "-c", "--concurrency"),
    rate_limit: Optional[str] = typer.Option(None, "-r", "--rate-limit"),
    extract: bool = typer.Option(True, "--extract/--no-extract"),
    classify: bool = typer.Option(False, "--classify/--no-classify"),
):
    """Process the download queue."""
    ...

@app.command("platforms")
def list_platforms_cmd():
    """List available platforms and their status."""
    ...

@app.command("credentials")
def manage_credentials_cmd(
    platform: str = typer.Argument(...),
    action: str = typer.Option("show", "-a", "--action",
                               help="show|set|delete"),
    value: Optional[str] = typer.Option(None, "-v", "--value"),
):
    """Manage platform API credentials."""
    ...
```

---

## Phase 3: Pipeline & Queue System

**Goal**: Implement full pipeline stages with checkpointing and resumability
**Files**: 6 new, 3 modified
**Spec Sections**: 3.1-3.4, 12.3

### 3.1 Pipeline Stages (`src/destill3d/core/pipeline.py`)

```python
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Any
from pathlib import Path
import numpy as np

class PipelineStage(Enum):
    """Pipeline processing stages."""
    QUEUED = "queued"           # Entry created, not yet downloaded
    ACQUIRED = "acquired"       # Files downloaded to temp directory
    EXTRACTED = "extracted"     # Snapshot created, pending classification
    CLASSIFIED = "classified"   # Classification complete
    STORED = "stored"           # Persisted to database, temp files cleaned
    FAILED = "failed"           # Error state with retry counter

@dataclass
class ProcessingCheckpoint:
    """Checkpoint record for resumable processing."""
    model_id: str                    # Unique identifier (platform:id)
    stage: PipelineStage
    created_at: datetime
    updated_at: datetime

    # Acquisition checkpoint
    source_url: str
    platform: str
    temp_path: Optional[Path] = None
    file_hash: Optional[str] = None  # SHA256 of downloaded file

    # Extraction checkpoint
    snapshot_path: Optional[Path] = None
    point_count: Optional[int] = None
    feature_version: Optional[str] = None

    # Classification checkpoint
    predictions: Optional[List["Prediction"]] = None
    embedding: Optional[np.ndarray] = None

    # Error tracking
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    error_stage: Optional[PipelineStage] = None

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def increment_retry(self, error: str):
        self.retry_count += 1
        self.last_error = error
        self.error_stage = self.stage
        self.updated_at = datetime.utcnow()

@dataclass
class BatchConfig:
    """Batch processing configuration."""
    download_batch_size: int = 10      # Concurrent downloads
    extraction_batch_size: int = 1     # Sequential (memory-bound)
    classification_batch_size: int = 32 # GPU batch size
    storage_batch_size: int = 100      # Database transaction size
```

### 3.2 Pipeline Executor (`src/destill3d/core/executor.py`)

```python
import asyncio
from typing import AsyncIterator, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .pipeline import PipelineStage, ProcessingCheckpoint, BatchConfig
from .database import Database
from ..acquire import DownloadQueue, PlatformRegistry
from ..extract import FeatureExtractor
from ..classify import Classifier

logger = logging.getLogger(__name__)

@dataclass
class PipelineProgress:
    """Progress update during pipeline execution."""
    stage: PipelineStage
    completed: int
    total: int
    current_id: Optional[str] = None
    rate: float = 0.0
    errors: int = 0

class PipelineExecutor:
    """Executes the full acquisition-extraction-classification pipeline."""

    def __init__(
        self,
        db: Database,
        extractor: FeatureExtractor,
        classifier: Classifier,
        platform_registry: "PlatformRegistry",
        config: BatchConfig = None,
    ):
        self.db = db
        self.extractor = extractor
        self.classifier = classifier
        self.platforms = platform_registry
        self.config = config or BatchConfig()
        self._running = False

    async def run(
        self,
        start_stage: PipelineStage = PipelineStage.QUEUED,
        end_stage: PipelineStage = PipelineStage.STORED,
    ) -> AsyncIterator[PipelineProgress]:
        """
        Run pipeline from start_stage to end_stage.
        Yields progress updates.
        """
        self._running = True

        try:
            # Get all items in appropriate stages
            items = await self._get_items_for_processing(start_stage, end_stage)
            total = len(items)
            completed = 0
            errors = 0

            for item in items:
                if not self._running:
                    break

                try:
                    await self._process_item(item, end_stage)
                    completed += 1
                except Exception as e:
                    logger.error(f"Pipeline error for {item.model_id}: {e}")
                    await self._handle_error(item, str(e))
                    errors += 1

                yield PipelineProgress(
                    stage=item.stage,
                    completed=completed,
                    total=total,
                    current_id=item.model_id,
                    errors=errors,
                )

        finally:
            self._running = False

    async def _process_item(
        self,
        checkpoint: ProcessingCheckpoint,
        end_stage: PipelineStage,
    ):
        """Process a single item through pipeline stages."""

        stages = [
            (PipelineStage.QUEUED, PipelineStage.ACQUIRED, self._acquire),
            (PipelineStage.ACQUIRED, PipelineStage.EXTRACTED, self._extract),
            (PipelineStage.EXTRACTED, PipelineStage.CLASSIFIED, self._classify),
            (PipelineStage.CLASSIFIED, PipelineStage.STORED, self._store),
        ]

        for from_stage, to_stage, handler in stages:
            if checkpoint.stage == from_stage:
                await handler(checkpoint)
                checkpoint.stage = to_stage
                checkpoint.updated_at = datetime.utcnow()
                await self.db.save_checkpoint(checkpoint)

                if to_stage == end_stage:
                    break

    async def _acquire(self, checkpoint: ProcessingCheckpoint):
        """Download files for checkpoint."""
        adapter = self.platforms.get(checkpoint.platform)
        model_id = checkpoint.source_url.split(":")[-1]

        result = await adapter.download(
            model_id,
            target_dir=self._temp_dir / checkpoint.model_id,
        )

        checkpoint.temp_path = result.files[0].parent
        checkpoint.file_hash = self._compute_hash(result.files[0])

    async def _extract(self, checkpoint: ProcessingCheckpoint):
        """Extract features from downloaded files."""
        # Find primary 3D file
        file_path = self._find_primary_file(checkpoint.temp_path)

        snapshot = self.extractor.extract_from_file(file_path)

        snapshot_path = self._snapshots_dir / f"{checkpoint.model_id}.d3d"
        snapshot.save(snapshot_path)

        checkpoint.snapshot_path = snapshot_path
        checkpoint.point_count = snapshot.geometry.point_count

    async def _classify(self, checkpoint: ProcessingCheckpoint):
        """Classify the extracted snapshot."""
        snapshot = Snapshot.load(checkpoint.snapshot_path)

        result = self.classifier.classify(snapshot)

        checkpoint.predictions = result.predictions
        checkpoint.embedding = result.embedding

    async def _store(self, checkpoint: ProcessingCheckpoint):
        """Store snapshot in database and cleanup."""
        snapshot = Snapshot.load(checkpoint.snapshot_path)
        snapshot.predictions = checkpoint.predictions

        await self.db.store_snapshot(snapshot)
        await self.db.store_embedding(
            checkpoint.model_id,
            checkpoint.embedding,
        )

        # Cleanup temp files
        if checkpoint.temp_path and checkpoint.temp_path.exists():
            import shutil
            shutil.rmtree(checkpoint.temp_path)

    async def _handle_error(
        self,
        checkpoint: ProcessingCheckpoint,
        error: str,
    ):
        """Handle pipeline error with retry logic."""
        checkpoint.increment_retry(error)

        if checkpoint.can_retry():
            # Schedule for retry
            checkpoint.stage = self._get_previous_stage(checkpoint.stage)
        else:
            checkpoint.stage = PipelineStage.FAILED

        await self.db.save_checkpoint(checkpoint)

    def stop(self):
        """Stop pipeline execution."""
        self._running = False
```

### 3.3 Recovery System (`src/destill3d/core/recovery.py`)

```python
from datetime import datetime, timedelta
from typing import List
import logging

from .pipeline import PipelineStage, ProcessingCheckpoint
from .database import Database

logger = logging.getLogger(__name__)

class RecoveryManager:
    """Handles recovery of interrupted processing."""

    def __init__(self, db: Database):
        self.db = db

    async def recover_on_startup(
        self,
        stuck_threshold_hours: int = 1,
    ) -> int:
        """
        Recover items stuck in intermediate states.
        Called on application startup.
        Returns number of items recovered.
        """
        cutoff = datetime.utcnow() - timedelta(hours=stuck_threshold_hours)

        # Find items stuck in non-terminal states
        intermediate_stages = [
            PipelineStage.ACQUIRED,
            PipelineStage.EXTRACTED,
            PipelineStage.CLASSIFIED,
        ]

        stuck_items = await self.db.query_checkpoints(
            stages=intermediate_stages,
            updated_before=cutoff,
        )

        recovered = 0
        for item in stuck_items:
            if item.can_retry():
                # Reset to previous stage for retry
                previous = self._get_previous_stage(item.stage)
                item.stage = previous
                item.retry_count += 1
                item.last_error = "Recovered from stuck state"
                item.updated_at = datetime.utcnow()

                await self.db.save_checkpoint(item)
                logger.info(f"Recovered {item.model_id}, retry {item.retry_count}")
                recovered += 1
            else:
                # Max retries exceeded
                item.stage = PipelineStage.FAILED
                item.last_error = "Max retries exceeded during recovery"
                await self.db.save_checkpoint(item)
                logger.error(f"Max retries exceeded for {item.model_id}")

        return recovered

    async def cleanup_temp_files(
        self,
        max_age_hours: int = 24,
    ) -> int:
        """Remove orphaned temp files."""
        # Implementation
        ...

    async def retry_failed(
        self,
        max_age_hours: int = 48,
        reset_retry_count: bool = False,
    ) -> int:
        """Retry failed items within age limit."""
        ...

    def _get_previous_stage(self, stage: PipelineStage) -> PipelineStage:
        """Get the stage before the given one."""
        order = [
            PipelineStage.QUEUED,
            PipelineStage.ACQUIRED,
            PipelineStage.EXTRACTED,
            PipelineStage.CLASSIFIED,
            PipelineStage.STORED,
        ]
        idx = order.index(stage)
        return order[max(0, idx - 1)]
```

---

## Phase 4: Zero-Shot Classification

**Goal**: OpenShape/CLIP integration for arbitrary class classification
**Files**: 5 new, 2 modified
**Spec Sections**: 7.1 (Zero-shot section), 7.2, 7.4

### 4.1 Zero-Shot Module (`src/destill3d/classify/zero_shot.py`)

```python
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import torch
from pathlib import Path

@dataclass
class ZeroShotConfig:
    """Configuration for zero-shot classification."""
    encoder_model: str = "openshape_pointbert"
    text_encoder: str = "openai/clip-vit-large-patch14"
    device: str = "auto"
    cache_embeddings: bool = True

@dataclass
class ZeroShotResult:
    """Result of zero-shot classification."""
    classes: List[str]
    probabilities: List[float]
    embedding_3d: np.ndarray
    embedding_dim: int

class ZeroShotClassifier:
    """Zero-shot 3D classification using OpenShape + CLIP."""

    def __init__(self, config: ZeroShotConfig = None):
        self.config = config or ZeroShotConfig()
        self._point_encoder = None
        self._text_encoder = None
        self._text_cache = {}
        self._device = None

    def _init_models(self):
        """Lazy initialization of models."""
        if self._point_encoder is not None:
            return

        # Determine device
        if self.config.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.config.device

        # Load OpenShape point encoder
        self._point_encoder = self._load_point_encoder()

        # Load CLIP text encoder
        from transformers import CLIPTokenizer, CLIPTextModel
        self._tokenizer = CLIPTokenizer.from_pretrained(self.config.text_encoder)
        self._text_encoder = CLIPTextModel.from_pretrained(
            self.config.text_encoder
        ).to(self._device)

    def _load_point_encoder(self):
        """Load the OpenShape point cloud encoder."""
        from ..classify.registry import MODEL_REGISTRY

        model_info = MODEL_REGISTRY.get(self.config.encoder_model)
        if not model_info:
            raise ValueError(f"Unknown encoder: {self.config.encoder_model}")

        # Load ONNX model
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self._device == "cpu":
            providers = ["CPUExecutionProvider"]

        return ort.InferenceSession(
            str(model_info.weights_path),
            providers=providers,
        )

    def encode_points(self, points: np.ndarray) -> np.ndarray:
        """
        Encode point cloud to embedding space.

        Args:
            points: (N, 3) point cloud, normalized

        Returns:
            (D,) embedding vector
        """
        self._init_models()

        # Prepare input
        points_input = points.astype(np.float32)
        if points_input.shape[0] != 2048:
            # Resample if needed
            points_input = self._resample_points(points_input, 2048)

        # Add batch dimension
        points_batch = points_input.reshape(1, -1, 3)

        # Run inference
        outputs = self._point_encoder.run(
            None,
            {"points": points_batch},
        )

        embedding = outputs[0][0]  # Remove batch dim

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode text labels to embedding space.

        Args:
            texts: List of class names

        Returns:
            (num_classes, D) embedding matrix
        """
        self._init_models()

        embeddings = []
        for text in texts:
            # Check cache
            if self.config.cache_embeddings and text in self._text_cache:
                embeddings.append(self._text_cache[text])
                continue

            # Encode with CLIP
            inputs = self._tokenizer(
                f"a 3D model of a {text}",
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._text_encoder(**inputs)
                embedding = outputs.pooler_output[0].cpu().numpy()

            # L2 normalize
            embedding = embedding / np.linalg.norm(embedding)

            if self.config.cache_embeddings:
                self._text_cache[text] = embedding

            embeddings.append(embedding)

        return np.array(embeddings)

    def classify(
        self,
        points: np.ndarray,
        classes: List[str],
        temperature: float = 0.07,
    ) -> ZeroShotResult:
        """
        Classify point cloud into arbitrary classes.

        Args:
            points: (N, 3) normalized point cloud
            classes: List of class names to classify into
            temperature: Softmax temperature (lower = sharper)

        Returns:
            ZeroShotResult with probabilities for each class
        """
        # Get embeddings
        point_embedding = self.encode_points(points)
        text_embeddings = self.encode_texts(classes)

        # Compute cosine similarities
        similarities = point_embedding @ text_embeddings.T

        # Apply temperature and softmax
        logits = similarities / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        return ZeroShotResult(
            classes=classes,
            probabilities=probabilities.tolist(),
            embedding_3d=point_embedding,
            embedding_dim=point_embedding.shape[0],
        )

    def _resample_points(
        self,
        points: np.ndarray,
        target_count: int,
    ) -> np.ndarray:
        """Resample points to target count."""
        current = points.shape[0]

        if current == target_count:
            return points
        elif current > target_count:
            # Random subsample
            indices = np.random.choice(current, target_count, replace=False)
            return points[indices]
        else:
            # Pad by repeating
            indices = np.random.choice(current, target_count - current, replace=True)
            return np.vstack([points, points[indices]])
```

### 4.2 CLI Integration (`src/destill3d/cli/commands/classify.py` modification)

```python
# Add to existing classify.py

@app.command("zero-shot")
def zero_shot_cmd(
    snapshot_path: Path = typer.Argument(..., help="Snapshot file or ID"),
    classes: str = typer.Option(..., "-c", "--classes",
                                help="Comma-separated class names"),
    model: str = typer.Option("openshape_pointbert", "-m", "--model"),
    temperature: float = typer.Option(0.07, "-t", "--temperature"),
    output: Optional[Path] = typer.Option(None, "-o", "--output"),
):
    """
    Zero-shot classification with custom classes.

    Example:
        destill3d classify zero-shot model.d3d -c "chair,table,lamp,sofa"
    """
    from destill3d.classify.zero_shot import ZeroShotClassifier, ZeroShotConfig

    # Parse classes
    class_list = [c.strip() for c in classes.split(",")]

    # Load snapshot
    snapshot = Snapshot.load(snapshot_path)

    # Run classification
    config = ZeroShotConfig(encoder_model=model)
    classifier = ZeroShotClassifier(config)

    result = classifier.classify(
        snapshot.geometry.points,
        class_list,
        temperature=temperature,
    )

    # Display results
    table = Table(title="Zero-Shot Classification")
    table.add_column("Class", style="cyan")
    table.add_column("Probability", style="green")

    sorted_results = sorted(
        zip(result.classes, result.probabilities),
        key=lambda x: x[1],
        reverse=True,
    )

    for cls, prob in sorted_results:
        table.add_row(cls, f"{prob:.1%}")

    console.print(table)
```

---

## Phase 5: Advanced Storage

**Goal**: PostgreSQL backend, FAISS similarity search, HDF5/TFRecord export
**Files**: 8 new, 4 modified
**Spec Sections**: 8.1-8.3

### 5.1 PostgreSQL Backend (`src/destill3d/core/database.py` modification)

```python
# Add to existing database.py

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from urllib.parse import quote_plus

class DatabaseFactory:
    """Factory for creating database connections."""

    @staticmethod
    def create(config: DatabaseConfig) -> "Database":
        """Create database instance from config."""
        if config.type == "sqlite":
            return SQLiteDatabase(config.path)
        elif config.type == "postgresql":
            return PostgreSQLDatabase(config)
        else:
            raise ValueError(f"Unknown database type: {config.type}")

class PostgreSQLDatabase(Database):
    """PostgreSQL database backend."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._engine = None
        self._session_factory = None

    def _get_connection_string(self) -> str:
        password = quote_plus(self.config.password) if self.config.password else ""
        return (
            f"postgresql://{self.config.user}:{password}"
            f"@{self.config.host}:{self.config.port}"
            f"/{self.config.database}"
        )

    def connect(self):
        """Establish database connection."""
        self._engine = create_engine(
            self._get_connection_string(),
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )

        # Create tables if needed
        Base.metadata.create_all(self._engine)

        self._session_factory = sessionmaker(bind=self._engine)

    # ... rest of implementation follows same interface as SQLite
```

### 5.2 FAISS Index (`src/destill3d/core/vector_index.py`)

```python
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class EmbeddingIndex:
    """FAISS-based similarity search for snapshot embeddings."""

    def __init__(
        self,
        dimension: int = 1024,
        index_type: str = "flat",  # flat, ivf, hnsw
        metric: str = "cosine",    # cosine, l2
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self._index = None
        self._id_map: List[str] = []  # snapshot_id at each index position
        self._reverse_map: dict = {}  # snapshot_id -> index position

    def _create_index(self):
        """Create FAISS index based on configuration."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss not installed. Install with: pip install faiss-cpu")

        if self.metric == "cosine":
            # For cosine similarity, use inner product after L2 normalization
            if self.index_type == "flat":
                self._index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self._index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, 100, faiss.METRIC_INNER_PRODUCT
                )
            elif self.index_type == "hnsw":
                self._index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            # L2 distance
            if self.index_type == "flat":
                self._index = faiss.IndexFlatL2(self.dimension)
            # ... other variants

    def add(self, snapshot_id: str, embedding: np.ndarray) -> None:
        """Add embedding to index."""
        if self._index is None:
            self._create_index()

        # L2 normalize for cosine similarity
        if self.metric == "cosine":
            embedding = embedding / np.linalg.norm(embedding)

        # Add to index
        self._index.add(embedding.reshape(1, -1).astype(np.float32))

        # Update maps
        idx = len(self._id_map)
        self._id_map.append(snapshot_id)
        self._reverse_map[snapshot_id] = idx

    def add_batch(
        self,
        snapshot_ids: List[str],
        embeddings: np.ndarray,
    ) -> None:
        """Add multiple embeddings at once."""
        if self._index is None:
            self._create_index()

        # Normalize
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        # Add all
        self._index.add(embeddings.astype(np.float32))

        # Update maps
        start_idx = len(self._id_map)
        for i, sid in enumerate(snapshot_ids):
            self._id_map.append(sid)
            self._reverse_map[sid] = start_idx + i

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Find k most similar snapshots.

        Returns:
            List of (snapshot_id, similarity_score) tuples
        """
        if self._index is None or len(self._id_map) == 0:
            return []

        # Normalize query
        if self.metric == "cosine":
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search (get extra results if we need to filter)
        search_k = k + len(exclude_ids or [])
        scores, indices = self._index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            min(search_k, len(self._id_map)),
        )

        results = []
        exclude_set = set(exclude_ids or [])

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue

            snapshot_id = self._id_map[idx]
            if snapshot_id in exclude_set:
                continue

            results.append((snapshot_id, float(score)))

            if len(results) >= k:
                break

        return results

    def remove(self, snapshot_id: str) -> bool:
        """Remove embedding from index. Returns True if found."""
        # Note: FAISS doesn't support efficient removal
        # For production, consider rebuilding index periodically
        if snapshot_id in self._reverse_map:
            del self._reverse_map[snapshot_id]
            # Mark as removed (actual cleanup on rebuild)
            return True
        return False

    def save(self, path: Path) -> None:
        """Persist index to disk."""
        import faiss

        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(path / "index.faiss"))

        # Save ID map
        with open(path / "id_map.json", "w") as f:
            json.dump({
                "id_map": self._id_map,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric,
            }, f)

    def load(self, path: Path) -> None:
        """Load index from disk."""
        import faiss

        self._index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "id_map.json") as f:
            data = json.load(f)
            self._id_map = data["id_map"]
            self._reverse_map = {sid: i for i, sid in enumerate(self._id_map)}
            self.dimension = data.get("dimension", self.dimension)
            self.index_type = data.get("index_type", self.index_type)
            self.metric = data.get("metric", self.metric)

    def __len__(self) -> int:
        return len(self._id_map)
```

### 5.3 Export Formats (`src/destill3d/export/`)

```python
# src/destill3d/export/__init__.py
from .base import ExportFormat, ExportResult, Exporter
from .numpy_export import NumpyExporter
from .hdf5_export import HDF5Exporter
from .tfrecord_export import TFRecordExporter
from .parquet_export import ParquetExporter

__all__ = [
    "ExportFormat", "ExportResult", "Exporter",
    "NumpyExporter", "HDF5Exporter", "TFRecordExporter", "ParquetExporter",
]
```

```python
# src/destill3d/export/base.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
from abc import ABC, abstractmethod

class ExportFormat(Enum):
    HDF5 = "hdf5"
    TFRECORD = "tfrecord"
    NUMPY = "numpy"
    PARQUET = "parquet"
    CSV = "csv"

@dataclass
class ExportResult:
    """Result of export operation."""
    format: ExportFormat
    output_path: Path
    num_samples: int
    file_size_bytes: int
    splits: Optional[Dict[str, int]] = None  # split_name -> count

@dataclass
class QueryFilters:
    """Filters for export query."""
    platform: Optional[str] = None
    label: Optional[str] = None
    min_confidence: Optional[float] = None
    taxonomy: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: Optional[int] = None

class Exporter(ABC):
    """Base class for dataset exporters."""

    @abstractmethod
    def export(
        self,
        db: "Database",
        output_path: Path,
        filters: Optional[QueryFilters] = None,
        include_views: bool = False,
        splits: Optional[Dict[str, float]] = None,
    ) -> ExportResult:
        """Export snapshots to format."""
        ...
```

```python
# src/destill3d/export/hdf5_export.py
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import h5py

from .base import Exporter, ExportFormat, ExportResult, QueryFilters

class HDF5Exporter(Exporter):
    """Export to HDF5 format for ML training."""

    def export(
        self,
        db: "Database",
        output_path: Path,
        filters: Optional[QueryFilters] = None,
        include_views: bool = False,
        splits: Optional[Dict[str, float]] = None,
    ) -> ExportResult:
        """
        Export to HDF5 with structure:

        /points          (N, 2048, 3)    float32
        /normals         (N, 2048, 3)    float32
        /labels          (N,)            int32
        /label_names     (C,)            string
        /snapshot_ids    (N,)            string
        /metadata        (N,)            JSON strings
        /embeddings      (N, 1024)       float32 (optional)
        /views           (N, 12, 224, 224) uint8 (optional)
        """
        # Query snapshots
        snapshots = db.query_snapshots(filters)

        if not snapshots:
            raise ValueError("No snapshots match the filters")

        # Build label mapping
        labels = sorted(set(s.predictions[0].label for s in snapshots if s.predictions))
        label_to_idx = {label: i for i, label in enumerate(labels)}

        # Create HDF5 file
        with h5py.File(output_path, 'w') as f:
            n = len(snapshots)
            point_count = snapshots[0].geometry.point_count

            # Create datasets
            points_ds = f.create_dataset(
                'points', (n, point_count, 3), dtype='float32',
                compression='gzip', compression_opts=4,
            )
            normals_ds = f.create_dataset(
                'normals', (n, point_count, 3), dtype='float32',
                compression='gzip', compression_opts=4,
            )
            labels_ds = f.create_dataset('labels', (n,), dtype='int32')

            # String datasets
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('label_names', data=labels)
            snapshot_ids_ds = f.create_dataset('snapshot_ids', (n,), dtype=dt)
            metadata_ds = f.create_dataset('metadata', (n,), dtype=dt)

            # Optional embeddings
            if any(s.embedding is not None for s in snapshots):
                embeddings_ds = f.create_dataset(
                    'embeddings', (n, 1024), dtype='float32',
                    compression='gzip',
                )

            # Write data
            for i, snapshot in enumerate(snapshots):
                points_ds[i] = snapshot.geometry.points
                normals_ds[i] = snapshot.geometry.normals

                if snapshot.predictions:
                    labels_ds[i] = label_to_idx.get(
                        snapshot.predictions[0].label, -1
                    )
                else:
                    labels_ds[i] = -1

                snapshot_ids_ds[i] = snapshot.snapshot_id
                metadata_ds[i] = snapshot.provenance.to_json()

                if snapshot.embedding is not None:
                    embeddings_ds[i] = snapshot.embedding

            # Handle splits if requested
            if splits:
                self._create_splits(f, n, splits)

        return ExportResult(
            format=ExportFormat.HDF5,
            output_path=output_path,
            num_samples=len(snapshots),
            file_size_bytes=output_path.stat().st_size,
            splits={k: int(v * len(snapshots)) for k, v in (splits or {}).items()},
        )

    def _create_splits(
        self,
        f: h5py.File,
        n: int,
        splits: Dict[str, float],
    ):
        """Create train/val/test split indices."""
        indices = np.random.permutation(n)

        offset = 0
        for name, ratio in splits.items():
            count = int(n * ratio)
            split_indices = indices[offset:offset + count]
            f.create_dataset(f'splits/{name}', data=split_indices)
            offset += count
```

### 5.4 Database Find Similar (`src/destill3d/core/database.py` addition)

```python
# Add to Database class

async def find_similar(
    self,
    snapshot_id: str,
    k: int = 10,
    min_similarity: float = 0.0,
    index: Optional["EmbeddingIndex"] = None,
) -> List[Tuple["Snapshot", float]]:
    """
    Find similar snapshots by embedding similarity.

    Args:
        snapshot_id: Source snapshot ID
        k: Number of similar items to return
        min_similarity: Minimum similarity threshold
        index: Optional pre-built FAISS index

    Returns:
        List of (snapshot, similarity) tuples
    """
    # Get source embedding
    source = await self.get_snapshot(snapshot_id)
    if source.embedding is None:
        raise ValueError(f"Snapshot {snapshot_id} has no embedding")

    if index is None:
        # Build index on the fly (slow for large DBs)
        index = await self._build_index()

    # Search
    results = index.search(
        source.embedding,
        k=k + 1,  # +1 to exclude self
        exclude_ids=[snapshot_id],
    )

    # Filter by similarity and fetch snapshots
    similar = []
    for sid, score in results:
        if score < min_similarity:
            continue
        snap = await self.get_snapshot(sid)
        similar.append((snap, score))

    return similar[:k]
```

---

## Phase 6: Multi-View Rendering

**Goal**: 12-view depth map generation for MVCNN hybrid classification
**Files**: 4 new, 2 modified
**Spec Sections**: 5.1 (Stage 6), 6.1 (view_images field)

### 6.1 View Renderer (`src/destill3d/extract/renderer.py`)

```python
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class ViewConfig:
    """Configuration for multi-view rendering."""
    num_views: int = 12
    resolution: int = 224
    elevation: float = 30.0  # degrees
    background: int = 255  # White background
    depth_range: Tuple[float, float] = (0.1, 10.0)

class MultiViewRenderer:
    """Renders multiple depth map views of a 3D model."""

    def __init__(self, config: ViewConfig = None):
        self.config = config or ViewConfig()
        self._renderer = None

    def _init_renderer(self):
        """Initialize the rendering backend."""
        if self._renderer is not None:
            return

        try:
            # Try pyrender first (GPU accelerated)
            import pyrender
            self._backend = "pyrender"
        except ImportError:
            try:
                # Fall back to trimesh's built-in
                import trimesh
                self._backend = "trimesh"
            except ImportError:
                raise ImportError(
                    "No rendering backend available. "
                    "Install pyrender or trimesh with rendering support."
                )

    def render_views(
        self,
        mesh: "trimesh.Trimesh",
        points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render multiple depth map views.

        Args:
            mesh: Input mesh (or use points if mesh unavailable)
            points: Point cloud fallback if no mesh

        Returns:
            (num_views, resolution, resolution) uint8 depth maps
        """
        self._init_renderer()

        views = []

        # Calculate camera positions around the object
        azimuth_step = 360.0 / self.config.num_views
        elevation_rad = math.radians(self.config.elevation)

        # Center and scale mesh
        if mesh is not None:
            mesh = mesh.copy()
            mesh.vertices -= mesh.centroid
            scale = 1.0 / mesh.bounding_box.extents.max()
            mesh.vertices *= scale

        for i in range(self.config.num_views):
            azimuth = i * azimuth_step

            depth_map = self._render_single_view(
                mesh, points, azimuth, self.config.elevation
            )
            views.append(depth_map)

        return np.array(views, dtype=np.uint8)

    def _render_single_view(
        self,
        mesh: "trimesh.Trimesh",
        points: Optional[np.ndarray],
        azimuth: float,
        elevation: float,
    ) -> np.ndarray:
        """Render a single depth map from given viewpoint."""
        if self._backend == "pyrender":
            return self._render_pyrender(mesh, azimuth, elevation)
        else:
            return self._render_trimesh(mesh, azimuth, elevation)

    def _render_pyrender(
        self,
        mesh: "trimesh.Trimesh",
        azimuth: float,
        elevation: float,
    ) -> np.ndarray:
        """Render using pyrender (GPU accelerated)."""
        import pyrender
        import trimesh

        # Create scene
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])

        # Add mesh
        py_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(py_mesh)

        # Calculate camera pose
        camera_pose = self._compute_camera_pose(azimuth, elevation, distance=3.0)

        # Add camera
        camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
        scene.add(camera, pose=camera_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(
            self.config.resolution,
            self.config.resolution,
        )

        _, depth = renderer.render(scene)
        renderer.delete()

        # Normalize depth to 0-255
        depth_normalized = self._normalize_depth(depth)

        return depth_normalized

    def _render_trimesh(
        self,
        mesh: "trimesh.Trimesh",
        azimuth: float,
        elevation: float,
    ) -> np.ndarray:
        """Render using trimesh (CPU fallback)."""
        import trimesh

        # Use trimesh's scene rendering
        scene = trimesh.Scene(mesh)

        # Set camera
        camera_pose = self._compute_camera_pose(azimuth, elevation, distance=3.0)
        scene.camera_transform = camera_pose

        # Render to image
        data = scene.save_image(resolution=(self.config.resolution,) * 2)

        # Convert to grayscale depth approximation
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(data)).convert('L')
        return np.array(img)

    def _compute_camera_pose(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
    ) -> np.ndarray:
        """Compute camera transformation matrix."""
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)

        # Camera position
        x = distance * math.cos(el_rad) * math.sin(az_rad)
        y = distance * math.cos(el_rad) * math.cos(az_rad)
        z = distance * math.sin(el_rad)

        # Look-at matrix
        eye = np.array([x, y, z])
        target = np.array([0, 0, 0])
        up = np.array([0, 0, 1])

        return self._look_at(eye, target, up)

    def _look_at(
        self,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray,
    ) -> np.ndarray:
        """Compute look-at transformation matrix."""
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up_new = np.cross(right, forward)

        matrix = np.eye(4)
        matrix[:3, 0] = right
        matrix[:3, 1] = up_new
        matrix[:3, 2] = -forward
        matrix[:3, 3] = eye

        return matrix

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth map to 0-255 range."""
        # Handle infinite/invalid depth
        valid_mask = np.isfinite(depth) & (depth > 0)

        if not valid_mask.any():
            return np.full(depth.shape, 255, dtype=np.uint8)

        min_depth = depth[valid_mask].min()
        max_depth = depth[valid_mask].max()

        if max_depth - min_depth < 1e-6:
            return np.full(depth.shape, 128, dtype=np.uint8)

        normalized = (depth - min_depth) / (max_depth - min_depth)
        normalized = np.clip(normalized * 255, 0, 255)
        normalized[~valid_mask] = 255  # Background

        return normalized.astype(np.uint8)
```

### 6.2 Integration with Extractor

```python
# Modify src/destill3d/extract/__init__.py

class FeatureExtractor:
    def __init__(
        self,
        config: ExtractionConfig = None,
        compute_views: bool = False,  # New parameter
        view_config: ViewConfig = None,
    ):
        self.config = config or ExtractionConfig()
        self.compute_views = compute_views
        self.view_config = view_config or ViewConfig()
        self._renderer = None

    def extract_from_file(self, file_path: Path, ...) -> Snapshot:
        # ... existing extraction code ...

        # Add view rendering if requested
        view_images = None
        if self.compute_views:
            if self._renderer is None:
                self._renderer = MultiViewRenderer(self.view_config)
            view_images = self._renderer.render_views(mesh)

        # Include in snapshot
        geometry = GeometryData(
            points=points,
            normals=normals,
            curvature=curvature,
            view_images=view_images,
            centroid=centroid,
            scale=scale,
        )

        # ...
```

---

## Phase 7: Security & Compliance

**Goal**: Input validation, credential management, license filtering
**Files**: 5 new, 3 modified
**Spec Sections**: 14.1-14.3

### 7.1 Input Validator (`src/destill3d/core/security.py`)

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Set, Optional
from urllib.parse import urlparse
import hashlib

@dataclass
class ValidationResult:
    """Result of validation check."""
    valid: bool
    error: Optional[str] = None
    warnings: list = None

class InputValidator:
    """Validate inputs to prevent security issues."""

    # File size limits
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

    # Allowed extensions (whitelist)
    ALLOWED_EXTENSIONS: Set[str] = {
        ".step", ".stp", ".iges", ".igs", ".brep", ".brp",
        ".stl", ".obj", ".ply", ".off", ".gltf", ".glb",
        ".3mf", ".dae", ".pcd", ".xyz", ".las", ".laz",
        ".fbx",
    }

    # Allowed domains for downloads
    ALLOWED_DOMAINS: Set[str] = {
        "www.thingiverse.com", "thingiverse.com",
        "api.thingiverse.com",
        "sketchfab.com", "api.sketchfab.com",
        "grabcad.com", "www.grabcad.com",
        "cults3d.com", "www.cults3d.com",
        "myminifactory.com", "www.myminifactory.com",
        "thangs.com", "www.thangs.com",
        "github.com", "raw.githubusercontent.com",
        "objects.githubusercontent.com",
    }

    def __init__(
        self,
        max_file_size: int = None,
        allowed_extensions: Set[str] = None,
        allowed_domains: Set[str] = None,
    ):
        self.max_file_size = max_file_size or self.MAX_FILE_SIZE
        self.allowed_extensions = allowed_extensions or self.ALLOWED_EXTENSIONS
        self.allowed_domains = allowed_domains or self.ALLOWED_DOMAINS

    def validate_file(self, path: Path) -> ValidationResult:
        """Validate file before processing."""
        warnings = []

        # Check path exists
        if not path.exists():
            return ValidationResult(valid=False, error="File does not exist")

        # Check for path traversal
        try:
            resolved = path.resolve()
            if ".." in str(path):
                return ValidationResult(valid=False, error="Invalid path (traversal detected)")
        except Exception:
            return ValidationResult(valid=False, error="Invalid path")

        # Check extension
        ext = path.suffix.lower()
        if ext not in self.allowed_extensions:
            return ValidationResult(
                valid=False,
                error=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(self.allowed_extensions))}"
            )

        # Check file size
        try:
            size = path.stat().st_size
            if size > self.max_file_size:
                return ValidationResult(
                    valid=False,
                    error=f"File too large: {size} bytes (max: {self.max_file_size})"
                )
            if size == 0:
                return ValidationResult(valid=False, error="File is empty")
        except OSError as e:
            return ValidationResult(valid=False, error=f"Cannot read file: {e}")

        # Warn about potentially problematic files
        if size > 100 * 1024 * 1024:  # 100 MB
            warnings.append("Large file may take longer to process")

        return ValidationResult(valid=True, warnings=warnings)

    def validate_url(self, url: str) -> ValidationResult:
        """Validate URL before download."""
        try:
            parsed = urlparse(url)
        except Exception:
            return ValidationResult(valid=False, error="Invalid URL format")

        # Must be HTTPS (allow HTTP for local dev)
        if parsed.scheme not in ("https", "http"):
            return ValidationResult(valid=False, error=f"Invalid scheme: {parsed.scheme}")

        # Check against allowed domains
        domain = parsed.netloc.lower()
        if domain not in self.allowed_domains:
            # Check if it's a subdomain of an allowed domain
            allowed = False
            for allowed_domain in self.allowed_domains:
                if domain.endswith("." + allowed_domain):
                    allowed = True
                    break

            if not allowed:
                return ValidationResult(
                    valid=False,
                    error=f"Domain not allowed: {domain}"
                )

        return ValidationResult(valid=True)

    def compute_file_hash(self, path: Path, algorithm: str = "sha256") -> str:
        """Compute hash of file for integrity verification."""
        h = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


class LicenseFilter:
    """Filter models by license for legal compliance."""

    # Licenses allowing redistribution of derived data
    PERMISSIVE_LICENSES: Set[str] = {
        "cc0", "cc-0", "public domain",
        "cc-by", "cc by",
        "cc-by-sa", "cc by-sa",
        "mit",
        "apache-2.0", "apache 2.0",
        "gpl-3.0", "gpl-3", "gplv3",
        "bsd-3-clause", "bsd-2-clause",
    }

    # Licenses prohibiting commercial use
    NON_COMMERCIAL: Set[str] = {
        "cc-by-nc", "cc by-nc",
        "cc-by-nc-sa", "cc by-nc-sa",
        "cc-by-nc-nd", "cc by-nc-nd",
    }

    # Licenses prohibiting derivatives
    NO_DERIVATIVES: Set[str] = {
        "cc-by-nd", "cc by-nd",
        "cc-by-nc-nd", "cc by-nc-nd",
    }

    def __init__(
        self,
        use_case: str = "research",  # research, commercial
        allow_derivatives: bool = True,
    ):
        self.use_case = use_case
        self.allow_derivatives = allow_derivatives

    def is_allowed(self, license_str: str) -> bool:
        """Check if license permits intended use."""
        if not license_str:
            return False  # Unknown license, be safe

        license_lower = license_str.lower().strip()

        # Check no-derivatives first
        if self.allow_derivatives:
            for nd in self.NO_DERIVATIVES:
                if nd in license_lower:
                    return False

        # Commercial use check
        if self.use_case == "commercial":
            for nc in self.NON_COMMERCIAL:
                if nc in license_lower:
                    return False

            # Must be explicitly permissive for commercial
            for perm in self.PERMISSIVE_LICENSES:
                if perm in license_lower:
                    return True
            return False

        # Research/personal use - more permissive
        all_allowed = self.PERMISSIVE_LICENSES | self.NON_COMMERCIAL
        for lic in all_allowed:
            if lic in license_lower:
                return True

        return False

    def filter_results(
        self,
        results: list,
        license_field: str = "license",
    ) -> list:
        """Filter search results by license."""
        return [
            r for r in results
            if self.is_allowed(getattr(r, license_field, None))
        ]
```

---

## Phase 8: Extended CLI & Testing

**Goal**: Complete CLI command tree, comprehensive test suite
**Files**: 10 new, 5 modified
**Spec Sections**: 9.1-9.3, 15.1-15.3

### 8.1 Models CLI (`src/destill3d/cli/commands/models.py`)

```python
import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Model management commands")
console = Console()

@app.command("list")
def list_models(
    taxonomy: Optional[str] = typer.Option(None, "-t", "--taxonomy"),
    downloaded_only: bool = typer.Option(False, "--downloaded"),
):
    """List available classification models."""
    from destill3d.classify.registry import MODEL_REGISTRY

    table = Table(title="Available Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Name")
    table.add_column("Taxonomy")
    table.add_column("Accuracy")
    table.add_column("Downloaded", style="green")

    for model_id, info in MODEL_REGISTRY.items():
        if taxonomy and info.taxonomy != taxonomy:
            continue

        downloaded = "✓" if info.weights_path and info.weights_path.exists() else "✗"

        if downloaded_only and downloaded == "✗":
            continue

        table.add_row(
            model_id,
            info.name,
            info.taxonomy,
            f"{info.modelnet40_accuracy:.1%}" if info.modelnet40_accuracy else "N/A",
            downloaded,
        )

    console.print(table)

@app.command("download")
def download_model(
    model_id: str = typer.Argument(..., help="Model ID to download"),
    force: bool = typer.Option(False, "-f", "--force", help="Re-download if exists"),
):
    """Download model weights."""
    from destill3d.classify.registry import MODEL_REGISTRY, download_model_weights

    if model_id not in MODEL_REGISTRY:
        console.print(f"[red]Unknown model: {model_id}[/red]")
        raise typer.Exit(1)

    info = MODEL_REGISTRY[model_id]

    if info.weights_path and info.weights_path.exists() and not force:
        console.print(f"[yellow]Model already downloaded: {info.weights_path}[/yellow]")
        return

    console.print(f"Downloading {info.name}...")

    with console.status("Downloading..."):
        path = download_model_weights(model_id)

    console.print(f"[green]Downloaded to: {path}[/green]")

@app.command("info")
def model_info(
    model_id: str = typer.Argument(..., help="Model ID"),
):
    """Show detailed model information."""
    from destill3d.classify.registry import MODEL_REGISTRY

    if model_id not in MODEL_REGISTRY:
        console.print(f"[red]Unknown model: {model_id}[/red]")
        raise typer.Exit(1)

    info = MODEL_REGISTRY[model_id]

    table = Table(title=f"Model: {model_id}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Name", info.name)
    table.add_row("Architecture", info.architecture)
    table.add_row("Taxonomy", info.taxonomy)
    table.add_row("Input Points", str(info.input_points))
    table.add_row("Requires Normals", "Yes" if info.requires_normals else "No")
    table.add_row("Format", info.format)
    table.add_row("Accuracy", f"{info.modelnet40_accuracy:.1%}" if info.modelnet40_accuracy else "N/A")
    table.add_row("Weights URL", info.weights_url)
    table.add_row("Local Path", str(info.weights_path) if info.weights_path else "Not downloaded")

    console.print(table)
```

### 8.2 Server CLI (`src/destill3d/cli/commands/server.py`)

```python
import typer
from typing import Optional

app = typer.Typer(help="REST API server commands")

@app.command("start")
def start_server(
    host: str = typer.Option("127.0.0.1", "-h", "--host"),
    port: int = typer.Option(8000, "-p", "--port"),
    workers: int = typer.Option(1, "-w", "--workers"),
    reload: bool = typer.Option(False, "--reload"),
):
    """Start the REST API server."""
    import uvicorn

    uvicorn.run(
        "destill3d.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )
```

### 8.3 Main CLI Update (`src/destill3d/cli/main.py`)

```python
import typer
from .commands import extract, classify, db, config, acquire, models, server

app = typer.Typer(
    name="destill3d",
    help="3D model feature extraction and classification toolkit",
    no_args_is_help=True,
)

# Register all command groups
app.add_typer(acquire.app, name="acquire")
app.add_typer(extract.app, name="extract")
app.add_typer(classify.app, name="classify")
app.add_typer(db.app, name="db")
app.add_typer(models.app, name="models")
app.add_typer(config.app, name="config")
app.add_typer(server.app, name="server")

# Quick commands at root level
@app.command("quick")
def quick_cmd(...):
    """Quick extract + classify shortcut."""
    ...

@app.command("info")
def info_cmd(
    file_path: Path = typer.Argument(...),
):
    """Show information about a 3D file."""
    ...

@app.command("version")
def version_cmd():
    """Show version information."""
    from destill3d import __version__
    print(f"destill3d {__version__}")
```

### 8.4 Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── unit/
│   ├── test_config.py
│   ├── test_exceptions.py
│   ├── test_snapshot.py
│   ├── test_sampling.py
│   ├── test_features.py
│   ├── test_loader.py
│   ├── test_tessellation.py
│   ├── test_registry.py
│   ├── test_inference.py
│   ├── test_zero_shot.py
│   ├── test_queue.py
│   ├── test_rate_limiter.py
│   ├── test_credentials.py
│   ├── test_pipeline.py
│   ├── test_vector_index.py
│   ├── test_security.py
│   └── test_renderer.py
├── integration/
│   ├── test_extraction_pipeline.py
│   ├── test_classification_pipeline.py
│   ├── test_database_operations.py
│   ├── test_platform_adapters.py
│   ├── test_export_formats.py
│   └── test_full_pipeline.py
├── e2e/
│   ├── test_cli_extract.py
│   ├── test_cli_classify.py
│   ├── test_cli_acquire.py
│   ├── test_cli_db.py
│   └── test_cli_workflow.py
├── benchmarks/
│   ├── bench_extraction.py
│   ├── bench_classification.py
│   ├── bench_database.py
│   └── bench_similarity.py
└── fixtures/
    ├── models/
    │   ├── cube.stl
    │   ├── sphere.obj
    │   ├── complex.step
    │   └── multipart.gltf
    └── snapshots/
        └── reference.d3d
```

### 8.5 Sample E2E Test (`tests/e2e/test_cli_workflow.py`)

```python
import pytest
from typer.testing import CliRunner
from pathlib import Path
import tempfile

from destill3d.cli.main import app

runner = CliRunner()

class TestCLIWorkflow:
    """End-to-end CLI workflow tests."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    def test_extract_classify_workflow(self, temp_dir, sample_stl):
        """Test extract → classify workflow."""
        snapshot_path = temp_dir / "output.d3d"

        # Extract
        result = runner.invoke(app, [
            "extract", "file", str(sample_stl),
            "-o", str(snapshot_path),
        ])
        assert result.exit_code == 0
        assert snapshot_path.exists()

        # Classify
        result = runner.invoke(app, [
            "classify", "snapshot", str(snapshot_path),
        ])
        assert result.exit_code == 0
        assert "confidence" in result.output.lower()

    def test_quick_command(self, temp_dir, sample_stl):
        """Test quick shortcut command."""
        result = runner.invoke(app, [
            "quick", str(sample_stl),
            "--classify",
        ])
        assert result.exit_code == 0

    def test_db_workflow(self, temp_dir, sample_stl):
        """Test database operations."""
        db_path = temp_dir / "test.db"

        # Extract with storage
        result = runner.invoke(app, [
            "extract", "file", str(sample_stl),
            "--store",
            "--db", str(db_path),
        ])
        assert result.exit_code == 0

        # Query
        result = runner.invoke(app, ["db", "stats", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "1" in result.output  # Should show 1 snapshot

        # Export
        export_path = temp_dir / "export.npz"
        result = runner.invoke(app, [
            "db", "export",
            "--format", "numpy",
            "-o", str(export_path),
            "--db", str(db_path),
        ])
        assert result.exit_code == 0
        assert export_path.exists()

class TestCLIAcquire:
    """Acquire command tests (requires mocked APIs)."""

    @pytest.mark.integration
    def test_search_command(self, mock_thingiverse_api):
        """Test search command with mocked API."""
        result = runner.invoke(app, [
            "acquire", "search",
            "chess piece",
            "-p", "thingiverse",
            "-l", "5",
        ])
        assert result.exit_code == 0
        assert "results" in result.output.lower()
```

### 8.6 Benchmark Suite (`tests/benchmarks/bench_extraction.py`)

```python
import pytest
import time
import numpy as np
from pathlib import Path

from destill3d.extract import FeatureExtractor
from destill3d.extract.sampling import sample_point_cloud, SamplingStrategy
from destill3d.extract.features import compute_normals, compute_curvature

class TestExtractionBenchmarks:
    """Extraction performance benchmarks."""

    @pytest.fixture
    def large_mesh(self):
        """Generate a large test mesh (~1M vertices)."""
        import trimesh
        # Create by subdividing a sphere
        mesh = trimesh.creation.icosphere(subdivisions=6)
        return mesh

    @pytest.mark.benchmark
    def test_sampling_performance(self, large_mesh, benchmark):
        """Benchmark point sampling."""
        def sample():
            return sample_point_cloud(
                large_mesh,
                n_points=2048,
                strategy=SamplingStrategy.HYBRID,
            )

        result = benchmark(sample)
        assert result.shape == (2048, 3)

    @pytest.mark.benchmark
    def test_normal_computation(self, benchmark):
        """Benchmark normal estimation."""
        points = np.random.randn(2048, 3).astype(np.float32)

        def compute():
            return compute_normals(points, k=30)

        result = benchmark(compute)
        assert result.shape == (2048, 3)

    @pytest.mark.benchmark
    def test_full_extraction(self, sample_stl, benchmark):
        """Benchmark full extraction pipeline."""
        extractor = FeatureExtractor()

        def extract():
            return extractor.extract_from_file(sample_stl)

        result = benchmark(extract)
        assert result.geometry.point_count == 2048
```

---

## Implementation Schedule

### Dependency Order

```
Phase 1 (Core Infrastructure)
    ↓
Phase 2 (Acquire) ←── Phase 3 (Pipeline)
    ↓                      ↓
Phase 4 (Zero-Shot)        │
    ↓                      │
Phase 5 (Storage) ←────────┘
    ↓
Phase 6 (Multi-View)
    ↓
Phase 7 (Security)
    ↓
Phase 8 (CLI & Testing)
```

### Estimated Scope per Phase

| Phase | New Files | Modified Files | Est. Lines | Priority |
|-------|-----------|----------------|------------|----------|
| 1. Core Infrastructure | 4 | 3 | 800 | Critical |
| 2. Acquire Module | 15 | 2 | 2,500 | High |
| 3. Pipeline System | 6 | 3 | 1,200 | High |
| 4. Zero-Shot | 5 | 2 | 800 | Medium |
| 5. Advanced Storage | 8 | 4 | 1,500 | Medium |
| 6. Multi-View | 4 | 2 | 600 | Low |
| 7. Security | 5 | 3 | 700 | Medium |
| 8. CLI & Testing | 10+ | 5 | 3,000+ | High |
| **Total** | **~57** | **~24** | **~11,100** | |

### Version Mapping

| Version | Phases | Key Deliverables |
|---------|--------|------------------|
| 0.2.0 | 1, 2, 3 | Acquire module, pipeline, Thingiverse+Sketchfab |
| 0.3.0 | 4 | Zero-shot classification, OpenShape |
| 0.4.0 | 5 | PostgreSQL, FAISS, HDF5 export |
| 0.5.0 | 6, 7 | Multi-view, security, more platforms |
| 1.0.0 | 8 | Full CLI, comprehensive tests, docs |

---

## Success Criteria

### Functional Requirements
- [ ] All CLI commands from spec implemented and working
- [ ] Platform adapters for Thingiverse, Sketchfab (P0)
- [ ] Zero-shot classification with arbitrary classes
- [ ] PostgreSQL backend operational
- [ ] FAISS similarity search < 50ms for 100K index
- [ ] HDF5/TFRecord export working
- [ ] Multi-view rendering produces valid depth maps

### Performance Requirements
- [ ] STEP tessellation < 5s (4-core CPU)
- [ ] Point sampling < 500ms (2048 points)
- [ ] Classification < 100ms (RTX 3090, single)
- [ ] Database insert < 10ms (NVMe SSD)

### Quality Requirements
- [ ] Unit test coverage ≥ 80%
- [ ] Integration test coverage ≥ 60%
- [ ] E2E tests for all CLI workflows
- [ ] All mypy type checks passing
- [ ] All ruff lint checks passing

---

*End of Implementation Plan*
