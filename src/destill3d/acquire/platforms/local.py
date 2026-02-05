"""
Local filesystem adapter for direct file input.

Allows treating local files and directories as a "platform"
for consistent pipeline processing.
"""

import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from destill3d.acquire.base import (
    DownloadResult,
    FileInfo,
    ModelMetadata,
    RateLimit,
    SearchFilters,
    SearchResult,
    SearchResults,
)

logger = logging.getLogger(__name__)

# Supported 3D file extensions
SUPPORTED_EXTENSIONS = {
    ".stl", ".obj", ".ply", ".gltf", ".glb",
    ".step", ".stp", ".iges", ".igs", ".brep",
    ".3mf", ".dae", ".fbx", ".off",
}


class LocalFilesystemAdapter:
    """Local filesystem adapter for direct file input."""

    def __init__(self):
        pass

    @property
    def platform_id(self) -> str:
        return "local"

    @property
    def rate_limit(self) -> RateLimit:
        return RateLimit(requests=1000, period_seconds=1)  # Effectively unlimited

    async def search(
        self,
        query: str,
        filters: SearchFilters,
        page: int = 1,
    ) -> SearchResults:
        """Search local directory for 3D files matching query."""
        search_path = Path(query)
        if not search_path.exists():
            return SearchResults(results=[], total_count=0, page=1, has_more=False)

        results: List[SearchResult] = []

        if search_path.is_file():
            if search_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                results.append(self._file_to_result(search_path))
        elif search_path.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                for f in search_path.rglob(f"*{ext}"):
                    results.append(self._file_to_result(f))

        # Apply format filter
        if filters.format:
            allowed = {f".{fmt.lstrip('.')}" for fmt in filters.format}
            results = [r for r in results if Path(r.url).suffix.lower() in allowed]

        # Paginate
        per_page = 20
        start = (page - 1) * per_page
        page_results = results[start:start + per_page]

        return SearchResults(
            results=page_results,
            total_count=len(results),
            page=page,
            has_more=start + per_page < len(results),
        )

    async def get_metadata(self, model_id: str) -> ModelMetadata:
        """Get metadata for a local file."""
        file_path = Path(model_id)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {model_id}")

        stat = file_path.stat()

        return ModelMetadata(
            platform=self.platform_id,
            model_id=model_id,
            title=file_path.stem,
            description=f"Local file: {file_path}",
            author="local",
            license="unknown",
            tags=[],
            files=[
                FileInfo(
                    filename=file_path.name,
                    url=str(file_path.absolute()),
                    size_bytes=stat.st_size,
                    format=file_path.suffix.lower(),
                )
            ],
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
        )

    async def download(
        self,
        model_id: str,
        target_dir: Path,
    ) -> DownloadResult:
        """Copy local file to target directory."""
        start = time.monotonic()

        source = Path(model_id)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {model_id}")

        metadata = await self.get_metadata(model_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files: List[Path] = []

        if source.is_file():
            target_path = target_dir / source.name
            shutil.copy2(source, target_path)
            downloaded_files.append(target_path)
        elif source.is_dir():
            for f in source.iterdir():
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    target_path = target_dir / f.name
                    shutil.copy2(f, target_path)
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
        """Parse local file path (file:// URL or absolute path)."""
        if url.startswith("file://"):
            return url[7:]

        path = Path(url)
        if path.is_absolute() and path.exists():
            return str(path)

        return None

    def _file_to_result(self, file_path: Path) -> SearchResult:
        """Convert a file path to a SearchResult."""
        stat = file_path.stat()
        return SearchResult(
            platform=self.platform_id,
            model_id=str(file_path.absolute()),
            title=file_path.stem,
            author="local",
            url=str(file_path.absolute()),
            created_at=datetime.fromtimestamp(stat.st_ctime),
        )
