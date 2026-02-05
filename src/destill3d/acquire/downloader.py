"""
Async download utilities with retry and hash verification.

Provides robust file downloading with progress tracking,
integrity verification, and automatic retry on failure.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from destill3d.core.exceptions import DownloadError
from destill3d.core.retry import download_retry

logger = logging.getLogger(__name__)


@download_retry()
async def download_file(
    url: str,
    target_path: Path,
    expected_hash: Optional[str] = None,
    timeout: float = 300.0,
    chunk_size: int = 8192,
) -> Path:
    """
    Download a file with retry logic and optional hash verification.

    Args:
        url: URL to download from.
        target_path: Local path to save the file.
        expected_hash: Optional SHA256 hash to verify after download.
        timeout: Download timeout in seconds.
        chunk_size: Size of download chunks in bytes.

    Returns:
        Path to the downloaded file.

    Raises:
        DownloadError: If download fails or hash verification fails.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(target_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)

    except httpx.HTTPStatusError as e:
        raise DownloadError(url, f"HTTP {e.response.status_code}")
    except httpx.RequestError as e:
        raise DownloadError(url, str(e))

    elapsed_ms = (time.monotonic() - start) * 1000
    file_size = target_path.stat().st_size

    logger.info(
        f"Downloaded {target_path.name} "
        f"({file_size / 1024:.1f} KB in {elapsed_ms:.0f}ms)"
    )

    # Verify hash if provided
    if expected_hash:
        actual_hash = compute_file_hash(target_path)
        if actual_hash != expected_hash:
            target_path.unlink(missing_ok=True)
            raise DownloadError(
                url,
                f"Hash mismatch: expected {expected_hash[:16]}..., "
                f"got {actual_hash[:16]}...",
            )

    return target_path


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file.

    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm (default: sha256).

    Returns:
        Hex digest of the file hash.
    """
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
