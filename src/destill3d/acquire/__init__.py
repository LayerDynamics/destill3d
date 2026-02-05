"""
Acquire module for Destill3D.

Provides platform adapters and utilities for automated
acquisition of 3D models from various hosting platforms.
"""

from destill3d.acquire.base import (
    DownloadResult,
    FileInfo,
    ModelMetadata,
    PlatformAdapter,
    RateLimit,
    SearchFilters,
    SearchResult,
    SearchResults,
)
from destill3d.acquire.credentials import CredentialManager
from destill3d.acquire.downloader import compute_file_hash, download_file
from destill3d.acquire.models import PlatformRegistry
from destill3d.acquire.queue import DownloadQueue, ProcessingProgress, QueueEntry, QueueStatus
from destill3d.acquire.rate_limiter import RateLimiter, TokenBucket

__all__ = [
    # Protocol
    "PlatformAdapter",
    # Data models
    "RateLimit",
    "SearchFilters",
    "SearchResult",
    "SearchResults",
    "ModelMetadata",
    "FileInfo",
    "DownloadResult",
    # Queue
    "DownloadQueue",
    "QueueEntry",
    "QueueStatus",
    "ProcessingProgress",
    # Utilities
    "PlatformRegistry",
    "RateLimiter",
    "TokenBucket",
    "CredentialManager",
    "download_file",
    "compute_file_hash",
]
