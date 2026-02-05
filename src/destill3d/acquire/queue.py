"""
Download queue management for batch acquisition.

Provides persistent queue with priority ordering,
retry logic, and concurrent processing.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Callable, List, Optional

logger = logging.getLogger(__name__)


class QueueStatus(Enum):
    """Status of a queue entry."""

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
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None  # For delayed retry


@dataclass
class ProcessingProgress:
    """Progress update during queue processing."""

    completed: int
    total: int
    current: Optional[QueueEntry] = None
    rate: float = 0.0  # items per minute
    eta_seconds: Optional[float] = None


class DownloadQueue:
    """Manages the download queue with persistence."""

    def __init__(self, db):
        self.db = db
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._entries: List[QueueEntry] = []
        self._next_id = 1

    async def add(
        self,
        url: str,
        priority: int = 0,
        platform: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> QueueEntry:
        """Add URL to download queue."""
        entry = QueueEntry(
            queue_id=self._next_id,
            platform=platform or "",
            model_id=model_id or url,
            source_url=url,
            status=QueueStatus.PENDING,
            priority=priority,
        )
        self._next_id += 1
        self._entries.append(entry)

        if hasattr(self.db, "save_queue_entry"):
            await self.db.save_queue_entry(entry)

        logger.info(f"Added to queue: {url} (priority={priority})")
        return entry

    async def add_batch(
        self,
        urls: List[str],
        priority: int = 0,
    ) -> List[QueueEntry]:
        """Add multiple URLs to queue."""
        entries = []
        for url in urls:
            entry = await self.add(url, priority=priority)
            entries.append(entry)
        return entries

    async def get_pending(
        self,
        limit: int = 100,
    ) -> List[QueueEntry]:
        """Get pending entries ordered by priority (descending)."""
        pending = [
            e for e in self._entries
            if e.status == QueueStatus.PENDING
        ]
        pending.sort(key=lambda e: e.priority, reverse=True)
        return pending[:limit]

    async def update_status(
        self,
        queue_id: int,
        status: QueueStatus,
        error: Optional[str] = None,
    ) -> None:
        """Update entry status."""
        for entry in self._entries:
            if entry.queue_id == queue_id:
                entry.status = status
                entry.updated_at = datetime.utcnow()
                if error:
                    entry.last_error = error
                    entry.retry_count += 1

                if hasattr(self.db, "save_queue_entry"):
                    await self.db.save_queue_entry(entry)
                break

    async def process(
        self,
        concurrency: int = 4,
        callback: Optional[Callable] = None,
    ) -> AsyncIterator[ProcessingProgress]:
        """
        Process queue with progress updates.

        Args:
            concurrency: Maximum concurrent downloads.
            callback: Optional callback for each completed item.

        Yields:
            ProcessingProgress updates.
        """
        self._running = True
        pending = await self.get_pending()
        total = len(pending)
        completed = 0
        start_time = datetime.utcnow()

        semaphore = asyncio.Semaphore(concurrency)

        for entry in pending:
            if not self._running:
                break

            async with semaphore:
                await self.update_status(entry.queue_id, QueueStatus.DOWNLOADING)

                try:
                    if callback:
                        await callback(entry)
                    await self.update_status(entry.queue_id, QueueStatus.COMPLETED)
                    completed += 1
                except Exception as e:
                    await self.update_status(
                        entry.queue_id, QueueStatus.FAILED, str(e)
                    )

                elapsed = (datetime.utcnow() - start_time).total_seconds()
                rate = (completed / elapsed * 60) if elapsed > 0 else 0.0
                remaining = total - completed
                eta = (remaining / rate * 60) if rate > 0 else None

                yield ProcessingProgress(
                    completed=completed,
                    total=total,
                    current=entry,
                    rate=rate,
                    eta_seconds=eta,
                )

        self._running = False

    async def retry_failed(
        self,
        max_age_hours: int = 24,
    ) -> int:
        """Retry failed entries within age limit."""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        retried = 0

        for entry in self._entries:
            if (
                entry.status == QueueStatus.FAILED
                and entry.retry_count < entry.max_retries
                and entry.updated_at >= cutoff
            ):
                entry.status = QueueStatus.PENDING
                entry.updated_at = datetime.utcnow()
                retried += 1

        if retried:
            logger.info(f"Queued {retried} failed entries for retry")

        return retried

    def stop(self) -> None:
        """Stop queue processing."""
        self._running = False
