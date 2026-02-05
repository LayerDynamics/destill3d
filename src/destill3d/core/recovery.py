"""
Recovery system for Destill3D pipeline.

Handles recovery of interrupted processing on application startup,
cleanup of orphaned temp files, and retry of failed items.
"""

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from destill3d.core.pipeline import PipelineStage, ProcessingCheckpoint, get_previous_stage

logger = logging.getLogger(__name__)


class RecoveryManager:
    """Handles recovery of interrupted processing."""

    def __init__(self, db, temp_dir: Path = None):
        self.db = db
        self._temp_dir = temp_dir or Path("/tmp/destill3d/processing")

    async def recover_on_startup(
        self,
        stuck_threshold_hours: int = 1,
    ) -> int:
        """
        Recover items stuck in intermediate states.

        Called on application startup to handle items left in
        non-terminal states from a previous run.

        Args:
            stuck_threshold_hours: Hours after which an item is considered stuck.

        Returns:
            Number of items recovered.
        """
        cutoff = datetime.utcnow() - timedelta(hours=stuck_threshold_hours)

        # Find items stuck in non-terminal states
        intermediate_stages = [
            PipelineStage.ACQUIRED,
            PipelineStage.EXTRACTED,
            PipelineStage.CLASSIFIED,
        ]

        if not hasattr(self.db, "query_checkpoints"):
            logger.warning("Database does not support checkpoint queries")
            return 0

        stuck_items = await self.db.query_checkpoints(
            stages=intermediate_stages,
            updated_before=cutoff,
        )

        recovered = 0
        for item in stuck_items:
            if item.can_retry():
                previous = get_previous_stage(item.stage)
                item.stage = previous
                item.retry_count += 1
                item.last_error = "Recovered from stuck state on startup"
                item.updated_at = datetime.utcnow()

                await self.db.save_checkpoint(item)
                logger.info(
                    f"Recovered {item.model_id} from {item.stage.value}, "
                    f"retry {item.retry_count}/{item.max_retries}"
                )
                recovered += 1
            else:
                item.stage = PipelineStage.FAILED
                item.last_error = "Max retries exceeded during startup recovery"
                item.updated_at = datetime.utcnow()
                await self.db.save_checkpoint(item)
                logger.error(
                    f"Max retries exceeded for {item.model_id}, marking as failed"
                )

        if recovered:
            logger.info(f"Recovered {recovered} stuck items on startup")

        return recovered

    async def cleanup_temp_files(
        self,
        max_age_hours: int = 24,
    ) -> int:
        """
        Remove orphaned temp files older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours before cleanup.

        Returns:
            Number of directories cleaned up.
        """
        if not self._temp_dir.exists():
            return 0

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        cleaned = 0

        for item_dir in self._temp_dir.iterdir():
            if not item_dir.is_dir():
                continue

            # Check modification time
            try:
                mtime = datetime.fromtimestamp(item_dir.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(item_dir, ignore_errors=True)
                    logger.info(f"Cleaned up orphaned temp directory: {item_dir.name}")
                    cleaned += 1
            except OSError as e:
                logger.warning(f"Error checking temp directory {item_dir}: {e}")

        if cleaned:
            logger.info(f"Cleaned up {cleaned} orphaned temp directories")

        return cleaned

    async def retry_failed(
        self,
        max_age_hours: int = 48,
        reset_retry_count: bool = False,
    ) -> int:
        """
        Retry failed items within age limit.

        Args:
            max_age_hours: Only retry items failed within this many hours.
            reset_retry_count: If True, reset retry count for another round.

        Returns:
            Number of items queued for retry.
        """
        if not hasattr(self.db, "query_checkpoints"):
            return 0

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        failed_items = await self.db.query_checkpoints(
            stages=[PipelineStage.FAILED],
            updated_after=cutoff,
        )

        retried = 0
        for item in failed_items:
            if reset_retry_count:
                item.retry_count = 0

            if item.can_retry():
                # Reset to queued for full reprocessing
                item.stage = PipelineStage.QUEUED
                item.last_error = None
                item.error_stage = None
                item.updated_at = datetime.utcnow()

                await self.db.save_checkpoint(item)
                logger.info(f"Queued {item.model_id} for retry")
                retried += 1

        if retried:
            logger.info(f"Queued {retried} failed items for retry")

        return retried

    async def get_status_summary(self) -> dict:
        """Get a summary of items in each pipeline stage."""
        if not hasattr(self.db, "query_checkpoints"):
            return {}

        summary = {}
        for stage in PipelineStage:
            items = await self.db.query_checkpoints(stages=[stage])
            summary[stage.value] = len(items)

        return summary
