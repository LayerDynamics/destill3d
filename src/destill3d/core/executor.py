"""
Pipeline executor for Destill3D.

Orchestrates the full acquisition-extraction-classification-storage pipeline
with progress tracking and error handling.
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

from destill3d.core.pipeline import (
    BatchConfig,
    PipelineStage,
    ProcessingCheckpoint,
    get_previous_stage,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineProgress:
    """Progress update during pipeline execution."""

    stage: PipelineStage
    completed: int
    total: int
    current_id: Optional[str] = None
    rate: float = 0.0  # items per minute
    errors: int = 0


class PipelineExecutor:
    """
    Executes the full acquisition-extraction-classification pipeline.

    Processes items through stages with checkpointing for resumability.
    """

    def __init__(
        self,
        db,
        extractor=None,
        classifier=None,
        platform_registry=None,
        config: Optional[BatchConfig] = None,
        temp_dir: Optional[Path] = None,
        snapshots_dir: Optional[Path] = None,
    ):
        self.db = db
        self.extractor = extractor
        self.classifier = classifier
        self.platforms = platform_registry
        self.config = config or BatchConfig()
        self._temp_dir = temp_dir or Path("/tmp/destill3d/processing")
        self._snapshots_dir = snapshots_dir or Path("/tmp/destill3d/snapshots")
        self._running = False

        # Ensure directories exist
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        start_stage: PipelineStage = PipelineStage.QUEUED,
        end_stage: PipelineStage = PipelineStage.STORED,
    ) -> AsyncIterator[PipelineProgress]:
        """
        Run pipeline from start_stage to end_stage.

        Yields progress updates as items are processed.
        """
        self._running = True
        start_time = datetime.utcnow()

        try:
            items = await self._get_items_for_processing(start_stage)
            total = len(items)
            completed = 0
            errors = 0

            for item in items:
                if not self._running:
                    logger.info("Pipeline stopped by user")
                    break

                try:
                    await self._process_item(item, end_stage)
                    completed += 1
                except Exception as e:
                    logger.error(f"Pipeline error for {item.model_id}: {e}")
                    await self._handle_error(item, str(e))
                    errors += 1

                elapsed = (datetime.utcnow() - start_time).total_seconds()
                rate = (completed / elapsed * 60) if elapsed > 0 else 0.0

                yield PipelineProgress(
                    stage=item.stage,
                    completed=completed,
                    total=total,
                    current_id=item.model_id,
                    rate=rate,
                    errors=errors,
                )

        finally:
            self._running = False

    async def _get_items_for_processing(
        self,
        start_stage: PipelineStage,
    ) -> list:
        """Get all items ready for processing at the given stage."""
        if hasattr(self.db, "query_checkpoints"):
            return await self.db.query_checkpoints(stages=[start_stage])
        return []

    async def _process_item(
        self,
        checkpoint: ProcessingCheckpoint,
        end_stage: PipelineStage,
    ) -> None:
        """Process a single item through pipeline stages."""
        stage_handlers = [
            (PipelineStage.QUEUED, PipelineStage.ACQUIRED, self._acquire),
            (PipelineStage.ACQUIRED, PipelineStage.EXTRACTED, self._extract),
            (PipelineStage.EXTRACTED, PipelineStage.CLASSIFIED, self._classify),
            (PipelineStage.CLASSIFIED, PipelineStage.STORED, self._store),
        ]

        for from_stage, to_stage, handler in stage_handlers:
            if checkpoint.stage == from_stage:
                await handler(checkpoint)
                checkpoint.advance_to(to_stage)

                if hasattr(self.db, "save_checkpoint"):
                    await self.db.save_checkpoint(checkpoint)

                if to_stage == end_stage:
                    break

    async def _acquire(self, checkpoint: ProcessingCheckpoint) -> None:
        """Download files for checkpoint."""
        if not self.platforms:
            raise RuntimeError("No platform registry configured for acquisition")

        adapter = self.platforms.get(checkpoint.platform)
        if not adapter:
            raise ValueError(f"No adapter for platform: {checkpoint.platform}")

        model_id = checkpoint.source_url.split(":")[-1] if ":" in checkpoint.source_url else checkpoint.model_id
        target_dir = self._temp_dir / checkpoint.model_id

        result = await adapter.download(model_id, target_dir=target_dir)

        checkpoint.temp_path = result.files[0].parent if result.files else target_dir
        if result.files:
            checkpoint.file_hash = self._compute_hash(result.files[0])

    async def _extract(self, checkpoint: ProcessingCheckpoint) -> None:
        """Extract features from downloaded files."""
        if not self.extractor:
            raise RuntimeError("No feature extractor configured")

        file_path = self._find_primary_file(checkpoint.temp_path)
        snapshot = self.extractor.extract_from_file(file_path)

        snapshot_path = self._snapshots_dir / f"{checkpoint.model_id}.d3d"
        snapshot.save(snapshot_path)

        checkpoint.snapshot_path = snapshot_path
        checkpoint.point_count = snapshot.geometry.point_count if snapshot.geometry else None

    async def _classify(self, checkpoint: ProcessingCheckpoint) -> None:
        """Classify the extracted snapshot."""
        if not self.classifier:
            raise RuntimeError("No classifier configured")

        from destill3d.core.snapshot import Snapshot
        snapshot = Snapshot.load(checkpoint.snapshot_path)

        predictions, embedding = self.classifier.classify(snapshot)

        checkpoint.predictions = predictions
        checkpoint.embedding = embedding

    async def _store(self, checkpoint: ProcessingCheckpoint) -> None:
        """Store snapshot in database and cleanup temp files."""
        from destill3d.core.snapshot import Snapshot
        snapshot = Snapshot.load(checkpoint.snapshot_path)

        if checkpoint.predictions:
            snapshot.predictions = checkpoint.predictions

        self.db.insert_snapshot(snapshot)

        if checkpoint.embedding is not None:
            self.db.update_embedding(
                snapshot.snapshot_id,
                "pipeline",
                checkpoint.embedding,
            )

        # Cleanup temp files
        if checkpoint.temp_path and checkpoint.temp_path.exists():
            shutil.rmtree(checkpoint.temp_path, ignore_errors=True)

    async def _handle_error(
        self,
        checkpoint: ProcessingCheckpoint,
        error: str,
    ) -> None:
        """Handle pipeline error with retry logic."""
        checkpoint.increment_retry(error)

        if checkpoint.can_retry():
            checkpoint.stage = get_previous_stage(checkpoint.stage)
        else:
            checkpoint.stage = PipelineStage.FAILED

        if hasattr(self.db, "save_checkpoint"):
            await self.db.save_checkpoint(checkpoint)

    def stop(self) -> None:
        """Stop pipeline execution after current item completes."""
        self._running = False

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        import hashlib
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _find_primary_file(self, directory: Path) -> Path:
        """Find the primary 3D file in a directory."""
        if directory is None:
            raise ValueError("No temp directory set for checkpoint")

        if directory.is_file():
            return directory

        # Priority order for 3D file formats
        priority_extensions = [
            ".step", ".stp", ".stl", ".obj", ".ply",
            ".gltf", ".glb", ".iges", ".igs", ".brep",
            ".3mf", ".dae", ".fbx", ".off",
        ]

        for ext in priority_extensions:
            files = list(directory.glob(f"*{ext}"))
            if files:
                # Return the largest file of this type
                return max(files, key=lambda f: f.stat().st_size)

        # Fallback: any file
        all_files = [f for f in directory.iterdir() if f.is_file()]
        if all_files:
            return max(all_files, key=lambda f: f.stat().st_size)

        raise FileNotFoundError(f"No 3D files found in {directory}")
