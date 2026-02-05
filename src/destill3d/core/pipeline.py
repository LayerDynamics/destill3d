"""
Pipeline stages and checkpoint system for Destill3D.

Implements the four-stage processing pipeline with checkpoint
capability for resumability per SPEC Section 3.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

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
    """
    Checkpoint record for resumable processing.

    Each model maintains a checkpoint enabling resumption
    at any pipeline stage.
    """

    model_id: str                        # Unique identifier (platform:id)
    stage: PipelineStage
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Acquisition checkpoint
    source_url: str = ""
    platform: str = ""
    temp_path: Optional[Path] = None
    file_hash: Optional[str] = None      # SHA256 of downloaded file

    # Extraction checkpoint
    snapshot_path: Optional[Path] = None
    point_count: Optional[int] = None
    feature_version: Optional[str] = None

    # Classification checkpoint
    predictions: Optional[list] = None
    embedding: Optional[np.ndarray] = None  # For similarity search

    # Error tracking
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    error_stage: Optional[PipelineStage] = None

    def can_retry(self) -> bool:
        """Check if this checkpoint can be retried."""
        return self.retry_count < self.max_retries

    def increment_retry(self, error: str) -> None:
        """Record a retry attempt with the error that caused it."""
        self.retry_count += 1
        self.last_error = error
        self.error_stage = self.stage
        self.updated_at = datetime.utcnow()

    def advance_to(self, stage: PipelineStage) -> None:
        """Advance the checkpoint to the next stage."""
        self.stage = stage
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict:
        """Serialize checkpoint to dictionary."""
        return {
            "model_id": self.model_id,
            "stage": self.stage.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_url": self.source_url,
            "platform": self.platform,
            "temp_path": str(self.temp_path) if self.temp_path else None,
            "file_hash": self.file_hash,
            "snapshot_path": str(self.snapshot_path) if self.snapshot_path else None,
            "point_count": self.point_count,
            "feature_version": self.feature_version,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_error": self.last_error,
            "error_stage": self.error_stage.value if self.error_stage else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingCheckpoint":
        """Deserialize checkpoint from dictionary."""
        return cls(
            model_id=data["model_id"],
            stage=PipelineStage(data["stage"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            source_url=data.get("source_url", ""),
            platform=data.get("platform", ""),
            temp_path=Path(data["temp_path"]) if data.get("temp_path") else None,
            file_hash=data.get("file_hash"),
            snapshot_path=Path(data["snapshot_path"]) if data.get("snapshot_path") else None,
            point_count=data.get("point_count"),
            feature_version=data.get("feature_version"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            last_error=data.get("last_error"),
            error_stage=PipelineStage(data["error_stage"]) if data.get("error_stage") else None,
        )


@dataclass
class BatchConfig:
    """Batch processing configuration for the pipeline."""

    download_batch_size: int = 10       # Concurrent downloads
    extraction_batch_size: int = 1      # Sequential (memory-bound)
    classification_batch_size: int = 32  # GPU batch size
    storage_batch_size: int = 100       # Database transaction size


# Stage ordering for transition logic
STAGE_ORDER: List[PipelineStage] = [
    PipelineStage.QUEUED,
    PipelineStage.ACQUIRED,
    PipelineStage.EXTRACTED,
    PipelineStage.CLASSIFIED,
    PipelineStage.STORED,
]


def get_previous_stage(stage: PipelineStage) -> PipelineStage:
    """Get the stage before the given one (for retry reset)."""
    if stage == PipelineStage.FAILED:
        return PipelineStage.QUEUED
    try:
        idx = STAGE_ORDER.index(stage)
        return STAGE_ORDER[max(0, idx - 1)]
    except ValueError:
        return PipelineStage.QUEUED


def get_next_stage(stage: PipelineStage) -> Optional[PipelineStage]:
    """Get the stage after the given one."""
    if stage == PipelineStage.FAILED:
        return None
    try:
        idx = STAGE_ORDER.index(stage)
        if idx + 1 < len(STAGE_ORDER):
            return STAGE_ORDER[idx + 1]
        return None
    except ValueError:
        return None
