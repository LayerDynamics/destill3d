"""Unit tests for pipeline module."""

import pytest

from destill3d.core.pipeline import (
    BatchConfig,
    PipelineStage,
    ProcessingCheckpoint,
    STAGE_ORDER,
    get_next_stage,
    get_previous_stage,
)


class TestPipelineStage:
    def test_enum_values(self):
        assert PipelineStage.QUEUED.value == "queued"
        assert PipelineStage.ACQUIRED.value == "acquired"
        assert PipelineStage.EXTRACTED.value == "extracted"
        assert PipelineStage.CLASSIFIED.value == "classified"
        assert PipelineStage.STORED.value == "stored"
        assert PipelineStage.FAILED.value == "failed"

    def test_stage_order(self):
        assert STAGE_ORDER == [
            PipelineStage.QUEUED,
            PipelineStage.ACQUIRED,
            PipelineStage.EXTRACTED,
            PipelineStage.CLASSIFIED,
            PipelineStage.STORED,
        ]

    def test_get_next_stage(self):
        assert get_next_stage(PipelineStage.QUEUED) == PipelineStage.ACQUIRED
        assert get_next_stage(PipelineStage.ACQUIRED) == PipelineStage.EXTRACTED
        assert get_next_stage(PipelineStage.EXTRACTED) == PipelineStage.CLASSIFIED
        assert get_next_stage(PipelineStage.CLASSIFIED) == PipelineStage.STORED
        assert get_next_stage(PipelineStage.STORED) is None

    def test_get_previous_stage(self):
        assert get_previous_stage(PipelineStage.STORED) == PipelineStage.CLASSIFIED
        assert get_previous_stage(PipelineStage.QUEUED) == PipelineStage.QUEUED  # First stage returns self

    def test_failed_has_no_next(self):
        assert get_next_stage(PipelineStage.FAILED) is None


class TestProcessingCheckpoint:
    def test_creation(self):
        cp = ProcessingCheckpoint(
            model_id="test:123",
            stage=PipelineStage.QUEUED,
            source_url="https://example.com/model",
        )
        assert cp.model_id == "test:123"
        assert cp.stage == PipelineStage.QUEUED

    def test_defaults(self):
        cp = ProcessingCheckpoint(model_id="test:1", stage=PipelineStage.QUEUED, source_url="url")
        assert cp.stage == PipelineStage.QUEUED
        assert cp.source_url == "url"
        assert cp.temp_path is None
        assert cp.snapshot_path is None


class TestBatchConfig:
    def test_defaults(self):
        bc = BatchConfig()
        assert bc.download_batch_size > 0
        assert bc.extraction_batch_size > 0
        assert bc.classification_batch_size > 0
        assert bc.storage_batch_size > 0
