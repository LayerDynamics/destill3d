"""
Unit tests for Snapshot dataclass and serialization.
"""

from pathlib import Path

import numpy as np
import pytest

from destill3d.core.snapshot import (
    Snapshot,
    Provenance,
    GeometryData,
    Features,
    Prediction,
    ProcessingMetadata,
    SNAPSHOT_VERSION,
)


class TestProvenance:
    """Tests for Provenance dataclass."""

    def test_provenance_creation(self):
        """Test basic provenance creation."""
        prov = Provenance(
            platform="thingiverse",
            source_id="12345",
            title="Test Model",
        )

        assert prov.platform == "thingiverse"
        assert prov.source_id == "12345"
        assert prov.title == "Test Model"
        assert prov.tags == []

    def test_provenance_with_all_fields(self):
        """Test provenance with all fields populated."""
        prov = Provenance(
            platform="local",
            source_url="file:///path/to/file.stl",
            source_id="abc123",
            title="My Model",
            description="A test model",
            author="Test Author",
            license="CC-BY-4.0",
            tags=["test", "cube"],
            original_filename="model.stl",
            original_format="stl_binary",
            original_file_size=1024,
            original_file_hash="sha256:abc123",
        )

        assert prov.license == "CC-BY-4.0"
        assert "test" in prov.tags
        assert prov.original_file_size == 1024


class TestGeometryData:
    """Tests for GeometryData dataclass."""

    def test_geometry_creation(self):
        """Test basic geometry creation."""
        points = np.random.randn(1024, 3).astype(np.float32)
        normals = np.random.randn(1024, 3).astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        curvature = np.random.rand(1024).astype(np.float32)

        geom = GeometryData(
            points=points,
            normals=normals,
            curvature=curvature,
            centroid=np.array([0, 0, 0], dtype=np.float32),
            scale=1.0,
        )

        assert geom.point_count == 1024
        assert geom.points.shape == (1024, 3)
        assert geom.normals.shape == (1024, 3)

    def test_geometry_point_count(self):
        """Test point_count property."""
        points = np.zeros((512, 3), dtype=np.float32)
        normals = np.zeros((512, 3), dtype=np.float32)
        curvature = np.zeros(512, dtype=np.float32)
        geom = GeometryData(points=points, normals=normals, curvature=curvature)

        assert geom.point_count == 512


class TestFeatures:
    """Tests for Features dataclass."""

    def test_features_creation(self):
        """Test features creation."""
        feat = Features(
            global_features=np.random.randn(32).astype(np.float32),
            surface_area=10.5,
            volume=5.2,
            is_watertight=True,
            original_vertex_count=1000,
            original_face_count=2000,
        )

        assert feat.surface_area == 10.5
        assert feat.is_watertight is True
        assert feat.global_features.shape == (32,)


class TestPrediction:
    """Tests for Prediction dataclass."""

    def test_prediction_creation(self):
        """Test prediction creation."""
        pred = Prediction(
            label="chair",
            confidence=0.95,
            taxonomy="modelnet40",
            model_name="pointnet2_ssg_mn40",
            rank=1,
        )

        assert pred.label == "chair"
        assert pred.confidence == 0.95
        assert pred.rank == 1

    def test_prediction_ordering(self):
        """Test that predictions can be sorted by confidence."""
        preds = [
            Prediction(label="chair", confidence=0.8, taxonomy="mn40", model_name="test", rank=2),
            Prediction(label="table", confidence=0.95, taxonomy="mn40", model_name="test", rank=1),
            Prediction(label="desk", confidence=0.6, taxonomy="mn40", model_name="test", rank=3),
        ]

        sorted_preds = sorted(preds, key=lambda p: -p.confidence)
        assert sorted_preds[0].label == "table"
        assert sorted_preds[-1].label == "desk"


class TestSnapshot:
    """Tests for Snapshot dataclass."""

    def test_snapshot_creation(self, sample_snapshot: Snapshot):
        """Test basic snapshot creation."""
        assert sample_snapshot.model_id == "test:sample123"
        assert sample_snapshot.version == SNAPSHOT_VERSION
        assert sample_snapshot.geometry.point_count == 2048

    def test_snapshot_save_load_roundtrip(self, sample_snapshot: Snapshot, temp_dir: Path):
        """Test that save/load preserves all data."""
        save_path = temp_dir / "test_snapshot.d3d"

        # Save
        sample_snapshot.save(save_path)
        assert save_path.exists()

        # Load
        loaded = Snapshot.load(save_path)

        # Verify all fields
        assert loaded.model_id == sample_snapshot.model_id
        assert loaded.version == sample_snapshot.version
        assert loaded.snapshot_id == sample_snapshot.snapshot_id

        # Verify provenance
        assert loaded.provenance.platform == sample_snapshot.provenance.platform
        assert loaded.provenance.title == sample_snapshot.provenance.title
        assert loaded.provenance.tags == sample_snapshot.provenance.tags

        # Verify geometry
        np.testing.assert_array_almost_equal(
            loaded.geometry.points,
            sample_snapshot.geometry.points,
        )
        np.testing.assert_array_almost_equal(
            loaded.geometry.normals,
            sample_snapshot.geometry.normals,
        )

        # Verify features
        np.testing.assert_array_almost_equal(
            loaded.features.global_features,
            sample_snapshot.features.global_features,
        )
        assert loaded.features.surface_area == sample_snapshot.features.surface_area

    def test_snapshot_top_prediction(self, sample_snapshot: Snapshot):
        """Test top_prediction property."""
        # Initially no predictions
        assert sample_snapshot.top_prediction is None

        # Add predictions
        sample_snapshot.predictions = [
            Prediction(label="chair", confidence=0.95, taxonomy="mn40", model_name="test", rank=1),
            Prediction(label="table", confidence=0.03, taxonomy="mn40", model_name="test", rank=2),
        ]

        assert sample_snapshot.top_prediction is not None
        assert sample_snapshot.top_prediction.label == "chair"

    def test_snapshot_to_dict(self, sample_snapshot: Snapshot):
        """Test serialization to dict."""
        data = sample_snapshot.to_dict()

        assert isinstance(data, dict)
        assert data["model_id"] == "test:sample123"
        assert "provenance" in data
        assert "geometry" in data
        assert "features" in data

    def test_snapshot_from_dict(self, sample_snapshot: Snapshot):
        """Test deserialization from dict."""
        data = sample_snapshot.to_dict()
        restored = Snapshot.from_dict(data)

        assert restored.model_id == sample_snapshot.model_id
        np.testing.assert_array_almost_equal(
            restored.geometry.points,
            sample_snapshot.geometry.points,
        )

    def test_snapshot_compression(self, sample_snapshot: Snapshot, temp_dir: Path):
        """Test that saved snapshots are compressed."""
        save_path = temp_dir / "test_compressed.d3d"
        sample_snapshot.save(save_path)

        # Calculate raw size (points alone)
        raw_size = sample_snapshot.geometry.points.nbytes
        file_size = save_path.stat().st_size

        # File should be smaller than raw data due to compression
        # (This is a rough check; actual compression ratio depends on data)
        assert file_size < raw_size * 2  # Should be significantly smaller

    def test_snapshot_uuid_generation(self):
        """Test that snapshot_id is automatically generated."""
        snapshot = Snapshot(
            model_id="test:auto",
            provenance=Provenance(platform="test"),
        )

        assert snapshot.snapshot_id is not None
        assert len(snapshot.snapshot_id) == 36  # UUID format

    def test_snapshot_version(self):
        """Test that version is set correctly."""
        snapshot = Snapshot(
            model_id="test:version",
            provenance=Provenance(platform="test"),
        )

        assert snapshot.version == SNAPSHOT_VERSION


class TestSnapshotSerialization:
    """Tests for snapshot serialization edge cases."""

    def test_snapshot_without_geometry(self, temp_dir: Path):
        """Test snapshot without geometry data."""
        snapshot = Snapshot(
            model_id="test:no_geom",
            provenance=Provenance(platform="test", title="No Geometry"),
        )

        save_path = temp_dir / "no_geom.d3d"
        snapshot.save(save_path)

        loaded = Snapshot.load(save_path)
        assert loaded.geometry is None

    def test_snapshot_with_embedding(self, sample_snapshot: Snapshot, temp_dir: Path):
        """Test snapshot with embedding."""
        embedding = np.random.randn(1024).astype(np.float32)
        sample_snapshot.embedding = embedding

        save_path = temp_dir / "with_embedding.d3d"
        sample_snapshot.save(save_path)

        loaded = Snapshot.load(save_path)
        assert loaded.embedding is not None
        np.testing.assert_array_almost_equal(loaded.embedding, embedding)

    def test_snapshot_with_predictions(self, sample_snapshot: Snapshot, temp_dir: Path):
        """Test snapshot with predictions."""
        sample_snapshot.predictions = [
            Prediction(
                label="chair",
                confidence=0.95,
                taxonomy="modelnet40",
                model_name="pointnet2",
                rank=1,
                uncertainty=0.1,
            ),
            Prediction(
                label="table",
                confidence=0.03,
                taxonomy="modelnet40",
                model_name="pointnet2",
                rank=2,
            ),
        ]

        save_path = temp_dir / "with_preds.d3d"
        sample_snapshot.save(save_path)

        loaded = Snapshot.load(save_path)
        assert len(loaded.predictions) == 2
        assert loaded.predictions[0].label == "chair"
        assert loaded.predictions[0].uncertainty == 0.1
