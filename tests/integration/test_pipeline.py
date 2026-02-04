"""
Integration tests for the full extraction pipeline.
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from destill3d.core.config import ExtractionConfig
from destill3d.core.database import Database
from destill3d.core.snapshot import Snapshot
from destill3d.extract import FeatureExtractor
from destill3d.extract.loader import FormatDetector, FileFormat, load_geometry


class TestFormatDetection:
    """Tests for format detection."""

    def test_detect_stl(self, sample_stl_file: Path):
        """Test STL format detection."""
        detector = FormatDetector()
        fmt = detector.detect(sample_stl_file)

        assert fmt in (FileFormat.STL_ASCII, FileFormat.STL_BINARY)

    def test_detect_obj(self, sample_obj_file: Path):
        """Test OBJ format detection."""
        detector = FormatDetector()
        fmt = detector.detect(sample_obj_file)

        assert fmt == FileFormat.OBJ

    def test_detect_ply(self, sample_ply_file: Path):
        """Test PLY format detection."""
        detector = FormatDetector()
        fmt = detector.detect(sample_ply_file)

        assert fmt == FileFormat.PLY


class TestGeometryLoading:
    """Tests for geometry loading."""

    def test_load_stl(self, sample_stl_file: Path):
        """Test loading STL file."""
        mesh = load_geometry(sample_stl_file)

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_load_obj(self, sample_obj_file: Path):
        """Test loading OBJ file."""
        mesh = load_geometry(sample_obj_file)

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0

    def test_load_ply(self, sample_ply_file: Path):
        """Test loading PLY file."""
        mesh = load_geometry(sample_ply_file)

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def test_extract_from_stl(self, sample_stl_file: Path, extraction_config: ExtractionConfig):
        """Test full extraction from STL file."""
        extractor = FeatureExtractor(extraction_config)
        snapshot = extractor.extract_from_file(sample_stl_file)

        assert isinstance(snapshot, Snapshot)
        assert snapshot.model_id.startswith("local:")
        assert snapshot.geometry is not None
        assert snapshot.geometry.point_count == extraction_config.target_points
        assert snapshot.features is not None

    def test_extract_from_obj(self, sample_obj_file: Path, extraction_config: ExtractionConfig):
        """Test extraction from OBJ file."""
        extractor = FeatureExtractor(extraction_config)
        snapshot = extractor.extract_from_file(sample_obj_file)

        assert snapshot.geometry.point_count == extraction_config.target_points
        assert snapshot.provenance.original_format == "obj"

    def test_extract_from_mesh(self, sample_mesh: trimesh.Trimesh, extraction_config: ExtractionConfig):
        """Test extraction from in-memory mesh."""
        extractor = FeatureExtractor(extraction_config)
        snapshot = extractor.extract_from_mesh(
            sample_mesh,
            model_id="test:inmemory",
            metadata={"title": "In-Memory Test"},
        )

        assert snapshot.model_id == "test:inmemory"
        assert snapshot.provenance.title == "In-Memory Test"

    def test_extract_with_metadata(self, sample_stl_file: Path, extraction_config: ExtractionConfig):
        """Test extraction with custom metadata."""
        extractor = FeatureExtractor(extraction_config)
        snapshot = extractor.extract_from_file(
            sample_stl_file,
            metadata={
                "title": "Custom Title",
                "author": "Test Author",
                "tags": ["test", "cube"],
            },
        )

        assert snapshot.provenance.title == "Custom Title"
        assert snapshot.provenance.author == "Test Author"
        assert "test" in snapshot.provenance.tags

    def test_extract_timing(self, sample_stl_file: Path, extraction_config: ExtractionConfig):
        """Test that extraction timing is recorded."""
        extractor = FeatureExtractor(extraction_config)
        snapshot = extractor.extract_from_file(sample_stl_file)

        assert snapshot.processing.extraction_time_ms > 0

    def test_extract_features_computed(self, sample_stl_file: Path, extraction_config: ExtractionConfig):
        """Test that all features are computed."""
        extractor = FeatureExtractor(extraction_config)
        snapshot = extractor.extract_from_file(sample_stl_file)

        # Check geometry
        assert snapshot.geometry.points is not None
        assert snapshot.geometry.normals is not None
        assert snapshot.geometry.curvature is not None
        assert snapshot.geometry.centroid is not None
        assert snapshot.geometry.scale > 0

        # Check features
        assert snapshot.features.global_features is not None
        assert len(snapshot.features.global_features) == 32
        assert snapshot.features.surface_area > 0


@pytest.mark.integration
class TestDatabaseIntegration:
    """Tests for database integration."""

    def test_insert_and_retrieve(
        self,
        sample_snapshot: Snapshot,
        db_path: Path,
    ):
        """Test inserting and retrieving snapshot."""
        db = Database(db_path)

        # Insert
        snapshot_id = db.insert(sample_snapshot)
        assert snapshot_id == sample_snapshot.snapshot_id

        # Retrieve
        retrieved = db.get(sample_snapshot.model_id)
        assert retrieved is not None
        assert retrieved.model_id == sample_snapshot.model_id

        np.testing.assert_array_almost_equal(
            retrieved.geometry.points,
            sample_snapshot.geometry.points,
        )

    def test_query_by_platform(self, sample_snapshot: Snapshot, db_path: Path):
        """Test querying by platform."""
        db = Database(db_path)
        db.insert(sample_snapshot)

        results = db.query(platform="test")
        assert len(results) == 1
        assert results[0].model_id == sample_snapshot.model_id

    def test_get_stats(self, sample_snapshot: Snapshot, db_path: Path):
        """Test getting database stats."""
        db = Database(db_path)
        db.insert(sample_snapshot)

        stats = db.get_stats()
        assert stats["total_snapshots"] == 1
        assert stats["unclassified_snapshots"] == 1

    def test_delete_snapshot(self, sample_snapshot: Snapshot, db_path: Path):
        """Test deleting snapshot."""
        db = Database(db_path)
        db.insert(sample_snapshot)

        # Delete
        db.delete(sample_snapshot.model_id)

        # Verify deleted
        from destill3d.core.exceptions import SnapshotNotFoundError
        with pytest.raises(SnapshotNotFoundError):
            db.get(sample_snapshot.model_id)


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(
        self,
        sample_stl_file: Path,
        temp_dir: Path,
        extraction_config: ExtractionConfig,
    ):
        """Test complete pipeline: extract -> save -> load -> database."""
        # Extract
        extractor = FeatureExtractor(extraction_config)
        snapshot = extractor.extract_from_file(sample_stl_file)

        # Save to file
        snapshot_path = temp_dir / "snapshot.d3d"
        snapshot.save(snapshot_path)

        # Load from file
        loaded = Snapshot.load(snapshot_path)

        # Store in database
        db = Database(temp_dir / "test.db")
        db.insert(loaded)

        # Query from database
        retrieved = db.get(loaded.model_id)

        # Verify integrity through entire pipeline
        assert retrieved.model_id == snapshot.model_id
        np.testing.assert_array_almost_equal(
            retrieved.geometry.points,
            snapshot.geometry.points,
        )

    def test_batch_extraction(
        self,
        temp_dir: Path,
        extraction_config: ExtractionConfig,
    ):
        """Test extracting multiple files."""
        # Create multiple test files with different content
        for i in range(3):
            # Create unique meshes by scaling
            mesh = trimesh.creation.box(extents=[1.0 + i * 0.1, 1.0, 1.0])
            path = temp_dir / f"mesh_{i}.stl"
            mesh.export(path)

        extractor = FeatureExtractor(extraction_config)

        # Extract all
        snapshots = []
        for path in sorted(temp_dir.glob("*.stl")):
            snapshot = extractor.extract_from_file(path)
            snapshots.append(snapshot)

        assert len(snapshots) == 3

        # All should have unique IDs (different content = different hash)
        ids = [s.model_id for s in snapshots]
        assert len(set(ids)) == 3
