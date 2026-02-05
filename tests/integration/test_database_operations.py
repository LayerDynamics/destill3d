"""
Integration tests for database operations.
"""

from pathlib import Path

import numpy as np
import pytest

from destill3d.core.database import Database
from destill3d.core.snapshot import (
    Features,
    GeometryData,
    Prediction,
    ProcessingMetadata,
    Provenance,
    Snapshot,
)


def _make_snapshot(model_id: str, platform: str = "test", label: str = None) -> Snapshot:
    """Helper to create test snapshots."""
    np.random.seed(hash(model_id) % 2**32)
    points = np.random.randn(256, 3).astype(np.float32)

    snapshot = Snapshot(
        model_id=model_id,
        provenance=Provenance(
            platform=platform,
            source_id=model_id,
            title=f"Model {model_id}",
            original_filename=f"{model_id}.stl",
            original_format="stl_binary",
            original_file_size=1024,
        ),
        geometry=GeometryData(
            points=points,
            normals=np.random.randn(256, 3).astype(np.float32),
            curvature=np.random.rand(256).astype(np.float32),
            centroid=np.zeros(3, dtype=np.float32),
            scale=1.0,
        ),
        features=Features(
            global_features=np.random.randn(32).astype(np.float32),
            surface_area=10.0,
            volume=5.0,
            is_watertight=True,
            original_vertex_count=100,
            original_face_count=200,
            bbox_min=np.array([-1, -1, -1], dtype=np.float32),
            bbox_max=np.array([1, 1, 1], dtype=np.float32),
        ),
        embedding=np.random.randn(1024).astype(np.float32),
        processing=ProcessingMetadata(
            target_points=256,
            sampling_strategy="uniform",
        ),
    )

    if label:
        snapshot.predictions = [
            Prediction(
                label=label,
                confidence=0.95,
                taxonomy="modelnet40",
                model_name="test_model",
                rank=1,
            )
        ]

    return snapshot


@pytest.mark.integration
class TestDatabaseCRUD:
    """Test database CRUD operations."""

    def test_insert_and_get(self, db_path: Path):
        db = Database(db_path)
        snapshot = _make_snapshot("crud_test_1")
        db.insert(snapshot)

        retrieved = db.get(snapshot.model_id)
        assert retrieved is not None
        assert retrieved.model_id == snapshot.model_id

    def test_insert_multiple(self, db_path: Path):
        db = Database(db_path)

        for i in range(5):
            snapshot = _make_snapshot(f"multi_{i}")
            db.insert(snapshot)

        stats = db.get_stats()
        assert stats["total_snapshots"] == 5

    def test_query_by_platform(self, db_path: Path):
        db = Database(db_path)

        db.insert(_make_snapshot("plat_1", platform="thingiverse"))
        db.insert(_make_snapshot("plat_2", platform="sketchfab"))
        db.insert(_make_snapshot("plat_3", platform="thingiverse"))

        results = db.query(platform="thingiverse")
        assert len(results) == 2

    def test_query_by_label(self, db_path: Path):
        db = Database(db_path)

        db.insert(_make_snapshot("label_1", label="chair"))
        db.insert(_make_snapshot("label_2", label="table"))
        db.insert(_make_snapshot("label_3", label="chair"))

        results = db.query(label="chair")
        assert len(results) == 2

    def test_delete_snapshot(self, db_path: Path):
        db = Database(db_path)
        snapshot = _make_snapshot("delete_test")
        db.insert(snapshot)

        db.delete(snapshot.model_id)

        from destill3d.core.exceptions import SnapshotNotFoundError

        with pytest.raises(SnapshotNotFoundError):
            db.get(snapshot.model_id)


@pytest.mark.integration
class TestDatabaseExport:
    """Test database export operations."""

    def test_export_numpy(self, db_path: Path, temp_dir: Path):
        db = Database(db_path)

        for i in range(10):
            db.insert(_make_snapshot(f"export_{i}", label="chair"))

        export_path = temp_dir / "export.npz"
        result = db.export(export_path, format="numpy")

        assert export_path.exists()
        assert result is not None
