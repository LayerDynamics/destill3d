"""Unit tests for the Database class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from destill3d.core.database import Database
from destill3d.core.exceptions import DuplicateSnapshotError, ExportError, SnapshotNotFoundError
from destill3d.core.snapshot import (
    Features, GeometryData, Prediction, ProcessingMetadata, Provenance, Snapshot,
)


@pytest.fixture
def db(temp_dir):
    return Database(temp_dir / "test.db")


def _make(model_id, platform="test", label="chair"):
    np.random.seed(hash(model_id) % 2**31)
    n = 32
    s = Snapshot(
        model_id=model_id,
        provenance=Provenance(
            platform=platform, source_id=model_id.split(":")[-1],
            title=f"Snap {model_id}", tags=["auto"],
            original_format="stl_binary", original_file_size=512,
        ),
        geometry=GeometryData(
            points=np.random.randn(n, 3).astype(np.float32),
            normals=np.random.randn(n, 3).astype(np.float32),
            curvature=np.random.rand(n).astype(np.float32),
        ),
        features=Features(
            global_features=np.random.randn(32).astype(np.float32),
            surface_area=6.0, volume=1.0, is_watertight=False,
        ),
        predictions=[
            Prediction(label=label, confidence=0.85, taxonomy="modelnet40",
                       model_name="pointnet2_ssg", rank=1),
        ],
        processing=ProcessingMetadata(),
    )
    return s


class TestDatabaseInit:
    def test_creates_file(self, temp_dir):
        p = temp_dir / "new.db"
        Database(p)
        assert p.exists()

    def test_engine_created(self, db):
        assert db.engine is not None


class TestInsertSnapshot:
    def test_returns_id(self, db, sample_snapshot):
        sid = db.insert(sample_snapshot)
        assert sid == sample_snapshot.snapshot_id

    def test_insert_and_retrieve(self, db, sample_snapshot):
        db.insert(sample_snapshot)
        r = db.get(sample_snapshot.model_id)
        assert r is not None
        assert r.model_id == sample_snapshot.model_id

    def test_duplicate_raises(self, db, sample_snapshot):
        db.insert(sample_snapshot)
        with pytest.raises(DuplicateSnapshotError):
            db.insert(sample_snapshot)


class TestQuerySnapshots:
    def test_by_platform(self, db):
        db.insert(_make("a:1", platform="alpha"))
        db.insert(_make("b:2", platform="beta"))
        r = db.query(platform="alpha")
        assert len(r) == 1

    def test_by_label(self, db):
        db.insert(_make("a:1", label="chair"))
        db.insert(_make("b:2", label="desk"))
        r = db.query(label="chair")
        assert len(r) == 1

    def test_with_limit(self, db):
        for i in range(5):
            db.insert(_make(f"p:{i}"))
        assert len(db.query(limit=3)) == 3


class TestDeleteSnapshot:
    def test_delete(self, db, sample_snapshot):
        db.insert(sample_snapshot)
        db.delete(sample_snapshot.model_id)
        with pytest.raises(SnapshotNotFoundError):
            db.get(sample_snapshot.model_id)


class TestStats:
    def test_empty(self, db):
        s = db.get_stats()
        assert s["total_snapshots"] == 0

    def test_after_inserts(self, db):
        db.insert(_make("s:1"))
        db.insert(_make("s:2"))
        s = db.get_stats()
        assert s["total_snapshots"] == 2


class TestExport:
    def test_numpy(self, db, temp_dir, sample_snapshot):
        db.insert(sample_snapshot)
        result = db.export(temp_dir / "out.npz", format="numpy")
        assert result is not None

    def test_tfrecord(self, db, temp_dir, sample_snapshot):
        db.insert(sample_snapshot)
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            pytest.skip("tensorflow not installed")
        result = db.export(temp_dir / "export.tfrecord", format="tfrecord")
        assert result.exists()
        assert result.stat().st_size > 0

    def test_tfrecord_no_tensorflow(self, db, temp_dir, sample_snapshot):
        """TFRecord export raises ExportError when tensorflow is missing."""
        import sys
        db.insert(sample_snapshot)

        # Temporarily hide tensorflow
        tf_modules = {k: v for k, v in sys.modules.items() if k.startswith("tensorflow")}
        for k in tf_modules:
            sys.modules[k] = None  # type: ignore

        try:
            with pytest.raises(ExportError):
                db.export(temp_dir / "export.tfrecord", format="tfrecord")
        finally:
            # Restore
            for k, v in tf_modules.items():
                sys.modules[k] = v

    def test_bad_format(self, db, temp_dir, sample_snapshot):
        db.insert(sample_snapshot)
        with pytest.raises(ExportError):
            db.export(temp_dir / "out.bad", format="xml")
