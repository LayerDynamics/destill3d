"""
Database layer for Destill3D using SQLAlchemy.

Provides SQLite storage with optional PostgreSQL support.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import gzip
import json

import numpy as np
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    LargeBinary,
    DateTime,
    ForeignKey,
    Index,
    Text,
    func,
    and_,
    or_,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

from destill3d.core.snapshot import Snapshot, Provenance, GeometryData, Features, Prediction, ProcessingMetadata
from destill3d.core.exceptions import (
    DatabaseError,
    SnapshotNotFoundError,
    DuplicateSnapshotError,
    ExportError,
)

Base = declarative_base()


# ─────────────────────────────────────────────────────────────────────────────
# SQLAlchemy Models
# ─────────────────────────────────────────────────────────────────────────────


class SnapshotModel(Base):
    """SQLAlchemy model for snapshots table."""

    __tablename__ = "snapshots"

    snapshot_id = Column(String(36), primary_key=True)
    model_id = Column(String(255), unique=True, nullable=False, index=True)

    # Provenance (denormalized for querying)
    platform = Column(String(50), nullable=False, default="local", index=True)
    source_url = Column(Text)
    source_id = Column(String(255))
    title = Column(String(500))
    author = Column(String(255))
    license = Column(String(100))
    original_format = Column(String(20))
    original_file_size = Column(Integer)
    original_file_hash = Column(String(64))

    # Snapshot data (compressed binary)
    snapshot_data = Column(LargeBinary, nullable=False)
    snapshot_version = Column(Integer, nullable=False)

    # Quick-access features (denormalized for filtering)
    point_count = Column(Integer)
    is_watertight = Column(Boolean)
    surface_area = Column(Float)
    volume = Column(Float)

    # Timestamps
    source_created_at = Column(DateTime)
    acquired_at = Column(DateTime, nullable=False)
    processed_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    tags = relationship("SnapshotTagModel", back_populates="snapshot", cascade="all, delete-orphan")
    classifications = relationship("ClassificationModel", back_populates="snapshot", cascade="all, delete-orphan")
    embedding_record = relationship("EmbeddingModel", back_populates="snapshot", uselist=False, cascade="all, delete-orphan")


class TagModel(Base):
    """SQLAlchemy model for tags table."""

    __tablename__ = "tags"

    tag_id = Column(Integer, primary_key=True, autoincrement=True)
    tag_name = Column(String(100), unique=True, nullable=False, index=True)


class SnapshotTagModel(Base):
    """SQLAlchemy model for snapshot_tags junction table."""

    __tablename__ = "snapshot_tags"

    snapshot_id = Column(String(36), ForeignKey("snapshots.snapshot_id", ondelete="CASCADE"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.tag_id", ondelete="CASCADE"), primary_key=True)
    source = Column(String(20), nullable=False, primary_key=True)  # 'platform' or 'predicted'
    confidence = Column(Float)  # NULL for platform tags

    snapshot = relationship("SnapshotModel", back_populates="tags")
    tag = relationship("TagModel")


class ClassificationModel(Base):
    """SQLAlchemy model for classifications table."""

    __tablename__ = "classifications"

    classification_id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(String(36), ForeignKey("snapshots.snapshot_id", ondelete="CASCADE"), nullable=False)

    model_id = Column(String(100), nullable=False)
    taxonomy = Column(String(50), nullable=False)
    label = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)
    uncertainty = Column(Float)
    classified_at = Column(DateTime, nullable=False)

    snapshot = relationship("SnapshotModel", back_populates="classifications")

    __table_args__ = (
        Index("idx_classifications_snapshot_model", "snapshot_id", "model_id"),
        Index("idx_classifications_label_confidence", "label", "confidence"),
    )


class EmbeddingModel(Base):
    """SQLAlchemy model for embeddings table."""

    __tablename__ = "embeddings"

    snapshot_id = Column(String(36), ForeignKey("snapshots.snapshot_id", ondelete="CASCADE"), primary_key=True)
    model_id = Column(String(100), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # float32 array as bytes
    embedding_dim = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    snapshot = relationship("SnapshotModel", back_populates="embedding_record")


class ProcessingQueueModel(Base):
    """SQLAlchemy model for processing_queue table (for future use)."""

    __tablename__ = "processing_queue"

    queue_id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(255), unique=True, nullable=False)
    platform = Column(String(50), nullable=False)
    source_url = Column(Text, nullable=False)

    stage = Column(String(20), nullable=False)  # QUEUED, ACQUIRED, EXTRACTED, CLASSIFIED
    priority = Column(Integer, default=0)

    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    last_error = Column(Text)

    temp_path = Column(String(500))

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    scheduled_at = Column(DateTime)

    __table_args__ = (
        Index("idx_queue_stage_priority", "stage", "priority"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Database Class
# ─────────────────────────────────────────────────────────────────────────────


class Database:
    """
    Database interface for Destill3D.

    Handles snapshot storage, querying, and export operations.
    """

    def __init__(self, db_path: Path, echo: bool = False):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            echo: Whether to echo SQL statements (for debugging)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=echo,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def _get_session(self) -> Session:
        """Get a new database session."""
        return self.Session()

    def insert_snapshot(self, snapshot: Snapshot) -> str:
        """
        Insert a new snapshot into the database.

        Args:
            snapshot: Snapshot to insert

        Returns:
            The snapshot_id of the inserted snapshot

        Raises:
            DuplicateSnapshotError: If a snapshot with the same model_id exists
        """
        session = self._get_session()
        try:
            # Check for duplicate
            existing = session.query(SnapshotModel).filter_by(model_id=snapshot.model_id).first()
            if existing:
                raise DuplicateSnapshotError(snapshot.model_id)

            # Serialize snapshot to compressed bytes
            snapshot_bytes = gzip.compress(
                json.dumps(snapshot.to_dict()).encode("utf-8"),
                compresslevel=6,
            )

            # Create model
            model = SnapshotModel(
                snapshot_id=snapshot.snapshot_id,
                model_id=snapshot.model_id,
                platform=snapshot.provenance.platform,
                source_url=snapshot.provenance.source_url,
                source_id=snapshot.provenance.source_id,
                title=snapshot.provenance.title,
                author=snapshot.provenance.author,
                license=snapshot.provenance.license,
                original_format=snapshot.provenance.original_format,
                original_file_size=snapshot.provenance.original_file_size,
                original_file_hash=snapshot.provenance.original_file_hash,
                snapshot_data=snapshot_bytes,
                snapshot_version=snapshot.version,
                point_count=snapshot.geometry.point_count if snapshot.geometry else 0,
                is_watertight=snapshot.features.is_watertight if snapshot.features else False,
                surface_area=snapshot.features.surface_area if snapshot.features else 0.0,
                volume=snapshot.features.volume if snapshot.features else 0.0,
                source_created_at=snapshot.provenance.source_created_at,
                acquired_at=snapshot.provenance.acquired_at,
                processed_at=snapshot.processing.processed_at,
            )

            session.add(model)

            # Add tags
            for tag_name in snapshot.provenance.tags:
                tag = session.query(TagModel).filter_by(tag_name=tag_name).first()
                if not tag:
                    tag = TagModel(tag_name=tag_name)
                    session.add(tag)
                    session.flush()

                snapshot_tag = SnapshotTagModel(
                    snapshot_id=snapshot.snapshot_id,
                    tag_id=tag.tag_id,
                    source="platform",
                    confidence=None,
                )
                session.add(snapshot_tag)

            # Add classifications
            for pred in snapshot.predictions:
                classification = ClassificationModel(
                    snapshot_id=snapshot.snapshot_id,
                    model_id=pred.model_name,
                    taxonomy=pred.taxonomy,
                    label=pred.label,
                    confidence=pred.confidence,
                    rank=pred.rank,
                    uncertainty=pred.uncertainty,
                    classified_at=snapshot.processing.processed_at,
                )
                session.add(classification)

            # Add embedding if present
            if snapshot.embedding is not None:
                embedding_model = EmbeddingModel(
                    snapshot_id=snapshot.snapshot_id,
                    model_id=snapshot.predictions[0].model_name if snapshot.predictions else "unknown",
                    embedding=snapshot.embedding.tobytes(),
                    embedding_dim=len(snapshot.embedding),
                )
                session.add(embedding_model)

            session.commit()
            return snapshot.snapshot_id

        except Exception as e:
            session.rollback()
            if isinstance(e, DuplicateSnapshotError):
                raise
            raise DatabaseError(f"Failed to insert snapshot: {e}")
        finally:
            session.close()

    def get_snapshot(self, snapshot_id: str) -> Optional[Snapshot]:
        """
        Get a snapshot by ID.

        Args:
            snapshot_id: UUID of the snapshot

        Returns:
            The Snapshot or None if not found
        """
        session = self._get_session()
        try:
            model = session.query(SnapshotModel).filter_by(snapshot_id=snapshot_id).first()
            if not model:
                return None

            return self._model_to_snapshot(model)
        finally:
            session.close()

    def get_snapshot_by_model_id(self, model_id: str) -> Optional[Snapshot]:
        """Get a snapshot by model_id (platform:source_id)."""
        session = self._get_session()
        try:
            model = session.query(SnapshotModel).filter_by(model_id=model_id).first()
            if not model:
                return None
            return self._model_to_snapshot(model)
        finally:
            session.close()

    def _model_to_snapshot(self, model: SnapshotModel) -> Snapshot:
        """Convert SQLAlchemy model to Snapshot."""
        data = json.loads(gzip.decompress(model.snapshot_data).decode("utf-8"))
        return Snapshot.from_dict(data)

    def query_snapshots(
        self,
        platform: Optional[str] = None,
        label: Optional[str] = None,
        min_confidence: Optional[float] = None,
        is_watertight: Optional[bool] = None,
        original_format: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Snapshot]:
        """
        Query snapshots with filters.

        Args:
            platform: Filter by source platform
            label: Filter by classification label
            min_confidence: Minimum classification confidence
            is_watertight: Filter by watertight status
            original_format: Filter by original file format
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            List of matching Snapshots
        """
        session = self._get_session()
        try:
            query = session.query(SnapshotModel)

            if platform:
                query = query.filter(SnapshotModel.platform == platform)

            if is_watertight is not None:
                query = query.filter(SnapshotModel.is_watertight == is_watertight)

            if original_format:
                query = query.filter(SnapshotModel.original_format == original_format)

            if label:
                # Join with classifications
                query = query.join(ClassificationModel).filter(
                    ClassificationModel.label == label
                )
                if min_confidence:
                    query = query.filter(ClassificationModel.confidence >= min_confidence)

            query = query.order_by(SnapshotModel.created_at.desc())
            query = query.offset(offset).limit(limit)

            return [self._model_to_snapshot(m) for m in query.all()]
        finally:
            session.close()

    def get_unclassified_snapshots(self, limit: Optional[int] = None) -> List[Snapshot]:
        """Get snapshots that haven't been classified yet."""
        session = self._get_session()
        try:
            query = session.query(SnapshotModel).outerjoin(ClassificationModel).filter(
                ClassificationModel.classification_id == None
            )
            if limit:
                query = query.limit(limit)
            return [self._model_to_snapshot(m) for m in query.all()]
        finally:
            session.close()

    def update_classifications(
        self,
        snapshot_id: str,
        predictions: List[Prediction],
    ) -> None:
        """Update classifications for a snapshot."""
        session = self._get_session()
        try:
            # Remove existing classifications for this model
            if predictions:
                model_name = predictions[0].model_name
                session.query(ClassificationModel).filter(
                    and_(
                        ClassificationModel.snapshot_id == snapshot_id,
                        ClassificationModel.model_id == model_name,
                    )
                ).delete()

            # Add new classifications
            for pred in predictions:
                classification = ClassificationModel(
                    snapshot_id=snapshot_id,
                    model_id=pred.model_name,
                    taxonomy=pred.taxonomy,
                    label=pred.label,
                    confidence=pred.confidence,
                    rank=pred.rank,
                    uncertainty=pred.uncertainty,
                    classified_at=datetime.utcnow(),
                )
                session.add(classification)

            session.commit()
        except Exception as e:
            session.rollback()
            raise DatabaseError(f"Failed to update classifications: {e}")
        finally:
            session.close()

    def update_embedding(
        self,
        snapshot_id: str,
        model_id: str,
        embedding: np.ndarray,
    ) -> None:
        """Update or insert embedding for a snapshot."""
        session = self._get_session()
        try:
            existing = session.query(EmbeddingModel).filter_by(snapshot_id=snapshot_id).first()
            if existing:
                existing.model_id = model_id
                existing.embedding = embedding.astype(np.float32).tobytes()
                existing.embedding_dim = len(embedding)
            else:
                emb_model = EmbeddingModel(
                    snapshot_id=snapshot_id,
                    model_id=model_id,
                    embedding=embedding.astype(np.float32).tobytes(),
                    embedding_dim=len(embedding),
                )
                session.add(emb_model)
            session.commit()
        except Exception as e:
            session.rollback()
            raise DatabaseError(f"Failed to update embedding: {e}")
        finally:
            session.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        session = self._get_session()
        try:
            total_count = session.query(func.count(SnapshotModel.snapshot_id)).scalar()

            # Count by platform
            by_platform = dict(
                session.query(
                    SnapshotModel.platform,
                    func.count(SnapshotModel.snapshot_id),
                ).group_by(SnapshotModel.platform).all()
            )

            # Count by format
            by_format = dict(
                session.query(
                    SnapshotModel.original_format,
                    func.count(SnapshotModel.snapshot_id),
                ).group_by(SnapshotModel.original_format).all()
            )

            # Count classified
            classified_count = session.query(
                func.count(func.distinct(ClassificationModel.snapshot_id))
            ).scalar()

            # Watertight count
            watertight_count = session.query(
                func.count(SnapshotModel.snapshot_id)
            ).filter(SnapshotModel.is_watertight == True).scalar()

            return {
                "total_count": total_count or 0,
                "by_platform": by_platform,
                "by_format": by_format,
                "classified_count": classified_count or 0,
                "watertight_count": watertight_count or 0,
            }
        finally:
            session.close()

    def export(
        self,
        output_path: Path,
        format: str = "hdf5",
        taxonomy: Optional[str] = None,
        min_confidence: Optional[float] = None,
        split: Optional[str] = None,  # e.g., "0.8:0.1:0.1" for train:val:test
    ) -> int:
        """
        Export snapshots to ML-ready format.

        Args:
            output_path: Output file path
            format: Export format (hdf5, numpy, parquet)
            taxonomy: Filter by taxonomy
            min_confidence: Minimum classification confidence
            split: Train/val/test split ratios

        Returns:
            Number of snapshots exported
        """
        output_path = Path(output_path)

        # Get all classified snapshots
        session = self._get_session()
        try:
            query = session.query(SnapshotModel).join(ClassificationModel)

            if taxonomy:
                query = query.filter(ClassificationModel.taxonomy == taxonomy)
            if min_confidence:
                query = query.filter(ClassificationModel.confidence >= min_confidence)

            # Only get top predictions (rank=1)
            query = query.filter(ClassificationModel.rank == 1)

            models = query.all()
            if not models:
                return 0

            snapshots = [self._model_to_snapshot(m) for m in models]

            if format == "hdf5":
                return self._export_hdf5(snapshots, output_path)
            elif format == "numpy":
                return self._export_numpy(snapshots, output_path)
            elif format == "parquet":
                return self._export_parquet(snapshots, output_path)
            else:
                raise ExportError(format, f"Unsupported format: {format}")

        finally:
            session.close()

    def _export_hdf5(self, snapshots: List[Snapshot], output_path: Path) -> int:
        """Export to HDF5 format."""
        try:
            import h5py
        except ImportError:
            raise ExportError("hdf5", "h5py not installed. Install with: pip install h5py")

        n = len(snapshots)
        point_count = snapshots[0].geometry.point_count if snapshots[0].geometry else 2048

        with h5py.File(output_path, "w") as f:
            # Create datasets
            points = f.create_dataset("points", shape=(n, point_count, 3), dtype="float32")
            normals = f.create_dataset("normals", shape=(n, point_count, 3), dtype="float32")

            # Get unique labels
            all_labels = list(set(s.top_prediction.label for s in snapshots if s.top_prediction))
            all_labels.sort()
            label_to_idx = {label: i for i, label in enumerate(all_labels)}

            labels = f.create_dataset("labels", shape=(n,), dtype="int32")
            label_names = f.create_dataset(
                "label_names",
                shape=(len(all_labels),),
                dtype=h5py.special_dtype(vlen=str),
            )
            label_names[:] = all_labels

            snapshot_ids = f.create_dataset(
                "snapshot_ids",
                shape=(n,),
                dtype=h5py.special_dtype(vlen=str),
            )

            for i, snapshot in enumerate(snapshots):
                if snapshot.geometry:
                    points[i] = snapshot.geometry.points
                    normals[i] = snapshot.geometry.normals
                if snapshot.top_prediction:
                    labels[i] = label_to_idx.get(snapshot.top_prediction.label, -1)
                snapshot_ids[i] = snapshot.snapshot_id

        return n

    def _export_numpy(self, snapshots: List[Snapshot], output_path: Path) -> int:
        """Export to NumPy .npz format."""
        n = len(snapshots)
        point_count = snapshots[0].geometry.point_count if snapshots[0].geometry else 2048

        points = np.zeros((n, point_count, 3), dtype=np.float32)
        normals = np.zeros((n, point_count, 3), dtype=np.float32)

        all_labels = list(set(s.top_prediction.label for s in snapshots if s.top_prediction))
        all_labels.sort()
        label_to_idx = {label: i for i, label in enumerate(all_labels)}

        labels = np.zeros(n, dtype=np.int32)
        snapshot_ids = []

        for i, snapshot in enumerate(snapshots):
            if snapshot.geometry:
                points[i] = snapshot.geometry.points
                normals[i] = snapshot.geometry.normals
            if snapshot.top_prediction:
                labels[i] = label_to_idx.get(snapshot.top_prediction.label, -1)
            snapshot_ids.append(snapshot.snapshot_id)

        np.savez_compressed(
            output_path,
            points=points,
            normals=normals,
            labels=labels,
            label_names=np.array(all_labels),
            snapshot_ids=np.array(snapshot_ids),
        )

        return n

    def _export_parquet(self, snapshots: List[Snapshot], output_path: Path) -> int:
        """Export metadata to Parquet format (no geometry)."""
        try:
            import pandas as pd
        except ImportError:
            raise ExportError("parquet", "pandas not installed. Install with: pip install pandas pyarrow")

        records = []
        for s in snapshots:
            record = {
                "snapshot_id": s.snapshot_id,
                "model_id": s.model_id,
                "platform": s.provenance.platform,
                "title": s.provenance.title,
                "original_format": s.provenance.original_format,
                "point_count": s.geometry.point_count if s.geometry else 0,
                "is_watertight": s.features.is_watertight if s.features else False,
                "surface_area": s.features.surface_area if s.features else 0.0,
                "volume": s.features.volume if s.features else 0.0,
                "label": s.top_prediction.label if s.top_prediction else None,
                "confidence": s.top_prediction.confidence if s.top_prediction else None,
            }
            records.append(record)

        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False)

        return len(records)

    def vacuum(self) -> None:
        """Optimize and cleanup database."""
        from sqlalchemy import text
        with self.engine.connect() as conn:
            conn.execute(text("VACUUM"))
            conn.commit()

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot by ID."""
        session = self._get_session()
        try:
            result = session.query(SnapshotModel).filter_by(snapshot_id=snapshot_id).delete()
            session.commit()
            return result > 0
        except Exception as e:
            session.rollback()
            raise DatabaseError(f"Failed to delete snapshot: {e}")
        finally:
            session.close()

    # ─────────────────────────────────────────────────────────────────────────────
    # Convenience methods (aliases used by CLI)
    # ─────────────────────────────────────────────────────────────────────────────

    def insert(self, snapshot: Snapshot) -> str:
        """Alias for insert_snapshot."""
        return self.insert_snapshot(snapshot)

    def get(self, model_id: str) -> Snapshot:
        """Get a snapshot by model_id, raising SnapshotNotFoundError if not found."""
        snapshot = self.get_snapshot_by_model_id(model_id)
        if snapshot is None:
            raise SnapshotNotFoundError(model_id)
        return snapshot

    def delete(self, model_id: str) -> bool:
        """Delete a snapshot by model_id."""
        session = self._get_session()
        try:
            model = session.query(SnapshotModel).filter_by(model_id=model_id).first()
            if not model:
                raise SnapshotNotFoundError(model_id)
            session.delete(model)
            session.commit()
            return True
        except SnapshotNotFoundError:
            raise
        except Exception as e:
            session.rollback()
            raise DatabaseError(f"Failed to delete snapshot: {e}")
        finally:
            session.close()

    def query(
        self,
        label: Optional[str] = None,
        platform: Optional[str] = None,
        tag: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 100,
    ) -> List[Snapshot]:
        """Query snapshots with filters (CLI-friendly interface)."""
        session = self._get_session()
        try:
            query = session.query(SnapshotModel)

            if platform:
                query = query.filter(SnapshotModel.platform == platform)

            if label:
                query = query.join(ClassificationModel).filter(
                    ClassificationModel.label == label,
                    ClassificationModel.rank == 1,
                )
                if min_confidence:
                    query = query.filter(ClassificationModel.confidence >= min_confidence)

            if tag:
                query = query.join(SnapshotTagModel).join(TagModel).filter(
                    TagModel.tag_name == tag
                )

            query = query.order_by(SnapshotModel.created_at.desc()).limit(limit)

            return [self._model_to_snapshot(m) for m in query.all()]
        finally:
            session.close()

    def query_all(self, limit: Optional[int] = None) -> List[Snapshot]:
        """Get all snapshots."""
        session = self._get_session()
        try:
            query = session.query(SnapshotModel).order_by(SnapshotModel.created_at.desc())
            if limit:
                query = query.limit(limit)
            return [self._model_to_snapshot(m) for m in query.all()]
        finally:
            session.close()

    def query_unclassified(self, limit: Optional[int] = None) -> List[Snapshot]:
        """Alias for get_unclassified_snapshots."""
        return self.get_unclassified_snapshots(limit=limit)

    def update_classification(self, snapshot: Snapshot) -> None:
        """Update classifications for a snapshot from its predictions."""
        if not snapshot.predictions:
            return

        self.update_classifications(snapshot.snapshot_id, snapshot.predictions)

        # Also update embedding if present
        if snapshot.embedding is not None and snapshot.predictions:
            self.update_embedding(
                snapshot.snapshot_id,
                snapshot.predictions[0].model_name,
                snapshot.embedding,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        session = self._get_session()
        try:
            total_snapshots = session.query(func.count(SnapshotModel.snapshot_id)).scalar() or 0

            # Count classified (distinct snapshots with classifications)
            classified_snapshots = session.query(
                func.count(func.distinct(ClassificationModel.snapshot_id))
            ).scalar() or 0

            unclassified_snapshots = total_snapshots - classified_snapshots

            # Count with embeddings
            with_embeddings = session.query(
                func.count(EmbeddingModel.snapshot_id)
            ).scalar() or 0

            # Count by platform
            platform_counts = dict(
                session.query(
                    SnapshotModel.platform,
                    func.count(SnapshotModel.snapshot_id),
                ).group_by(SnapshotModel.platform).all()
            )

            # Count by top label
            label_counts = dict(
                session.query(
                    ClassificationModel.label,
                    func.count(ClassificationModel.classification_id),
                ).filter(ClassificationModel.rank == 1).group_by(
                    ClassificationModel.label
                ).all()
            )

            # Database file size
            db_size_mb = 0.0
            if self.db_path.exists():
                db_size_mb = self.db_path.stat().st_size / (1024 * 1024)

            return {
                "total_snapshots": total_snapshots,
                "classified_snapshots": classified_snapshots,
                "unclassified_snapshots": unclassified_snapshots,
                "with_embeddings": with_embeddings,
                "platform_counts": platform_counts,
                "label_counts": label_counts,
                "db_size_mb": db_size_mb,
            }
        finally:
            session.close()

    def export(
        self,
        output_path: Path,
        format: str = "hdf5",
        label: Optional[str] = None,
        include_geometry: bool = True,
        include_embeddings: bool = True,
        limit: Optional[int] = None,
        taxonomy: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> Path:
        """
        Export snapshots to ML-ready format.

        Args:
            output_path: Output file path
            format: Export format (hdf5, numpy, parquet, json)
            label: Filter by label
            include_geometry: Include point cloud geometry
            include_embeddings: Include embeddings
            limit: Maximum snapshots to export
            taxonomy: Filter by taxonomy
            min_confidence: Minimum classification confidence

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)

        # Get snapshots
        session = self._get_session()
        try:
            query = session.query(SnapshotModel)

            if label or taxonomy or min_confidence:
                query = query.join(ClassificationModel)
                if label:
                    query = query.filter(ClassificationModel.label == label)
                if taxonomy:
                    query = query.filter(ClassificationModel.taxonomy == taxonomy)
                if min_confidence:
                    query = query.filter(ClassificationModel.confidence >= min_confidence)

            if limit:
                query = query.limit(limit)

            models = query.all()
            if not models:
                raise ExportError(format, "No snapshots match the criteria")

            snapshots = [self._model_to_snapshot(m) for m in models]

            if format == "hdf5":
                return self._export_hdf5_full(snapshots, output_path, include_geometry, include_embeddings)
            elif format == "numpy":
                return self._export_numpy_full(snapshots, output_path, include_geometry)
            elif format == "parquet":
                self._export_parquet(snapshots, output_path)
                return output_path
            elif format == "json":
                return self._export_json(snapshots, output_path)
            elif format == "tfrecord":
                return self._export_tfrecord(snapshots, output_path, include_geometry)
            else:
                raise ExportError(format, f"Unsupported format: {format}")

        finally:
            session.close()

    def _export_hdf5_full(
        self,
        snapshots: List[Snapshot],
        output_path: Path,
        include_geometry: bool,
        include_embeddings: bool,
    ) -> Path:
        """Export to HDF5 format with options."""
        try:
            import h5py
        except ImportError:
            raise ExportError("hdf5", "h5py not installed. Install with: pip install h5py")

        n = len(snapshots)

        with h5py.File(output_path, "w") as f:
            # Metadata
            ids = f.create_dataset("model_ids", (n,), dtype=h5py.special_dtype(vlen=str))
            for i, s in enumerate(snapshots):
                ids[i] = s.model_id

            # Geometry
            if include_geometry:
                point_count = snapshots[0].geometry.point_count if snapshots[0].geometry else 2048
                points = f.create_dataset("points", shape=(n, point_count, 3), dtype="float32")
                normals = f.create_dataset("normals", shape=(n, point_count, 3), dtype="float32")

                for i, s in enumerate(snapshots):
                    if s.geometry:
                        points[i] = s.geometry.points
                        if s.geometry.normals is not None:
                            normals[i] = s.geometry.normals

            # Labels
            all_labels = list(set(s.top_prediction.label for s in snapshots if s.top_prediction))
            all_labels.sort()
            label_to_idx = {label: i for i, label in enumerate(all_labels)}

            labels = f.create_dataset("labels", shape=(n,), dtype="int32")
            confidences = f.create_dataset("confidences", shape=(n,), dtype="float32")

            label_names = f.create_dataset("label_names", (len(all_labels),), dtype=h5py.special_dtype(vlen=str))
            for i, label in enumerate(all_labels):
                label_names[i] = label

            for i, s in enumerate(snapshots):
                if s.top_prediction:
                    labels[i] = label_to_idx.get(s.top_prediction.label, -1)
                    confidences[i] = s.top_prediction.confidence
                else:
                    labels[i] = -1
                    confidences[i] = 0.0

            # Embeddings
            if include_embeddings:
                has_emb = [s for s in snapshots if s.embedding is not None]
                if has_emb:
                    emb_dim = len(has_emb[0].embedding)
                    embeddings = f.create_dataset("embeddings", shape=(n, emb_dim), dtype="float32")
                    for i, s in enumerate(snapshots):
                        if s.embedding is not None:
                            embeddings[i] = s.embedding

        return output_path

    def _export_numpy_full(
        self,
        snapshots: List[Snapshot],
        output_path: Path,
        include_geometry: bool,
    ) -> Path:
        """Export to NumPy .npz format with options."""
        n = len(snapshots)
        data = {}

        data["model_ids"] = np.array([s.model_id for s in snapshots])

        if include_geometry:
            point_count = snapshots[0].geometry.point_count if snapshots[0].geometry else 2048
            points = np.zeros((n, point_count, 3), dtype=np.float32)
            normals = np.zeros((n, point_count, 3), dtype=np.float32)

            for i, s in enumerate(snapshots):
                if s.geometry:
                    points[i] = s.geometry.points
                    if s.geometry.normals is not None:
                        normals[i] = s.geometry.normals

            data["points"] = points
            data["normals"] = normals

        # Labels
        all_labels = list(set(s.top_prediction.label for s in snapshots if s.top_prediction))
        all_labels.sort()
        label_to_idx = {label: i for i, label in enumerate(all_labels)}

        labels = np.zeros(n, dtype=np.int32)
        for i, s in enumerate(snapshots):
            if s.top_prediction:
                labels[i] = label_to_idx.get(s.top_prediction.label, -1)
            else:
                labels[i] = -1

        data["labels"] = labels
        data["label_names"] = np.array(all_labels)

        np.savez_compressed(output_path, **data)
        return output_path

    # ─────────────────────────────────────────────────────────────────────────────
    # Vector Similarity Search
    # ─────────────────────────────────────────────────────────────────────────────

    def find_similar(
        self,
        snapshot_id: str,
        k: int = 10,
        min_similarity: float = 0.0,
        index: Optional["EmbeddingIndex"] = None,
    ) -> List[dict]:
        """
        Find similar snapshots using embedding similarity.

        Args:
            snapshot_id: Snapshot to find similar items for.
            k: Number of results.
            min_similarity: Minimum similarity threshold.
            index: Pre-built EmbeddingIndex (builds one if None).

        Returns:
            List of dicts with 'snapshot_id' and 'similarity' keys.
        """
        if index is None:
            index = self._build_index()

        # Get the query embedding
        session = self._get_session()
        try:
            emb_record = session.query(EmbeddingModel).filter_by(
                snapshot_id=snapshot_id
            ).first()
            if not emb_record:
                return []

            query_embedding = np.frombuffer(emb_record.embedding, dtype=np.float32)
        finally:
            session.close()

        results = index.search(
            query_embedding, k=k + 1, exclude_ids=[snapshot_id]
        )

        return [
            {"snapshot_id": sid, "similarity": score}
            for sid, score in results[:k]
            if score >= min_similarity
        ]

    def _build_index(self) -> "EmbeddingIndex":
        """Build an EmbeddingIndex from all stored embeddings."""
        from destill3d.core.vector_index import EmbeddingIndex

        session = self._get_session()
        try:
            records = session.query(EmbeddingModel).all()
            if not records:
                return EmbeddingIndex()

            dimension = records[0].embedding_dim
            index = EmbeddingIndex(dimension=dimension)

            for record in records:
                embedding = np.frombuffer(record.embedding, dtype=np.float32)
                index.add(record.snapshot_id, embedding)

            return index
        finally:
            session.close()

    # ─────────────────────────────────────────────────────────────────────────────
    # Checkpoint Persistence (for pipeline)
    # ─────────────────────────────────────────────────────────────────────────────

    async def save_checkpoint(self, checkpoint) -> None:
        """Save or update a processing checkpoint."""
        from destill3d.core.pipeline import ProcessingCheckpoint

        session = self._get_session()
        try:
            existing = session.query(ProcessingQueueModel).filter_by(
                model_id=checkpoint.model_id
            ).first()

            if existing:
                existing.stage = checkpoint.stage.value
                existing.retry_count = checkpoint.retry_count
                existing.max_retries = checkpoint.max_retries
                existing.last_error = checkpoint.last_error
                existing.temp_path = str(checkpoint.temp_path) if checkpoint.temp_path else None
            else:
                entry = ProcessingQueueModel(
                    model_id=checkpoint.model_id,
                    platform=checkpoint.platform,
                    source_url=checkpoint.source_url,
                    stage=checkpoint.stage.value,
                    retry_count=checkpoint.retry_count,
                    max_retries=checkpoint.max_retries,
                    last_error=checkpoint.last_error,
                    temp_path=str(checkpoint.temp_path) if checkpoint.temp_path else None,
                )
                session.add(entry)

            session.commit()
        except Exception as e:
            session.rollback()
            raise DatabaseError(f"Failed to save checkpoint: {e}")
        finally:
            session.close()

    async def query_checkpoints(
        self,
        stages: Optional[List] = None,
        updated_before=None,
        updated_after=None,
    ) -> list:
        """Query checkpoints by stage and time filters."""
        from destill3d.core.pipeline import PipelineStage, ProcessingCheckpoint

        session = self._get_session()
        try:
            query = session.query(ProcessingQueueModel)

            if stages:
                stage_values = [s.value for s in stages]
                query = query.filter(ProcessingQueueModel.stage.in_(stage_values))

            if updated_before:
                query = query.filter(ProcessingQueueModel.updated_at < updated_before)

            if updated_after:
                query = query.filter(ProcessingQueueModel.updated_at > updated_after)

            results = []
            for record in query.all():
                checkpoint = ProcessingCheckpoint(
                    model_id=record.model_id,
                    stage=PipelineStage(record.stage),
                    source_url=record.source_url or "",
                    platform=record.platform or "",
                    retry_count=record.retry_count or 0,
                    max_retries=record.max_retries or 3,
                    last_error=record.last_error,
                )
                if record.temp_path:
                    checkpoint.temp_path = Path(record.temp_path)
                results.append(checkpoint)

            return results
        finally:
            session.close()

    async def get_checkpoint(self, model_id: str):
        """Get a single checkpoint by model_id."""
        from destill3d.core.pipeline import PipelineStage, ProcessingCheckpoint

        session = self._get_session()
        try:
            record = session.query(ProcessingQueueModel).filter_by(
                model_id=model_id
            ).first()

            if not record:
                return None

            checkpoint = ProcessingCheckpoint(
                model_id=record.model_id,
                stage=PipelineStage(record.stage),
                source_url=record.source_url or "",
                platform=record.platform or "",
                retry_count=record.retry_count or 0,
                max_retries=record.max_retries or 3,
                last_error=record.last_error,
            )
            if record.temp_path:
                checkpoint.temp_path = Path(record.temp_path)
            return checkpoint
        finally:
            session.close()

    def _export_tfrecord(
        self,
        snapshots: List[Snapshot],
        output_path: Path,
        include_geometry: bool,
    ) -> Path:
        """Export to TensorFlow TFRecord format."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ExportError(
                "tfrecord",
                "tensorflow not installed. Install with: pip install destill3d[tfrecord]",
            )

        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for snapshot in snapshots:
                feature = {
                    "model_id": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[snapshot.model_id.encode()]
                        )
                    ),
                    "platform": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[snapshot.provenance.platform.encode()]
                        )
                    ),
                }

                if include_geometry and snapshot.geometry:
                    feature["points"] = tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=snapshot.geometry.points.flatten().tolist()
                        )
                    )
                    feature["point_count"] = tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=[snapshot.geometry.point_count]
                        )
                    )
                    if snapshot.geometry.normals is not None:
                        feature["normals"] = tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=snapshot.geometry.normals.flatten().tolist()
                            )
                        )

                if snapshot.top_prediction:
                    feature["label"] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[snapshot.top_prediction.label.encode()]
                        )
                    )
                    feature["confidence"] = tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[snapshot.top_prediction.confidence]
                        )
                    )
                    feature["taxonomy"] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[snapshot.top_prediction.taxonomy.encode()]
                        )
                    )

                if snapshot.features:
                    feature["surface_area"] = tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[snapshot.features.surface_area]
                        )
                    )
                    feature["volume"] = tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[snapshot.features.volume]
                        )
                    )
                    feature["is_watertight"] = tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=[int(snapshot.features.is_watertight)]
                        )
                    )

                example = tf.train.Example(
                    features=tf.train.Features(feature=feature)
                )
                writer.write(example.SerializeToString())

        return output_path

    def _export_json(self, snapshots: List[Snapshot], output_path: Path) -> Path:
        """Export metadata to JSON format."""
        records = []
        for s in snapshots:
            record = {
                "model_id": s.model_id,
                "snapshot_id": s.snapshot_id,
                "platform": s.provenance.platform,
                "title": s.provenance.title,
                "original_format": s.provenance.original_format,
                "tags": s.provenance.tags,
                "point_count": s.geometry.point_count if s.geometry else 0,
                "is_watertight": s.features.is_watertight if s.features else False,
                "surface_area": s.features.surface_area if s.features else 0.0,
                "volume": s.features.volume if s.features else 0.0,
            }
            if s.top_prediction:
                record["label"] = s.top_prediction.label
                record["confidence"] = s.top_prediction.confidence
                record["taxonomy"] = s.top_prediction.taxonomy
            records.append(record)

        with open(output_path, "w") as f:
            json.dump(records, f, indent=2, default=str)

        return output_path
