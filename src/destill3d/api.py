"""
Unified API for Destill3D.

Provides a single entry point for all Destill3D functionality:
acquisition, extraction, classification, and database operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from destill3d.core.config import Destill3DConfig
from destill3d.core.snapshot import Snapshot

if TYPE_CHECKING:
    from destill3d.acquire.base import DownloadResult

logger = logging.getLogger(__name__)


class ExtractAPI:
    """Feature extraction operations."""

    def __init__(self, parent: "Destill3D"):
        self._parent = parent

    def from_file(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Snapshot:
        """
        Extract features from a local 3D file.

        Args:
            file_path: Path to the 3D file.
            metadata: Optional metadata (title, tags, etc.).

        Returns:
            Snapshot with extracted features.
        """
        from destill3d.extract import FeatureExtractor

        extractor = FeatureExtractor(self._parent.config.extraction)
        return extractor.extract_from_file(Path(file_path), metadata)

    def from_directory(
        self,
        directory: Path,
        pattern: str = "**/*",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Snapshot]:
        """
        Extract features from all 3D files in a directory.

        Args:
            directory: Directory to scan.
            pattern: Glob pattern for file matching.
            metadata: Optional metadata applied to all files.

        Returns:
            List of Snapshots.
        """
        from destill3d.extract import FeatureExtractor
        from destill3d.extract.loader import FormatDetector

        extractor = FeatureExtractor(self._parent.config.extraction)
        detector = FormatDetector()
        directory = Path(directory)

        snapshots = []
        for file_path in sorted(directory.glob(pattern)):
            if not file_path.is_file():
                continue
            try:
                detector.detect(file_path)
            except Exception:
                continue

            try:
                snapshot = extractor.extract_from_file(file_path, metadata)
                snapshots.append(snapshot)
            except Exception as e:
                logger.warning(f"Failed to extract {file_path}: {e}")

        return snapshots

    def from_mesh(
        self,
        mesh,
        model_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Snapshot:
        """
        Extract features from an already-loaded mesh.

        Args:
            mesh: trimesh.Trimesh object.
            model_id: Unique identifier for the model.
            metadata: Optional metadata.

        Returns:
            Snapshot with extracted features.
        """
        from destill3d.extract import FeatureExtractor

        extractor = FeatureExtractor(self._parent.config.extraction)
        return extractor.extract_from_mesh(mesh, model_id, metadata)


class ClassifyAPI:
    """Classification operations."""

    def __init__(self, parent: "Destill3D"):
        self._parent = parent
        self._classifier = None

    def _get_classifier(self):
        """Lazy-load classifier."""
        if self._classifier is None:
            from destill3d.classify.inference import Classifier

            self._classifier = Classifier(
                models_dir=self._parent.config.data_dir / "models",
                device=self._parent.config.classification.device,
            )
        return self._classifier

    def classify(
        self,
        snapshot: Snapshot,
        model_id: str = "pointnet2_ssg_mn40",
        top_k: int = 5,
    ) -> Snapshot:
        """
        Classify a snapshot.

        Args:
            snapshot: Snapshot to classify.
            model_id: Model to use.
            top_k: Number of top predictions.

        Returns:
            Snapshot updated with predictions and embedding.
        """
        classifier = self._get_classifier()
        predictions, embedding = classifier.classify(
            snapshot, model_id=model_id, top_k=top_k
        )
        snapshot.predictions = predictions
        if embedding is not None:
            snapshot.embedding = embedding
        return snapshot

    def zero_shot(
        self,
        snapshot: Snapshot,
        classes: List[str],
        top_k: int = 5,
    ) -> Snapshot:
        """
        Zero-shot classify a snapshot with arbitrary class names.

        Args:
            snapshot: Snapshot to classify.
            classes: List of class label strings.
            top_k: Number of top predictions.

        Returns:
            Snapshot updated with predictions.
        """
        from destill3d.classify.zero_shot import ZeroShotClassifier

        zs = ZeroShotClassifier()
        result = zs.classify(
            snapshot.geometry.points,
            classes,
            top_k=top_k,
        )

        from destill3d.core.snapshot import Prediction

        snapshot.predictions = [
            Prediction(
                label=cls,
                confidence=float(prob),
                taxonomy="zero-shot",
                model_name="openshape",
                rank=i + 1,
            )
            for i, (cls, prob) in enumerate(
                zip(result.classes[:top_k], result.probabilities[:top_k])
            )
        ]
        if result.embedding_3d is not None:
            snapshot.embedding = result.embedding_3d
        return snapshot

    def batch(
        self,
        snapshots: List[Snapshot],
        model_id: str = "pointnet2_ssg_mn40",
        batch_size: int = 32,
        top_k: int = 5,
    ) -> List[Snapshot]:
        """
        Classify multiple snapshots in batches.

        Args:
            snapshots: Snapshots to classify.
            model_id: Model to use.
            batch_size: Inference batch size.
            top_k: Number of top predictions.

        Returns:
            List of updated Snapshots.
        """
        classifier = self._get_classifier()
        results = classifier.classify_batch(
            snapshots,
            model_id=model_id,
            batch_size=batch_size,
            top_k=top_k,
        )
        for snapshot, (predictions, embedding) in zip(snapshots, results):
            snapshot.predictions = predictions
            if embedding is not None:
                snapshot.embedding = embedding
        return snapshots


class DatabaseAPI:
    """Database operations."""

    def __init__(self, parent: "Destill3D"):
        self._parent = parent
        self._db = None

    def _get_db(self):
        """Lazy-load database."""
        if self._db is None:
            from destill3d.core.database import Database

            self._db = Database(self._parent.config.database.path)
        return self._db

    def store(self, snapshot: Snapshot) -> None:
        """Store a snapshot in the database."""
        db = self._get_db()
        db.store(snapshot)

    def get(self, snapshot_id: str) -> Optional[Snapshot]:
        """Get a snapshot by ID."""
        db = self._get_db()
        return db.get(snapshot_id)

    def query(
        self,
        platform: Optional[str] = None,
        label: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 100,
    ) -> List[Snapshot]:
        """
        Query snapshots from the database.

        Args:
            platform: Filter by platform.
            label: Filter by classification label.
            min_confidence: Minimum confidence threshold.
            limit: Maximum results.

        Returns:
            List of matching Snapshots.
        """
        db = self._get_db()
        return db.query(
            platform=platform,
            label=label,
            min_confidence=min_confidence,
            limit=limit,
        )

    def find_similar(
        self,
        snapshot_id: str,
        k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Find similar snapshots by embedding similarity.

        Args:
            snapshot_id: Query snapshot ID.
            k: Number of results.
            min_similarity: Minimum similarity score.

        Returns:
            List of (snapshot_id, similarity) tuples.
        """
        db = self._get_db()
        return db.find_similar(
            snapshot_id=snapshot_id, k=k, min_similarity=min_similarity
        )

    def stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        db = self._get_db()
        return db.stats()

    def export(
        self,
        output_path: Path,
        format: str = "numpy",
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Export database to ML-ready format.

        Args:
            output_path: Output file/directory path.
            format: Export format (numpy, hdf5, csv).
            train_split: Training set fraction.
            val_split: Validation set fraction.
            test_split: Test set fraction.
            seed: Random seed for splits.

        Returns:
            Dict with export statistics.
        """
        db = self._get_db()
        return db.export(
            output_path=Path(output_path),
            format=format,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )


class AcquireAPI:
    """Model acquisition operations."""

    def __init__(self, parent: "Destill3D"):
        self._parent = parent

    def search(
        self,
        query: str,
        platform: str = "thingiverse",
        limit: int = 20,
        license_filter: Optional[str] = None,
    ) -> list:
        """
        Search for 3D models on a platform.

        Args:
            query: Search query.
            platform: Platform to search.
            limit: Maximum results.
            license_filter: License filter string.

        Returns:
            List of SearchResult objects.
        """
        from destill3d.acquire.models import PlatformRegistry

        registry = PlatformRegistry()
        adapter = registry.get(platform)
        if adapter is None:
            raise ValueError(f"Unknown platform: {platform}")

        from destill3d.acquire.base import SearchFilters

        filters = SearchFilters(license=[license_filter]) if license_filter else SearchFilters()

        import asyncio

        results = asyncio.get_event_loop().run_until_complete(
            adapter.search(query, filters=filters, limit=limit)
        )
        return results

    def from_url(
        self,
        url: str,
        output_dir: Optional[Path] = None,
    ) -> "DownloadResult":
        """
        Download a model from a URL.

        Args:
            url: Model URL.
            output_dir: Output directory.

        Returns:
            DownloadResult with file paths.
        """
        from destill3d.acquire.models import PlatformRegistry

        registry = PlatformRegistry()
        platform, model_id = registry.resolve_url(url)

        adapter = registry.get(platform)
        if adapter is None:
            raise ValueError(f"No adapter for platform: {platform}")

        import asyncio

        metadata = asyncio.get_event_loop().run_until_complete(
            adapter.get_metadata(model_id)
        )

        download_dir = output_dir or self._parent.config.data_dir / "downloads"
        result = asyncio.get_event_loop().run_until_complete(
            adapter.download(metadata, download_dir)
        )
        return result

    def queue(
        self,
        urls: List[str],
        priority: int = 0,
    ) -> int:
        """
        Add URLs to the download queue.

        Args:
            urls: List of URLs to queue.
            priority: Priority level (higher = sooner).

        Returns:
            Number of items queued.
        """
        import asyncio

        from destill3d.acquire.queue import DownloadQueue

        db = self._parent.db._get_db()
        q = DownloadQueue(db)
        asyncio.run(q.add_batch(urls, priority=priority))
        return len(urls)

    def process_queue(
        self,
        concurrency: int = 4,
        extract: bool = True,
        classify: bool = False,
    ) -> Dict[str, int]:
        """
        Process the download queue.

        Args:
            concurrency: Number of concurrent downloads.
            extract: Run extraction after download.
            classify: Run classification after extraction.

        Returns:
            Dict with success/failure counts.
        """
        import asyncio

        from destill3d.acquire.queue import DownloadQueue

        db = self._parent.db._get_db()
        q = DownloadQueue(db)

        async def run_processing():
            completed = 0
            total = 0
            failed = 0
            async for progress in q.process(concurrency=concurrency):
                completed = progress.completed
                total = progress.total
                failed = total - completed
            return {"completed": completed, "failed": failed, "total": total}

        return asyncio.run(run_processing())


class Destill3D:
    """
    Unified API for Destill3D.

    Provides access to all major subsystems:
    - extract: Feature extraction from 3D files
    - classify: ML-based classification
    - db: Database storage and queries
    - acquire: Model acquisition from platforms

    Example::

        d3d = Destill3D()
        snapshot = d3d.extract.from_file("model.stl")
        snapshot = d3d.classify.classify(snapshot)
        d3d.db.store(snapshot)
    """

    def __init__(self, config: Optional[Destill3DConfig] = None):
        """
        Initialize Destill3D.

        Args:
            config: Configuration (uses defaults if None).
        """
        self.config = config or Destill3DConfig()
        self.extract = ExtractAPI(self)
        self.classify = ClassifyAPI(self)
        self.db = DatabaseAPI(self)
        self.acquire = AcquireAPI(self)

    def process_file(
        self,
        file_path: Path,
        classify: bool = True,
        store: bool = True,
        model_id: str = "pointnet2_ssg_mn40",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Snapshot:
        """
        Full pipeline: extract + classify + store.

        Args:
            file_path: Path to 3D file.
            classify: Run classification.
            store: Store in database.
            model_id: Classification model.
            metadata: Optional metadata.

        Returns:
            Processed Snapshot.
        """
        snapshot = self.extract.from_file(file_path, metadata)

        if classify:
            snapshot = self.classify.classify(snapshot, model_id=model_id)

        if store:
            self.db.store(snapshot)

        return snapshot

    def process_directory(
        self,
        directory: Path,
        pattern: str = "**/*",
        classify: bool = True,
        store: bool = True,
        model_id: str = "pointnet2_ssg_mn40",
    ) -> List[Snapshot]:
        """
        Process all 3D files in a directory.

        Args:
            directory: Directory to scan.
            pattern: Glob pattern.
            classify: Run classification.
            store: Store in database.
            model_id: Classification model.

        Returns:
            List of processed Snapshots.
        """
        snapshots = self.extract.from_directory(directory, pattern)

        if classify and snapshots:
            snapshots = self.classify.batch(snapshots, model_id=model_id)

        if store:
            for snapshot in snapshots:
                self.db.store(snapshot)

        return snapshots

    def export(
        self,
        output_path: Path,
        format: str = "numpy",
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Export database to ML-ready format with splits.

        Args:
            output_path: Output path.
            format: Export format.
            train_split: Training fraction.
            val_split: Validation fraction.
            test_split: Test fraction.

        Returns:
            Export statistics.
        """
        return self.db.export(
            output_path=output_path,
            format=format,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
        )
