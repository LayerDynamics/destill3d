"""
Feature extraction module for Destill3D.

Handles geometry loading, point cloud sampling, and feature computation.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import time
import hashlib

from destill3d.core.config import ExtractionConfig, TessellationConfig
from destill3d.core.snapshot import (
    Snapshot,
    Provenance,
    GeometryData,
    Features,
    ProcessingMetadata,
    SNAPSHOT_VERSION,
)
from destill3d.core.exceptions import ExtractionError

from destill3d.extract.loader import load_geometry, FormatDetector, get_mesh_info
from destill3d.extract.sampling import (
    sample_point_cloud,
    SamplingConfig,
    SamplingStrategy,
)
from destill3d.extract.tessellation import TessellationConfig as TessConfig
from destill3d.extract.features import (
    compute_normals,
    compute_curvature,
    compute_global_features,
    compute_local_density,
    compute_per_point_features,
)
from destill3d.extract.preprocessing import preprocess_mesh, MemoryLimits
from destill3d.extract.renderer import MultiViewRenderer, ViewConfig


__all__ = [
    "FeatureExtractor",
    "load_geometry",
    "FormatDetector",
    "sample_point_cloud",
    "SamplingConfig",
    "SamplingStrategy",
    "TessConfig",
    "compute_normals",
    "compute_curvature",
    "compute_global_features",
    "compute_local_density",
    "compute_per_point_features",
    "preprocess_mesh",
    "MemoryLimits",
    "MultiViewRenderer",
    "ViewConfig",
]


class FeatureExtractor:
    """
    Main feature extraction pipeline for Destill3D.

    Orchestrates the full pipeline:
    1. Load geometry (mesh or CAD)
    2. Sample point cloud
    3. Compute normals and curvature
    4. Compute global features
    5. Create Snapshot
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize the feature extractor.

        Args:
            config: Extraction configuration (uses defaults if None)
        """
        self.config = config or ExtractionConfig()

    def extract_from_file(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Snapshot:
        """
        Extract features from a local 3D file.

        Full pipeline: load -> tessellate (if CAD) -> sample -> compute features -> create snapshot

        Args:
            file_path: Path to the 3D file
            metadata: Optional metadata to attach (title, tags, etc.)

        Returns:
            Snapshot containing extracted features

        Raises:
            ExtractionError: If extraction fails
        """
        start_time = time.time()
        file_path = Path(file_path)
        warnings: list[str] = []
        metadata = metadata or {}

        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Compute file hash and detect format
        # ─────────────────────────────────────────────────────────────────────

        file_hash = self._compute_file_hash(file_path)
        detector = FormatDetector()
        file_format = detector.detect(file_path)

        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Load geometry
        # ─────────────────────────────────────────────────────────────────────

        tess_config = TessConfig(
            linear_deflection=self.config.tessellation.linear_deflection,
            angular_deflection=self.config.tessellation.angular_deflection,
            relative=self.config.tessellation.relative,
        )

        try:
            mesh = load_geometry(file_path, tess_config)
        except Exception as e:
            raise ExtractionError(f"Failed to load geometry: {e}")

        # Preprocess mesh (cleanup + simplification)
        try:
            mesh = preprocess_mesh(mesh)
        except Exception as e:
            warnings.append(f"Mesh preprocessing warning: {e}")

        mesh_info = get_mesh_info(mesh)

        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Sample point cloud
        # ─────────────────────────────────────────────────────────────────────

        sampling_config = SamplingConfig(
            strategy=SamplingStrategy(self.config.sampling_strategy),
            target_points=self.config.target_points,
            oversample_ratio=self.config.oversample_ratio,
        )

        try:
            sample_result = sample_point_cloud(mesh, sampling_config)
        except Exception as e:
            raise ExtractionError(f"Failed to sample point cloud: {e}")

        points = sample_result.points

        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Compute normals and curvature
        # ─────────────────────────────────────────────────────────────────────

        try:
            normals = compute_normals(
                points,
                k=self.config.normal_estimation_k,
                orient_normals=True,
            )
        except Exception as e:
            warnings.append(f"Normal estimation warning: {e}")
            # Fallback to zero normals
            normals = np.zeros_like(points)

        try:
            curvature = compute_curvature(
                points,
                k=self.config.curvature_estimation_k,
            )
        except Exception as e:
            warnings.append(f"Curvature estimation warning: {e}")
            curvature = np.zeros(len(points), dtype=np.float32)

        # ─────────────────────────────────────────────────────────────────────
        # Step 5: Compute global features
        # ─────────────────────────────────────────────────────────────────────

        try:
            global_features = compute_global_features(
                points,
                normals,
                curvature,
                mesh,
            )
        except Exception as e:
            warnings.append(f"Global feature computation warning: {e}")
            global_features = np.zeros(32, dtype=np.float32)

        # ─────────────────────────────────────────────────────────────────────
        # Step 5b: Render multi-view depth maps (optional)
        # ─────────────────────────────────────────────────────────────────────

        view_images = None
        if self.config.compute_views:
            try:
                view_config = ViewConfig(
                    num_views=self.config.view_count,
                    resolution=self.config.view_resolution,
                )
                renderer = MultiViewRenderer(config=view_config)
                view_images = renderer.render_views(mesh, points)
            except Exception as e:
                warnings.append(f"View rendering warning: {e}")

        extraction_time = (time.time() - start_time) * 1000

        # ─────────────────────────────────────────────────────────────────────
        # Step 6: Build Snapshot
        # ─────────────────────────────────────────────────────────────────────

        # Generate model_id from platform and file hash
        model_id = f"local:{file_hash[:16]}"

        snapshot = Snapshot(
            model_id=model_id,
            version=SNAPSHOT_VERSION,
            provenance=Provenance(
                platform="local",
                source_url=f"file://{file_path.absolute()}",
                source_id=file_hash[:16],
                title=metadata.get("title", file_path.stem),
                description=metadata.get("description"),
                author=metadata.get("author"),
                license=metadata.get("license"),
                tags=metadata.get("tags", []),
                original_filename=file_path.name,
                original_format=file_format.value,
                original_file_size=file_path.stat().st_size,
                original_file_hash=file_hash,
                acquired_at=datetime.utcnow(),
            ),
            geometry=GeometryData(
                points=points,
                normals=normals,
                curvature=curvature,
                centroid=sample_result.centroid,
                scale=sample_result.scale,
                view_images=view_images,
            ),
            features=Features(
                global_features=global_features,
                surface_area=mesh_info["surface_area"],
                volume=mesh_info["volume"],
                is_watertight=mesh_info["is_watertight"],
                original_vertex_count=mesh_info["vertex_count"],
                original_face_count=mesh_info["face_count"],
                bbox_min=np.array(mesh_info["bounds_min"], dtype=np.float32),
                bbox_max=np.array(mesh_info["bounds_max"], dtype=np.float32),
            ),
            processing=ProcessingMetadata(
                destill3d_version="0.1.0",
                feature_extractor_version="1",
                target_points=self.config.target_points,
                sampling_strategy=self.config.sampling_strategy,
                tessellation_deflection=tess_config.linear_deflection,
                extraction_time_ms=extraction_time,
                warnings=warnings,
                processed_at=datetime.utcnow(),
            ),
        )

        return snapshot

    def extract_from_mesh(
        self,
        mesh,  # trimesh.Trimesh
        model_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Snapshot:
        """
        Extract features from an already-loaded mesh.

        Args:
            mesh: trimesh.Trimesh object
            model_id: Unique identifier for this model
            metadata: Optional metadata

        Returns:
            Snapshot containing extracted features
        """
        start_time = time.time()
        metadata = metadata or {}
        warnings: list[str] = []

        mesh_info = get_mesh_info(mesh)

        # Sample point cloud
        sampling_config = SamplingConfig(
            strategy=SamplingStrategy(self.config.sampling_strategy),
            target_points=self.config.target_points,
            oversample_ratio=self.config.oversample_ratio,
        )
        sample_result = sample_point_cloud(mesh, sampling_config)
        points = sample_result.points

        # Compute features
        normals = compute_normals(points, k=self.config.normal_estimation_k)
        curvature = compute_curvature(points, k=self.config.curvature_estimation_k)
        global_features = compute_global_features(points, normals, curvature, mesh)

        extraction_time = (time.time() - start_time) * 1000

        return Snapshot(
            model_id=model_id,
            provenance=Provenance(
                platform=metadata.get("platform", "memory"),
                title=metadata.get("title"),
                tags=metadata.get("tags", []),
            ),
            geometry=GeometryData(
                points=points,
                normals=normals,
                curvature=curvature,
                centroid=sample_result.centroid,
                scale=sample_result.scale,
            ),
            features=Features(
                global_features=global_features,
                surface_area=mesh_info["surface_area"],
                volume=mesh_info["volume"],
                is_watertight=mesh_info["is_watertight"],
                original_vertex_count=mesh_info["vertex_count"],
                original_face_count=mesh_info["face_count"],
                bbox_min=np.array(mesh_info["bounds_min"], dtype=np.float32),
                bbox_max=np.array(mesh_info["bounds_max"], dtype=np.float32),
            ),
            processing=ProcessingMetadata(
                target_points=self.config.target_points,
                sampling_strategy=self.config.sampling_strategy,
                extraction_time_ms=extraction_time,
                warnings=warnings,
            ),
        )

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


# Import numpy for array operations
import numpy as np
