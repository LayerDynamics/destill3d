"""
Point cloud sampling algorithms for Destill3D.

Provides multiple sampling strategies:
- Uniform: Random sampling from mesh surface
- FPS: Farthest Point Sampling for even coverage
- Poisson: Poisson disk sampling
- Hybrid: Uniform oversample + FPS (default)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
import trimesh

from destill3d.core.exceptions import SamplingError, NormalizationError


class SamplingStrategy(Enum):
    """Point cloud sampling strategies."""

    UNIFORM = "uniform"
    FPS = "fps"
    POISSON = "poisson"
    HYBRID = "hybrid"


@dataclass
class SamplingConfig:
    """Configuration for point cloud sampling."""

    strategy: SamplingStrategy = SamplingStrategy.HYBRID
    target_points: int = 2048
    oversample_ratio: float = 5.0
    seed: int | None = None  # For reproducibility


@dataclass
class SamplingResult:
    """Result of point cloud sampling."""

    points: np.ndarray  # (N, 3) float32 - normalized points
    centroid: np.ndarray  # (3,) float32 - original centroid
    scale: float  # normalization scale factor


def sample_point_cloud(
    mesh: trimesh.Trimesh,
    config: SamplingConfig | None = None,
) -> SamplingResult:
    """
    Sample points from mesh surface and normalize.

    Args:
        mesh: Input trimesh mesh
        config: Sampling configuration

    Returns:
        SamplingResult with normalized points and normalization params

    Raises:
        SamplingError: If sampling fails
    """
    if config is None:
        config = SamplingConfig()

    if config.seed is not None:
        np.random.seed(config.seed)

    n_points = config.target_points

    try:
        if config.strategy == SamplingStrategy.UNIFORM:
            points = _sample_uniform(mesh, n_points)

        elif config.strategy == SamplingStrategy.FPS:
            # Oversample then apply FPS
            n_oversample = int(n_points * config.oversample_ratio)
            points = _sample_uniform(mesh, n_oversample)
            points = farthest_point_sampling(points, n_points)

        elif config.strategy == SamplingStrategy.HYBRID:
            # Same as FPS but with different name for clarity
            n_oversample = int(n_points * config.oversample_ratio)
            points = _sample_uniform(mesh, n_oversample)
            points = farthest_point_sampling(points, n_points)

        elif config.strategy == SamplingStrategy.POISSON:
            points = _sample_poisson(mesh, n_points)

        else:
            raise SamplingError(f"Unknown sampling strategy: {config.strategy}")

    except Exception as e:
        if isinstance(e, SamplingError):
            raise
        raise SamplingError(f"Sampling failed: {e}")

    # Normalize points
    try:
        points, centroid, scale = normalize_point_cloud(points)
    except Exception as e:
        raise NormalizationError(f"Normalization failed: {e}")

    return SamplingResult(
        points=points.astype(np.float32),
        centroid=centroid.astype(np.float32),
        scale=float(scale),
    )


def _sample_uniform(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
    """Uniformly sample points from mesh surface."""
    if mesh.area < 1e-10:
        raise SamplingError("Mesh has zero surface area")

    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def _sample_poisson(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
    """Sample points using Poisson disk sampling via Open3D."""
    try:
        import open3d as o3d
    except ImportError:
        raise SamplingError("Open3D required for Poisson sampling. Install with: pip install open3d")

    # Convert trimesh to Open3D
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # Sample with Poisson disk
    pcd = o3d_mesh.sample_points_poisson_disk(n_points)
    points = np.asarray(pcd.points)

    # Poisson may return fewer points than requested
    if len(points) < n_points:
        # Fill with uniform samples
        additional = _sample_uniform(mesh, n_points - len(points))
        points = np.vstack([points, additional])

    return points[:n_points]


def farthest_point_sampling(
    points: np.ndarray,
    n_samples: int,
    start_idx: int | None = None,
) -> np.ndarray:
    """
    Farthest Point Sampling algorithm.

    Iteratively selects points that maximize the minimum distance to
    the already selected set, ensuring even spatial coverage.

    Args:
        points: Input points (N, 3)
        n_samples: Number of points to sample
        start_idx: Starting point index (random if None)

    Returns:
        Sampled points (n_samples, 3)
    """
    n_points = len(points)

    if n_samples >= n_points:
        return points.copy()

    # Initialize
    selected_indices = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(n_points, np.inf, dtype=np.float64)

    # Start with random or specified point
    if start_idx is None:
        start_idx = np.random.randint(n_points)
    selected_indices[0] = start_idx

    # Iteratively select farthest points
    for i in range(1, n_samples):
        last_selected = selected_indices[i - 1]
        last_point = points[last_selected]

        # Update distances to nearest selected point
        dist_to_last = np.sum((points - last_point) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_last)

        # Select point with maximum distance to selected set
        selected_indices[i] = np.argmax(distances)

    return points[selected_indices].copy()


def normalize_point_cloud(
    points: np.ndarray,
    method: str = "unit_sphere",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize point cloud to standard position and scale.

    Args:
        points: Input points (N, 3)
        method: Normalization method ('unit_sphere' or 'unit_box')

    Returns:
        Tuple of (normalized_points, centroid, scale)
    """
    # Center at origin
    centroid = points.mean(axis=0)
    centered = points - centroid

    if method == "unit_sphere":
        # Scale to fit in unit sphere
        scale = np.max(np.linalg.norm(centered, axis=1))
        if scale < 1e-10:
            raise NormalizationError("Point cloud has zero extent")
        normalized = centered / scale

    elif method == "unit_box":
        # Scale to fit in [-1, 1]^3 box
        max_extent = np.max(np.abs(centered))
        if max_extent < 1e-10:
            raise NormalizationError("Point cloud has zero extent")
        scale = max_extent
        normalized = centered / scale

    else:
        raise NormalizationError(f"Unknown normalization method: {method}")

    return normalized, centroid, scale


def denormalize_point_cloud(
    normalized_points: np.ndarray,
    centroid: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Reverse point cloud normalization.

    Args:
        normalized_points: Normalized points (N, 3)
        centroid: Original centroid (3,)
        scale: Normalization scale factor

    Returns:
        Original-scale points (N, 3)
    """
    return normalized_points * scale + centroid


def augment_point_cloud(
    points: np.ndarray,
    rotation: bool = True,
    jitter: bool = True,
    scale_range: Tuple[float, float] | None = (0.8, 1.2),
    jitter_sigma: float = 0.01,
    jitter_clip: float = 0.05,
) -> np.ndarray:
    """
    Apply data augmentation to point cloud.

    Useful for training data generation.

    Args:
        points: Input points (N, 3)
        rotation: Apply random rotation around Y axis
        jitter: Add Gaussian noise
        scale_range: Random scale range (min, max) or None to disable
        jitter_sigma: Noise standard deviation
        jitter_clip: Maximum noise magnitude

    Returns:
        Augmented points (N, 3)
    """
    augmented = points.copy()

    # Random rotation around Y axis (up)
    if rotation:
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [cos_t, 0, sin_t],
            [0, 1, 0],
            [-sin_t, 0, cos_t],
        ])
        augmented = augmented @ rotation_matrix.T

    # Random scaling
    if scale_range is not None:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        augmented = augmented * scale

    # Add jitter
    if jitter:
        noise = np.random.normal(0, jitter_sigma, augmented.shape)
        noise = np.clip(noise, -jitter_clip, jitter_clip)
        augmented = augmented + noise

    return augmented.astype(np.float32)
