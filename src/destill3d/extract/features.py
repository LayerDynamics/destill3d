"""
Feature computation for Destill3D.

Computes:
- Surface normals via PCA on k-NN
- Local curvature via eigenvalue analysis
- Global shape descriptors (32-dimensional vector)
"""

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import KDTree
import trimesh


def compute_normals(
    points: np.ndarray,
    k: int = 30,
    orient_normals: bool = True,
) -> np.ndarray:
    """
    Estimate surface normals via PCA on k-nearest neighbors.

    Args:
        points: Point cloud (N, 3)
        k: Number of neighbors for PCA
        orient_normals: Whether to orient normals consistently

    Returns:
        Unit normals (N, 3) float32
    """
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
        )

        if orient_normals:
            pcd.orient_normals_consistent_tangent_plane(k=k)

        normals = np.asarray(pcd.normals).astype(np.float32)

    except ImportError:
        # Fallback to manual PCA-based normal estimation
        normals = _compute_normals_pca(points, k)

    return normals


def _compute_normals_pca(points: np.ndarray, k: int) -> np.ndarray:
    """Compute normals using manual PCA (fallback when Open3D unavailable)."""
    n_points = len(points)
    normals = np.zeros((n_points, 3), dtype=np.float32)

    tree = KDTree(points)

    for i in range(n_points):
        # Find k nearest neighbors
        _, indices = tree.query(points[i], k=k)
        neighbors = points[indices]

        # PCA: normal is eigenvector of smallest eigenvalue
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered.T)

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Smallest eigenvalue corresponds to normal direction
            normal = eigenvectors[:, 0]
        except np.linalg.LinAlgError:
            # Fallback to arbitrary normal if PCA fails
            normal = np.array([0, 1, 0], dtype=np.float32)

        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm

        normals[i] = normal

    # Simple orientation: point normals away from centroid
    centroid = points.mean(axis=0)
    for i in range(n_points):
        direction = points[i] - centroid
        if np.dot(normals[i], direction) < 0:
            normals[i] = -normals[i]

    return normals


def compute_curvature(
    points: np.ndarray,
    k: int = 30,
) -> np.ndarray:
    """
    Estimate local curvature via eigenvalue analysis.

    Curvature is computed as the ratio of the smallest eigenvalue
    to the sum of all eigenvalues of the local covariance matrix.

    Args:
        points: Point cloud (N, 3)
        k: Number of neighbors for estimation

    Returns:
        Curvature values (N,) float32 in range [0, ~0.33]
    """
    n_points = len(points)
    curvatures = np.zeros(n_points, dtype=np.float32)

    tree = KDTree(points)

    for i in range(n_points):
        _, indices = tree.query(points[i], k=k)
        neighbors = points[indices]

        # PCA on neighbors
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered.T)

        try:
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(np.abs(eigenvalues))  # ascending order

            # Curvature = lambda_min / sum(lambdas)
            total = np.sum(eigenvalues)
            if total > 1e-10:
                curvatures[i] = eigenvalues[0] / total
        except np.linalg.LinAlgError:
            curvatures[i] = 0.0

    return curvatures


def compute_global_features(
    points: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
    mesh: Optional[trimesh.Trimesh] = None,
) -> np.ndarray:
    """
    Compute rotation-invariant global shape descriptors.

    Returns a 32-dimensional feature vector containing:
    - [0:3] Normalized bounding box dimensions (sorted)
    - [3:6] Aspect ratios
    - [6] Surface area (log-scaled)
    - [7] Volume (log-scaled, 0 if non-watertight)
    - [8] Sphericity
    - [9] Convexity (V / V_hull)
    - [10:14] Curvature statistics (mean, std, min, max)
    - [14:24] Curvature histogram (10 bins)
    - [24:27] PCA eigenvalue ratios
    - [27:30] Anisotropy, planarity, linearity
    - [30:32] Reserved

    Args:
        points: Normalized point cloud (N, 3)
        normals: Surface normals (N, 3)
        curvature: Local curvature values (N,)
        mesh: Original mesh (optional, for surface area/volume)

    Returns:
        Feature vector (32,) float32
    """
    features = np.zeros(32, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Bounding box features [0:6]
    # ─────────────────────────────────────────────────────────────────────────

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    dims = bbox_max - bbox_min
    dims_sorted = np.sort(dims)[::-1]  # Descending order

    # Normalized dimensions (largest = 1)
    if dims_sorted[0] > 1e-10:
        features[0:3] = dims_sorted / dims_sorted[0]

    # Aspect ratios
    if dims_sorted[1] > 1e-10:
        features[3] = dims_sorted[0] / dims_sorted[1]
    if dims_sorted[2] > 1e-10:
        features[4] = dims_sorted[0] / dims_sorted[2]
        features[5] = dims_sorted[1] / dims_sorted[2]

    # ─────────────────────────────────────────────────────────────────────────
    # Mesh-derived features [6:10]
    # ─────────────────────────────────────────────────────────────────────────

    if mesh is not None:
        # Surface area (log-scaled)
        if mesh.area > 0:
            features[6] = np.log1p(mesh.area)

        # Volume (log-scaled, only if watertight)
        if mesh.is_watertight:
            volume = abs(mesh.volume)
            features[7] = np.log1p(volume)

            # Sphericity: how sphere-like is the shape
            # Perfect sphere has sphericity = 1
            if mesh.area > 0:
                sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / mesh.area
                features[8] = min(sphericity, 1.0)

            # Convexity: V / V_hull
            try:
                hull = mesh.convex_hull
                if hull.is_watertight and hull.volume > 0:
                    features[9] = volume / hull.volume
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Curvature statistics [10:24]
    # ─────────────────────────────────────────────────────────────────────────

    if len(curvature) > 0:
        features[10] = np.mean(curvature)
        features[11] = np.std(curvature)
        features[12] = np.min(curvature)
        features[13] = np.max(curvature)

        # Curvature histogram (10 bins, range [0, 0.5])
        hist, _ = np.histogram(curvature, bins=10, range=(0, 0.5), density=False)
        hist = hist.astype(np.float32)
        hist_sum = hist.sum()
        if hist_sum > 0:
            features[14:24] = hist / hist_sum

    # ─────────────────────────────────────────────────────────────────────────
    # PCA-derived features [24:32]
    # ─────────────────────────────────────────────────────────────────────────

    # Compute PCA on point cloud
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)

    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]  # Descending

        total_eig = eigenvalues.sum()
        if total_eig > 1e-10:
            # Eigenvalue ratios (variance explained by each axis)
            features[24:27] = eigenvalues / total_eig

        if eigenvalues[0] > 1e-10:
            # Anisotropy: (e1 - e3) / e1
            features[27] = (eigenvalues[0] - eigenvalues[2]) / eigenvalues[0]

            # Planarity: (e2 - e3) / e1
            features[28] = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]

            # Linearity: (e1 - e2) / e1
            features[29] = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]

    except np.linalg.LinAlgError:
        pass

    # [30:32] Reserved for future features

    return features


def compute_fpfh(
    points: np.ndarray,
    normals: np.ndarray,
    radius: float = 0.05,
) -> np.ndarray:
    """
    Compute Fast Point Feature Histograms (FPFH) descriptors.

    FPFH is a local geometric descriptor that captures the spatial
    relationship between a point and its neighbors.

    Args:
        points: Point cloud (N, 3)
        normals: Surface normals (N, 3)
        radius: Search radius for FPFH computation

    Returns:
        FPFH descriptors (N, 33) float32
    """
    try:
        import open3d as o3d
    except ImportError:
        # Return zeros if Open3D not available
        return np.zeros((len(points), 33), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100),
    )

    return np.asarray(fpfh.data).T.astype(np.float32)


def compute_normal_coherence(
    normals: np.ndarray,
    points: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute normal coherence score.

    Measures how consistently normals are oriented across the surface.
    Higher values indicate smoother surfaces.

    Args:
        normals: Surface normals (N, 3)
        points: Point cloud (N, 3)
        k: Number of neighbors to consider

    Returns:
        Coherence score in [0, 1]
    """
    tree = KDTree(points)
    coherences = []

    for i in range(len(points)):
        _, indices = tree.query(points[i], k=k)
        neighbor_normals = normals[indices]

        # Compute average dot product with neighbors
        dots = np.abs(np.dot(neighbor_normals, normals[i]))
        coherences.append(dots.mean())

    return float(np.mean(coherences))


def estimate_surface_area_from_points(
    points: np.ndarray,
    normals: np.ndarray,
    k: int = 10,
) -> float:
    """
    Estimate surface area from point cloud.

    Uses local neighborhood density and normal consistency.

    Args:
        points: Point cloud (N, 3)
        normals: Surface normals (N, 3)
        k: Number of neighbors for density estimation

    Returns:
        Estimated surface area
    """
    tree = KDTree(points)
    n_points = len(points)

    # Estimate local area around each point
    local_areas = []
    for i in range(n_points):
        distances, _ = tree.query(points[i], k=k)
        # Average distance to neighbors gives local density
        avg_dist = distances[1:].mean()  # Skip self
        # Approximate local area
        local_area = np.pi * avg_dist ** 2
        local_areas.append(local_area)

    # Total area is sum of local areas weighted by point density
    return float(np.sum(local_areas))
