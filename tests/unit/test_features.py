"""
Unit tests for feature computation.
"""

import numpy as np
import pytest
import trimesh

from destill3d.extract.features import (
    compute_normals,
    compute_curvature,
    compute_global_features,
)


class TestComputeNormals:
    """Tests for normal estimation."""

    def test_normals_shape(self, sample_points: np.ndarray):
        """Test that normals have correct shape."""
        normals = compute_normals(sample_points, k=10)

        assert normals.shape == sample_points.shape
        assert normals.dtype == np.float32

    def test_normals_unit_length(self, sample_points: np.ndarray):
        """Test that normals are unit vectors."""
        normals = compute_normals(sample_points, k=10)

        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_array_almost_equal(lengths, np.ones_like(lengths), decimal=5)

    def test_normals_sphere(self, sample_sphere_mesh: trimesh.Trimesh):
        """Test normals on a sphere point outward."""
        # Sample points from sphere
        points = sample_sphere_mesh.sample(500).astype(np.float32)
        normals = compute_normals(points, k=15, orient_normals=True)

        # For a unit sphere centered at origin, normals should point outward
        # (same direction as position vector)
        dot_products = np.sum(points * normals, axis=1)

        # Most normals should point outward (positive dot product)
        outward_ratio = np.sum(dot_products > 0) / len(dot_products)
        assert outward_ratio > 0.8  # At least 80% pointing outward

    def test_normals_k_parameter(self, sample_points: np.ndarray):
        """Test that k parameter affects results."""
        normals_k5 = compute_normals(sample_points, k=5)
        normals_k20 = compute_normals(sample_points, k=20)

        # Different k values should give different results
        assert not np.allclose(normals_k5, normals_k20)


class TestComputeCurvature:
    """Tests for curvature estimation."""

    def test_curvature_shape(self, sample_points: np.ndarray):
        """Test that curvature has correct shape."""
        curvature = compute_curvature(sample_points, k=10)

        assert curvature.shape == (len(sample_points),)
        assert curvature.dtype == np.float32

    def test_curvature_range(self, sample_points: np.ndarray):
        """Test that curvature values are in valid range."""
        curvature = compute_curvature(sample_points, k=10)

        # Curvature should be non-negative
        assert np.all(curvature >= 0)
        # And bounded (normalized)
        assert np.all(curvature <= 1.0)

    def test_curvature_flat_surface(self):
        """Test curvature on a flat surface is low."""
        # Create a flat plane of points
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())], axis=1).astype(np.float32)

        curvature = compute_curvature(points, k=8)

        # Interior points should have low curvature
        # (excluding boundary points which have edge effects)
        interior_mask = (np.abs(points[:, 0]) < 0.8) & (np.abs(points[:, 1]) < 0.8)
        interior_curvature = curvature[interior_mask]

        assert np.mean(interior_curvature) < 0.2

    def test_curvature_sphere(self, sample_sphere_mesh: trimesh.Trimesh):
        """Test curvature on a sphere is relatively uniform."""
        points = sample_sphere_mesh.sample(500).astype(np.float32)
        curvature = compute_curvature(points, k=15)

        # Curvature on a sphere should be fairly uniform
        std = np.std(curvature)
        mean = np.mean(curvature)

        # Standard deviation should be small relative to mean
        assert std < mean * 0.5


class TestComputeGlobalFeatures:
    """Tests for global feature computation."""

    def test_global_features_shape(self, sample_points: np.ndarray, sample_mesh: trimesh.Trimesh):
        """Test that global features have correct shape."""
        normals = compute_normals(sample_points, k=10)
        curvature = compute_curvature(sample_points, k=10)

        features = compute_global_features(sample_points, normals, curvature, sample_mesh)

        assert features.shape == (32,)
        assert features.dtype == np.float32

    def test_global_features_no_nan(self, sample_points: np.ndarray, sample_mesh: trimesh.Trimesh):
        """Test that global features don't contain NaN."""
        normals = compute_normals(sample_points, k=10)
        curvature = compute_curvature(sample_points, k=10)

        features = compute_global_features(sample_points, normals, curvature, sample_mesh)

        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_global_features_different_meshes(
        self,
        sample_mesh: trimesh.Trimesh,
        sample_sphere_mesh: trimesh.Trimesh,
    ):
        """Test that different meshes produce different features."""
        # Cube features
        cube_points = sample_mesh.sample(1024).astype(np.float32)
        cube_normals = compute_normals(cube_points, k=10)
        cube_curvature = compute_curvature(cube_points, k=10)
        cube_features = compute_global_features(cube_points, cube_normals, cube_curvature, sample_mesh)

        # Sphere features
        sphere_points = sample_sphere_mesh.sample(1024).astype(np.float32)
        sphere_normals = compute_normals(sphere_points, k=10)
        sphere_curvature = compute_curvature(sphere_points, k=10)
        sphere_features = compute_global_features(sphere_points, sphere_normals, sphere_curvature, sample_sphere_mesh)

        # Features should be different
        assert not np.allclose(cube_features, sphere_features)

    def test_global_features_without_mesh(self, sample_points: np.ndarray):
        """Test global features computation without mesh."""
        normals = compute_normals(sample_points, k=10)
        curvature = compute_curvature(sample_points, k=10)

        features = compute_global_features(sample_points, normals, curvature, mesh=None)

        assert features.shape == (32,)
        assert not np.any(np.isnan(features))
