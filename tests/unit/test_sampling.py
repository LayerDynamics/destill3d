"""
Unit tests for point cloud sampling.
"""

import numpy as np
import pytest
import trimesh

from destill3d.extract.sampling import (
    SamplingStrategy,
    SamplingConfig,
    SamplingResult,
    farthest_point_sampling,
    sample_point_cloud,
    augment_point_cloud,
)


class TestFarthestPointSampling:
    """Tests for FPS algorithm."""

    def test_fps_basic(self, sample_points: np.ndarray):
        """Test basic FPS functionality."""
        n_samples = 64
        sampled = farthest_point_sampling(sample_points, n_samples)

        assert len(sampled) == n_samples
        assert sampled.shape == (n_samples, 3)
        assert sampled.dtype == sample_points.dtype

    def test_fps_coverage(self, sample_points: np.ndarray):
        """Test that FPS samples cover the point cloud well."""
        n_samples = 32
        sampled = farthest_point_sampling(sample_points, n_samples)

        # Check that sampled points are spread out
        # Compute pairwise distances
        dists = np.linalg.norm(
            sampled[:, None, :] - sampled[None, :, :],
            axis=-1,
        )
        np.fill_diagonal(dists, np.inf)

        # Minimum distance should be reasonable
        min_dist = np.min(dists)
        assert min_dist > 0.0

    def test_fps_deterministic_with_seed(self, sample_points: np.ndarray):
        """Test FPS is deterministic with same seed."""
        np.random.seed(42)
        sampled1 = farthest_point_sampling(sample_points, 32)

        np.random.seed(42)
        sampled2 = farthest_point_sampling(sample_points, 32)

        np.testing.assert_array_equal(sampled1, sampled2)

    def test_fps_small_pointcloud(self):
        """Test FPS with very small point cloud."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        sampled = farthest_point_sampling(points, 3)

        assert len(sampled) == 3
        # All original points should be in the result
        for pt in points:
            assert any(np.allclose(pt, s) for s in sampled)

    def test_fps_more_than_available(self):
        """Test FPS when requesting more points than available."""
        points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        sampled = farthest_point_sampling(points, 10)

        # Should return all available points
        assert len(sampled) == 2


class TestSamplePointCloud:
    """Tests for point cloud sampling from mesh."""

    def test_sample_uniform(self, sample_mesh: trimesh.Trimesh):
        """Test uniform sampling."""
        config = SamplingConfig(
            strategy=SamplingStrategy.UNIFORM,
            target_points=1024,
        )
        result = sample_point_cloud(sample_mesh, config)

        assert isinstance(result, SamplingResult)
        assert result.points.shape == (1024, 3)

    def test_sample_fps(self, sample_mesh: trimesh.Trimesh):
        """Test FPS sampling."""
        config = SamplingConfig(
            strategy=SamplingStrategy.FPS,
            target_points=512,
            oversample_ratio=2.0,
        )
        result = sample_point_cloud(sample_mesh, config)

        assert result.points.shape == (512, 3)

    def test_sample_hybrid(self, sample_mesh: trimesh.Trimesh):
        """Test hybrid sampling."""
        config = SamplingConfig(
            strategy=SamplingStrategy.HYBRID,
            target_points=1024,
            oversample_ratio=2.0,
        )
        result = sample_point_cloud(sample_mesh, config)

        assert result.points.shape == (1024, 3)

    def test_sample_normalized(self, sample_mesh: trimesh.Trimesh):
        """Test that sampled points are normalized."""
        config = SamplingConfig(
            strategy=SamplingStrategy.UNIFORM,
            target_points=1024,
        )
        result = sample_point_cloud(sample_mesh, config)

        # Check centroid is near origin
        centroid = np.mean(result.points, axis=0)
        np.testing.assert_array_almost_equal(centroid, [0, 0, 0], decimal=1)

        # Check points are scaled to unit sphere
        max_dist = np.max(np.linalg.norm(result.points, axis=1))
        assert max_dist <= 1.1  # Allow small margin

    def test_sample_result_metadata(self, sample_mesh: trimesh.Trimesh):
        """Test that sampling result contains correct metadata."""
        config = SamplingConfig(
            strategy=SamplingStrategy.UNIFORM,
            target_points=512,
        )
        result = sample_point_cloud(sample_mesh, config)

        assert result.centroid is not None
        assert result.scale > 0
        assert result.centroid.shape == (3,)


class TestAugmentation:
    """Tests for point cloud augmentation."""

    def test_augment_rotation(self, sample_points: np.ndarray):
        """Test rotation augmentation."""
        augmented = augment_point_cloud(
            sample_points,
            rotation=True,
            jitter=False,
            scale_range=None,
        )

        assert augmented.shape == sample_points.shape
        # Points should be different (rotated)
        assert not np.allclose(augmented, sample_points)

        # But distances from centroid should be preserved
        orig_dists = np.linalg.norm(sample_points - sample_points.mean(axis=0), axis=1)
        aug_dists = np.linalg.norm(augmented - augmented.mean(axis=0), axis=1)
        np.testing.assert_array_almost_equal(sorted(orig_dists), sorted(aug_dists), decimal=5)

    def test_augment_jitter(self, sample_points: np.ndarray):
        """Test jitter augmentation."""
        np.random.seed(42)
        augmented = augment_point_cloud(
            sample_points,
            rotation=False,
            jitter=True,
            jitter_sigma=0.01,
            scale_range=None,
        )

        assert augmented.shape == sample_points.shape
        # Points should be slightly different
        diff = np.abs(augmented - sample_points)
        assert np.max(diff) > 0
        assert np.max(diff) < 0.1  # Jitter should be small

    def test_augment_scale(self, sample_points: np.ndarray):
        """Test scale augmentation."""
        np.random.seed(42)
        augmented = augment_point_cloud(
            sample_points,
            rotation=False,
            jitter=False,
            scale_range=(0.9, 1.1),
        )

        assert augmented.shape == sample_points.shape
        # Points should be scaled (shape preserved)

    def test_augment_combined(self, sample_points: np.ndarray):
        """Test combined augmentation."""
        augmented = augment_point_cloud(
            sample_points,
            rotation=True,
            jitter=True,
            scale_range=(0.8, 1.2),
        )

        assert augmented.shape == sample_points.shape
        assert not np.allclose(augmented, sample_points)


class TestSamplingConfig:
    """Tests for SamplingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SamplingConfig()

        assert config.strategy == SamplingStrategy.HYBRID
        assert config.target_points == 2048
        assert config.oversample_ratio == 5.0  # Default is 5.0

    def test_config_with_values(self):
        """Test configuration with custom values."""
        config = SamplingConfig(
            strategy=SamplingStrategy.FPS,
            target_points=1024,
            oversample_ratio=3.0,
        )

        assert config.strategy == SamplingStrategy.FPS
        assert config.target_points == 1024
        assert config.oversample_ratio == 3.0
