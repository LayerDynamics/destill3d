"""Unit tests for MultiViewRenderer."""

import numpy as np
import pytest

from destill3d.extract.renderer import MultiViewRenderer, ViewConfig


class TestViewConfig:
    def test_defaults(self):
        vc = ViewConfig()
        assert vc.num_views == 12
        assert vc.resolution == 224
        assert vc.elevation == 30.0
        assert vc.background == 255

    def test_custom(self):
        vc = ViewConfig(num_views=6, resolution=128, elevation=45.0)
        assert vc.num_views == 6
        assert vc.resolution == 128


class TestMultiViewRenderer:
    def test_compute_camera_pose(self):
        r = MultiViewRenderer()
        pose = r._compute_camera_pose(azimuth=0, elevation=30, distance=3.0)
        assert pose.shape == (4, 4)
        assert np.isfinite(pose).all()

    def test_compute_camera_pose_multiple_angles(self):
        r = MultiViewRenderer()
        for az in [0, 90, 180, 270]:
            pose = r._compute_camera_pose(azimuth=az, elevation=30, distance=3.0)
            assert pose.shape == (4, 4)

    def test_look_at(self):
        r = MultiViewRenderer()
        eye = np.array([3.0, 0.0, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        matrix = r._look_at(eye, target, up)
        assert matrix.shape == (4, 4)
        np.testing.assert_array_almost_equal(matrix[:3, 3], eye)

    def test_normalize_depth_normal(self):
        r = MultiViewRenderer()
        depth = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = r._normalize_depth(depth)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_normalize_depth_all_zero(self):
        r = MultiViewRenderer()
        depth = np.zeros((4, 4))
        result = r._normalize_depth(depth)
        assert result.dtype == np.uint8

    def test_normalize_depth_infinite(self):
        r = MultiViewRenderer()
        depth = np.full((4, 4), np.inf)
        result = r._normalize_depth(depth)
        assert (result == 255).all()

    def test_normalize_depth_same_value(self):
        r = MultiViewRenderer()
        depth = np.full((4, 4), 2.0)
        result = r._normalize_depth(depth)
        assert result.dtype == np.uint8
        assert (result == 128).all()
