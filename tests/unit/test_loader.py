"""Unit tests for point cloud loading in loader module."""

import numpy as np
import pytest
from pathlib import Path

from destill3d.core.exceptions import FormatError
from destill3d.extract.loader import (
    FileFormat,
    FormatDetector,
    load_point_cloud,
    load_geometry,
)


class TestFormatDetector:
    """Test format detection for point cloud files."""

    def test_detect_pcd(self, temp_dir):
        pcd_path = temp_dir / "test.pcd"
        pcd_path.write_text("# .PCD v0.7\nVERSION 0.7\n")
        detector = FormatDetector()
        assert detector.detect(pcd_path) == FileFormat.PCD

    def test_detect_xyz(self, temp_dir):
        xyz_path = temp_dir / "test.xyz"
        xyz_path.write_text("0.0 0.0 0.0\n1.0 0.0 0.0\n")
        detector = FormatDetector()
        assert detector.detect(xyz_path) == FileFormat.XYZ

    def test_detect_las(self, temp_dir):
        las_path = temp_dir / "test.las"
        las_path.write_bytes(b"\x00" * 100)
        detector = FormatDetector()
        assert detector.detect(las_path) == FileFormat.LAS

    def test_detect_laz(self, temp_dir):
        laz_path = temp_dir / "test.laz"
        laz_path.write_bytes(b"\x00" * 100)
        detector = FormatDetector()
        assert detector.detect(laz_path) == FileFormat.LAZ


class TestFileFormatProperties:
    """Test FileFormat enum properties."""

    def test_pcd_is_point_cloud(self):
        assert FileFormat.PCD.is_point_cloud

    def test_xyz_is_point_cloud(self):
        assert FileFormat.XYZ.is_point_cloud

    def test_las_is_point_cloud(self):
        assert FileFormat.LAS.is_point_cloud

    def test_laz_is_point_cloud(self):
        assert FileFormat.LAZ.is_point_cloud

    def test_stl_not_point_cloud(self):
        assert not FileFormat.STL_BINARY.is_point_cloud

    def test_pcd_not_mesh(self):
        assert not FileFormat.PCD.is_mesh


class TestLoadPointCloud:
    """Test point cloud loading functionality."""

    def test_load_pcd_with_open3d(self, temp_dir):
        """Test loading PCD file via Open3D."""
        try:
            import open3d as o3d
        except ImportError:
            pytest.skip("Open3D not installed")

        # Create a PCD file using Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        )
        pcd_path = temp_dir / "test.pcd"
        o3d.io.write_point_cloud(str(pcd_path), pcd)

        points = load_point_cloud(pcd_path)
        assert points.shape == (3, 3)
        assert points.dtype == np.float32

    def test_load_xyz_with_open3d(self, temp_dir):
        """Test loading XYZ file via Open3D."""
        try:
            import open3d as o3d
        except ImportError:
            pytest.skip("Open3D not installed")

        # Create a simple XYZ file
        xyz_content = "0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n"
        xyz_path = temp_dir / "test.xyz"
        xyz_path.write_text(xyz_content)

        points = load_point_cloud(xyz_path)
        assert points.shape == (4, 3)
        assert points.dtype == np.float32

    def test_load_pcd_no_open3d(self, temp_dir):
        """Test that PCD loading raises FormatError without Open3D."""
        import sys

        pcd_path = temp_dir / "test.pcd"
        pcd_path.write_text("# .PCD v0.7\n")

        # Temporarily hide open3d
        open3d_mod = sys.modules.get("open3d")
        sys.modules["open3d"] = None  # type: ignore

        try:
            with pytest.raises(FormatError):
                load_point_cloud(pcd_path)
        finally:
            if open3d_mod is not None:
                sys.modules["open3d"] = open3d_mod
            else:
                sys.modules.pop("open3d", None)

    def test_load_las_no_laspy(self, temp_dir):
        """Test that LAS loading raises FormatError without laspy."""
        import sys

        las_path = temp_dir / "test.las"
        las_path.write_bytes(b"\x00" * 100)

        # Temporarily hide laspy
        laspy_mod = sys.modules.get("laspy")
        sys.modules["laspy"] = None  # type: ignore

        try:
            with pytest.raises(FormatError):
                load_point_cloud(las_path)
        finally:
            if laspy_mod is not None:
                sys.modules["laspy"] = laspy_mod
            else:
                sys.modules.pop("laspy", None)

    def test_unsupported_format(self, temp_dir):
        """Test that unsupported formats raise FormatError."""
        bad_path = temp_dir / "test.foo"
        bad_path.write_text("data")

        with pytest.raises(FormatError):
            load_point_cloud(bad_path)


class TestLoadGeometryPointCloud:
    """Test load_geometry with point cloud files."""

    def test_load_pcd_geometry(self, temp_dir):
        """Test that load_geometry handles PCD files end-to-end."""
        try:
            import open3d as o3d
        except ImportError:
            pytest.skip("Open3D not installed")

        # Create a PCD file with enough points for mesh reconstruction
        np.random.seed(42)
        sphere_points = np.random.randn(100, 3).astype(np.float64)
        sphere_points = sphere_points / np.linalg.norm(sphere_points, axis=1, keepdims=True)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sphere_points)
        pcd_path = temp_dir / "sphere.pcd"
        o3d.io.write_point_cloud(str(pcd_path), pcd)

        result = load_geometry(pcd_path)
        # Should return either Trimesh or PointCloud
        assert result is not None
        assert hasattr(result, "vertices") or hasattr(result, "shape")
