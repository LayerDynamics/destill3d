"""
Pytest fixtures for Destill3D tests.
"""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import trimesh

from destill3d.core.config import Destill3DConfig, ExtractionConfig
from destill3d.core.snapshot import (
    Snapshot,
    Provenance,
    GeometryData,
    Features,
    ProcessingMetadata,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_points() -> np.ndarray:
    """Generate sample point cloud (unit cube vertices)."""
    # 8 corners of a unit cube
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)

    # Add more points by interpolating
    points = []
    for i in range(len(corners)):
        for j in range(len(corners)):
            for t in np.linspace(0, 1, 5):
                point = corners[i] * (1 - t) + corners[j] * t
                points.append(point)

    return np.array(points, dtype=np.float32)


@pytest.fixture
def sample_mesh() -> trimesh.Trimesh:
    """Create a simple cube mesh for testing."""
    return trimesh.creation.box(extents=[1.0, 1.0, 1.0])


@pytest.fixture
def sample_sphere_mesh() -> trimesh.Trimesh:
    """Create a sphere mesh for testing."""
    return trimesh.creation.icosphere(subdivisions=2, radius=1.0)


@pytest.fixture
def sample_stl_file(temp_dir: Path, sample_mesh: trimesh.Trimesh) -> Path:
    """Create a temporary STL file for testing."""
    stl_path = temp_dir / "test_cube.stl"
    sample_mesh.export(stl_path)
    return stl_path


@pytest.fixture
def sample_obj_file(temp_dir: Path, sample_mesh: trimesh.Trimesh) -> Path:
    """Create a temporary OBJ file for testing."""
    obj_path = temp_dir / "test_cube.obj"
    sample_mesh.export(obj_path)
    return obj_path


@pytest.fixture
def sample_ply_file(temp_dir: Path, sample_mesh: trimesh.Trimesh) -> Path:
    """Create a temporary PLY file for testing."""
    ply_path = temp_dir / "test_cube.ply"
    sample_mesh.export(ply_path)
    return ply_path


@pytest.fixture
def sample_snapshot() -> Snapshot:
    """Create a sample snapshot for testing."""
    # Generate random point cloud
    np.random.seed(42)
    points = np.random.randn(2048, 3).astype(np.float32)
    # Normalize to unit sphere
    points = points / np.linalg.norm(points, axis=1, keepdims=True)

    normals = points.copy()  # For a sphere, normals point outward
    curvature = np.ones(2048, dtype=np.float32) * 0.5

    return Snapshot(
        model_id="test:sample123",
        provenance=Provenance(
            platform="test",
            source_id="sample123",
            title="Test Snapshot",
            tags=["test", "sample"],
            original_filename="test.stl",
            original_format="stl_binary",
            original_file_size=1024,
        ),
        geometry=GeometryData(
            points=points,
            normals=normals,
            curvature=curvature,
            centroid=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            scale=1.0,
        ),
        features=Features(
            global_features=np.random.randn(32).astype(np.float32),
            surface_area=12.56,
            volume=4.19,
            is_watertight=True,
            original_vertex_count=1000,
            original_face_count=2000,
            bbox_min=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            bbox_max=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        ),
        processing=ProcessingMetadata(
            destill3d_version="0.1.0",
            target_points=2048,
            sampling_strategy="hybrid",
            extraction_time_ms=100.0,
        ),
    )


@pytest.fixture
def extraction_config() -> ExtractionConfig:
    """Create extraction config for testing."""
    return ExtractionConfig(
        target_points=1024,  # Smaller for faster tests
        sampling_strategy="uniform",
        oversample_ratio=1.5,
    )


@pytest.fixture
def config(temp_dir: Path) -> Destill3DConfig:
    """Create test configuration with temp paths."""
    return Destill3DConfig(
        config_dir=temp_dir / "config",
        data_dir=temp_dir / "data",
    )


@pytest.fixture
def db_path(temp_dir: Path) -> Path:
    """Path for temporary test database."""
    return temp_dir / "test.db"


# ─────────────────────────────────────────────────────────────────────────────
# Markers
# ─────────────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "cad: marks tests that require pythonocc (CAD support)"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
