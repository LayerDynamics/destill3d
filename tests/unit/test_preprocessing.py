"""Unit tests for mesh preprocessing."""

import numpy as np
import pytest
import trimesh

from destill3d.extract.preprocessing import MemoryLimits, preprocess_mesh


class TestMemoryLimits:
    def test_defaults(self):
        limits = MemoryLimits()
        assert limits.max_mesh_vertices == 10_000_000
        assert limits.max_mesh_faces == 20_000_000
        assert limits.max_point_cloud == 1_000_000
        assert limits.simplification_target == 1_000_000

    def test_custom(self):
        limits = MemoryLimits(max_mesh_vertices=100, max_mesh_faces=200)
        assert limits.max_mesh_vertices == 100


class TestPreprocessMesh:
    def test_clean_mesh_passthrough(self):
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        original_faces = len(mesh.faces)
        result = preprocess_mesh(mesh)
        assert isinstance(result, trimesh.Trimesh)
        assert len(result.faces) > 0
        # Clean mesh should be similar size
        assert abs(len(result.faces) - original_faces) < original_faces * 0.5

    def test_sphere_mesh(self):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        result = preprocess_mesh(mesh)
        assert isinstance(result, trimesh.Trimesh)
        assert len(result.vertices) > 0
        assert len(result.faces) > 0

    def test_preserves_mesh_quality(self):
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        result = preprocess_mesh(mesh)
        # Volume should be approximately preserved
        assert abs(result.volume - mesh.volume) < mesh.volume * 0.1

    def test_with_custom_limits(self):
        mesh = trimesh.creation.icosphere(subdivisions=3)
        limits = MemoryLimits(
            max_mesh_faces=100,
            simplification_target=50,
        )
        result = preprocess_mesh(mesh, limits)
        assert isinstance(result, trimesh.Trimesh)
        # Should have been simplified
        assert len(result.faces) <= len(mesh.faces)
