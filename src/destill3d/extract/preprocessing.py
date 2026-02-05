"""
Mesh preprocessing for Destill3D.

Provides mesh cleanup and simplification before feature extraction.
Handles degenerate geometry, duplicate vertices, hole filling,
and mesh simplification for memory-constrained processing.
"""

import logging
from dataclasses import dataclass

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


@dataclass
class MemoryLimits:
    """Memory limits for mesh processing."""

    max_mesh_vertices: int = 10_000_000
    max_mesh_faces: int = 20_000_000
    max_point_cloud: int = 1_000_000
    simplification_target: int = 1_000_000


def preprocess_mesh(
    mesh: trimesh.Trimesh,
    limits: MemoryLimits = None,
) -> trimesh.Trimesh:
    """
    Preprocess mesh for feature extraction.

    Performs the following cleanup steps:
    1. Remove degenerate faces (zero-area triangles)
    2. Merge duplicate vertices
    3. Repair small holes
    4. Simplify if exceeding face limit

    Args:
        mesh: Input mesh to preprocess.
        limits: Memory limits for processing.

    Returns:
        Cleaned and optionally simplified mesh.
    """
    limits = limits or MemoryLimits()

    initial_verts = len(mesh.vertices)
    initial_faces = len(mesh.faces)

    # Step 1: Remove degenerate faces
    mesh = _remove_degenerate_faces(mesh)

    # Step 2: Merge duplicate vertices
    mesh = _merge_duplicate_vertices(mesh)

    # Step 3: Repair small holes
    mesh = _repair_holes(mesh)

    # Step 4: Simplify if too large
    if len(mesh.faces) > limits.simplification_target:
        mesh = _simplify_mesh(mesh, limits.simplification_target)

    final_verts = len(mesh.vertices)
    final_faces = len(mesh.faces)

    if initial_verts != final_verts or initial_faces != final_faces:
        logger.info(
            f"Preprocessed mesh: {initial_verts} -> {final_verts} vertices, "
            f"{initial_faces} -> {final_faces} faces"
        )

    return mesh


def _remove_degenerate_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove zero-area and degenerate triangles."""
    # trimesh has built-in degenerate face detection
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        # Manual removal: find faces with zero area
        face_areas = mesh.area_faces
        valid_mask = face_areas > 1e-10
        if not valid_mask.all():
            removed = (~valid_mask).sum()
            mesh.update_faces(valid_mask)
            logger.debug(f"Removed {removed} degenerate faces")

    return mesh


def _merge_duplicate_vertices(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge vertices that are at the same position."""
    mesh.merge_vertices()
    return mesh


def _repair_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fill small holes in the mesh."""
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception as e:
        logger.debug(f"Hole repair skipped: {e}")
    return mesh


def _simplify_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int,
) -> trimesh.Trimesh:
    """
    Simplify mesh to target face count.

    Uses quadric decimation via Open3D if available,
    falls back to trimesh simplification.
    """
    if len(mesh.faces) <= target_faces:
        return mesh

    try:
        import open3d as o3d

        # Convert to Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

        # Simplify using quadric decimation
        simplified = o3d_mesh.simplify_quadric_decimation(target_faces)

        # Convert back to trimesh
        result = trimesh.Trimesh(
            vertices=np.asarray(simplified.vertices),
            faces=np.asarray(simplified.triangles),
        )

        logger.info(
            f"Simplified mesh: {len(mesh.faces)} -> {len(result.faces)} faces"
        )
        return result

    except ImportError:
        # Fallback: use trimesh's simplification
        try:
            result = mesh.simplify_quadric_decimation(target_faces)
            logger.info(
                f"Simplified mesh: {len(mesh.faces)} -> {len(result.faces)} faces"
            )
            return result
        except Exception as e:
            logger.warning(f"Mesh simplification failed: {e}")
            return mesh
