"""
Multi-view depth map rendering for 3D models.

Renders multiple depth maps from different viewpoints around a 3D model,
primarily for MVCNN-based hybrid classification.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ViewConfig:
    """Configuration for multi-view rendering."""

    num_views: int = 12
    resolution: int = 224
    elevation: float = 30.0  # degrees
    background: int = 255  # White background
    depth_range: Tuple[float, float] = (0.1, 10.0)


class MultiViewRenderer:
    """Renders multiple depth map views of a 3D model."""

    def __init__(self, config: ViewConfig = None):
        self.config = config or ViewConfig()
        self._renderer = None
        self._backend = None

    def _init_renderer(self):
        """Initialize the rendering backend."""
        if self._backend is not None:
            return

        try:
            # Try pyrender first (GPU accelerated)
            import pyrender  # noqa: F401

            self._backend = "pyrender"
        except ImportError:
            try:
                # Fall back to trimesh's built-in
                import trimesh  # noqa: F401

                self._backend = "trimesh"
            except ImportError:
                raise ImportError(
                    "No rendering backend available. "
                    "Install pyrender or trimesh with rendering support."
                )

    def render_views(
        self,
        mesh: "trimesh.Trimesh",
        points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render multiple depth map views.

        Args:
            mesh: Input mesh (or use points if mesh unavailable).
            points: Point cloud fallback if no mesh.

        Returns:
            (num_views, resolution, resolution) uint8 depth maps.
        """
        self._init_renderer()

        views = []

        # Calculate camera positions around the object
        azimuth_step = 360.0 / self.config.num_views

        # Center and scale mesh
        if mesh is not None:
            mesh = mesh.copy()
            mesh.vertices -= mesh.centroid
            scale = 1.0 / mesh.bounding_box.extents.max()
            mesh.vertices *= scale

        for i in range(self.config.num_views):
            azimuth = i * azimuth_step

            depth_map = self._render_single_view(
                mesh, points, azimuth, self.config.elevation
            )
            views.append(depth_map)

        return np.array(views, dtype=np.uint8)

    def _render_single_view(
        self,
        mesh: "trimesh.Trimesh",
        points: Optional[np.ndarray],
        azimuth: float,
        elevation: float,
    ) -> np.ndarray:
        """Render a single depth map from given viewpoint."""
        if self._backend == "pyrender":
            return self._render_pyrender(mesh, azimuth, elevation)
        else:
            return self._render_trimesh(mesh, azimuth, elevation)

    def _render_pyrender(
        self,
        mesh: "trimesh.Trimesh",
        azimuth: float,
        elevation: float,
    ) -> np.ndarray:
        """Render using pyrender (GPU accelerated)."""
        import pyrender

        # Create scene
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])

        # Add mesh
        py_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(py_mesh)

        # Calculate camera pose
        camera_pose = self._compute_camera_pose(azimuth, elevation, distance=3.0)

        # Add camera
        camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
        scene.add(camera, pose=camera_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(
            self.config.resolution,
            self.config.resolution,
        )

        _, depth = renderer.render(scene)
        renderer.delete()

        # Normalize depth to 0-255
        depth_normalized = self._normalize_depth(depth)

        return depth_normalized

    def _render_trimesh(
        self,
        mesh: "trimesh.Trimesh",
        azimuth: float,
        elevation: float,
    ) -> np.ndarray:
        """Render using trimesh (CPU fallback)."""
        import trimesh

        # Use trimesh's scene rendering
        scene = trimesh.Scene(mesh)

        # Set camera
        camera_pose = self._compute_camera_pose(azimuth, elevation, distance=3.0)
        scene.camera_transform = camera_pose

        # Render to image
        data = scene.save_image(resolution=(self.config.resolution,) * 2)

        # Convert to grayscale depth approximation
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("L")
        return np.array(img)

    def _compute_camera_pose(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
    ) -> np.ndarray:
        """Compute camera transformation matrix."""
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)

        # Camera position
        x = distance * math.cos(el_rad) * math.sin(az_rad)
        y = distance * math.cos(el_rad) * math.cos(az_rad)
        z = distance * math.sin(el_rad)

        # Look-at matrix
        eye = np.array([x, y, z])
        target = np.array([0, 0, 0])
        up = np.array([0, 0, 1])

        return self._look_at(eye, target, up)

    def _look_at(
        self,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray,
    ) -> np.ndarray:
        """Compute look-at transformation matrix."""
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up_new = np.cross(right, forward)

        matrix = np.eye(4)
        matrix[:3, 0] = right
        matrix[:3, 1] = up_new
        matrix[:3, 2] = -forward
        matrix[:3, 3] = eye

        return matrix

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth map to 0-255 range."""
        # Handle infinite/invalid depth
        valid_mask = np.isfinite(depth) & (depth > 0)

        if not valid_mask.any():
            return np.full(depth.shape, self.config.background, dtype=np.uint8)

        min_depth = depth[valid_mask].min()
        max_depth = depth[valid_mask].max()

        if max_depth - min_depth < 1e-6:
            return np.full(depth.shape, 128, dtype=np.uint8)

        normalized = (depth - min_depth) / (max_depth - min_depth)
        normalized = np.clip(normalized * 255, 0, 255)
        normalized[~valid_mask] = self.config.background  # Background

        return normalized.astype(np.uint8)
