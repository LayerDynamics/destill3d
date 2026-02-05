"""
Format detection and geometry loading for Destill3D.

Supports multiple 3D file formats and provides unified mesh loading.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from destill3d.core.exceptions import FormatError, GeometryError


class FileFormat(Enum):
    """Supported 3D file formats."""

    # Mesh formats
    STL_ASCII = "stl_ascii"
    STL_BINARY = "stl_binary"
    OBJ = "obj"
    PLY = "ply"
    OFF = "off"
    GLTF = "gltf"
    GLTF_BINARY = "glb"

    # CAD formats (require tessellation)
    STEP = "step"
    IGES = "iges"
    BREP = "brep"

    # Additional mesh formats
    FBX = "fbx"
    THREE_MF = "3mf"
    DAE = "dae"

    # Point cloud formats
    PCD = "pcd"
    XYZ = "xyz"
    LAS = "las"
    LAZ = "laz"

    UNKNOWN = "unknown"

    @property
    def is_mesh(self) -> bool:
        """Check if format is a mesh format."""
        return self in {
            FileFormat.STL_ASCII,
            FileFormat.STL_BINARY,
            FileFormat.OBJ,
            FileFormat.PLY,
            FileFormat.OFF,
            FileFormat.GLTF,
            FileFormat.GLTF_BINARY,
            FileFormat.FBX,
            FileFormat.THREE_MF,
            FileFormat.DAE,
        }

    @property
    def is_cad(self) -> bool:
        """Check if format is a CAD format requiring tessellation."""
        return self in {
            FileFormat.STEP,
            FileFormat.IGES,
            FileFormat.BREP,
        }

    @property
    def is_point_cloud(self) -> bool:
        """Check if format is a point cloud format."""
        return self in {
            FileFormat.PCD,
            FileFormat.XYZ,
            FileFormat.LAS,
            FileFormat.LAZ,
        }


class FormatDetector:
    """
    Multi-stage format detection for 3D files.

    Uses magic bytes and file extension for reliable detection.
    """

    EXTENSION_MAP = {
        ".stl": FileFormat.STL_BINARY,  # May be overridden by magic bytes
        ".obj": FileFormat.OBJ,
        ".ply": FileFormat.PLY,
        ".off": FileFormat.OFF,
        ".gltf": FileFormat.GLTF,
        ".glb": FileFormat.GLTF_BINARY,
        ".step": FileFormat.STEP,
        ".stp": FileFormat.STEP,
        ".iges": FileFormat.IGES,
        ".igs": FileFormat.IGES,
        ".brep": FileFormat.BREP,
        ".brp": FileFormat.BREP,
        ".fbx": FileFormat.FBX,
        ".3mf": FileFormat.THREE_MF,
        ".dae": FileFormat.DAE,
        ".pcd": FileFormat.PCD,
        ".xyz": FileFormat.XYZ,
        ".las": FileFormat.LAS,
        ".laz": FileFormat.LAZ,
    }

    # Magic bytes for format identification
    MAGIC_BYTES = {
        b"solid": FileFormat.STL_ASCII,
        b"ply": FileFormat.PLY,
        b"OFF": FileFormat.OFF,
        b"glTF": FileFormat.GLTF_BINARY,
    }

    def detect(self, file_path: Path) -> FileFormat:
        """
        Detect file format using magic bytes and extension.

        Args:
            file_path: Path to the 3D file

        Returns:
            Detected FileFormat
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FormatError("unknown", f"File not found: {file_path}")

        # Stage 1: Try magic bytes
        magic = self._read_magic_bytes(file_path, 16)

        # Check for STL ASCII (starts with "solid")
        if magic.startswith(b"solid"):
            # Verify it's actually ASCII STL (not just a coincidence)
            if self._verify_ascii_stl(file_path):
                return FileFormat.STL_ASCII

        # Check other magic bytes
        for magic_prefix, fmt in self.MAGIC_BYTES.items():
            if magic.startswith(magic_prefix):
                return fmt

        # Check for binary GLTF
        if len(magic) >= 4 and magic[:4] == b"glTF":
            return FileFormat.GLTF_BINARY

        # Stage 2: Check for XML-based formats (COLLADA, 3MF)
        if magic.startswith(b"<?xml") or magic.startswith(b"<"):
            xml_content = self._read_magic_bytes(file_path, 4096).lower()
            if b"collada" in xml_content:
                return FileFormat.DAE
            if b"<model" in xml_content and b"3dmanufacturing" in xml_content:
                return FileFormat.THREE_MF

        # Stage 3: Fall back to extension mapping
        ext = file_path.suffix.lower()
        if ext in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[ext]

        return FileFormat.UNKNOWN

    def _read_magic_bytes(self, file_path: Path, n: int) -> bytes:
        """Read the first n bytes of a file."""
        try:
            with open(file_path, "rb") as f:
                return f.read(n)
        except Exception:
            return b""

    def _verify_ascii_stl(self, file_path: Path) -> bool:
        """
        Verify that a file is actually ASCII STL.

        Binary STL can coincidentally start with "solid".
        """
        try:
            with open(file_path, "rb") as f:
                # Read first 1KB
                header = f.read(1024)

            # ASCII STL should have "facet" or "endsolid" keywords
            header_lower = header.lower()
            return b"facet" in header_lower or b"endsolid" in header_lower

        except Exception:
            return False


def load_geometry(
    file_path: Path,
    tessellation_config: Optional["TessellationConfig"] = None,
) -> trimesh.Trimesh:
    """
    Load 3D geometry from file.

    Auto-detects format and handles:
    - Mesh formats: Loaded directly
    - CAD formats: Tessellated via pythonocc
    - Scenes: Flattened to single mesh

    Args:
        file_path: Path to the 3D file
        tessellation_config: Configuration for CAD tessellation

    Returns:
        Unified trimesh.Trimesh

    Raises:
        FormatError: If format is unsupported or detection fails
        GeometryError: If geometry is invalid
    """
    file_path = Path(file_path)

    # Detect format
    detector = FormatDetector()
    file_format = detector.detect(file_path)

    if file_format == FileFormat.UNKNOWN:
        raise FormatError("unknown", f"Cannot detect format of: {file_path}")

    # Handle CAD formats
    if file_format.is_cad:
        from destill3d.extract.tessellation import tessellate_cad_file
        return tessellate_cad_file(file_path, tessellation_config)

    # Handle point cloud formats
    if file_format.is_point_cloud:
        points = load_point_cloud(file_path)
        return _point_cloud_to_mesh(points)

    # Load mesh formats with trimesh
    try:
        mesh = trimesh.load(str(file_path), force="mesh")
    except Exception as e:
        raise FormatError(file_format.value, f"Failed to load mesh: {e}")

    # Handle scenes (multiple objects)
    if isinstance(mesh, trimesh.Scene):
        mesh = _flatten_scene(mesh)

    # Validate mesh
    _validate_mesh(mesh)

    return mesh


def load_point_cloud(file_path: Path) -> np.ndarray:
    """
    Load point cloud from PCD, XYZ, PTS, LAS, or LAZ files.

    Args:
        file_path: Path to point cloud file

    Returns:
        numpy array of shape (N, 3) with float32 dtype

    Raises:
        FormatError: If format unsupported or library missing
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext in {".pcd", ".xyz", ".pts"}:
        try:
            import open3d as o3d
        except ImportError:
            raise FormatError(
                ext,
                "Open3D not installed. Install with: pip install destill3d[open3d]",
            )

        fmt = "xyz" if ext in {".xyz", ".pts"} else "auto"
        pcd = o3d.io.read_point_cloud(str(file_path), format=fmt)
        points = np.asarray(pcd.points, dtype=np.float32)

        if len(points) == 0:
            raise FormatError(ext, f"Point cloud file contains no points: {file_path}")

        return points

    elif ext in {".las", ".laz"}:
        try:
            import laspy
        except ImportError:
            raise FormatError(
                ext,
                "laspy not installed. Install with: pip install destill3d[lidar]",
            )

        las = laspy.read(str(file_path))
        points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)

        if len(points) == 0:
            raise FormatError(ext, f"LAS file contains no points: {file_path}")

        return points

    else:
        raise FormatError(ext, f"Unsupported point cloud format: {ext}")


def _point_cloud_to_mesh(points: np.ndarray) -> trimesh.Trimesh:
    """
    Convert point cloud to mesh via surface reconstruction.

    Falls back to PointCloud if reconstruction fails.

    Args:
        points: numpy array of shape (N, 3)

    Returns:
        trimesh.Trimesh or trimesh.PointCloud
    """
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()

        # Ball pivoting reconstruction
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2])
        )

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        if len(faces) > 0:
            return trimesh.Trimesh(vertices=vertices, faces=faces)
    except Exception:
        pass

    # Fallback: return as point cloud wrapped in a trimesh PointCloud
    return trimesh.PointCloud(points)


def _flatten_scene(scene: trimesh.Scene) -> trimesh.Trimesh:
    """
    Flatten a trimesh Scene to a single mesh.

    Args:
        scene: trimesh.Scene containing multiple geometries

    Returns:
        Single combined trimesh.Trimesh
    """
    # Get all meshes from scene
    meshes = []
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            meshes.append(geom)

    if not meshes:
        raise GeometryError("Scene contains no valid meshes")

    if len(meshes) == 1:
        return meshes[0]

    # Concatenate all meshes
    try:
        combined = trimesh.util.concatenate(meshes)
        return combined
    except Exception as e:
        raise GeometryError(f"Failed to combine scene meshes: {e}")


def _validate_mesh(mesh: trimesh.Trimesh) -> None:
    """
    Validate mesh geometry.

    Args:
        mesh: Mesh to validate

    Raises:
        GeometryError: If mesh is invalid
    """
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise GeometryError("Invalid mesh: missing vertices or faces")

    if len(mesh.vertices) == 0:
        raise GeometryError("Mesh has no vertices")

    if len(mesh.faces) == 0:
        raise GeometryError("Mesh has no faces")

    if mesh.area < 1e-10:
        raise GeometryError("Mesh has zero surface area")

    # Check for degenerate geometry
    if not mesh.is_volume and not mesh.is_watertight:
        # This is fine - many models aren't watertight
        pass

    # Check for NaN/Inf values
    if not np.isfinite(mesh.vertices).all():
        raise GeometryError("Mesh contains NaN or Inf vertex coordinates")


def get_mesh_info(mesh: trimesh.Trimesh) -> dict:
    """
    Get detailed information about a mesh.

    Args:
        mesh: Input mesh

    Returns:
        Dictionary of mesh properties
    """
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]

    return {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "is_watertight": mesh.is_watertight,
        "is_convex": mesh.is_convex,
        "surface_area": float(mesh.area),
        "volume": float(abs(mesh.volume)) if mesh.is_watertight else 0.0,
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "dimensions": dimensions.tolist(),
        "euler_number": mesh.euler_number,
        "center_of_mass": mesh.center_mass.tolist() if mesh.is_watertight else None,
    }
