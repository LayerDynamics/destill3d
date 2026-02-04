"""
CAD B-rep tessellation for Destill3D.

Converts STEP, IGES, and BREP files to triangular meshes using pythonocc-core.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import trimesh

from destill3d.core.exceptions import TessellationError, FormatError


@dataclass
class TessellationConfig:
    """Configuration for B-rep tessellation."""

    linear_deflection: float = 0.001
    """Linear deflection tolerance (smaller = finer mesh)."""

    angular_deflection: float = 0.5
    """Angular deflection in radians (smaller = finer mesh)."""

    relative: bool = False
    """Use relative deflection (scaled to model bounding box)."""

    @classmethod
    def preview(cls) -> "TessellationConfig":
        """Quick preview quality - fast but coarse."""
        return cls(linear_deflection=0.01, angular_deflection=0.8)

    @classmethod
    def standard(cls) -> "TessellationConfig":
        """Standard quality - balanced speed and quality."""
        return cls(linear_deflection=0.001, angular_deflection=0.5)

    @classmethod
    def high_quality(cls) -> "TessellationConfig":
        """High quality - slow but detailed."""
        return cls(linear_deflection=0.0001, angular_deflection=0.2)


def tessellate_cad_file(
    file_path: Path,
    config: Optional[TessellationConfig] = None,
) -> trimesh.Trimesh:
    """
    Convert a CAD file (STEP/IGES/BREP) to a triangular mesh.

    Uses pythonocc-core for B-rep loading and tessellation.

    Args:
        file_path: Path to CAD file
        config: Tessellation configuration

    Returns:
        trimesh.Trimesh of the tessellated geometry

    Raises:
        TessellationError: If tessellation fails
        FormatError: If file format is not supported
    """
    if config is None:
        config = TessellationConfig.standard()

    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    # Check for pythonocc-core
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IGESControl import IGESControl_Reader
        from OCC.Core.BRepTools import breptools_Read
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.TopoDS import TopoDS_Shape
    except ImportError:
        raise TessellationError(
            "pythonocc-core is required for CAD file processing",
            "Install with: pip install destill3d[cad] or conda install -c conda-forge pythonocc-core",
        )

    # Load the shape based on file type
    try:
        if ext in (".step", ".stp"):
            shape = _load_step(file_path)
        elif ext in (".iges", ".igs"):
            shape = _load_iges(file_path)
        elif ext in (".brep", ".brp"):
            shape = _load_brep(file_path)
        else:
            raise FormatError(ext, f"Unsupported CAD format: {ext}")
    except Exception as e:
        if isinstance(e, (FormatError, TessellationError)):
            raise
        raise TessellationError(f"Failed to load CAD file: {e}")

    # Tessellate the shape
    try:
        vertices, faces = _tessellate_shape(shape, config)
    except Exception as e:
        raise TessellationError(f"Tessellation failed: {e}")

    if len(vertices) == 0 or len(faces) == 0:
        raise TessellationError("Tessellation produced empty mesh")

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def _load_step(file_path: Path):
    """Load STEP file and return TopoDS_Shape."""
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(file_path))

    if status != IFSelect_RetDone:
        raise TessellationError(f"Failed to read STEP file: status {status}")

    reader.TransferRoots()
    shape = reader.OneShape()

    if shape.IsNull():
        raise TessellationError("STEP file contains no valid geometry")

    return shape


def _load_iges(file_path: Path):
    """Load IGES file and return TopoDS_Shape."""
    from OCC.Core.IGESControl import IGESControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone

    reader = IGESControl_Reader()
    status = reader.ReadFile(str(file_path))

    if status != IFSelect_RetDone:
        raise TessellationError(f"Failed to read IGES file: status {status}")

    reader.TransferRoots()
    shape = reader.OneShape()

    if shape.IsNull():
        raise TessellationError("IGES file contains no valid geometry")

    return shape


def _load_brep(file_path: Path):
    """Load BREP file and return TopoDS_Shape."""
    from OCC.Core.BRepTools import breptools_Read
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Shape

    builder = BRep_Builder()
    shape = TopoDS_Shape()

    success = breptools_Read(shape, str(file_path), builder)

    if not success:
        raise TessellationError("Failed to read BREP file")

    if shape.IsNull():
        raise TessellationError("BREP file contains no valid geometry")

    return shape


def _tessellate_shape(
    shape,
    config: TessellationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tessellate a TopoDS_Shape to vertices and faces.

    Args:
        shape: OCC TopoDS_Shape
        config: Tessellation configuration

    Returns:
        Tuple of (vertices, faces) as numpy arrays
    """
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods_Face

    # Perform tessellation
    mesh = BRepMesh_IncrementalMesh(
        shape,
        config.linear_deflection,
        config.relative,
        config.angular_deflection,
        True,  # InParallel
    )
    mesh.Perform()

    if not mesh.IsDone():
        raise TessellationError("Tessellation did not complete")

    # Extract vertices and faces from all faces
    all_vertices: List[List[float]] = []
    all_faces: List[List[int]] = []
    vertex_offset = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods_Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)

        if triangulation is not None:
            # Extract vertices
            transformation = location.Transformation()

            for i in range(1, triangulation.NbNodes() + 1):
                node = triangulation.Node(i)
                # Apply transformation if not identity
                if not location.IsIdentity():
                    node = node.Transformed(transformation)
                all_vertices.append([node.X(), node.Y(), node.Z()])

            # Extract triangles
            for i in range(1, triangulation.NbTriangles() + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                # OCC uses 1-based indexing, convert to 0-based
                all_faces.append([
                    n1 - 1 + vertex_offset,
                    n2 - 1 + vertex_offset,
                    n3 - 1 + vertex_offset,
                ])

            vertex_offset = len(all_vertices)

        explorer.Next()

    vertices = np.array(all_vertices, dtype=np.float64)
    faces = np.array(all_faces, dtype=np.int64)

    return vertices, faces


def get_cad_info(file_path: Path) -> dict:
    """
    Get information about a CAD file without full tessellation.

    Args:
        file_path: Path to CAD file

    Returns:
        Dictionary with CAD file information
    """
    try:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib_Add
    except ImportError:
        return {"error": "pythonocc-core not installed"}

    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    try:
        if ext in (".step", ".stp"):
            shape = _load_step(file_path)
        elif ext in (".iges", ".igs"):
            shape = _load_iges(file_path)
        elif ext in (".brep", ".brp"):
            shape = _load_brep(file_path)
        else:
            return {"error": f"Unsupported format: {ext}"}

        # Get bounding box
        bbox = Bnd_Box()
        brepbndlib_Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        return {
            "format": ext.replace(".", "").upper(),
            "bounds_min": [xmin, ymin, zmin],
            "bounds_max": [xmax, ymax, zmax],
            "dimensions": [xmax - xmin, ymax - ymin, zmax - zmin],
        }

    except Exception as e:
        return {"error": str(e)}
