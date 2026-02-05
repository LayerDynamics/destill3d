"""
Destill3D: 3D model feature extraction and classification toolkit.

Destill3D is a command-line application and library for automated acquisition,
feature extraction, and classification-ready distillation of 3D models.
"""

__version__ = "0.1.0"
__author__ = "LayerDynamics / Lattice Labs"

from destill3d.core.config import Destill3DConfig
from destill3d.core.snapshot import (
    Snapshot,
    Provenance,
    GeometryData,
    Features,
    Prediction,
    ProcessingMetadata,
)
from destill3d.core.exceptions import (
    Destill3DError,
    ExtractionError,
    ClassificationError,
    DatabaseError,
)
from destill3d.api import Destill3D

__all__ = [
    # Version
    "__version__",
    # Config
    "Destill3DConfig",
    # Unified API
    "Destill3D",
    # Snapshot
    "Snapshot",
    "Provenance",
    "GeometryData",
    "Features",
    "Prediction",
    "ProcessingMetadata",
    # Exceptions
    "Destill3DError",
    "ExtractionError",
    "ClassificationError",
    "DatabaseError",
]
