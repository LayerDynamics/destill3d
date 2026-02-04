"""
Classification module for Destill3D.

Handles model registry and inference for 3D point cloud classification.
"""

from destill3d.classify.inference import Classifier
from destill3d.classify.registry import (
    ModelRegistry,
    RegisteredModel,
    ModelFormat,
    TaxonomyConfig,
    TAXONOMIES,
    MODELNET40,
    MODELNET10,
    SCANOBJECTNN,
)

__all__ = [
    "Classifier",
    "ModelRegistry",
    "RegisteredModel",
    "ModelFormat",
    "TaxonomyConfig",
    "TAXONOMIES",
    "MODELNET40",
    "MODELNET10",
    "SCANOBJECTNN",
]
