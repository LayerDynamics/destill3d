"""
Platform-specific adapters for model acquisition.

Each adapter implements the PlatformAdapter protocol for a
specific 3D model hosting platform.
"""

from destill3d.acquire.platforms.cults3d import Cults3DAdapter
from destill3d.acquire.platforms.github import GitHubAdapter
from destill3d.acquire.platforms.grabcad import GrabCADAdapter
from destill3d.acquire.platforms.huggingface import HuggingFaceAdapter
from destill3d.acquire.platforms.local import LocalFilesystemAdapter
from destill3d.acquire.platforms.sketchfab import SketchfabAdapter
from destill3d.acquire.platforms.thangs import ThangsAdapter
from destill3d.acquire.platforms.thingiverse import ThingiverseAdapter

__all__ = [
    "ThingiverseAdapter",
    "SketchfabAdapter",
    "LocalFilesystemAdapter",
    "GrabCADAdapter",
    "Cults3DAdapter",
    "ThangsAdapter",
    "GitHubAdapter",
    "HuggingFaceAdapter",
]
