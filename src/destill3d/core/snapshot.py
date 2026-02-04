"""
Snapshot data structure for Destill3D.

A Snapshot is the central data structure that contains:
- Provenance: where the model came from
- GeometryData: sampled points, normals, curvature
- Features: global shape descriptors
- Predictions: classification results
- ProcessingMetadata: extraction/classification timing and settings
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import gzip
import json
import struct
import uuid

import numpy as np

# Snapshot format version - increment when format changes
SNAPSHOT_VERSION = 1

# Magic bytes for snapshot files
SNAPSHOT_MAGIC = b"D3DS"  # Destill3D Snapshot


@dataclass
class Provenance:
    """Source information and metadata for a 3D model."""

    platform: str = "local"
    source_url: str = ""
    source_id: str = ""
    title: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    original_filename: str = ""
    original_format: str = ""
    original_file_size: int = 0
    original_file_hash: str = ""  # SHA256
    source_created_at: Optional[datetime] = None
    acquired_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "platform": self.platform,
            "source_url": self.source_url,
            "source_id": self.source_id,
            "title": self.title,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "tags": self.tags,
            "original_filename": self.original_filename,
            "original_format": self.original_format,
            "original_file_size": self.original_file_size,
            "original_file_hash": self.original_file_hash,
            "source_created_at": self.source_created_at.isoformat() if self.source_created_at else None,
            "acquired_at": self.acquired_at.isoformat() if self.acquired_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Provenance":
        """Create from dictionary."""
        data = data.copy()
        if data.get("source_created_at"):
            data["source_created_at"] = datetime.fromisoformat(data["source_created_at"])
        if data.get("acquired_at"):
            data["acquired_at"] = datetime.fromisoformat(data["acquired_at"])
        return cls(**data)


@dataclass
class GeometryData:
    """Sampled point cloud geometry with per-point features."""

    points: np.ndarray  # (N, 3) float32 - normalized coordinates
    normals: np.ndarray  # (N, 3) float32 - unit normals
    curvature: np.ndarray  # (N,) float32 - local curvature values

    # Normalization parameters (to recover original scale)
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    scale: float = 1.0

    # Computed after init
    point_count: int = 0

    def __post_init__(self):
        # Ensure correct dtypes
        self.points = np.asarray(self.points, dtype=np.float32)
        self.normals = np.asarray(self.normals, dtype=np.float32)
        self.curvature = np.asarray(self.curvature, dtype=np.float32)
        self.centroid = np.asarray(self.centroid, dtype=np.float32)
        self.point_count = len(self.points)

    def to_dict(self) -> dict:
        """Convert to dictionary (arrays as lists for JSON)."""
        return {
            "points": self.points.tolist(),
            "normals": self.normals.tolist(),
            "curvature": self.curvature.tolist(),
            "centroid": self.centroid.tolist(),
            "scale": self.scale,
            "point_count": self.point_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeometryData":
        """Create from dictionary."""
        return cls(
            points=np.array(data["points"], dtype=np.float32),
            normals=np.array(data["normals"], dtype=np.float32),
            curvature=np.array(data["curvature"], dtype=np.float32),
            centroid=np.array(data["centroid"], dtype=np.float32),
            scale=data["scale"],
        )

    def to_bytes(self) -> bytes:
        """Serialize geometry to compact binary format."""
        # Header: point_count (4 bytes), scale (4 bytes), centroid (12 bytes)
        header = struct.pack("<If3f", self.point_count, self.scale, *self.centroid)
        # Data: points (N*12), normals (N*12), curvature (N*4)
        data = self.points.tobytes() + self.normals.tobytes() + self.curvature.tobytes()
        return header + data

    @classmethod
    def from_bytes(cls, data: bytes) -> "GeometryData":
        """Deserialize from binary format."""
        # Parse header
        point_count, scale, cx, cy, cz = struct.unpack("<If3f", data[:20])
        centroid = np.array([cx, cy, cz], dtype=np.float32)

        # Parse arrays
        offset = 20
        points_size = point_count * 3 * 4
        normals_size = point_count * 3 * 4
        curvature_size = point_count * 4

        points = np.frombuffer(data[offset:offset + points_size], dtype=np.float32).reshape(-1, 3)
        offset += points_size

        normals = np.frombuffer(data[offset:offset + normals_size], dtype=np.float32).reshape(-1, 3)
        offset += normals_size

        curvature = np.frombuffer(data[offset:offset + curvature_size], dtype=np.float32)

        return cls(
            points=points.copy(),
            normals=normals.copy(),
            curvature=curvature.copy(),
            centroid=centroid,
            scale=scale,
        )


@dataclass
class Features:
    """Global shape features and mesh properties."""

    global_features: np.ndarray  # (32,) float32 - rotation-invariant descriptors

    # Mesh properties
    surface_area: float = 0.0
    volume: float = 0.0
    is_watertight: bool = False

    # Original mesh stats
    original_vertex_count: int = 0
    original_face_count: int = 0

    # Bounding box (in original scale)
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def __post_init__(self):
        self.global_features = np.asarray(self.global_features, dtype=np.float32)
        self.bbox_min = np.asarray(self.bbox_min, dtype=np.float32)
        self.bbox_max = np.asarray(self.bbox_max, dtype=np.float32)

    @property
    def bbox_dimensions(self) -> np.ndarray:
        """Get bounding box dimensions."""
        return self.bbox_max - self.bbox_min

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "global_features": self.global_features.tolist(),
            "surface_area": self.surface_area,
            "volume": self.volume,
            "is_watertight": self.is_watertight,
            "original_vertex_count": self.original_vertex_count,
            "original_face_count": self.original_face_count,
            "bbox_min": self.bbox_min.tolist(),
            "bbox_max": self.bbox_max.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Features":
        """Create from dictionary."""
        return cls(
            global_features=np.array(data["global_features"], dtype=np.float32),
            surface_area=data["surface_area"],
            volume=data["volume"],
            is_watertight=data["is_watertight"],
            original_vertex_count=data["original_vertex_count"],
            original_face_count=data["original_face_count"],
            bbox_min=np.array(data["bbox_min"], dtype=np.float32),
            bbox_max=np.array(data["bbox_max"], dtype=np.float32),
        )


@dataclass
class Prediction:
    """A single classification prediction."""

    label: str
    confidence: float
    taxonomy: str
    model_name: str
    rank: int = 1
    uncertainty: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "taxonomy": self.taxonomy,
            "model_name": self.model_name,
            "rank": self.rank,
            "uncertainty": self.uncertainty,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Prediction":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ProcessingMetadata:
    """Metadata about how the snapshot was created."""

    destill3d_version: str = "0.1.0"
    feature_extractor_version: str = "1"

    # Extraction settings
    target_points: int = 2048
    sampling_strategy: str = "hybrid"
    tessellation_deflection: float = 0.001

    # Timing
    extraction_time_ms: float = 0.0
    classification_time_ms: float = 0.0

    # Warnings/issues encountered
    warnings: List[str] = field(default_factory=list)

    # Timestamps
    processed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "destill3d_version": self.destill3d_version,
            "feature_extractor_version": self.feature_extractor_version,
            "target_points": self.target_points,
            "sampling_strategy": self.sampling_strategy,
            "tessellation_deflection": self.tessellation_deflection,
            "extraction_time_ms": self.extraction_time_ms,
            "classification_time_ms": self.classification_time_ms,
            "warnings": self.warnings,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingMetadata":
        """Create from dictionary."""
        data = data.copy()
        if data.get("processed_at"):
            data["processed_at"] = datetime.fromisoformat(data["processed_at"])
        return cls(**data)


@dataclass
class Snapshot:
    """
    Complete snapshot of a 3D model.

    A snapshot is a compressed, self-contained representation that includes:
    - Identity: unique IDs and version
    - Provenance: source information
    - Geometry: sampled point cloud with features
    - Features: global shape descriptors
    - Predictions: classification results
    - Metadata: processing information
    """

    # Identity
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""  # platform:source_id
    version: int = SNAPSHOT_VERSION

    # Components
    provenance: Provenance = field(default_factory=Provenance)
    geometry: Optional[GeometryData] = None
    features: Optional[Features] = None
    predictions: List[Prediction] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None  # (1024,) float32 from classifier
    processing: ProcessingMetadata = field(default_factory=ProcessingMetadata)

    def save(self, path: Path, compress: bool = True) -> None:
        """
        Save snapshot to file.

        Format: Magic (4) + Version (4) + Compressed/Raw JSON+Binary
        """
        path = Path(path)

        # Build the data structure
        data = {
            "snapshot_id": self.snapshot_id,
            "model_id": self.model_id,
            "version": self.version,
            "provenance": self.provenance.to_dict(),
            "geometry_bytes": None,  # Placeholder for binary data
            "features": self.features.to_dict() if self.features else None,
            "predictions": [p.to_dict() for p in self.predictions],
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "processing": self.processing.to_dict(),
        }

        # Serialize geometry separately as binary
        geometry_bytes = self.geometry.to_bytes() if self.geometry else b""

        # Create the final payload
        json_bytes = json.dumps(data).encode("utf-8")

        # Combine: json_length (4) + json_data + geometry_data
        payload = struct.pack("<I", len(json_bytes)) + json_bytes + geometry_bytes

        if compress:
            payload = gzip.compress(payload, compresslevel=6)

        # Write with header
        with open(path, "wb") as f:
            f.write(SNAPSHOT_MAGIC)
            f.write(struct.pack("<I", SNAPSHOT_VERSION))
            f.write(struct.pack("<B", 1 if compress else 0))  # Compression flag
            f.write(payload)

    @classmethod
    def load(cls, path: Path) -> "Snapshot":
        """Load snapshot from file."""
        path = Path(path)

        with open(path, "rb") as f:
            # Read header
            magic = f.read(4)
            if magic != SNAPSHOT_MAGIC:
                raise ValueError(f"Invalid snapshot file: wrong magic bytes")

            version = struct.unpack("<I", f.read(4))[0]
            if version > SNAPSHOT_VERSION:
                raise ValueError(f"Snapshot version {version} is newer than supported {SNAPSHOT_VERSION}")

            compressed = struct.unpack("<B", f.read(1))[0]
            payload = f.read()

        if compressed:
            payload = gzip.decompress(payload)

        # Parse payload
        json_length = struct.unpack("<I", payload[:4])[0]
        json_bytes = payload[4:4 + json_length]
        geometry_bytes = payload[4 + json_length:]

        data = json.loads(json_bytes.decode("utf-8"))

        # Reconstruct snapshot
        snapshot = cls(
            snapshot_id=data["snapshot_id"],
            model_id=data["model_id"],
            version=data["version"],
            provenance=Provenance.from_dict(data["provenance"]),
            geometry=GeometryData.from_bytes(geometry_bytes) if geometry_bytes else None,
            features=Features.from_dict(data["features"]) if data["features"] else None,
            predictions=[Prediction.from_dict(p) for p in data["predictions"]],
            embedding=np.array(data["embedding"], dtype=np.float32) if data["embedding"] else None,
            processing=ProcessingMetadata.from_dict(data["processing"]),
        )

        return snapshot

    def to_dict(self) -> dict:
        """Convert snapshot to dictionary (for JSON export)."""
        return {
            "snapshot_id": self.snapshot_id,
            "model_id": self.model_id,
            "version": self.version,
            "provenance": self.provenance.to_dict(),
            "geometry": self.geometry.to_dict() if self.geometry else None,
            "features": self.features.to_dict() if self.features else None,
            "predictions": [p.to_dict() for p in self.predictions],
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "processing": self.processing.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Snapshot":
        """Create snapshot from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            model_id=data["model_id"],
            version=data["version"],
            provenance=Provenance.from_dict(data["provenance"]),
            geometry=GeometryData.from_dict(data["geometry"]) if data["geometry"] else None,
            features=Features.from_dict(data["features"]) if data["features"] else None,
            predictions=[Prediction.from_dict(p) for p in data["predictions"]],
            embedding=np.array(data["embedding"], dtype=np.float32) if data["embedding"] else None,
            processing=ProcessingMetadata.from_dict(data["processing"]),
        )

    @property
    def top_prediction(self) -> Optional[Prediction]:
        """Get the top classification prediction."""
        if not self.predictions:
            return None
        return min(self.predictions, key=lambda p: p.rank)

    @property
    def compression_ratio(self) -> float:
        """Estimate compression ratio from original file."""
        if self.provenance.original_file_size == 0:
            return 0.0
        # Estimate snapshot size (rough)
        snapshot_size = (
            self.geometry.point_count * (3 + 3 + 1) * 4  # points + normals + curvature
            + 32 * 4  # global features
            + 1000  # metadata overhead estimate
        )
        return self.provenance.original_file_size / snapshot_size

    def __repr__(self) -> str:
        top = self.top_prediction
        pred_str = f"{top.label} ({top.confidence:.1%})" if top else "unclassified"
        return (
            f"Snapshot(id={self.snapshot_id[:8]}..., "
            f"points={self.geometry.point_count if self.geometry else 0}, "
            f"prediction={pred_str})"
        )
