"""
Protocol Buffers module for Destill3D.

Contains snapshot serialization schema and helpers for
converting between Snapshot objects and protobuf format.

Note: The .proto file defines the schema. For full protobuf
serialization, compile with protoc. These helpers provide
a Python-native serialization that follows the protobuf schema
structure without requiring compiled proto files.
"""

import json
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def snapshot_to_proto_dict(snapshot) -> dict:
    """
    Convert a Snapshot to a protobuf-compatible dictionary structure.

    This follows the schema defined in snapshot.proto and can be
    used with protobuf JSON serialization or as a standalone format.

    Args:
        snapshot: Snapshot object to serialize.

    Returns:
        Dictionary matching the protobuf message structure.
    """
    from destill3d.core.snapshot import Snapshot

    result = {
        "snapshot_id": snapshot.snapshot_id,
        "model_id": snapshot.model_id,
        "version": snapshot.version,
    }

    # Provenance
    if snapshot.provenance:
        p = snapshot.provenance
        result["provenance"] = {
            "platform": p.platform,
            "source_url": p.source_url,
            "source_id": p.source_id,
            "title": p.title or "",
            "description": p.description or "",
            "author": p.author or "",
            "license": p.license or "",
            "tags": p.tags,
            "original_filename": p.original_filename,
            "original_format": p.original_format,
            "original_file_size": p.original_file_size,
            "original_file_hash": p.original_file_hash,
            "source_created_at": p.source_created_at.isoformat() if p.source_created_at else "",
            "source_modified_at": p.source_modified_at.isoformat() if p.source_modified_at else "",
            "acquired_at": p.acquired_at.isoformat() if p.acquired_at else "",
        }

    # Geometry - store arrays as base64-encoded bytes
    if snapshot.geometry:
        g = snapshot.geometry
        import base64
        result["geometry"] = {
            "points": base64.b64encode(g.points.tobytes()).decode("ascii"),
            "normals": base64.b64encode(g.normals.tobytes()).decode("ascii"),
            "curvature": base64.b64encode(g.curvature.tobytes()).decode("ascii"),
            "centroid": base64.b64encode(g.centroid.tobytes()).decode("ascii"),
            "scale": g.scale,
            "point_count": g.point_count,
        }
        if g.view_images is not None:
            result["geometry"]["view_images"] = base64.b64encode(
                g.view_images.tobytes()
            ).decode("ascii")

    # Features
    if snapshot.features:
        f = snapshot.features
        import base64
        result["features"] = {
            "global_features": base64.b64encode(f.global_features.tobytes()).decode("ascii"),
            "surface_area": f.surface_area,
            "volume": f.volume,
            "is_watertight": f.is_watertight,
            "original_vertex_count": f.original_vertex_count,
            "original_face_count": f.original_face_count,
            "bounding_box": {
                "min_x": float(f.bbox_min[0]),
                "min_y": float(f.bbox_min[1]),
                "min_z": float(f.bbox_min[2]),
                "max_x": float(f.bbox_max[0]),
                "max_y": float(f.bbox_max[1]),
                "max_z": float(f.bbox_max[2]),
            },
        }

    # Predictions
    result["predictions"] = [
        {
            "label": pred.label,
            "confidence": pred.confidence,
            "taxonomy": pred.taxonomy,
            "model_name": pred.model_name,
            "rank": pred.rank,
            "uncertainty": pred.uncertainty or 0.0,
        }
        for pred in snapshot.predictions
    ]

    # Embedding
    if snapshot.embedding is not None:
        import base64
        result["embedding"] = base64.b64encode(
            snapshot.embedding.astype(np.float32).tobytes()
        ).decode("ascii")

    # Processing
    if snapshot.processing:
        pm = snapshot.processing
        result["processing"] = {
            "destill3d_version": pm.destill3d_version,
            "feature_extractor_version": pm.feature_extractor_version,
            "target_points": pm.target_points,
            "sampling_strategy": pm.sampling_strategy,
            "tessellation_deflection": pm.tessellation_deflection,
            "mesh_quality_score": pm.mesh_quality_score,
            "extraction_time_ms": pm.extraction_time_ms,
            "classification_time_ms": pm.classification_time_ms,
            "download_time_ms": pm.download_time_ms,
            "warnings": pm.warnings,
            "processed_at": pm.processed_at.isoformat() if pm.processed_at else "",
        }

    return result


def proto_dict_to_snapshot(data: dict):
    """
    Convert a protobuf-compatible dictionary back to a Snapshot.

    Args:
        data: Dictionary from snapshot_to_proto_dict.

    Returns:
        Snapshot object.
    """
    import base64
    from datetime import datetime

    from destill3d.core.snapshot import (
        Features,
        GeometryData,
        Prediction,
        ProcessingMetadata,
        Provenance,
        Snapshot,
    )

    # Provenance
    prov_data = data.get("provenance", {})
    provenance = Provenance(
        platform=prov_data.get("platform", "local"),
        source_url=prov_data.get("source_url", ""),
        source_id=prov_data.get("source_id", ""),
        title=prov_data.get("title") or None,
        description=prov_data.get("description") or None,
        author=prov_data.get("author") or None,
        license=prov_data.get("license") or None,
        tags=prov_data.get("tags", []),
        original_filename=prov_data.get("original_filename", ""),
        original_format=prov_data.get("original_format", ""),
        original_file_size=prov_data.get("original_file_size", 0),
        original_file_hash=prov_data.get("original_file_hash", ""),
    )

    if prov_data.get("source_created_at"):
        provenance.source_created_at = datetime.fromisoformat(prov_data["source_created_at"])
    if prov_data.get("source_modified_at"):
        provenance.source_modified_at = datetime.fromisoformat(prov_data["source_modified_at"])
    if prov_data.get("acquired_at"):
        provenance.acquired_at = datetime.fromisoformat(prov_data["acquired_at"])

    # Geometry
    geometry = None
    if "geometry" in data:
        g = data["geometry"]
        geometry = GeometryData(
            points=np.frombuffer(base64.b64decode(g["points"]), dtype=np.float32).reshape(-1, 3).copy(),
            normals=np.frombuffer(base64.b64decode(g["normals"]), dtype=np.float32).reshape(-1, 3).copy(),
            curvature=np.frombuffer(base64.b64decode(g["curvature"]), dtype=np.float32).copy(),
            centroid=np.frombuffer(base64.b64decode(g["centroid"]), dtype=np.float32).copy(),
            scale=g["scale"],
        )

    # Features
    features = None
    if "features" in data:
        f = data["features"]
        bb = f.get("bounding_box", {})
        features = Features(
            global_features=np.frombuffer(base64.b64decode(f["global_features"]), dtype=np.float32).copy(),
            surface_area=f["surface_area"],
            volume=f["volume"],
            is_watertight=f["is_watertight"],
            original_vertex_count=f["original_vertex_count"],
            original_face_count=f["original_face_count"],
            bbox_min=np.array([bb.get("min_x", 0), bb.get("min_y", 0), bb.get("min_z", 0)], dtype=np.float32),
            bbox_max=np.array([bb.get("max_x", 0), bb.get("max_y", 0), bb.get("max_z", 0)], dtype=np.float32),
        )

    # Predictions
    predictions = [
        Prediction(
            label=p["label"],
            confidence=p["confidence"],
            taxonomy=p["taxonomy"],
            model_name=p["model_name"],
            rank=p["rank"],
            uncertainty=p.get("uncertainty"),
        )
        for p in data.get("predictions", [])
    ]

    # Embedding
    embedding = None
    if data.get("embedding"):
        embedding = np.frombuffer(
            base64.b64decode(data["embedding"]), dtype=np.float32
        ).copy()

    # Processing
    processing = ProcessingMetadata()
    if "processing" in data:
        pm = data["processing"]
        processing = ProcessingMetadata(
            destill3d_version=pm.get("destill3d_version", "0.1.0"),
            feature_extractor_version=pm.get("feature_extractor_version", "1"),
            target_points=pm.get("target_points", 2048),
            sampling_strategy=pm.get("sampling_strategy", "hybrid"),
            tessellation_deflection=pm.get("tessellation_deflection", 0.001),
            mesh_quality_score=pm.get("mesh_quality_score", 0.0),
            extraction_time_ms=pm.get("extraction_time_ms", 0.0),
            classification_time_ms=pm.get("classification_time_ms", 0.0),
            download_time_ms=pm.get("download_time_ms", 0.0),
            warnings=pm.get("warnings", []),
        )
        if pm.get("processed_at"):
            processing.processed_at = datetime.fromisoformat(pm["processed_at"])

    return Snapshot(
        snapshot_id=data["snapshot_id"],
        model_id=data["model_id"],
        version=data.get("version", 1),
        provenance=provenance,
        geometry=geometry,
        features=features,
        predictions=predictions,
        embedding=embedding,
        processing=processing,
    )
