"""
REST API server for Destill3D.

Provides HTTP endpoints for feature extraction, classification,
database operations, and model acquisition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from destill3d.api import Destill3D
from destill3d.core.config import Destill3DConfig

logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(
    title="Destill3D API",
    description="REST API for 3D model feature extraction and classification",
    version="0.1.0",
)

# Global Destill3D instance (lazy initialized)
_d3d: Optional[Destill3D] = None


def get_d3d() -> Destill3D:
    """Get or initialize the Destill3D instance."""
    global _d3d
    if _d3d is None:
        config = Destill3DConfig.load()
        _d3d = Destill3D(config)
    return _d3d


# ─────────────────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"


class ExtractRequest(BaseModel):
    """Request for file extraction."""

    file_path: str = Field(..., description="Path to 3D file")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class ExtractResponse(BaseModel):
    """Response from extraction."""

    snapshot_id: str
    model_id: str
    point_count: int
    is_watertight: bool
    surface_area: float
    volume: float


class ClassifyRequest(BaseModel):
    """Request for classification."""

    snapshot_id: str = Field(..., description="Snapshot ID to classify")
    model_id: str = Field("pointnet2_ssg_mn40", description="Classification model")
    top_k: int = Field(5, ge=1, le=40, description="Number of top predictions")


class PredictionResponse(BaseModel):
    """Classification prediction."""

    label: str
    confidence: float
    rank: int
    taxonomy: str


class ClassifyResponse(BaseModel):
    """Response from classification."""

    snapshot_id: str
    predictions: List[PredictionResponse]


class ZeroShotRequest(BaseModel):
    """Request for zero-shot classification."""

    snapshot_id: str = Field(..., description="Snapshot ID to classify")
    classes: List[str] = Field(..., description="Class labels for zero-shot")
    top_k: int = Field(5, ge=1, le=40, description="Number of top predictions")


class SearchRequest(BaseModel):
    """Request for platform search."""

    query: str = Field(..., description="Search query")
    platform: str = Field("thingiverse", description="Platform to search")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")
    license_filter: Optional[str] = Field(None, description="License filter")


class SearchResultResponse(BaseModel):
    """Search result item."""

    platform: str
    model_id: str
    title: str
    author: str
    url: str
    thumbnail_url: Optional[str] = None
    download_count: Optional[int] = None
    license: Optional[str] = None


class SearchResponse(BaseModel):
    """Response from platform search."""

    results: List[SearchResultResponse]
    total_count: int
    page: int
    has_more: bool


class QueueRequest(BaseModel):
    """Request to add URLs to queue."""

    urls: List[str] = Field(..., description="URLs to queue for download")
    priority: int = Field(0, ge=0, le=100, description="Priority level")


class QueueResponse(BaseModel):
    """Response from queue operation."""

    queued_count: int


class StatsResponse(BaseModel):
    """Database statistics."""

    total_snapshots: int
    classified_snapshots: int
    unclassified_snapshots: int
    with_embeddings: int
    platform_counts: Dict[str, int]
    label_counts: Dict[str, int]
    db_size_mb: float


class QueryRequest(BaseModel):
    """Request for database query."""

    platform: Optional[str] = Field(None, description="Filter by platform")
    label: Optional[str] = Field(None, description="Filter by label")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    limit: int = Field(100, ge=1, le=1000)


class SnapshotSummary(BaseModel):
    """Snapshot summary for list responses."""

    snapshot_id: str
    model_id: str
    platform: str
    title: Optional[str] = None
    point_count: int
    top_label: Optional[str] = None
    top_confidence: Optional[float] = None


class QueryResponse(BaseModel):
    """Response from database query."""

    snapshots: List[SnapshotSummary]
    count: int


class SimilarityResult(BaseModel):
    """Similarity search result."""

    snapshot_id: str
    similarity: float


class SimilarityResponse(BaseModel):
    """Response from similarity search."""

    query_id: str
    results: List[SimilarityResult]


# ─────────────────────────────────────────────────────────────────────────────
# Health Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status."""
    return HealthResponse()


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint with API info."""
    return HealthResponse()


# ─────────────────────────────────────────────────────────────────────────────
# Extract Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/extract/file", response_model=ExtractResponse, tags=["Extract"])
async def extract_file(request: ExtractRequest):
    """
    Extract features from a local 3D file.

    Processes the file and returns a snapshot with extracted features.
    """
    d3d = get_d3d()
    file_path = Path(request.file_path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        snapshot = d3d.extract.from_file(file_path, request.metadata)

        return ExtractResponse(
            snapshot_id=snapshot.snapshot_id,
            model_id=snapshot.model_id,
            point_count=snapshot.geometry.point_count if snapshot.geometry else 0,
            is_watertight=snapshot.features.is_watertight if snapshot.features else False,
            surface_area=snapshot.features.surface_area if snapshot.features else 0.0,
            volume=snapshot.features.volume if snapshot.features else 0.0,
        )
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/upload", response_model=ExtractResponse, tags=["Extract"])
async def extract_upload(file: UploadFile = File(...)):
    """
    Extract features from an uploaded 3D file.

    Accepts file upload and returns a snapshot with extracted features.
    """
    d3d = get_d3d()

    # Save uploaded file to temp directory
    temp_path = d3d.config.temp_dir / file.filename
    try:
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        snapshot = d3d.extract.from_file(temp_path)

        return ExtractResponse(
            snapshot_id=snapshot.snapshot_id,
            model_id=snapshot.model_id,
            point_count=snapshot.geometry.point_count if snapshot.geometry else 0,
            is_watertight=snapshot.features.is_watertight if snapshot.features else False,
            surface_area=snapshot.features.surface_area if snapshot.features else 0.0,
            volume=snapshot.features.volume if snapshot.features else 0.0,
        )
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Classify Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/classify", response_model=ClassifyResponse, tags=["Classify"])
async def classify_snapshot(request: ClassifyRequest):
    """
    Classify a snapshot using a trained model.

    Returns top-K predictions with confidence scores.
    """
    d3d = get_d3d()

    snapshot = d3d.db.get(request.snapshot_id)
    if snapshot is None:
        raise HTTPException(
            status_code=404, detail=f"Snapshot not found: {request.snapshot_id}"
        )

    try:
        snapshot = d3d.classify.classify(
            snapshot, model_id=request.model_id, top_k=request.top_k
        )

        predictions = [
            PredictionResponse(
                label=p.label,
                confidence=p.confidence,
                rank=p.rank,
                taxonomy=p.taxonomy,
            )
            for p in snapshot.predictions
        ]

        return ClassifyResponse(
            snapshot_id=snapshot.snapshot_id, predictions=predictions
        )
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/zero-shot", response_model=ClassifyResponse, tags=["Classify"])
async def zero_shot_classify(request: ZeroShotRequest):
    """
    Zero-shot classify a snapshot with arbitrary class labels.

    Uses CLIP/OpenShape embeddings for classification.
    """
    d3d = get_d3d()

    snapshot = d3d.db.get(request.snapshot_id)
    if snapshot is None:
        raise HTTPException(
            status_code=404, detail=f"Snapshot not found: {request.snapshot_id}"
        )

    try:
        snapshot = d3d.classify.zero_shot(
            snapshot, classes=request.classes, top_k=request.top_k
        )

        predictions = [
            PredictionResponse(
                label=p.label,
                confidence=p.confidence,
                rank=p.rank,
                taxonomy=p.taxonomy,
            )
            for p in snapshot.predictions
        ]

        return ClassifyResponse(
            snapshot_id=snapshot.snapshot_id, predictions=predictions
        )
    except Exception as e:
        logger.error(f"Zero-shot classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Database Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/db/stats", response_model=StatsResponse, tags=["Database"])
async def get_stats():
    """Get database statistics."""
    d3d = get_d3d()

    try:
        stats = d3d.db.stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/db/query", response_model=QueryResponse, tags=["Database"])
async def query_snapshots(request: QueryRequest):
    """Query snapshots with filters."""
    d3d = get_d3d()

    try:
        snapshots = d3d.db.query(
            platform=request.platform,
            label=request.label,
            min_confidence=request.min_confidence,
            limit=request.limit,
        )

        summaries = [
            SnapshotSummary(
                snapshot_id=s.snapshot_id,
                model_id=s.model_id,
                platform=s.provenance.platform,
                title=s.provenance.title,
                point_count=s.geometry.point_count if s.geometry else 0,
                top_label=s.top_prediction.label if s.top_prediction else None,
                top_confidence=s.top_prediction.confidence if s.top_prediction else None,
            )
            for s in snapshots
        ]

        return QueryResponse(snapshots=summaries, count=len(summaries))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/db/snapshot/{snapshot_id}", tags=["Database"])
async def get_snapshot(snapshot_id: str):
    """Get a snapshot by ID."""
    d3d = get_d3d()

    snapshot = d3d.db.get(snapshot_id)
    if snapshot is None:
        raise HTTPException(
            status_code=404, detail=f"Snapshot not found: {snapshot_id}"
        )

    return JSONResponse(content=snapshot.to_dict())


@app.delete("/db/snapshot/{snapshot_id}", tags=["Database"])
async def delete_snapshot(snapshot_id: str):
    """Delete a snapshot by ID."""
    d3d = get_d3d()

    try:
        # Get the snapshot first to check it exists
        snapshot = d3d.db.get(snapshot_id)
        if snapshot is None:
            raise HTTPException(
                status_code=404, detail=f"Snapshot not found: {snapshot_id}"
            )

        db = d3d.db._get_db()
        db.delete(snapshot.model_id)
        return {"status": "deleted", "snapshot_id": snapshot_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/db/similar/{snapshot_id}", response_model=SimilarityResponse, tags=["Database"])
async def find_similar(
    snapshot_id: str,
    k: int = Query(10, ge=1, le=100),
    min_similarity: float = Query(0.0, ge=0.0, le=1.0),
):
    """Find similar snapshots by embedding similarity."""
    d3d = get_d3d()

    try:
        db = d3d.db._get_db()
        results = db.find_similar(snapshot_id, k=k, min_similarity=min_similarity)

        return SimilarityResponse(
            query_id=snapshot_id,
            results=[
                SimilarityResult(snapshot_id=r["snapshot_id"], similarity=r["similarity"])
                for r in results
            ],
        )
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Acquire Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/acquire/search", response_model=SearchResponse, tags=["Acquire"])
async def search_platform(request: SearchRequest):
    """Search a platform for 3D models."""
    d3d = get_d3d()

    try:
        results = d3d.acquire.search(
            query=request.query,
            platform=request.platform,
            limit=request.limit,
            license_filter=request.license_filter,
        )

        return SearchResponse(
            results=[
                SearchResultResponse(
                    platform=r.platform,
                    model_id=r.model_id,
                    title=r.title,
                    author=r.author,
                    url=r.url,
                    thumbnail_url=r.thumbnail_url,
                    download_count=r.download_count,
                    license=r.license,
                )
                for r in results.items
            ],
            total_count=results.total_count,
            page=results.page,
            has_more=results.has_more,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/acquire/queue", response_model=QueueResponse, tags=["Acquire"])
async def add_to_queue(request: QueueRequest):
    """Add URLs to the download queue."""
    d3d = get_d3d()

    try:
        count = d3d.acquire.queue(urls=request.urls, priority=request.priority)
        return QueueResponse(queued_count=count)
    except Exception as e:
        logger.error(f"Queue operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/acquire/platforms", tags=["Acquire"])
async def list_platforms():
    """List available platforms."""
    from destill3d.acquire.models import PlatformRegistry

    registry = PlatformRegistry()
    platforms = registry.list_platforms()

    return {
        "platforms": [
            {
                "id": pid,
                "status": "available",
            }
            for pid in platforms
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Process Endpoints
# ─────────────────────────────────────────────────────────────────────────────


class ProcessRequest(BaseModel):
    """Request for full pipeline processing."""

    file_path: str = Field(..., description="Path to 3D file")
    classify: bool = Field(True, description="Run classification")
    store: bool = Field(True, description="Store in database")
    model_id: str = Field("pointnet2_ssg_mn40", description="Classification model")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class ProcessResponse(BaseModel):
    """Response from pipeline processing."""

    snapshot_id: str
    model_id: str
    point_count: int
    is_watertight: bool
    predictions: List[PredictionResponse]
    stored: bool


@app.post("/process/file", response_model=ProcessResponse, tags=["Process"])
async def process_file(request: ProcessRequest):
    """
    Full pipeline: extract + classify + store.

    Processes a file through the complete pipeline.
    """
    d3d = get_d3d()
    file_path = Path(request.file_path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        snapshot = d3d.process_file(
            file_path=file_path,
            classify=request.classify,
            store=request.store,
            model_id=request.model_id,
            metadata=request.metadata,
        )

        predictions = [
            PredictionResponse(
                label=p.label,
                confidence=p.confidence,
                rank=p.rank,
                taxonomy=p.taxonomy,
            )
            for p in snapshot.predictions
        ] if snapshot.predictions else []

        return ProcessResponse(
            snapshot_id=snapshot.snapshot_id,
            model_id=snapshot.model_id,
            point_count=snapshot.geometry.point_count if snapshot.geometry else 0,
            is_watertight=snapshot.features.is_watertight if snapshot.features else False,
            predictions=predictions,
            stored=request.store,
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
