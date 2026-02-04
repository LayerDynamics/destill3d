# Destill3D: Technical Specification Document

**Version**: 1.0.0-draft  
**Author**: LayerDynamics / Lattice Labs  
**Status**: Planning Phase  
**Last Updated**: February 2026

---

## Executive Summary

Destill3D is a command-line application and library for automated acquisition, feature extraction, and classification-ready distillation of 3D models from public hosting platforms. The system downloads models, extracts geometric features and metadata into a condensed representation, classifies them using trained or zero-shot models, then discards source files—retaining only the distilled data necessary for downstream ML training and analysis.

**Core Value Proposition**: Transform terabytes of raw 3D model downloads into gigabytes of classification-ready feature vectors, enabling large-scale 3D dataset construction without persistent storage of original assets.

---

## Table of Contents

1. [Problem Statement & Goals](#1-problem-statement--goals)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Supported Platforms & Formats](#4-supported-platforms--formats)
5. [Feature Extraction Specification](#5-feature-extraction-specification)
6. [Snapshot Format Specification](#6-snapshot-format-specification)
7. [Classification System](#7-classification-system)
8. [Storage & Database Design](#8-storage--database-design)
9. [CLI Interface](#9-cli-interface)
10. [Python API](#10-python-api)
11. [Configuration System](#11-configuration-system)
12. [Error Handling & Recovery](#12-error-handling--recovery)
13. [Performance Requirements](#13-performance-requirements)
14. [Security Considerations](#14-security-considerations)
15. [Testing Strategy](#15-testing-strategy)
16. [Deployment & Distribution](#16-deployment--distribution)
17. [Future Roadmap](#17-future-roadmap)

---

## 1. Problem Statement & Goals

### 1.1 Problem Statement

Building large-scale 3D classification datasets requires:

- Downloading models from multiple platforms with varying APIs and formats
- Processing diverse file formats (STEP, STL, OBJ, GLTF, etc.)
- Extracting consistent feature representations across formats
- Managing storage for potentially millions of models
- Maintaining provenance and metadata for reproducibility

Current approaches require retaining full model files, leading to:

- Storage costs scaling linearly with dataset size
- Redundant reprocessing when features need updates
- Legal/licensing complexity around storing original assets

### 1.2 Goals

| Goal | Success Metric |
|------|----------------|
| **G1**: Automated multi-platform acquisition | Support ≥5 major platforms with unified interface |
| **G2**: Format-agnostic feature extraction | Process ≥10 common 3D formats to unified representation |
| **G3**: Extreme compression ratio | Achieve ≥100:1 reduction (raw model → distilled snapshot) |
| **G4**: Classification-ready output | Direct compatibility with PointNet++, DGCNN, Point-MAE |
| **G5**: Provenance preservation | Full metadata chain from source URL to classification |
| **G6**: Resumable processing | Handle interruptions without data loss |
| **G7**: Scalable architecture | Process 10,000+ models/day on commodity hardware |

### 1.3 Non-Goals (Explicit Exclusions)

- Real-time 3D rendering or visualization
- Model editing or modification
- Hosting or serving original model files
- Training classification models (inference only)
- Format conversion for downstream CAD use

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DESTILL3D CORE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Acquire    │───▶│   Extract    │───▶│   Classify   │───▶│   Store   │ │
│  │   Module     │    │   Module     │    │   Module     │    │   Module  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  Platform    │    │   Format     │    │   Model      │    │  Database │ │
│  │  Adapters    │    │   Parsers    │    │   Registry   │    │  Backend  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           INFRASTRUCTURE LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Config    │  │   Logging   │  │   Metrics   │  │   Job Queue         │ │
│  │   Manager   │  │   System    │  │   Collector │  │   (Redis/SQLite)    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Responsibilities

#### 2.2.1 Acquire Module

- Platform adapter registration and selection
- Rate limiting and respectful crawling
- Authentication management (API keys, OAuth)
- Download queue management with retry logic
- Temporary file management

#### 2.2.2 Extract Module

- Format detection and parser routing
- Geometry loading via pythonocc-core, trimesh, pygltflib
- Point cloud generation (FPS, Poisson disk, uniform)
- Feature computation (normals, curvature, FPFH)
- Multi-view rendering for hybrid approaches
- Metadata extraction from embedded model data

#### 2.2.3 Classify Module

- Model loading (ONNX, TorchScript, SavedModel)
- Inference batching and GPU scheduling
- Zero-shot classification via CLIP/OpenShape embeddings
- Confidence scoring and uncertainty estimation
- Hierarchical taxonomy mapping

#### 2.2.4 Store Module

- Snapshot serialization (Protocol Buffers / MessagePack)
- Database transactions (SQLite / PostgreSQL)
- Index management for similarity search
- Export utilities (HDF5, TFRecord, NumPy)

### 2.3 Data Flow Diagram

```
                                    ┌─────────────────┐
                                    │  Platform APIs  │
                                    │  (Thingiverse,  │
                                    │   Sketchfab,    │
                                    │   GrabCAD...)   │
                                    └────────┬────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ACQUISITION PHASE                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Search/   │───▶│   Queue     │───▶│  Download   │───▶│   Validate  │   │
│  │   Discover  │    │   Entry     │    │   Files     │    │   Files     │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                  │          │
│                                                                  ▼          │
│                                                          ┌─────────────┐    │
│                                                          │  Temp Dir   │    │
│                                                          │  /tmp/d3d/  │    │
│                                                          └──────┬──────┘    │
└─────────────────────────────────────────────────────────────────┼───────────┘
                                                                  │
                                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXTRACTION PHASE                                  │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Detect    │───▶│   Load      │───▶│  Tessellate │───▶│   Sample    │   │
│  │   Format    │    │   Geometry  │    │   (if CAD)  │    │   Points    │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                  │          │
│                                                                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Compute   │◀───│   Compute   │◀───│   Compute   │◀───│   Normalize │   │
│  │   Global    │    │   Local     │    │   Normals   │    │   Scale     │   │
│  │   Features  │    │   Features  │    │             │    │   Center    │   │
│  └──────┬──────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      SNAPSHOT ASSEMBLY                              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │    │
│  │  │  Points  │  │ Normals  │  │ Features │  │ Metadata │             │    │
│  │  │  (2048)  │  │  (2048)  │  │  (1024)  │  │  (JSON)  │             │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │ 
│                                         │                                   │
└─────────────────────────────────────────┼───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CLASSIFICATION PHASE                               │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Load      │───▶│   Encode    │───▶│   Classify  │───▶│   Compute   │   │
│  │   Model     │    │   Features  │    │   (Top-K)   │    │   Confidence│   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                  │          │
│                                                                  ▼          │
│                                                          ┌─────────────┐    │
│                                                          │ Predictions │    │
│                                                          │ + Scores    │    │
│                                                          └──────┬──────┘    │
└─────────────────────────────────────────────────────────────────┼───────────┘
                                                                  │
                                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            STORAGE PHASE                                    │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  Serialize  │───▶│   Write     │───▶│   Index     │───▶│   Cleanup   │   │
│  │  Snapshot   │    │   Database  │    │   Embeddings│    │   Temp Dir  │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Language** | Python 3.11+ | Ecosystem compatibility, pythonocc bindings |
| **CAD Processing** | pythonocc-core 7.7+ | STEP/IGES B-rep handling, tessellation |
| **Mesh Processing** | trimesh, Open3D | Point sampling, normal estimation, simplification |
| **Point Cloud ML** | PyTorch, ONNX Runtime | Model inference, GPU acceleration |
| **Database** | SQLite (default), PostgreSQL (scale) | Snapshot storage, metadata indexing |
| **Vector Search** | FAISS, Qdrant (optional) | Similarity search on embeddings |
| **Serialization** | Protocol Buffers, MessagePack | Compact binary snapshots |
| **HTTP Client** | httpx (async) | Platform API communication |
| **CLI** | Typer | Modern CLI with rich output |
| **Config** | Pydantic Settings | Validation, env var support |

---

## 3. Data Pipeline

### 3.1 Pipeline Stages

The pipeline processes models through four discrete stages, each with checkpoint capability for resumability.

```python
class PipelineStage(Enum):
    QUEUED = "queued"           # Entry created, not yet downloaded
    ACQUIRED = "acquired"       # Files downloaded to temp directory
    EXTRACTED = "extracted"     # Snapshot created, pending classification
    CLASSIFIED = "classified"   # Classification complete
    STORED = "stored"           # Persisted to database, temp files cleaned
    FAILED = "failed"           # Error state with retry counter
```

### 3.2 Stage Transitions

```
QUEUED ──download──▶ ACQUIRED ──extract──▶ EXTRACTED ──classify──▶ CLASSIFIED ──store──▶ STORED
   │                    │                      │                        │
   │                    │                      │                        │
   ▼                    ▼                      ▼                        ▼
FAILED ◀────────────────┴──────────────────────┴────────────────────────┘
   │
   │ (retry < max_retries)
   ▼
QUEUED
```

### 3.3 Checkpoint Schema

Each model maintains a checkpoint record enabling resumption:

```python
@dataclass
class ProcessingCheckpoint:
    model_id: str                    # Unique identifier (platform:id)
    stage: PipelineStage
    created_at: datetime
    updated_at: datetime
    
    # Acquisition checkpoint
    source_url: str
    platform: str
    temp_path: Optional[Path]
    file_hash: Optional[str]         # SHA256 of downloaded file
    
    # Extraction checkpoint
    snapshot_path: Optional[Path]
    point_count: Optional[int]
    feature_version: Optional[str]
    
    # Classification checkpoint
    predictions: Optional[List[Prediction]]
    embedding: Optional[np.ndarray]   # For similarity search
    
    # Error tracking
    retry_count: int = 0
    last_error: Optional[str] = None
    error_stage: Optional[PipelineStage] = None
```

### 3.4 Batch Processing

Models are processed in configurable batches to optimize GPU utilization and network efficiency:

```python
class BatchConfig:
    download_batch_size: int = 10      # Concurrent downloads
    extraction_batch_size: int = 1     # Sequential (memory-bound)
    classification_batch_size: int = 32 # GPU batch size
    storage_batch_size: int = 100      # Database transaction size
```

---

## 4. Supported Platforms & Formats

### 4.1 Platform Adapters

Each platform requires a dedicated adapter implementing the `PlatformAdapter` protocol:

```python
class PlatformAdapter(Protocol):
    """Protocol for platform-specific acquisition logic."""
    
    @property
    def platform_id(self) -> str:
        """Unique identifier (e.g., 'thingiverse', 'sketchfab')"""
        ...
    
    @property
    def rate_limit(self) -> RateLimit:
        """Platform-specific rate limiting configuration"""
        ...
    
    async def search(
        self, 
        query: str, 
        filters: SearchFilters,
        page: int = 1
    ) -> SearchResults:
        """Search for models matching criteria"""
        ...
    
    async def get_metadata(self, model_id: str) -> ModelMetadata:
        """Fetch detailed metadata for a specific model"""
        ...
    
    async def download(
        self, 
        model_id: str, 
        target_dir: Path
    ) -> DownloadResult:
        """Download model files to target directory"""
        ...
    
    def parse_url(self, url: str) -> Optional[str]:
        """Extract model_id from platform URL, or None if not matching"""
        ...
```

### 4.2 Platform Support Matrix

| Platform | Status | API Type | Auth Required | Rate Limit | Notes |
|----------|--------|----------|---------------|------------|-------|
| **Thingiverse** | P0 | REST API v1 | API Key | 300/5min | Largest CC dataset |
| **Sketchfab** | P0 | REST API | OAuth2 | 1000/day (free) | High-quality scans |
| **GrabCAD** | P1 | Web Scraping | Session | 100/hour | CAD-heavy, STEP/IGES |
| **Cults3D** | P1 | Web Scraping | Login | 50/hour | Design files |
| **MyMiniFactory** | P2 | REST API | API Key | 500/day | Curated prints |
| **Thangs** | P2 | REST API | API Key | TBD | Search aggregator |
| **GitHub** | P2 | REST API | Token | 5000/hour | Open-source models |
| **Turbosquid** | P3 | Web Scraping | Login | TBD | Commercial focus |
| **CGTrader** | P3 | Web Scraping | Login | TBD | Commercial focus |
| **Local Files** | P0 | Filesystem | None | N/A | Direct file input |

**Priority Key**: P0 = MVP, P1 = v1.1, P2 = v1.2, P3 = Future

### 4.3 File Format Support

#### 4.3.1 CAD Formats (via pythonocc-core)

| Format | Extension | Read | Tessellation Required | Notes |
|--------|-----------|------|----------------------|-------|
| STEP | .step, .stp | ✅ | Yes | AP203/AP214/AP242 |
| IGES | .iges, .igs | ✅ | Yes | Legacy CAD exchange |
| BREP | .brep, .brp | ✅ | Yes | OpenCASCADE native |

#### 4.3.2 Mesh Formats (via trimesh/Open3D)

| Format | Extension | Read | Notes |
|--------|-----------|------|-------|
| STL | .stl | ✅ | Binary and ASCII |
| OBJ | .obj | ✅ | With MTL materials |
| PLY | .ply | ✅ | With vertex colors/normals |
| OFF | .off | ✅ | ModelNet native format |
| GLTF | .gltf, .glb | ✅ | Sketchfab primary format |
| FBX | .fbx | ⚠️ | Via trimesh (limited) |
| 3MF | .3mf | ✅ | Modern printing format |
| DAE | .dae | ✅ | Collada (ShapeNet native) |

#### 4.3.3 Point Cloud Formats (via Open3D)

| Format | Extension | Read | Notes |
|--------|-----------|------|-------|
| PCD | .pcd | ✅ | PCL native format |
| XYZ | .xyz, .pts | ✅ | Simple ASCII points |
| LAS/LAZ | .las, .laz | ✅ | LiDAR standard |

### 4.4 Format Detection

Format detection uses a multi-stage approach:

```python
class FormatDetector:
    def detect(self, file_path: Path) -> FileFormat:
        """
        Detection order:
        1. Magic bytes (binary signatures)
        2. File extension mapping
        3. Content sniffing (XML, JSON structure)
        4. Fallback to extension-based guess
        """
        
        # Stage 1: Magic bytes
        magic = self._read_magic_bytes(file_path, 16)
        if magic.startswith(b'solid'):
            return FileFormat.STL_ASCII
        if magic.startswith(b'\x00\x00\x00'):  # Binary STL
            return FileFormat.STL_BINARY
        if magic.startswith(b'glTF'):
            return FileFormat.GLTF_BINARY
        if magic.startswith(b'ISO-10303'):
            return FileFormat.STEP
        
        # Stage 2: Extension mapping
        ext = file_path.suffix.lower()
        if ext in self.extension_map:
            return self.extension_map[ext]
        
        # Stage 3: Content sniffing
        if self._is_xml(file_path):
            return self._detect_xml_format(file_path)
        
        # Stage 4: Fallback
        raise FormatDetectionError(f"Unknown format: {file_path}")
```

---

## 5. Feature Extraction Specification

### 5.1 Extraction Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE EXTRACTION PIPELINE                          │
│                                                                             │
│  INPUT: Raw 3D File                                                         │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 1: GEOMETRY LOADING                                           │    │
│  │                                                                     │    │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐              │    │
│  │  │  STEP   │   │   STL   │   │   OBJ   │   │  GLTF   │   ...        │    │
│  │  │ Loader  │   │ Loader  │   │ Loader  │   │ Loader  │              │    │
│  │  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘              │    │
│  │       └─────────────┴───────┬─────┴─────────────┘                  │   │
│  │                             ▼                                      │   │
│  │                    ┌─────────────────┐                             │   │
│  │                    │ Unified Mesh    │                             │   │
│  │                    │ (vertices,faces)│                             │   │
│  │                    └────────┬────────┘                             │   │
│  └─────────────────────────────┼───────────────────────────────────────┘   │
│                                │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: MESH PREPROCESSING                                         │   │
│  │                                                                     │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐       │   │
│  │  │  Remove   │─▶│  Merge    │─▶│  Repair   │─▶│  Simplify │       │   │
│  │  │Degenerate │  │  Vertices │  │  Holes    │  │  (if >1M) │       │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘       │   │
│  │                                                     │              │   │
│  └─────────────────────────────────────────────────────┼──────────────┘   │
│                                                        │                   │
│                                                        ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: POINT CLOUD GENERATION                                     │   │
│  │                                                                     │   │
│  │  Sampling Strategy: Configurable (default: FPS + Uniform hybrid)   │   │
│  │                                                                     │   │
│  │  ┌────────────────────────────────────────────────────────────┐    │   │
│  │  │ 1. Sample 10,000 points uniform on surface                 │    │   │
│  │  │ 2. Apply FPS to reduce to 2,048 (or configured N)          │    │   │
│  │  │ 3. Center at origin (subtract centroid)                    │    │   │
│  │  │ 4. Scale to unit sphere (divide by max distance)           │    │   │
│  │  └────────────────────────────────────────────────────────────┘    │   │
│  │                                                     │              │   │
│  │  Output: points ∈ ℝ^(N×3), normalized to [-1, 1]   │              │   │
│  └─────────────────────────────────────────────────────┼──────────────┘   │
│                                                        │                   │
│                                                        ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: PER-POINT FEATURE COMPUTATION                              │   │
│  │                                                                     │   │
│  │  For each point p_i:                                               │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ a) Surface Normal (n_i ∈ ℝ³)                                │   │   │
│  │  │    - PCA on k-NN (k=30)                                     │   │   │
│  │  │    - Eigenvector of smallest eigenvalue                     │   │   │
│  │  │    - Orient consistently via MST propagation                │   │   │
│  │  ├─────────────────────────────────────────────────────────────┤   │   │
│  │  │ b) Curvature Estimate (κ_i ∈ ℝ)                             │   │   │
│  │  │    - κ = λ_min / (λ_1 + λ_2 + λ_3)                         │   │   │
│  │  │    - Approximates local surface curvature                   │   │   │
│  │  ├─────────────────────────────────────────────────────────────┤   │   │
│  │  │ c) Local Density (ρ_i ∈ ℝ)                                  │   │   │
│  │  │    - 1 / mean_distance_to_k_neighbors                       │   │   │
│  │  │    - Normalized to [0, 1] across point cloud                │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  Output: per_point_features ∈ ℝ^(N×7) = [x,y,z,nx,ny,nz,κ]        │   │
│  │          (density stored separately for weighting)                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                        │                   │
│                                                        ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 5: GLOBAL FEATURE COMPUTATION                                 │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ Geometric Statistics:                                       │   │   │
│  │  │ - Bounding box dimensions (W, H, D)                        │   │   │
│  │  │ - Aspect ratios (W/H, W/D, H/D)                            │   │   │
│  │  │ - Surface area estimate (sum of triangle areas)             │   │   │
│  │  │ - Volume estimate (if watertight, via signed tetrahedra)   │   │   │
│  │  │ - Sphericity = (π^(1/3) * (6V)^(2/3)) / A                  │   │   │
│  │  │ - Convex hull volume ratio = V / V_hull                    │   │   │
│  │  ├─────────────────────────────────────────────────────────────┤   │   │
│  │  │ Curvature Distribution:                                     │   │   │
│  │  │ - Mean, std, min, max curvature                            │   │   │
│  │  │ - 10-bin histogram of curvature values                     │   │   │
│  │  ├─────────────────────────────────────────────────────────────┤   │   │
│  │  │ Principal Axes:                                             │   │   │
│  │  │ - PCA eigenvalues (λ_1, λ_2, λ_3) / sum                    │   │   │
│  │  │ - Anisotropy = (λ_1 - λ_3) / λ_1                          │   │   │
│  │  │ - Planarity = (λ_2 - λ_3) / λ_1                           │   │   │
│  │  │ - Linearity = (λ_1 - λ_2) / λ_1                           │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  Output: global_features ∈ ℝ^32                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                        │                   │
│                                                        ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 6: MULTI-VIEW RENDERING (Optional)                            │   │
│  │                                                                     │   │
│  │  Generate 12 views at 30° intervals (elevation 30°)                │   │
│  │  Resolution: 224×224 grayscale depth maps                          │   │
│  │  Used for: MVCNN hybrid classification, visual thumbnails          │   │
│  │                                                                     │   │
│  │  Output: view_images ∈ ℝ^(12×224×224) (optional, configurable)     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  FINAL OUTPUT: ExtractedFeatures                                           │
│    - points: float32[N, 3]                                                 │
│    - normals: float32[N, 3]                                                │
│    - curvature: float32[N]                                                 │
│    - global_features: float32[32]                                          │
│    - views: Optional[uint8[12, 224, 224]]                                  │
│    - extraction_metadata: ExtractionMetadata                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Tessellation Parameters (CAD Files)

For STEP/IGES files processed via pythonocc-core:

```python
@dataclass
class TessellationConfig:
    """Configuration for B-rep to mesh conversion."""
    
    # Linear deflection: max distance between curve and tessellation
    # Smaller = finer mesh, more points
    linear_deflection: float = 0.001  # 1mm for typical CAD
    
    # Angular deflection: max angle between adjacent segments (radians)
    # Smaller = smoother curves
    angular_deflection: float = 0.5   # ~28.6 degrees
    
    # Relative mode: deflection as fraction of bounding box
    relative: bool = False
    relative_linear: float = 0.001    # 0.1% of bbox diagonal
    
    # Quality presets
    @classmethod
    def preview(cls) -> "TessellationConfig":
        """Fast tessellation for preview/thumbnails"""
        return cls(linear_deflection=0.01, angular_deflection=0.8)
    
    @classmethod
    def standard(cls) -> "TessellationConfig":
        """Balanced quality/speed for classification"""
        return cls(linear_deflection=0.001, angular_deflection=0.5)
    
    @classmethod
    def high_quality(cls) -> "TessellationConfig":
        """Maximum quality for detailed analysis"""
        return cls(linear_deflection=0.0001, angular_deflection=0.2)
```

### 5.3 Point Sampling Strategies

```python
class SamplingStrategy(Enum):
    UNIFORM = "uniform"           # Random uniform on surface
    FPS = "fps"                   # Farthest Point Sampling
    POISSON = "poisson"           # Poisson disk sampling
    VOXEL = "voxel"               # Voxel grid downsampling
    HYBRID = "hybrid"             # Uniform oversample → FPS

@dataclass
class SamplingConfig:
    strategy: SamplingStrategy = SamplingStrategy.HYBRID
    target_points: int = 2048
    oversample_ratio: float = 5.0  # For HYBRID: sample 5x, then FPS
    
    # Poisson-specific
    poisson_radius: Optional[float] = None  # Auto-computed if None
    
    # Voxel-specific  
    voxel_size: Optional[float] = None  # Auto-computed if None
```

**Why HYBRID is default**: Pure uniform sampling may cluster points in high-detail regions. Pure FPS is expensive (O(N²) naive, O(N log N) with KD-tree). Hybrid approach samples many points cheaply, then uses FPS to ensure spatial coverage—best of both.

### 5.4 Feature Computation Functions

```python
def compute_normals(
    points: np.ndarray,  # (N, 3)
    k: int = 30,
    orient_normals: bool = True
) -> np.ndarray:  # (N, 3)
    """
    Estimate surface normals via PCA on k-nearest neighbors.
    
    For each point p_i:
    1. Find k nearest neighbors
    2. Compute covariance matrix C of neighbors
    3. Normal = eigenvector of smallest eigenvalue of C
    4. If orient_normals: propagate consistent orientation via MST
    """
    pass

def compute_curvature(
    points: np.ndarray,  # (N, 3)
    k: int = 30
) -> np.ndarray:  # (N,)
    """
    Estimate local curvature as ratio of smallest eigenvalue.
    
    κ_i = λ_min / (λ_1 + λ_2 + λ_3)
    
    Interpretation:
    - κ ≈ 0: Planar region
    - κ ≈ 0.33: Isotropic curvature (sphere-like)
    - κ > 0.33: Ridge or corner
    """
    pass

def compute_global_features(
    points: np.ndarray,       # (N, 3)
    normals: np.ndarray,      # (N, 3)
    curvature: np.ndarray,    # (N,)
    mesh: Optional[Mesh] = None  # For watertight volume
) -> np.ndarray:  # (32,)
    """
    Compute rotation-invariant global shape descriptors.
    
    Feature vector composition:
    [0:3]   - Normalized bbox dimensions (sorted)
    [3:6]   - Aspect ratios
    [6]     - Surface area (log-scaled)
    [7]     - Volume (log-scaled, 0 if non-watertight)
    [8]     - Sphericity
    [9]     - Convexity (V / V_hull)
    [10:14] - Curvature statistics (mean, std, min, max)
    [14:24] - Curvature histogram (10 bins)
    [24:27] - PCA eigenvalue ratios
    [27:30] - Anisotropy, planarity, linearity
    [30:32] - Reserved
    """
    pass
```

---

## 6. Snapshot Format Specification

### 6.1 Snapshot Structure

The snapshot is the core data unit—a self-contained, compressed representation of a 3D model optimized for classification tasks.

```protobuf
// destill3d/proto/snapshot.proto
syntax = "proto3";

package destill3d;

message Snapshot {
    // Identity
    string snapshot_id = 1;          // UUID v4
    string model_id = 2;             // platform:original_id
    uint32 version = 3;              // Snapshot format version
    
    // Provenance
    Provenance provenance = 4;
    
    // Geometric Data
    GeometryData geometry = 5;
    
    // Extracted Features  
    Features features = 6;
    
    // Classification Results
    repeated Prediction predictions = 7;
    bytes embedding = 8;             // float32[1024] for similarity search
    
    // Processing Metadata
    ProcessingMetadata processing = 9;
}

message Provenance {
    string platform = 1;             // e.g., "thingiverse"
    string source_url = 2;           // Original download URL
    string source_id = 3;            // Platform-specific ID
    
    // Original metadata from platform
    string title = 4;
    string description = 5;
    string author = 6;
    string license = 7;
    repeated string tags = 8;
    
    // File information
    string original_filename = 9;
    string original_format = 10;     // e.g., "STEP", "STL"
    uint64 original_file_size = 11;  // bytes
    string original_file_hash = 12;  // SHA256
    
    // Timestamps
    google.protobuf.Timestamp source_created = 13;
    google.protobuf.Timestamp source_modified = 14;
    google.protobuf.Timestamp acquired_at = 15;
}

message GeometryData {
    // Point cloud: N points × 3 coordinates
    // Stored as flat float32 array, row-major
    bytes points = 1;                // float32[N*3]
    uint32 point_count = 2;          // N
    
    // Per-point normals
    bytes normals = 3;               // float32[N*3]
    
    // Per-point curvature
    bytes curvature = 4;             // float32[N]
    
    // Optional: Multi-view depth images
    repeated bytes view_images = 5;  // uint8[224*224] × 12
    
    // Normalization parameters (for denormalization)
    repeated float centroid = 6;     // Original centroid [x, y, z]
    float scale = 7;                 // Original scale factor
}

message Features {
    // Global shape features
    bytes global_features = 1;       // float32[32]
    
    // Bounding box in original scale
    BoundingBox original_bbox = 2;
    
    // Summary statistics
    float surface_area = 3;
    float volume = 4;                // 0 if non-watertight
    bool is_watertight = 5;
    uint32 original_vertex_count = 6;
    uint32 original_face_count = 7;
}

message BoundingBox {
    float min_x = 1;
    float min_y = 2;
    float min_z = 3;
    float max_x = 4;
    float max_y = 5;
    float max_z = 6;
}

message Prediction {
    string label = 1;                // Class name
    float confidence = 2;            // [0, 1]
    string taxonomy = 3;             // e.g., "modelnet40", "shapenet55"
    string model_name = 4;           // e.g., "pointnet2_ssg"
}

message ProcessingMetadata {
    string destill3d_version = 1;
    string feature_extractor_version = 2;
    
    // Extraction parameters used
    uint32 target_points = 3;
    string sampling_strategy = 4;
    float tessellation_deflection = 5;
    
    // Timing
    float download_time_ms = 6;
    float extraction_time_ms = 7;
    float classification_time_ms = 8;
    
    // Quality indicators
    float mesh_quality_score = 9;    // 0-1, based on degenerate faces etc.
    repeated string warnings = 10;   // Non-fatal issues encountered
}
```

### 6.2 Size Estimation

For a typical snapshot with 2048 points and 12 views:

| Component | Size | Notes |
|-----------|------|-------|
| Points (2048 × 3 × float32) | 24,576 bytes | ~24 KB |
| Normals (2048 × 3 × float32) | 24,576 bytes | ~24 KB |
| Curvature (2048 × float32) | 8,192 bytes | ~8 KB |
| Global features (32 × float32) | 128 bytes | <1 KB |
| Embedding (1024 × float32) | 4,096 bytes | ~4 KB |
| Views (12 × 224 × 224 × uint8) | 602,112 bytes | ~588 KB |
| Metadata (JSON-like) | ~2,000 bytes | ~2 KB |
| **Total (with views)** | **~666 KB** | ~650 KB |
| **Total (without views)** | **~63 KB** | ~62 KB |

**Compression**: With gzip, typical snapshot compresses to ~40% of raw size:

- With views: ~260 KB
- Without views: ~25 KB

**Compression ratio** vs original:

- Typical STL (5 MB) → 25 KB snapshot = **200:1 ratio**
- Typical STEP (10 MB) → 25 KB snapshot = **400:1 ratio**

### 6.3 Versioning

Snapshot format versions follow semantic versioning:

```python
SNAPSHOT_VERSION = 1  # Current version

# Version history:
# v1: Initial release
#     - 2048 points, 32 global features, 1024-dim embedding
#     - Protobuf serialization with optional gzip
```

Readers must handle version mismatches gracefully:

- Same major version: Backward compatible, ignore unknown fields
- Different major version: Error with migration guidance

---

## 7. Classification System

### 7.1 Classification Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLASSIFICATION SYSTEM                                │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        MODEL REGISTRY                                  │ │
│  │                                                                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
│  │  │ PointNet++  │  │   DGCNN     │  │ Point-MAE   │  │  OpenShape  │  │ │
│  │  │ (ModelNet)  │  │ (ModelNet)  │  │ (Pretrain)  │  │ (Zero-shot) │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │ │
│  │                                                                       │ │
│  │  Model artifacts stored in: ~/.destill3d/models/                      │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        INFERENCE ENGINE                                │ │
│  │                                                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    BATCH PROCESSOR                              │  │ │
│  │  │                                                                 │  │ │
│  │  │  Input: List[Snapshot]                                         │  │ │
│  │  │    │                                                           │  │ │
│  │  │    ▼                                                           │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────┐   │  │ │
│  │  │  │ Collate: Stack points/normals into batch tensors        │   │  │ │
│  │  │  │ points_batch: (B, N, 3)                                 │   │  │ │
│  │  │  │ normals_batch: (B, N, 3)                                │   │  │ │
│  │  │  └─────────────────────────────────────────────────────────┘   │  │ │
│  │  │    │                                                           │  │ │
│  │  │    ▼                                                           │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────┐   │  │ │
│  │  │  │ Augmentation (optional, for test-time augmentation):    │   │  │ │
│  │  │  │ - Random rotation around Z-axis                         │   │  │ │
│  │  │  │ - Random scaling (0.95-1.05)                            │   │  │ │
│  │  │  │ - Multiple augmented passes, average predictions        │   │  │ │
│  │  │  └─────────────────────────────────────────────────────────┘   │  │ │
│  │  │    │                                                           │  │ │
│  │  │    ▼                                                           │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────┐   │  │ │
│  │  │  │ Model Forward Pass:                                     │   │  │ │
│  │  │  │ - Load model to device (GPU if available)               │   │  │ │
│  │  │  │ - Run inference: logits = model(points_batch)           │   │  │ │
│  │  │  │ - Apply softmax: probs = softmax(logits)                │   │  │ │
│  │  │  │ - Extract embeddings from penultimate layer             │   │  │ │
│  │  │  └─────────────────────────────────────────────────────────┘   │  │ │
│  │  │    │                                                           │  │ │
│  │  │    ▼                                                           │  │ │
│  │  │  Output: List[ClassificationResult]                            │  │ │
│  │  │    - predictions: List[Prediction] (top-5)                    │  │ │
│  │  │    - embedding: ndarray[1024]                                  │  │ │
│  │  │    - uncertainty: float (entropy or MC dropout std)            │  │ │
│  │  │                                                                 │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      ZERO-SHOT CLASSIFICATION                          │ │
│  │                                                                       │ │
│  │  For arbitrary class names without retraining:                        │ │
│  │                                                                       │ │
│  │  1. Encode point cloud → 3D embedding (via OpenShape encoder)         │ │
│  │  2. Encode class names → text embeddings (via CLIP text encoder)      │ │
│  │  3. Compute cosine similarity: sim = dot(3d_emb, text_embs.T)        │ │
│  │  4. Softmax over similarities → class probabilities                   │ │
│  │                                                                       │ │
│  │  Example:                                                             │ │
│  │    classes = ["firearm", "tool", "toy", "furniture"]                 │ │
│  │    probs = zero_shot_classify(snapshot, classes)                      │ │
│  │    # → [0.85, 0.08, 0.05, 0.02]                                      │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Supported Taxonomies

```python
class Taxonomy:
    """Predefined classification taxonomies with label mappings."""
    
    MODELNET40 = TaxonomyConfig(
        name="modelnet40",
        num_classes=40,
        labels=[
            "airplane", "bathtub", "bed", "bench", "bookshelf",
            "bottle", "bowl", "car", "chair", "cone",
            "cup", "curtain", "desk", "door", "dresser",
            "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
            "laptop", "mantel", "monitor", "night_stand", "person",
            "piano", "plant", "radio", "range_hood", "sink",
            "sofa", "stairs", "stool", "table", "tent",
            "toilet", "tv_stand", "vase", "wardrobe", "xbox"
        ]
    )
    
    SHAPENET55 = TaxonomyConfig(
        name="shapenet55",
        num_classes=55,
        labels=[...]  # ShapeNet category names
    )
    
    DEFCAD = TaxonomyConfig(
        name="defcad",
        num_classes=12,
        labels=[
            "lower_receiver", "upper_receiver", "barrel", "stock",
            "grip", "magazine", "trigger_group", "rail",
            "sight", "suppressor", "accessory", "other"
        ],
        hierarchical=True,
        parent_map={...}  # For hierarchical classification
    )
```

### 7.3 Model Registry

```python
@dataclass
class RegisteredModel:
    """A classification model registered for use."""
    
    model_id: str                    # Unique identifier
    name: str                        # Human-readable name
    taxonomy: str                    # Target taxonomy
    architecture: str                # e.g., "pointnet2_ssg"
    
    # Model files
    weights_url: str                 # Download URL
    weights_hash: str                # SHA256 for verification
    weights_path: Optional[Path]     # Local path after download
    
    # Input requirements
    input_points: int = 2048
    input_features: List[str] = field(default_factory=lambda: ["xyz"])
    requires_normals: bool = False
    
    # Performance metadata
    modelnet40_accuracy: Optional[float] = None
    inference_time_ms: Optional[float] = None
    
    # Runtime
    format: str = "onnx"             # "onnx", "torchscript", "savedmodel"
    device: str = "cuda"             # "cuda", "cpu"

# Default model registry
MODEL_REGISTRY = {
    "pointnet2_ssg_mn40": RegisteredModel(
        model_id="pointnet2_ssg_mn40",
        name="PointNet++ SSG (ModelNet40)",
        taxonomy="modelnet40",
        architecture="pointnet2_ssg",
        weights_url="https://github.com/.../pointnet2_ssg_mn40.onnx",
        weights_hash="sha256:abc123...",
        modelnet40_accuracy=0.927,
    ),
    "dgcnn_mn40": RegisteredModel(...),
    "openshape_pointbert": RegisteredModel(
        model_id="openshape_pointbert",
        name="OpenShape PointBERT (Zero-shot)",
        taxonomy="zero_shot",
        architecture="pointbert",
        weights_url="https://huggingface.co/.../openshape.onnx",
        requires_normals=True,
    ),
}
```

### 7.4 Uncertainty Quantification

```python
class UncertaintyMethod(Enum):
    ENTROPY = "entropy"              # Shannon entropy of softmax
    MC_DROPOUT = "mc_dropout"        # Monte Carlo dropout (if supported)
    ENSEMBLE = "ensemble"            # Variance across model ensemble

def compute_uncertainty(
    probs: np.ndarray,  # (num_classes,)
    method: UncertaintyMethod = UncertaintyMethod.ENTROPY
) -> float:
    """
    Compute classification uncertainty.
    
    Returns value in [0, 1]:
    - 0: Completely certain (one class has prob=1)
    - 1: Maximum uncertainty (uniform distribution)
    """
    if method == UncertaintyMethod.ENTROPY:
        # Normalized entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        return entropy / max_entropy
    # ... other methods
```

---

## 8. Storage & Database Design

### 8.1 Database Schema

```sql
-- SQLite/PostgreSQL compatible schema

-- Core snapshot storage
CREATE TABLE snapshots (
    snapshot_id TEXT PRIMARY KEY,        -- UUID
    model_id TEXT NOT NULL UNIQUE,       -- platform:id
    
    -- Provenance
    platform TEXT NOT NULL,
    source_url TEXT,
    source_id TEXT,
    title TEXT,
    author TEXT,
    license TEXT,
    original_format TEXT,
    original_file_size INTEGER,
    original_file_hash TEXT,
    
    -- Snapshot data (compressed protobuf)
    snapshot_data BLOB NOT NULL,
    snapshot_version INTEGER NOT NULL,
    
    -- Quick-access features (denormalized)
    point_count INTEGER,
    is_watertight BOOLEAN,
    surface_area REAL,
    volume REAL,
    
    -- Timestamps
    source_created_at TIMESTAMP,
    acquired_at TIMESTAMP NOT NULL,
    processed_at TIMESTAMP NOT NULL,
    
    -- Indexing
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tags (many-to-many)
CREATE TABLE tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT NOT NULL UNIQUE
);

CREATE TABLE snapshot_tags (
    snapshot_id TEXT REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES tags(tag_id) ON DELETE CASCADE,
    source TEXT NOT NULL,                -- 'platform' or 'predicted'
    confidence REAL,                     -- NULL for platform tags
    PRIMARY KEY (snapshot_id, tag_id, source)
);

-- Classifications
CREATE TABLE classifications (
    classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    
    model_id TEXT NOT NULL,              -- From model registry
    taxonomy TEXT NOT NULL,
    
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    rank INTEGER NOT NULL,               -- 1 = top prediction
    
    uncertainty REAL,
    
    classified_at TIMESTAMP NOT NULL,
    
    UNIQUE (snapshot_id, model_id, rank)
);

-- Embeddings for similarity search (stored separately for efficient access)
CREATE TABLE embeddings (
    snapshot_id TEXT PRIMARY KEY REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,              -- Which model produced embedding
    embedding BLOB NOT NULL,             -- float32[1024] as binary
    embedding_dim INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing queue (for resumability)
CREATE TABLE processing_queue (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL UNIQUE,
    platform TEXT NOT NULL,
    source_url TEXT NOT NULL,
    
    stage TEXT NOT NULL,                 -- PipelineStage enum
    priority INTEGER DEFAULT 0,
    
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    last_error TEXT,
    
    temp_path TEXT,                      -- Path during processing
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    scheduled_at TIMESTAMP               -- For delayed retry
);

-- Indexes
CREATE INDEX idx_snapshots_platform ON snapshots(platform);
CREATE INDEX idx_snapshots_acquired ON snapshots(acquired_at);
CREATE INDEX idx_classifications_label ON classifications(label);
CREATE INDEX idx_classifications_snapshot ON classifications(snapshot_id);
CREATE INDEX idx_queue_stage ON processing_queue(stage, priority DESC);
CREATE INDEX idx_queue_scheduled ON processing_queue(scheduled_at) WHERE scheduled_at IS NOT NULL;
```

### 8.2 Vector Search Integration

For similarity search on embeddings, integrate with FAISS or Qdrant:

```python
class EmbeddingIndex:
    """FAISS-based similarity search for snapshot embeddings."""
    
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
        self.id_map: List[str] = []  # snapshot_id at each index position
    
    def add(self, snapshot_id: str, embedding: np.ndarray):
        """Add embedding to index."""
        embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        self.id_map.append(snapshot_id)
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find k most similar snapshots."""
        query = query_embedding / np.linalg.norm(query_embedding)
        scores, indices = self.index.search(
            query.reshape(1, -1).astype(np.float32), k
        )
        return [
            (self.id_map[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx < len(self.id_map)
        ]
    
    def save(self, path: Path):
        """Persist index to disk."""
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "id_map.json", "w") as f:
            json.dump(self.id_map, f)
    
    def load(self, path: Path):
        """Load index from disk."""
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "id_map.json") as f:
            self.id_map = json.load(f)
```

### 8.3 Export Formats

```python
class ExportFormat(Enum):
    HDF5 = "hdf5"           # Hierarchical, good for large datasets
    TFRECORD = "tfrecord"   # TensorFlow native
    NUMPY = "numpy"         # Simple .npz archives
    PARQUET = "parquet"     # Columnar, good for metadata queries

def export_dataset(
    db: Database,
    output_path: Path,
    format: ExportFormat,
    filters: Optional[QueryFilters] = None,
    include_views: bool = False
) -> ExportResult:
    """
    Export snapshots to ML-ready format.
    
    HDF5 structure:
    /points          (N, 2048, 3)    float32
    /normals         (N, 2048, 3)    float32
    /labels          (N,)            int32
    /label_names     (C,)            string
    /snapshot_ids    (N,)            string
    /metadata        (N,)            JSON strings
    
    For train/val/test splits, export separately or use attributes.
    """
    pass
```

---

## 9. CLI Interface

### 9.1 Command Structure

```
destill3d
├── acquire              # Download models from platforms
│   ├── search           # Search platforms and queue results
│   ├── url              # Add specific URL to queue
│   ├── list             # List files in queue
│   └── run              # Process download queue
│
├── extract              # Extract features from local files
│   ├── file             # Extract single file
│   ├── dir              # Extract directory of files
│   └── queue            # Process extraction queue
│
├── classify             # Run classification on snapshots
│   ├── snapshot         # Classify specific snapshot
│   ├── all              # Classify all unclassified
│   └── zero-shot        # Zero-shot with custom classes
│
├── db                   # Database operations
│   ├── stats            # Show database statistics
│   ├── query            # Query snapshots
│   ├── export           # Export to ML formats
│   └── vacuum           # Cleanup and optimize
│
├── models               # Model management
│   ├── list             # List available models
│   ├── download         # Download model weights
│   └── info             # Show model details
│
├── config               # Configuration management
│   ├── show             # Show current config
│   ├── set              # Set config value
│   └── init             # Initialize config file
│
└── server               # (Future) REST API server
    └── start            # Start API server
```

### 9.2 Example Commands

```bash
# Search Thingiverse for "chess piece" and queue first 100 results
destill3d acquire search --platform thingiverse --query "chess piece" --limit 100

# Add specific URL
destill3d acquire url "https://www.thingiverse.com/thing:12345"

# Process queue with 4 concurrent downloads
destill3d acquire run --concurrency 4 --rate-limit 60/min

# Extract features from local STEP file
destill3d extract file ./model.step --points 2048 --output ./snapshot.d3d

# Extract all STL files in directory
destill3d extract dir ./models/ --pattern "*.stl" --recursive

# Classify all pending snapshots using PointNet++
destill3d classify all --model pointnet2_ssg_mn40

# Zero-shot classification with custom classes
destill3d classify zero-shot \
    --snapshot abc123 \
    --classes "firearm,tool,toy,decoration"

# Query database
destill3d db query --platform thingiverse --label chair --confidence 0.8

# Export to HDF5 for training
destill3d db export \
    --format hdf5 \
    --output ./dataset.h5 \
    --taxonomy modelnet40 \
    --split 0.8:0.1:0.1  # train:val:test

# Show statistics
destill3d db stats
# Output:
# Total snapshots: 45,231
# By platform:
#   thingiverse: 32,451 (71.7%)
#   sketchfab: 8,234 (18.2%)
#   grabcad: 4,546 (10.1%)
# By format:
#   STL: 28,234 (62.4%)
#   STEP: 12,456 (27.5%)
#   OBJ: 4,541 (10.1%)
# Classification coverage: 98.2%
```

### 9.3 Output Formatting

```python
# Rich console output for interactive use
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()

def display_snapshot_info(snapshot: Snapshot):
    """Display snapshot details in rich format."""
    table = Table(title=f"Snapshot: {snapshot.snapshot_id[:8]}")
    
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Source", f"{snapshot.provenance.platform}:{snapshot.provenance.source_id}")
    table.add_row("Title", snapshot.provenance.title or "N/A")
    table.add_row("Format", snapshot.provenance.original_format)
    table.add_row("Points", str(snapshot.geometry.point_count))
    table.add_row("Watertight", "✓" if snapshot.features.is_watertight else "✗")
    
    if snapshot.predictions:
        pred = snapshot.predictions[0]
        table.add_row("Top Class", f"{pred.label} ({pred.confidence:.1%})")
    
    console.print(table)
```

---

## 10. Python API

### 10.1 Core API

```python
from destill3d import Destill3D, Snapshot, ClassificationResult
from destill3d.acquire import ThingiverseAdapter
from destill3d.extract import FeatureExtractor
from destill3d.classify import Classifier

# Initialize with default config
d3d = Destill3D()

# Or with custom config
d3d = Destill3D(
    db_path="./my_dataset.db",
    models_dir="./models",
    temp_dir="/tmp/destill3d",
)

# ─────────────────────────────────────────────────────────────────────────────
# ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────

# Search and download
results = await d3d.acquire.search(
    platform="thingiverse",
    query="mechanical keyboard",
    limit=50,
    filters={"license": "cc-by"}
)

# Queue for processing
for result in results:
    d3d.acquire.queue(result.url)

# Process queue
async for progress in d3d.acquire.process_queue():
    print(f"Downloaded: {progress.model_id} ({progress.completed}/{progress.total})")

# Or add single URL
snapshot = await d3d.acquire.from_url(
    "https://www.thingiverse.com/thing:12345"
)

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION (from local files)
# ─────────────────────────────────────────────────────────────────────────────

# Extract from local file
snapshot = d3d.extract.from_file(
    path="./model.step",
    point_count=2048,
    compute_views=False,
    metadata={
        "custom_tag": "experiment_1"
    }
)

# Extract with custom parameters
extractor = FeatureExtractor(
    sampling_strategy="fps",
    tessellation=TessellationConfig.high_quality(),
    normal_estimation_k=50,
)
snapshot = extractor.extract(mesh)

# Batch extraction
snapshots = d3d.extract.from_directory(
    path="./models/",
    pattern="**/*.stl",
    parallel=True,
    max_workers=4,
)

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

# Classify snapshot
result = d3d.classify(
    snapshot,
    model="pointnet2_ssg_mn40",
    top_k=5,
)
print(f"Top prediction: {result.predictions[0].label} ({result.predictions[0].confidence:.1%})")

# Zero-shot classification
result = d3d.classify.zero_shot(
    snapshot,
    classes=["chair", "table", "lamp", "sofa"],
)

# Batch classification
results = d3d.classify.batch(
    snapshots,
    model="dgcnn_mn40",
    batch_size=32,
    device="cuda",
)

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE QUERIES
# ─────────────────────────────────────────────────────────────────────────────

# Query snapshots
snapshots = d3d.db.query(
    platform="thingiverse",
    label="chair",
    min_confidence=0.8,
    limit=100,
)

# Get snapshot by ID
snapshot = d3d.db.get("abc123-def456")

# Find similar snapshots
similar = d3d.db.find_similar(
    snapshot,
    k=10,
    min_similarity=0.7,
)

# Statistics
stats = d3d.db.stats()
print(f"Total snapshots: {stats.total_count}")
print(f"Storage used: {stats.storage_bytes / 1e9:.2f} GB")

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

# Export to HDF5
d3d.export(
    output="./dataset.h5",
    format="hdf5",
    taxonomy="modelnet40",
    filters={"min_confidence": 0.9},
    splits={"train": 0.8, "val": 0.1, "test": 0.1},
)

# Export to NumPy
arrays = d3d.export(
    format="numpy",
    return_arrays=True,  # Return instead of save
)
# arrays = {"points": ndarray, "labels": ndarray, ...}
```

### 10.2 Low-Level API

```python
from destill3d.extract import (
    load_geometry,
    tessellate_brep,
    sample_point_cloud,
    compute_normals,
    compute_curvature,
    compute_global_features,
)
from destill3d.snapshot import Snapshot, GeometryData, Features

# Load geometry (auto-detects format)
mesh = load_geometry("./model.step")

# Or explicit CAD processing
from OCC.Core.STEPControl import STEPControl_Reader
reader = STEPControl_Reader()
reader.ReadFile("./model.step")
reader.TransferRoots()
shape = reader.OneShape()

mesh = tessellate_brep(
    shape,
    linear_deflection=0.001,
    angular_deflection=0.5,
)

# Sample point cloud
points = sample_point_cloud(
    mesh,
    n_points=2048,
    strategy="fps",
)

# Compute features
normals = compute_normals(points, k=30)
curvature = compute_curvature(points, k=30)
global_features = compute_global_features(points, normals, curvature, mesh)

# Assemble snapshot
snapshot = Snapshot(
    geometry=GeometryData(
        points=points,
        normals=normals,
        curvature=curvature,
    ),
    features=Features(
        global_features=global_features,
    ),
    provenance=Provenance(
        platform="local",
        source_url="file://./model.step",
    ),
)

# Serialize
snapshot.save("./snapshot.d3d")
loaded = Snapshot.load("./snapshot.d3d")
```

---

## 11. Configuration System

### 11.1 Configuration File

```yaml
# ~/.destill3d/config.yaml

# Database configuration
database:
  path: ~/.destill3d/destill3d.db
  type: sqlite  # sqlite or postgresql
  # PostgreSQL connection (if type: postgresql)
  # host: localhost
  # port: 5432
  # database: destill3d
  # user: destill3d
  # password: ${DESTILL3D_DB_PASSWORD}  # Env var expansion

# Model storage
models:
  directory: ~/.destill3d/models
  auto_download: true
  default_model: pointnet2_ssg_mn40

# Extraction defaults
extraction:
  target_points: 2048
  sampling_strategy: hybrid
  oversample_ratio: 5.0
  compute_views: false
  view_count: 12
  view_resolution: 224
  
  tessellation:
    linear_deflection: 0.001
    angular_deflection: 0.5
  
  normal_estimation:
    k_neighbors: 30
    orient: true

# Classification defaults
classification:
  batch_size: 32
  device: auto  # auto, cuda, cpu
  top_k: 5
  uncertainty_method: entropy

# Acquisition settings
acquisition:
  temp_directory: /tmp/destill3d
  max_concurrent_downloads: 4
  retry_attempts: 3
  retry_delay_seconds: 60
  
  # Platform-specific settings
  platforms:
    thingiverse:
      api_key: ${THINGIVERSE_API_KEY}
      rate_limit: 300/5min
    sketchfab:
      api_key: ${SKETCHFAB_API_KEY}
      rate_limit: 1000/day

# Logging
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR
  file: ~/.destill3d/destill3d.log
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Telemetry (optional, for monitoring)
telemetry:
  enabled: false
  endpoint: http://localhost:4317  # OTLP endpoint
```

### 11.2 Configuration Schema

```python
from pydantic import BaseSettings, Field
from typing import Optional, Dict, Any

class DatabaseConfig(BaseSettings):
    path: Path = Path("~/.destill3d/destill3d.db").expanduser()
    type: str = "sqlite"
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None

class ExtractionConfig(BaseSettings):
    target_points: int = Field(2048, ge=256, le=16384)
    sampling_strategy: str = "hybrid"
    oversample_ratio: float = Field(5.0, ge=1.0)
    compute_views: bool = False
    view_count: int = 12
    view_resolution: int = 224
    
    class TessellationConfig(BaseSettings):
        linear_deflection: float = 0.001
        angular_deflection: float = 0.5
    
    tessellation: TessellationConfig = TessellationConfig()

class Destill3DConfig(BaseSettings):
    database: DatabaseConfig = DatabaseConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    # ... other sections
    
    class Config:
        env_prefix = "DESTILL3D_"
        env_nested_delimiter = "__"
```

---

## 12. Error Handling & Recovery

### 12.1 Error Taxonomy

```python
class Destill3DError(Exception):
    """Base exception for all Destill3D errors."""
    pass

# Acquisition errors
class AcquisitionError(Destill3DError):
    """Error during model acquisition."""
    pass

class PlatformError(AcquisitionError):
    """Platform-specific error (API issues, auth failures)."""
    platform: str
    status_code: Optional[int]

class RateLimitError(AcquisitionError):
    """Rate limit exceeded."""
    retry_after: Optional[int]  # Seconds to wait

class DownloadError(AcquisitionError):
    """File download failed."""
    url: str
    reason: str

# Extraction errors
class ExtractionError(Destill3DError):
    """Error during feature extraction."""
    pass

class FormatError(ExtractionError):
    """Unsupported or malformed file format."""
    format: Optional[str]
    details: str

class GeometryError(ExtractionError):
    """Invalid or degenerate geometry."""
    issue: str  # e.g., "non-manifold", "zero-area-faces"

class TessellationError(ExtractionError):
    """B-rep tessellation failed."""
    shape_type: str

# Classification errors
class ClassificationError(Destill3DError):
    """Error during classification."""
    pass

class ModelNotFoundError(ClassificationError):
    """Requested model not available."""
    model_id: str

class InferenceError(ClassificationError):
    """Model inference failed."""
    model_id: str
    reason: str

# Database errors
class DatabaseError(Destill3DError):
    """Database operation failed."""
    pass
```

### 12.2 Retry Strategy

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class RetryConfig:
    download_attempts: int = 3
    download_wait_min: int = 1
    download_wait_max: int = 60
    
    extraction_attempts: int = 2
    
    classification_attempts: int = 2

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((DownloadError, RateLimitError)),
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} for {retry_state.args[0]}"
    ),
)
async def download_with_retry(url: str, target: Path) -> Path:
    """Download with automatic retry on transient failures."""
    pass
```

### 12.3 Recovery Procedures

```python
async def recover_processing_queue(db: Database):
    """
    Recover interrupted processing.
    
    Called on startup to handle items left in intermediate states.
    """
    # Find items stuck in non-terminal states
    stuck = await db.query_queue(
        stages=[PipelineStage.ACQUIRED, PipelineStage.EXTRACTED],
        updated_before=datetime.utcnow() - timedelta(hours=1),
    )
    
    for item in stuck:
        if item.retry_count < item.max_retries:
            # Reset to previous stage for retry
            if item.stage == PipelineStage.EXTRACTED:
                await db.update_queue(item.model_id, stage=PipelineStage.ACQUIRED)
            else:
                await db.update_queue(item.model_id, stage=PipelineStage.QUEUED)
            
            item.retry_count += 1
            logger.info(f"Recovered {item.model_id}, retry {item.retry_count}")
        else:
            # Max retries exceeded, mark as failed
            await db.update_queue(item.model_id, stage=PipelineStage.FAILED)
            logger.error(f"Max retries exceeded for {item.model_id}")
```

---

## 13. Performance Requirements

### 13.1 Benchmarks

| Operation | Target | Hardware Baseline |
|-----------|--------|-------------------|
| Download (per model) | <10s (network dependent) | 100 Mbps connection |
| STEP tessellation | <5s | 4-core CPU |
| STL loading | <1s | 4-core CPU |
| Point sampling (2048 pts) | <500ms | 4-core CPU |
| Feature extraction (full) | <2s | 4-core CPU |
| Classification (single) | <100ms | RTX 3090 |
| Classification (batch 32) | <200ms | RTX 3090 |
| Database insert | <10ms | NVMe SSD |
| Similarity search (100K index) | <50ms | 16GB RAM |

### 13.2 Memory Constraints

```python
class MemoryLimits:
    # Per-model processing limits
    max_mesh_vertices: int = 10_000_000      # 10M vertices
    max_mesh_faces: int = 20_000_000         # 20M faces
    max_point_cloud: int = 1_000_000         # 1M points (before sampling)
    
    # Simplify if exceeded
    simplification_target: int = 1_000_000   # Simplify to 1M faces
    
    # Batch processing limits
    max_batch_gpu_memory: int = 8 * 1024**3  # 8 GB
    max_batch_size: int = 64                 # Classification batch
```

### 13.3 Scalability Targets

| Scale | Snapshots | Storage | Query Time | Notes |
|-------|-----------|---------|------------|-------|
| Small | <10K | <1 GB | <10ms | Single SQLite file |
| Medium | 10K-100K | 1-10 GB | <50ms | SQLite with indexes |
| Large | 100K-1M | 10-100 GB | <100ms | PostgreSQL recommended |
| XL | >1M | >100 GB | <200ms | PostgreSQL + FAISS index |

---

## 14. Security Considerations

### 14.1 Input Validation

```python
class InputValidator:
    """Validate inputs to prevent security issues."""
    
    # File size limits
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
    
    # Allowed extensions (whitelist)
    ALLOWED_EXTENSIONS = {
        ".step", ".stp", ".iges", ".igs", ".brep",
        ".stl", ".obj", ".ply", ".off", ".gltf", ".glb",
        ".3mf", ".dae", ".pcd", ".xyz", ".las", ".laz"
    }
    
    def validate_file(self, path: Path) -> ValidationResult:
        """Validate file before processing."""
        # Check extension
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            return ValidationResult(valid=False, error="Unsupported file type")
        
        # Check file size
        if path.stat().st_size > self.MAX_FILE_SIZE:
            return ValidationResult(valid=False, error="File too large")
        
        # Check for path traversal
        if ".." in str(path):
            return ValidationResult(valid=False, error="Invalid path")
        
        return ValidationResult(valid=True)
    
    def validate_url(self, url: str) -> ValidationResult:
        """Validate URL before download."""
        parsed = urlparse(url)
        
        # Must be HTTPS
        if parsed.scheme not in ("https", "http"):
            return ValidationResult(valid=False, error="Invalid scheme")
        
        # Check against allowed domains
        if parsed.netloc not in self.ALLOWED_DOMAINS:
            return ValidationResult(valid=False, error="Domain not allowed")
        
        return ValidationResult(valid=True)
```

### 14.2 API Key Management

```python
class CredentialManager:
    """Secure storage and retrieval of API credentials."""
    
    def __init__(self, keyring_service: str = "destill3d"):
        self.service = keyring_service
    
    def store(self, platform: str, api_key: str):
        """Store API key in system keyring."""
        keyring.set_password(self.service, platform, api_key)
    
    def retrieve(self, platform: str) -> Optional[str]:
        """Retrieve API key from keyring or environment."""
        # Try environment variable first
        env_var = f"{platform.upper()}_API_KEY"
        if env_key := os.environ.get(env_var):
            return env_key
        
        # Fall back to keyring
        return keyring.get_password(self.service, platform)
    
    def delete(self, platform: str):
        """Remove stored credential."""
        keyring.delete_password(self.service, platform)
```

### 14.3 License Compliance

```python
class LicenseFilter:
    """Filter models by license for legal compliance."""
    
    # Licenses allowing redistribution of derived data
    PERMISSIVE_LICENSES = {
        "cc0",           # Public domain
        "cc-by",         # Attribution
        "cc-by-sa",      # Attribution-ShareAlike
        "mit",
        "apache-2.0",
        "gpl-3.0",
    }
    
    # Licenses prohibiting commercial use
    NON_COMMERCIAL = {
        "cc-by-nc",
        "cc-by-nc-sa",
        "cc-by-nc-nd",
    }
    
    def is_allowed(
        self, 
        license: str, 
        use_case: str = "research"
    ) -> bool:
        """Check if license permits intended use."""
        license = license.lower()
        
        if use_case == "commercial":
            return license in self.PERMISSIVE_LICENSES
        
        # Research/personal use
        return (
            license in self.PERMISSIVE_LICENSES or 
            license in self.NON_COMMERCIAL
        )
```

---

## 15. Testing Strategy

### 15.1 Test Categories

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_extraction.py
│   ├── test_sampling.py
│   ├── test_features.py
│   ├── test_snapshot.py
│   └── test_database.py
│
├── integration/             # Component interaction tests
│   ├── test_pipeline.py
│   ├── test_platform_adapters.py
│   └── test_classification.py
│
├── e2e/                     # End-to-end workflow tests
│   ├── test_acquire_extract_classify.py
│   └── test_cli.py
│
├── fixtures/                # Test data
│   ├── models/
│   │   ├── cube.stl
│   │   ├── sphere.step
│   │   └── complex.obj
│   └── snapshots/
│       └── reference.d3d
│
└── benchmarks/              # Performance tests
    ├── bench_extraction.py
    └── bench_classification.py
```

### 15.2 Key Test Cases

```python
# Unit test: Point sampling produces correct count
def test_fps_sampling_count():
    mesh = trimesh.creation.icosphere()
    points = sample_point_cloud(mesh, n_points=1024, strategy="fps")
    assert points.shape == (1024, 3)

# Unit test: Normals are unit vectors
def test_normals_are_normalized():
    points = np.random.randn(100, 3)
    normals = compute_normals(points, k=10)
    norms = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

# Integration test: Full extraction pipeline
def test_extraction_pipeline_step_file():
    snapshot = extract_from_file("fixtures/models/sphere.step")
    assert snapshot.geometry.point_count == 2048
    assert snapshot.features.is_watertight == True
    assert 0.9 < snapshot.features.sphericity < 1.0  # Near-sphere

# E2E test: CLI workflow
def test_cli_extract_classify(cli_runner, tmp_path):
    result = cli_runner.invoke(
        ["extract", "file", "fixtures/models/cube.stl", 
         "--output", str(tmp_path / "out.d3d")]
    )
    assert result.exit_code == 0
    
    result = cli_runner.invoke(
        ["classify", "snapshot", str(tmp_path / "out.d3d")]
    )
    assert result.exit_code == 0
    assert "chair" in result.output or "table" in result.output  # Some prediction
```

### 15.3 Test Coverage Requirements

| Category | Minimum Coverage | Critical Paths |
|----------|------------------|----------------|
| Unit | 80% | Feature computation, serialization |
| Integration | 60% | Pipeline stages, database operations |
| E2E | Key workflows | Acquire→Extract→Classify→Export |

---

## 16. Deployment & Distribution

### 16.1 Package Structure

```
destill3d/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── destill3d/
│       ├── __init__.py
│       ├── __main__.py          # CLI entry point
│       ├── core/
│       │   ├── config.py
│       │   ├── database.py
│       │   └── snapshot.py
│       ├── acquire/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── platforms/
│       │       ├── thingiverse.py
│       │       └── sketchfab.py
│       ├── extract/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── tessellation.py
│       │   ├── sampling.py
│       │   └── features.py
│       ├── classify/
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   └── inference.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── commands/
│       └── proto/
│           └── snapshot.proto
├── tests/
└── docs/
```

### 16.2 Installation Options

```bash
# From PyPI (standard)
pip install destill3d

# With GPU support
pip install destill3d[gpu]

# With all optional dependencies
pip install destill3d[full]

# Development install
pip install -e ".[dev]"

# From source with specific extras
pip install "destill3d[postgresql,faiss-gpu] @ git+https://github.com/..."
```

### 16.3 Dependency Groups

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "trimesh>=4.0",
    "open3d>=0.17",
    "httpx>=0.25",
    "typer>=0.9",
    "rich>=13.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "protobuf>=4.0",
    "sqlalchemy>=2.0",
]

[project.optional-dependencies]
cad = [
    "pythonocc-core>=7.7",
]
gpu = [
    "torch>=2.0",
    "onnxruntime-gpu>=1.16",
]
postgresql = [
    "psycopg2-binary>=2.9",
]
faiss = [
    "faiss-cpu>=1.7",
]
faiss-gpu = [
    "faiss-gpu>=1.7",
]
full = [
    "destill3d[cad,gpu,postgresql,faiss-gpu]",
]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "mypy>=1.0",
    "ruff>=0.1",
]
```

---

## 17. Future Roadmap

### 17.1 Version Milestones

| Version | Target | Key Features |
|---------|--------|--------------|
| **0.1.0** | MVP | Local file extraction, SQLite storage, PointNet++ classification |
| **0.2.0** | +Acquire | Thingiverse + Sketchfab adapters, queue management |
| **0.3.0** | +Zero-shot | OpenShape integration, custom class classification |
| **0.4.0** | +Scale | PostgreSQL backend, FAISS similarity search |
| **0.5.0** | +Platforms | GrabCAD, Cults3D, MyMiniFactory adapters |
| **1.0.0** | Production | Full feature set, comprehensive docs, stable API |

### 17.2 Future Features

**Near-term (v0.x)**:

- Multi-view CNN hybrid classification
- Batch export to TFRecord for TensorFlow training
- REST API server mode
- Docker container with GPU support

**Medium-term (v1.x)**:

- Hierarchical classification with part segmentation
- Active learning: flag uncertain samples for human review
- Distributed processing with Celery/Redis
- Real-time streaming classification

**Long-term (v2.x)**:

- Self-supervised pre-training on acquired dataset
- 3D-language model fine-tuning
- Generative model integration (Point-E, Shap-E)
- Browser extension for one-click acquisition

### 17.3 Research Integrations

| Capability | Target Research | Integration Path |
|------------|-----------------|------------------|
| Zero-shot | OpenShape, ULIP | ONNX export, embedding alignment |
| Pre-training | Point-MAE, Point-M2AE | Custom training scripts |
| Generation | Point-E, Shap-E | Text-to-point-cloud for augmentation |
| Multimodal | Point-Bind, Point-LLM | Unified embedding space |

---

## Appendices

### A. Glossary

| Term | Definition |
|------|------------|
| **B-rep** | Boundary Representation: CAD solid defined by its boundary surfaces |
| **FPS** | Farthest Point Sampling: iteratively select points maximizing coverage |
| **FPFH** | Fast Point Feature Histograms: local geometric descriptor |
| **SDF** | Signed Distance Field: implicit surface representation |
| **Snapshot** | Compressed, self-contained representation of a 3D model |
| **Taxonomy** | Predefined classification scheme (e.g., ModelNet40) |
| **Tessellation** | Conversion of analytical geometry to triangular mesh |

### B. File Format Reference

| Format | Extension | Type | Typical Use |
|--------|-----------|------|-------------|
| STEP | .step, .stp | CAD B-rep | Engineering exchange |
| IGES | .iges, .igs | CAD B-rep | Legacy CAD exchange |
| STL | .stl | Mesh | 3D printing |
| OBJ | .obj | Mesh | Graphics, games |
| GLTF | .gltf, .glb | Mesh | Web, Sketchfab |
| PLY | .ply | Mesh/Points | Research, scanning |
| PCD | .pcd | Points | PCL native |

### C. Platform API Reference

See individual adapter documentation:

- [Thingiverse API v1](https://www.thingiverse.com/developers)
- [Sketchfab Data API](https://sketchfab.com/developers/data-api)
- [GrabCAD](https://grabcad.com/) (no official API)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0-draft | 2026-02 | LayerDynamics | Initial specification |

---

*End of Specification Document*
