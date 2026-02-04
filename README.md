# Destill3D

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**3D Model Feature Extraction & Classification Toolkit**

Destill3D transforms 3D files into compact, classification-ready snapshots with 100:1+ compression ratios. Extract geometric features, classify objects using PointNet++, and build searchable databases of 3D model fingerprints.

## Features

- **Multi-format Support**: STL, OBJ, PLY, GLTF, STEP, IGES, and more
- **Intelligent Sampling**: FPS, uniform, hybrid, and Poisson disk sampling
- **Rich Feature Extraction**: Normals, curvature, 32-dim global shape descriptors
- **ML Classification**: PointNet++ with ModelNet40 taxonomy (40 object classes)
- **Compact Storage**: Binary snapshots with gzip compression
- **SQLite Database**: Query, search, and export your model collection
- **Extensible CLI**: Full-featured command-line interface

## Installation

### Basic Installation

```bash
pip install destill3d
```

### With Classification Support (CPU)

```bash
pip install "destill3d[cpu]"
```

### With GPU Acceleration

```bash
pip install "destill3d[gpu]"
```

### Development Installation

```bash
git clone https://github.com/LayerDynamics/destill3d.git
cd destill3d
pip install -e ".[cpu,dev]"
```

### Optional Dependencies

| Extra | Description |
|-------|-------------|
| `cpu` | ONNX Runtime + PyTorch for CPU inference |
| `gpu` | ONNX Runtime GPU + PyTorch CUDA |
| `cad` | pythonocc for STEP/IGES CAD files |
| `open3d` | Open3D for advanced point cloud processing |
| `dev` | pytest, mypy, ruff for development |

## Quick Start

### Extract Features from a 3D File

```bash
# Quick extraction
destill3d quick model.stl

# With classification
destill3d quick model.stl --classify

# Batch extraction
destill3d extract dir ./models/ --recursive
```

### Python API

```python
from destill3d import FeatureExtractor
from destill3d.core.config import ExtractionConfig

# Configure extraction
config = ExtractionConfig(
    target_points=2048,
    sampling_strategy="hybrid",
)

# Extract features
extractor = FeatureExtractor(config)
snapshot = extractor.extract_from_file("model.stl")

# Access extracted data
print(f"Points: {snapshot.geometry.point_count}")
print(f"Surface Area: {snapshot.features.surface_area:.2f}")
print(f"Global Features: {snapshot.features.global_features.shape}")

# Save snapshot (typically 100x smaller than original)
snapshot.save("model.d3d")
```

### Classification

```python
from destill3d.classify import Classifier

classifier = Classifier()
predictions, embedding = classifier.classify(snapshot)

for pred in predictions[:3]:
    print(f"{pred.label}: {pred.confidence:.1%}")
# Output:
# chair: 94.2%
# armchair: 3.1%
# stool: 1.4%
```

## CLI Reference

### System Information

```bash
destill3d info
```

### Feature Extraction

```bash
# Single file
destill3d extract file model.stl --output model.d3d

# Directory (batch)
destill3d extract dir ./models/ --recursive --workers 4

# File info without extraction
destill3d extract info model.step
```

### Classification

```bash
# Classify a snapshot
destill3d classify snapshot model.d3d

# Classify all unclassified in database
destill3d classify all --model pointnet2_ssg_mn40

# List available models
destill3d classify models
```

### Database Operations

```bash
# View statistics
destill3d db stats

# Query snapshots
destill3d db query --label chair --min-confidence 0.9

# Export to various formats
destill3d db export --format numpy --output embeddings.npz
destill3d db export --format parquet --output data.parquet

# Show snapshot details
destill3d db show local:abc123def
```

### Configuration

```bash
# Show current config
destill3d config show

# Initialize config file
destill3d config init

# Set values
destill3d config set extraction.target_points 4096
destill3d config set classification.device cuda
```

## Snapshot Format

A Destill3D snapshot (`.d3d`) contains:

| Component | Description |
|-----------|-------------|
| **Provenance** | Source URL, platform, metadata, file hash |
| **Geometry** | Normalized point cloud (N×3), normals, curvature |
| **Features** | 32-dim global features, surface area, volume, bbox |
| **Predictions** | Classification labels with confidence scores |
| **Embedding** | 1024-dim latent vector for similarity search |

### Compression Ratio

| Original Format | Typical Size | Snapshot Size | Ratio |
|-----------------|--------------|---------------|-------|
| STL (binary) | 5 MB | 50 KB | 100:1 |
| OBJ | 10 MB | 50 KB | 200:1 |
| STEP | 2 MB | 50 KB | 40:1 |

## Supported File Formats

### Mesh Formats
- **STL** (ASCII & Binary)
- **OBJ** (Wavefront)
- **PLY** (Stanford)
- **OFF** (Object File Format)
- **GLTF/GLB** (GL Transmission Format)

### CAD Formats (requires `[cad]` extra)
- **STEP/STP** (ISO 10303)
- **IGES/IGS** (Initial Graphics Exchange)
- **BREP/BRP** (OpenCASCADE)

## Configuration

Destill3D can be configured via:

1. **Environment variables**: `DESTILL3D_EXTRACTION_TARGET_POINTS=4096`
2. **Config file**: `~/.destill3d/config.yaml`
3. **Python API**: Direct configuration objects

### Example Config File

```yaml
extraction:
  target_points: 2048
  sampling_strategy: hybrid
  oversample_ratio: 5.0

classification:
  batch_size: 32
  device: auto
  top_k: 5

database:
  type: sqlite
  path: ~/.destill3d/destill3d.db
```

## Architecture

```
Input File (.stl/.step/.obj)
    │
    ▼
┌─────────────────┐
│  FormatDetector │ ─► Detect file type via magic bytes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  load_geometry  │ ─► trimesh.Trimesh (tessellate if CAD)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│sample_point_cloud│ ─► (N, 3) normalized points
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│compute_features │ ─► normals, curvature, 32-dim global
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Snapshot     │ ─► Serialize to .d3d (gzip + binary)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Classifier    │ ─► PointNet++ inference ─► predictions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Database     │ ─► SQLite storage + embeddings
└─────────────────┘
```

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=destill3d --cov-report=html
```

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/destill3d/

# Format
ruff format src/
```

## Roadmap

- [ ] Platform adapters (Thingiverse, Sketchfab, Printables)
- [ ] Multi-view rendering
- [ ] FAISS similarity search
- [ ] PostgreSQL support
- [ ] REST API server
- [ ] Web UI dashboard

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Destill3D in your research, please cite:

```bibtex
@software{destill3d,
  title = {Destill3D: 3D Model Feature Extraction and Classification Toolkit},
  author = {LayerDynamics},
  year = {2024},
  url = {https://github.com/LayerDynamics/destill3d}
}
```
