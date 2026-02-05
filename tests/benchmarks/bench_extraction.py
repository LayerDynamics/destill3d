"""
Benchmarks for feature extraction pipeline.

Run with: python -m pytest tests/benchmarks/bench_extraction.py -v --benchmark
Or standalone: python tests/benchmarks/bench_extraction.py
"""

import time
from pathlib import Path

import numpy as np
import trimesh

from destill3d.core.config import ExtractionConfig
from destill3d.extract import FeatureExtractor
from destill3d.extract.features import (
    compute_curvature,
    compute_global_features,
    compute_normals,
)
from destill3d.extract.sampling import SamplingConfig, SamplingStrategy, sample_point_cloud


def bench_sampling(mesh: trimesh.Trimesh, n_points: int = 2048, n_iters: int = 10):
    """Benchmark point cloud sampling."""
    config = SamplingConfig(
        strategy=SamplingStrategy.HYBRID,
        target_points=n_points,
    )

    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        sample_point_cloud(mesh, config)
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
    }


def bench_normals(points: np.ndarray, n_iters: int = 10):
    """Benchmark normal estimation."""
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        compute_normals(points, k=30)
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
    }


def bench_features(points: np.ndarray, normals: np.ndarray, curvature: np.ndarray, mesh, n_iters: int = 10):
    """Benchmark global feature computation."""
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        compute_global_features(points, normals, curvature, mesh)
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
    }


def bench_full_pipeline(stl_path: Path, n_iters: int = 5):
    """Benchmark full extraction pipeline."""
    config = ExtractionConfig(target_points=2048)
    extractor = FeatureExtractor(config)

    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        extractor.extract_from_file(stl_path)
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
    }


if __name__ == "__main__":
    import tempfile

    print("Destill3D Extraction Benchmarks")
    print("=" * 50)

    # Create test meshes
    meshes = {
        "cube": trimesh.creation.box(extents=[1, 1, 1]),
        "sphere_low": trimesh.creation.icosphere(subdivisions=2),
        "sphere_high": trimesh.creation.icosphere(subdivisions=4),
    }

    for name, mesh in meshes.items():
        print(f"\nMesh: {name} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")

        # Sampling
        result = bench_sampling(mesh)
        print(f"  Sampling:   {result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms")

        # Normals
        config = SamplingConfig(target_points=2048)
        sample = sample_point_cloud(mesh, config)
        result = bench_normals(sample.points)
        print(f"  Normals:    {result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms")

    # Full pipeline with file I/O
    print("\nFull pipeline (with file I/O):")
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        mesh = trimesh.creation.icosphere(subdivisions=3)
        mesh.export(f.name)
        result = bench_full_pipeline(Path(f.name))
        print(f"  Pipeline:   {result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms")
