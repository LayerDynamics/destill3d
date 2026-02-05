"""
Benchmarks for classification pipeline.

Run standalone: python tests/benchmarks/bench_classification.py
"""

import time

import numpy as np

from destill3d.core.snapshot import (
    Features,
    GeometryData,
    ProcessingMetadata,
    Provenance,
    Snapshot,
)


def create_benchmark_snapshot(n_points: int = 2048) -> Snapshot:
    """Create a snapshot for benchmarking."""
    np.random.seed(42)
    points = np.random.randn(n_points, 3).astype(np.float32)
    normals = np.random.randn(n_points, 3).astype(np.float32)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    return Snapshot(
        model_id="bench:test",
        provenance=Provenance(
            platform="benchmark",
            source_id="test",
            original_filename="bench.stl",
            original_format="stl_binary",
            original_file_size=1024,
        ),
        geometry=GeometryData(
            points=points,
            normals=normals,
            curvature=np.random.rand(n_points).astype(np.float32),
            centroid=np.zeros(3, dtype=np.float32),
            scale=1.0,
        ),
        features=Features(
            global_features=np.random.randn(32).astype(np.float32),
            surface_area=10.0,
            volume=5.0,
            is_watertight=True,
            original_vertex_count=1000,
            original_face_count=2000,
            bbox_min=np.array([-1, -1, -1], dtype=np.float32),
            bbox_max=np.array([1, 1, 1], dtype=np.float32),
        ),
        processing=ProcessingMetadata(
            target_points=n_points,
            sampling_strategy="hybrid",
        ),
    )


def bench_snapshot_serialization(n_iters: int = 100):
    """Benchmark snapshot serialization/deserialization."""
    snapshot = create_benchmark_snapshot()

    # Serialize
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        data = snapshot.to_dict()
        times.append(time.perf_counter() - start)

    print(f"  Serialize:    {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")

    # Deserialize
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        Snapshot.from_dict(data)
        times.append(time.perf_counter() - start)

    print(f"  Deserialize:  {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")


def bench_proto_serialization(n_iters: int = 100):
    """Benchmark protobuf dict serialization."""
    from destill3d.proto import proto_dict_to_snapshot, snapshot_to_proto_dict

    snapshot = create_benchmark_snapshot()

    # Serialize
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        data = snapshot_to_proto_dict(snapshot)
        times.append(time.perf_counter() - start)

    print(f"  Proto serialize:    {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")

    # Deserialize
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        proto_dict_to_snapshot(data)
        times.append(time.perf_counter() - start)

    print(f"  Proto deserialize:  {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")


if __name__ == "__main__":
    print("Destill3D Classification Benchmarks")
    print("=" * 50)

    print("\nSnapshot serialization:")
    bench_snapshot_serialization()

    print("\nProtobuf dict serialization:")
    bench_proto_serialization()
