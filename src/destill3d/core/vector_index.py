"""
FAISS-based similarity search for snapshot embeddings.

Provides efficient nearest-neighbor search in embedding space
for finding similar 3D models.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingIndex:
    """FAISS-based similarity search for snapshot embeddings."""

    def __init__(
        self,
        dimension: int = 1024,
        index_type: str = "flat",  # flat, ivf, hnsw
        metric: str = "cosine",    # cosine, l2
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self._index = None
        self._id_map: List[str] = []  # snapshot_id at each index position
        self._reverse_map: dict = {}  # snapshot_id -> index position

    def _create_index(self):
        """Create FAISS index based on configuration."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss not installed. Install with: pip install faiss-cpu"
            )

        if self.metric == "cosine":
            # For cosine similarity, use inner product after L2 normalization
            if self.index_type == "flat":
                self._index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self._index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, 100, faiss.METRIC_INNER_PRODUCT
                )
            elif self.index_type == "hnsw":
                self._index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
        else:
            # L2 distance
            if self.index_type == "flat":
                self._index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.dimension)
                self._index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, 100
                )
            elif self.index_type == "hnsw":
                self._index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")

    def add(self, snapshot_id: str, embedding: np.ndarray) -> None:
        """Add a single embedding to the index."""
        if self._index is None:
            self._create_index()

        embedding = embedding.astype(np.float32).flatten()

        # L2 normalize for cosine similarity
        if self.metric == "cosine":
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        self._index.add(embedding.reshape(1, -1))

        idx = len(self._id_map)
        self._id_map.append(snapshot_id)
        self._reverse_map[snapshot_id] = idx

    def add_batch(
        self,
        snapshot_ids: List[str],
        embeddings: np.ndarray,
    ) -> None:
        """Add multiple embeddings at once."""
        if self._index is None:
            self._create_index()

        embeddings = embeddings.astype(np.float32)

        # Normalize
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            embeddings = embeddings / norms

        self._index.add(embeddings)

        start_idx = len(self._id_map)
        for i, sid in enumerate(snapshot_ids):
            self._id_map.append(sid)
            self._reverse_map[sid] = start_idx + i

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Find k most similar snapshots.

        Args:
            query_embedding: Query embedding vector.
            k: Number of results.
            exclude_ids: Snapshot IDs to exclude from results.

        Returns:
            List of (snapshot_id, similarity_score) tuples.
        """
        if self._index is None or len(self._id_map) == 0:
            return []

        query = query_embedding.astype(np.float32).flatten()

        # Normalize query
        if self.metric == "cosine":
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

        # Search (get extra results if we need to filter)
        search_k = k + len(exclude_ids or [])
        scores, indices = self._index.search(
            query.reshape(1, -1),
            min(search_k, len(self._id_map)),
        )

        results = []
        exclude_set = set(exclude_ids or [])

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue

            snapshot_id = self._id_map[idx]
            if snapshot_id in exclude_set:
                continue

            results.append((snapshot_id, float(score)))

            if len(results) >= k:
                break

        return results

    def remove(self, snapshot_id: str) -> bool:
        """
        Remove embedding from index.

        Note: FAISS doesn't support efficient removal.
        This marks the ID as removed; actual cleanup happens on rebuild.

        Returns:
            True if the ID was found.
        """
        if snapshot_id in self._reverse_map:
            del self._reverse_map[snapshot_id]
            return True
        return False

    def save(self, path: Path) -> None:
        """Persist index to disk."""
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(path / "index.faiss"))

        # Save ID map
        with open(path / "id_map.json", "w") as f:
            json.dump({
                "id_map": self._id_map,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric,
            }, f)

        logger.info(f"Saved embedding index ({len(self._id_map)} entries) to {path}")

    @classmethod
    def load(cls, path: Path) -> "EmbeddingIndex":
        """Load index from disk."""
        import faiss

        path = Path(path)

        with open(path / "id_map.json") as f:
            data = json.load(f)

        index = cls(
            dimension=data.get("dimension", 1024),
            index_type=data.get("index_type", "flat"),
            metric=data.get("metric", "cosine"),
        )

        index._index = faiss.read_index(str(path / "index.faiss"))
        index._id_map = data["id_map"]
        index._reverse_map = {sid: i for i, sid in enumerate(index._id_map)}

        logger.info(f"Loaded embedding index ({len(index._id_map)} entries) from {path}")
        return index

    def __len__(self) -> int:
        return len(self._id_map)

    def __contains__(self, snapshot_id: str) -> bool:
        return snapshot_id in self._reverse_map
