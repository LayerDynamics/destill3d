"""Unit tests for EmbeddingIndex (FAISS vector search)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
class TestEmbeddingIndex:
    def test_create_flat_cosine(self):
        from destill3d.core.vector_index import EmbeddingIndex
        idx = EmbeddingIndex(dimension=128, index_type="flat", metric="cosine")
        assert len(idx) == 0

    def test_add_single(self):
        from destill3d.core.vector_index import EmbeddingIndex
        idx = EmbeddingIndex(dimension=128, metric="cosine")
        emb = np.random.randn(128).astype(np.float32)
        idx.add("snap1", emb)
        assert len(idx) == 1
        assert "snap1" in idx

    def test_add_batch(self):
        from destill3d.core.vector_index import EmbeddingIndex
        idx = EmbeddingIndex(dimension=128, metric="cosine")
        ids = [f"snap{i}" for i in range(10)]
        embs = np.random.randn(10, 128).astype(np.float32)
        idx.add_batch(ids, embs)
        assert len(idx) == 10
        assert "snap5" in idx

    def test_search(self):
        from destill3d.core.vector_index import EmbeddingIndex
        idx = EmbeddingIndex(dimension=128, metric="cosine")
        ids = [f"snap{i}" for i in range(20)]
        embs = np.random.randn(20, 128).astype(np.float32)
        idx.add_batch(ids, embs)

        results = idx.search(embs[0], k=5)
        assert len(results) == 5
        assert results[0][0] == "snap0"  # Self should be most similar

    def test_search_exclude(self):
        from destill3d.core.vector_index import EmbeddingIndex
        idx = EmbeddingIndex(dimension=128, metric="cosine")
        ids = [f"snap{i}" for i in range(10)]
        embs = np.random.randn(10, 128).astype(np.float32)
        idx.add_batch(ids, embs)

        results = idx.search(embs[0], k=5, exclude_ids=["snap0"])
        assert all(r[0] != "snap0" for r in results)

    def test_remove(self):
        from destill3d.core.vector_index import EmbeddingIndex
        idx = EmbeddingIndex(dimension=128, metric="cosine")
        idx.add("snap1", np.random.randn(128).astype(np.float32))
        assert idx.remove("snap1") is True
        assert idx.remove("nonexistent") is False

    def test_search_empty(self):
        from destill3d.core.vector_index import EmbeddingIndex
        idx = EmbeddingIndex(dimension=128, metric="cosine")
        results = idx.search(np.random.randn(128).astype(np.float32), k=5)
        assert results == []

    def test_l2_metric(self):
        from destill3d.core.vector_index import EmbeddingIndex
        idx = EmbeddingIndex(dimension=64, metric="l2")
        embs = np.random.randn(5, 64).astype(np.float32)
        idx.add_batch([f"s{i}" for i in range(5)], embs)
        results = idx.search(embs[0], k=3)
        assert len(results) == 3
