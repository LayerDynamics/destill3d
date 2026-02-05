"""
Classification inference engine for Destill3D.

Provides ONNX-based inference for point cloud classification
with multiple uncertainty estimation methods.
"""

from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
import time

import numpy as np

from destill3d.core.snapshot import Snapshot, Prediction
from destill3d.core.exceptions import InferenceError, ModelNotFoundError
from destill3d.classify.registry import (
    ModelRegistry,
    RegisteredModel,
    ModelFormat,
    TaxonomyConfig,
)


class UncertaintyMethod(Enum):
    """Methods for estimating classification uncertainty."""

    ENTROPY = "entropy"
    MC_DROPOUT = "mc_dropout"
    ENSEMBLE = "ensemble"


class Classifier:
    """
    Point cloud classification using pre-trained models.

    Supports ONNX Runtime for inference with automatic GPU detection.
    """

    def __init__(
        self,
        models_dir: Path,
        device: str = "auto",
    ):
        """
        Initialize the classifier.

        Args:
            models_dir: Directory containing model weights
            device: Device for inference ('auto', 'cuda', 'cpu')
        """
        self.registry = ModelRegistry(models_dir)
        self.device = self._resolve_device(device)
        self._loaded_models: dict = {}

    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device == "auto":
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    return "cuda"
            except ImportError:
                pass

            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass

            return "cpu"

        return device

    def _load_model(self, model_id: str):
        """Load model into memory."""
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        model_info = self.registry.get(model_id)
        model_path = self.registry.get_model_path(model_id)

        if model_info.format == ModelFormat.ONNX:
            session = self._load_onnx_model(model_path)
        else:
            raise InferenceError(
                model_id,
                f"Unsupported model format: {model_info.format}",
            )

        self._loaded_models[model_id] = (session, model_info)
        return self._loaded_models[model_id]

    def _load_onnx_model(self, model_path: Path):
        """Load ONNX model using ONNX Runtime."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise InferenceError(
                "onnx",
                "onnxruntime not installed. Install with: pip install onnxruntime",
            )

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        try:
            session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as e:
            raise InferenceError("onnx", f"Failed to load model: {e}")

        return session

    def classify(
        self,
        snapshot: Snapshot,
        model_id: str = "pointnet2_ssg_mn40",
        top_k: int = 5,
    ) -> Tuple[List[Prediction], Optional[np.ndarray]]:
        """
        Classify a single snapshot.

        Args:
            snapshot: Snapshot to classify
            model_id: Model to use for classification
            top_k: Number of top predictions to return

        Returns:
            Tuple of (predictions, embedding)
            - predictions: List of Prediction objects
            - embedding: Feature embedding (1024,) or None if not available

        Raises:
            InferenceError: If classification fails
        """
        start_time = time.time()

        session, model_info = self._load_model(model_id)
        taxonomy = self.registry.get_taxonomy(model_info.taxonomy)

        # Prepare input
        if snapshot.geometry is None:
            raise InferenceError(model_id, "Snapshot has no geometry data")

        points = snapshot.geometry.points.astype(np.float32)

        # Ensure correct number of points
        if len(points) != model_info.input_points:
            points = self._adjust_point_count(points, model_info.input_points)

        # Add batch dimension: (1, N, 3)
        points_batch = points.reshape(1, -1, 3)

        # Run inference
        try:
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: points_batch})
        except Exception as e:
            raise InferenceError(model_id, f"Inference failed: {e}")

        # Parse outputs
        logits = outputs[0]  # (1, num_classes)
        embedding = outputs[1] if len(outputs) > 1 else None  # (1, 1024) if available

        # Apply softmax
        probs = self._softmax(logits[0])

        # Compute uncertainty (entropy-based)
        uncertainty = self._compute_uncertainty(probs)

        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        predictions = [
            Prediction(
                label=taxonomy.labels[idx],
                confidence=float(probs[idx]),
                taxonomy=taxonomy.name,
                model_name=model_id,
                rank=rank + 1,
                uncertainty=uncertainty if rank == 0 else None,
            )
            for rank, idx in enumerate(top_indices)
        ]

        classification_time = (time.time() - start_time) * 1000
        snapshot.processing.classification_time_ms = classification_time

        embedding_flat = embedding.flatten() if embedding is not None else None

        return predictions, embedding_flat

    def classify_batch(
        self,
        snapshots: List[Snapshot],
        model_id: str = "pointnet2_ssg_mn40",
        batch_size: int = 32,
        top_k: int = 5,
    ) -> List[Tuple[List[Prediction], Optional[np.ndarray]]]:
        """
        Classify multiple snapshots with batching.

        Args:
            snapshots: List of snapshots to classify
            model_id: Model to use
            batch_size: Batch size for inference
            top_k: Number of top predictions per snapshot

        Returns:
            List of (predictions, embedding) tuples
        """
        results = []

        for i in range(0, len(snapshots), batch_size):
            batch = snapshots[i:i + batch_size]
            batch_results = self._classify_batch_internal(batch, model_id, top_k)
            results.extend(batch_results)

        return results

    def _classify_batch_internal(
        self,
        snapshots: List[Snapshot],
        model_id: str,
        top_k: int,
    ) -> List[Tuple[List[Prediction], Optional[np.ndarray]]]:
        """Internal batch classification."""
        if not snapshots:
            return []

        session, model_info = self._load_model(model_id)
        taxonomy = self.registry.get_taxonomy(model_info.taxonomy)

        # Prepare batched input
        points_list = []
        for snapshot in snapshots:
            if snapshot.geometry is None:
                # Use zeros for missing geometry
                points = np.zeros((model_info.input_points, 3), dtype=np.float32)
            else:
                points = snapshot.geometry.points.astype(np.float32)
                if len(points) != model_info.input_points:
                    points = self._adjust_point_count(points, model_info.input_points)
            points_list.append(points)

        points_batch = np.stack(points_list)  # (B, N, 3)

        # Run inference
        try:
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: points_batch})
        except Exception as e:
            raise InferenceError(model_id, f"Batch inference failed: {e}")

        logits = outputs[0]  # (B, num_classes)
        embeddings = outputs[1] if len(outputs) > 1 else [None] * len(snapshots)

        # Process each result
        results = []
        for j, (snapshot, logit_row) in enumerate(zip(snapshots, logits)):
            probs = self._softmax(logit_row)
            uncertainty = self._compute_uncertainty(probs)

            top_indices = np.argsort(probs)[::-1][:top_k]
            predictions = [
                Prediction(
                    label=taxonomy.labels[idx],
                    confidence=float(probs[idx]),
                    taxonomy=taxonomy.name,
                    model_name=model_id,
                    rank=rank + 1,
                    uncertainty=uncertainty if rank == 0 else None,
                )
                for rank, idx in enumerate(top_indices)
            ]

            emb = embeddings[j].flatten() if embeddings[j] is not None else None
            results.append((predictions, emb))

        return results

    def _adjust_point_count(self, points: np.ndarray, target: int) -> np.ndarray:
        """Adjust point cloud to target point count."""
        n = len(points)

        if n == target:
            return points

        if n > target:
            # Subsample randomly
            indices = np.random.choice(n, target, replace=False)
            return points[indices]

        # Upsample by repeating points
        indices = np.random.choice(n, target, replace=True)
        return points[indices]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def _compute_uncertainty(
        self,
        probs: np.ndarray,
        method: UncertaintyMethod = UncertaintyMethod.ENTROPY,
        session=None,
        points_batch: np.ndarray = None,
        input_name: str = None,
    ) -> float:
        """
        Compute classification uncertainty.

        Args:
            probs: Softmax probabilities from a single forward pass.
            method: Uncertainty estimation method.
            session: ONNX session (needed for MC Dropout/Ensemble).
            points_batch: Input batch (needed for MC Dropout/Ensemble).
            input_name: Input tensor name (needed for MC Dropout/Ensemble).

        Returns:
            Uncertainty value in [0, 1].
        """
        if method == UncertaintyMethod.ENTROPY:
            return self._compute_uncertainty_entropy(probs)
        elif method == UncertaintyMethod.MC_DROPOUT:
            if session is not None and points_batch is not None and input_name is not None:
                return self._compute_uncertainty_mc_dropout(
                    session, points_batch, input_name
                )
            return self._compute_uncertainty_entropy(probs)
        elif method == UncertaintyMethod.ENSEMBLE:
            if session is not None and points_batch is not None and input_name is not None:
                return self._compute_uncertainty_ensemble(
                    session, points_batch, input_name
                )
            return self._compute_uncertainty_entropy(probs)
        else:
            return self._compute_uncertainty_entropy(probs)

    def _compute_uncertainty_entropy(self, probs: np.ndarray) -> float:
        """
        Compute uncertainty using normalized entropy.

        Returns value in [0, 1]:
        - 0: Completely certain (one class has prob=1)
        - 1: Maximum uncertainty (uniform distribution)
        """
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        if max_entropy > 0:
            return float(entropy / max_entropy)
        return 0.0

    def _compute_uncertainty_mc_dropout(
        self,
        session,
        points_batch: np.ndarray,
        input_name: str,
        n_samples: int = 10,
    ) -> float:
        """
        Compute uncertainty via MC Dropout.

        Runs multiple forward passes and measures prediction variance.
        Note: Requires model to have dropout layers active during inference.
        Falls back to entropy-based uncertainty if MC Dropout isn't supported.
        """
        all_probs = []

        for _ in range(n_samples):
            try:
                outputs = session.run(None, {input_name: points_batch})
                logits = outputs[0][0]
                probs = self._softmax(logits)
                all_probs.append(probs)
            except Exception:
                break

        if len(all_probs) < 2:
            return self._compute_uncertainty_entropy(all_probs[0] if all_probs else np.ones(1))

        all_probs = np.array(all_probs)
        mean_probs = all_probs.mean(axis=0)
        variance = all_probs.var(axis=0).mean()

        # Normalize variance to [0, 1] range
        max_var = 0.25  # Maximum variance for binary classification
        return float(min(variance / max_var, 1.0))

    def _compute_uncertainty_ensemble(
        self,
        session,
        points_batch: np.ndarray,
        input_name: str,
        n_augmentations: int = 5,
    ) -> float:
        """
        Compute uncertainty via ensemble of augmented inputs.

        Applies random augmentations and measures prediction consistency.
        """
        all_probs = []

        # Original prediction
        try:
            outputs = session.run(None, {input_name: points_batch})
            logits = outputs[0][0]
            all_probs.append(self._softmax(logits))
        except Exception:
            return 0.5

        # Augmented predictions
        for _ in range(n_augmentations):
            try:
                # Apply small random perturbation
                noise = np.random.normal(0, 0.01, points_batch.shape).astype(np.float32)
                augmented = points_batch + noise

                outputs = session.run(None, {input_name: augmented})
                logits = outputs[0][0]
                all_probs.append(self._softmax(logits))
            except Exception:
                continue

        if len(all_probs) < 2:
            return self._compute_uncertainty_entropy(all_probs[0])

        all_probs = np.array(all_probs)
        variance = all_probs.var(axis=0).mean()
        max_var = 0.25
        return float(min(variance / max_var, 1.0))

    def get_model_info(self, model_id: str) -> RegisteredModel:
        """Get information about a registered model."""
        return self.registry.get(model_id)

    def list_models(self) -> List[RegisteredModel]:
        """List all available models."""
        return self.registry.list_models()

    def unload_model(self, model_id: str) -> None:
        """Unload a model from memory."""
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]

    def unload_all_models(self) -> None:
        """Unload all models from memory."""
        self._loaded_models.clear()
