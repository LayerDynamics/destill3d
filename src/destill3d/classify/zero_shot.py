"""
Zero-shot 3D classification using OpenShape + CLIP.

Allows classification of point clouds into arbitrary text-described
categories without task-specific training.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ZeroShotConfig:
    """Configuration for zero-shot classification."""

    encoder_model: str = "openshape_pointbert"
    text_encoder: str = "openai/clip-vit-large-patch14"
    device: str = "auto"
    cache_embeddings: bool = True


@dataclass
class ZeroShotResult:
    """Result of zero-shot classification."""

    classes: List[str]
    probabilities: List[float]
    embedding_3d: np.ndarray
    embedding_dim: int


class ZeroShotClassifier:
    """Zero-shot 3D classification using OpenShape + CLIP."""

    def __init__(self, config: ZeroShotConfig = None):
        self.config = config or ZeroShotConfig()
        self._point_encoder = None
        self._text_encoder = None
        self._tokenizer = None
        self._text_cache: dict = {}
        self._device = None

    def _init_models(self):
        """Lazy initialization of models."""
        if self._point_encoder is not None:
            return

        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch required for zero-shot classification. "
                "Install with: pip install destill3d[zero-shot]"
            )

        # Determine device
        if self.config.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.config.device

        # Load OpenShape point encoder
        self._point_encoder = self._load_point_encoder()

        # Load CLIP text encoder
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError:
            raise ImportError(
                "transformers required for zero-shot classification. "
                "Install with: pip install destill3d[zero-shot]"
            )

        self._tokenizer = CLIPTokenizer.from_pretrained(self.config.text_encoder)
        self._text_encoder = CLIPTextModel.from_pretrained(
            self.config.text_encoder
        ).to(self._device)

        logger.info(
            f"Initialized zero-shot classifier on {self._device} "
            f"with {self.config.encoder_model}"
        )

    def _load_point_encoder(self):
        """Load the OpenShape point cloud encoder."""
        from destill3d.classify.registry import MODEL_REGISTRY

        model_info = MODEL_REGISTRY.get(self.config.encoder_model)
        if not model_info:
            raise ValueError(f"Unknown encoder: {self.config.encoder_model}")

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime required for point cloud encoding. "
                "Install with: pip install onnxruntime"
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self._device == "cpu":
            providers = ["CPUExecutionProvider"]

        return ort.InferenceSession(
            str(model_info.weights_path),
            providers=providers,
        )

    def encode_points(self, points: np.ndarray) -> np.ndarray:
        """
        Encode point cloud to embedding space.

        Args:
            points: (N, 3) point cloud, normalized.

        Returns:
            (D,) embedding vector, L2-normalized.
        """
        self._init_models()

        # Prepare input
        points_input = points.astype(np.float32)
        if points_input.shape[0] != 2048:
            points_input = self._resample_points(points_input, 2048)

        # Add batch dimension
        points_batch = points_input.reshape(1, -1, 3)

        # Run inference
        outputs = self._point_encoder.run(
            None,
            {"points": points_batch},
        )

        embedding = outputs[0][0]  # Remove batch dim

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode text labels to embedding space.

        Args:
            texts: List of class names.

        Returns:
            (num_classes, D) embedding matrix, L2-normalized.
        """
        import torch

        self._init_models()

        embeddings = []
        for text in texts:
            # Check cache
            if self.config.cache_embeddings and text in self._text_cache:
                embeddings.append(self._text_cache[text])
                continue

            # Encode with CLIP
            inputs = self._tokenizer(
                f"a 3D model of a {text}",
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._text_encoder(**inputs)
                embedding = outputs.pooler_output[0].cpu().numpy()

            # L2 normalize
            embedding = embedding / np.linalg.norm(embedding)

            if self.config.cache_embeddings:
                self._text_cache[text] = embedding

            embeddings.append(embedding)

        return np.array(embeddings)

    def classify(
        self,
        points: np.ndarray,
        classes: List[str],
        temperature: float = 0.07,
    ) -> ZeroShotResult:
        """
        Classify point cloud into arbitrary classes.

        Args:
            points: (N, 3) normalized point cloud.
            classes: List of class names to classify into.
            temperature: Softmax temperature (lower = sharper).

        Returns:
            ZeroShotResult with probabilities for each class.
        """
        # Get embeddings
        point_embedding = self.encode_points(points)
        text_embeddings = self.encode_texts(classes)

        # Compute cosine similarities
        similarities = point_embedding @ text_embeddings.T

        # Apply temperature and softmax
        logits = similarities / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        return ZeroShotResult(
            classes=classes,
            probabilities=probabilities.tolist(),
            embedding_3d=point_embedding,
            embedding_dim=point_embedding.shape[0],
        )

    def _resample_points(
        self,
        points: np.ndarray,
        target_count: int,
    ) -> np.ndarray:
        """Resample points to target count."""
        current = points.shape[0]

        if current == target_count:
            return points
        elif current > target_count:
            indices = np.random.choice(current, target_count, replace=False)
            return points[indices]
        else:
            indices = np.random.choice(current, target_count - current, replace=True)
            return np.vstack([points, points[indices]])
