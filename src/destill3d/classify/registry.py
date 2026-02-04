"""
Model registry for Destill3D.

Manages classification models and their taxonomies.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

import httpx

from destill3d.core.exceptions import ModelNotFoundError, ModelDownloadError


class ModelFormat(Enum):
    """Supported model formats."""

    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    PYTORCH = "pytorch"


@dataclass
class TaxonomyConfig:
    """Configuration for a classification taxonomy."""

    name: str
    num_classes: int
    labels: List[str]

    def label_to_index(self, label: str) -> int:
        """Get index for a label."""
        return self.labels.index(label)

    def index_to_label(self, index: int) -> str:
        """Get label for an index."""
        return self.labels[index]


# ─────────────────────────────────────────────────────────────────────────────
# Standard Taxonomies
# ─────────────────────────────────────────────────────────────────────────────

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
        "toilet", "tv_stand", "vase", "wardrobe", "xbox",
    ],
)

MODELNET10 = TaxonomyConfig(
    name="modelnet10",
    num_classes=10,
    labels=[
        "bathtub", "bed", "chair", "desk", "dresser",
        "monitor", "night_stand", "sofa", "table", "toilet",
    ],
)

SCANOBJECTNN = TaxonomyConfig(
    name="scanobjectnn",
    num_classes=15,
    labels=[
        "bag", "bin", "box", "cabinet", "chair",
        "desk", "display", "door", "pillow", "shelf",
        "sink", "sofa", "table", "bed", "toilet",
    ],
)

# Taxonomy registry
TAXONOMIES: Dict[str, TaxonomyConfig] = {
    "modelnet40": MODELNET40,
    "modelnet10": MODELNET10,
    "scanobjectnn": SCANOBJECTNN,
}


@dataclass
class RegisteredModel:
    """A classification model registered for use."""

    model_id: str
    name: str
    taxonomy: str
    architecture: str

    # Model files
    weights_url: str
    weights_hash: str  # SHA256 for verification
    weights_path: Optional[Path] = None  # Local path after download

    # Input requirements
    input_points: int = 2048
    input_features: List[str] = field(default_factory=lambda: ["xyz"])
    requires_normals: bool = False

    # Performance metadata
    modelnet40_accuracy: Optional[float] = None
    inference_time_ms: Optional[float] = None

    # Runtime
    format: ModelFormat = ModelFormat.ONNX
    device: str = "auto"


# ─────────────────────────────────────────────────────────────────────────────
# Default Model Registry
# ─────────────────────────────────────────────────────────────────────────────

# Note: In production, these would point to actual hosted model files
# For MVP, we provide stub entries that users can populate

MODEL_REGISTRY: Dict[str, RegisteredModel] = {
    "pointnet2_ssg_mn40": RegisteredModel(
        model_id="pointnet2_ssg_mn40",
        name="PointNet++ SSG (ModelNet40)",
        taxonomy="modelnet40",
        architecture="pointnet2_ssg",
        weights_url="https://github.com/layerdynamics/destill3d-models/releases/download/v0.1.0/pointnet2_ssg_mn40.onnx",
        weights_hash="",  # Will be computed when model is available
        modelnet40_accuracy=0.927,
        format=ModelFormat.ONNX,
        input_points=2048,
    ),
    "pointnet2_msg_mn40": RegisteredModel(
        model_id="pointnet2_msg_mn40",
        name="PointNet++ MSG (ModelNet40)",
        taxonomy="modelnet40",
        architecture="pointnet2_msg",
        weights_url="https://github.com/layerdynamics/destill3d-models/releases/download/v0.1.0/pointnet2_msg_mn40.onnx",
        weights_hash="",
        modelnet40_accuracy=0.933,
        format=ModelFormat.ONNX,
        input_points=2048,
    ),
    "dgcnn_mn40": RegisteredModel(
        model_id="dgcnn_mn40",
        name="DGCNN (ModelNet40)",
        taxonomy="modelnet40",
        architecture="dgcnn",
        weights_url="https://github.com/layerdynamics/destill3d-models/releases/download/v0.1.0/dgcnn_mn40.onnx",
        weights_hash="",
        modelnet40_accuracy=0.926,
        format=ModelFormat.ONNX,
        input_points=2048,
    ),
}


class ModelRegistry:
    """
    Manages classification model registration and downloads.

    Models are downloaded on-demand and cached locally.
    """

    def __init__(self, models_dir: Path):
        """
        Initialize the model registry.

        Args:
            models_dir: Directory for storing model weights
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._registry = MODEL_REGISTRY.copy()
        self._custom_models: Dict[str, RegisteredModel] = {}

    def get(self, model_id: str) -> RegisteredModel:
        """
        Get a registered model by ID.

        Args:
            model_id: Model identifier

        Returns:
            RegisteredModel

        Raises:
            ModelNotFoundError: If model is not registered
        """
        if model_id in self._custom_models:
            return self._custom_models[model_id]
        if model_id in self._registry:
            return self._registry[model_id]
        raise ModelNotFoundError(model_id)

    def list_models(self) -> List[RegisteredModel]:
        """List all registered models."""
        models = list(self._registry.values())
        models.extend(self._custom_models.values())
        return models

    def list_model_ids(self) -> List[str]:
        """List all registered model IDs."""
        return list(self._registry.keys()) + list(self._custom_models.keys())

    def is_downloaded(self, model_id: str) -> bool:
        """Check if model weights are downloaded."""
        model = self.get(model_id)
        expected_path = self._get_model_path(model)
        return expected_path.exists()

    def get_model_path(self, model_id: str) -> Path:
        """
        Get the local path for model weights.

        Downloads if not already present.

        Args:
            model_id: Model identifier

        Returns:
            Path to model weights file
        """
        model = self.get(model_id)
        local_path = self._get_model_path(model)

        if not local_path.exists():
            self.download_model(model_id)

        return local_path

    def _get_model_path(self, model: RegisteredModel) -> Path:
        """Get expected local path for model weights."""
        ext = {
            ModelFormat.ONNX: ".onnx",
            ModelFormat.TORCHSCRIPT: ".pt",
            ModelFormat.PYTORCH: ".pth",
        }.get(model.format, ".bin")

        return self.models_dir / f"{model.model_id}{ext}"

    def download_model(self, model_id: str, force: bool = False) -> Path:
        """
        Download model weights from remote URL.

        Args:
            model_id: Model identifier
            force: Re-download even if exists

        Returns:
            Path to downloaded model

        Raises:
            ModelDownloadError: If download fails
        """
        model = self.get(model_id)
        target_path = self._get_model_path(model)

        if target_path.exists() and not force:
            return target_path

        if not model.weights_url:
            raise ModelDownloadError(
                model_id,
                "N/A",
                "Model URL not configured. Please provide model weights manually.",
            )

        try:
            with httpx.stream("GET", model.weights_url, follow_redirects=True, timeout=300) as response:
                response.raise_for_status()

                # Download to temp file first
                temp_path = target_path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)

                # Verify hash if provided
                if model.weights_hash:
                    actual_hash = self._compute_file_hash(temp_path)
                    if not actual_hash.startswith(model.weights_hash.replace("sha256:", "")):
                        temp_path.unlink()
                        raise ModelDownloadError(
                            model_id,
                            model.weights_url,
                            f"Hash mismatch: expected {model.weights_hash}, got sha256:{actual_hash}",
                        )

                # Move to final location
                temp_path.rename(target_path)

        except httpx.HTTPError as e:
            raise ModelDownloadError(model_id, model.weights_url, str(e))
        except Exception as e:
            raise ModelDownloadError(model_id, model.weights_url, str(e))

        return target_path

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 hash of a file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def register_custom_model(
        self,
        model_id: str,
        name: str,
        taxonomy: str,
        architecture: str,
        weights_path: Path,
        format: ModelFormat = ModelFormat.ONNX,
        input_points: int = 2048,
        requires_normals: bool = False,
    ) -> RegisteredModel:
        """
        Register a custom local model.

        Args:
            model_id: Unique identifier for the model
            name: Human-readable name
            taxonomy: Target taxonomy name
            architecture: Model architecture
            weights_path: Path to model weights
            format: Model format
            input_points: Expected number of input points
            requires_normals: Whether model needs normals

        Returns:
            The registered model
        """
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        model = RegisteredModel(
            model_id=model_id,
            name=name,
            taxonomy=taxonomy,
            architecture=architecture,
            weights_url="",
            weights_hash="",
            weights_path=weights_path,
            format=format,
            input_points=input_points,
            requires_normals=requires_normals,
        )

        self._custom_models[model_id] = model
        return model

    def get_taxonomy(self, taxonomy_name: str) -> TaxonomyConfig:
        """
        Get a taxonomy configuration.

        Args:
            taxonomy_name: Name of the taxonomy

        Returns:
            TaxonomyConfig

        Raises:
            KeyError: If taxonomy is unknown
        """
        if taxonomy_name not in TAXONOMIES:
            raise KeyError(f"Unknown taxonomy: {taxonomy_name}")
        return TAXONOMIES[taxonomy_name]

    def list_taxonomies(self) -> List[str]:
        """List all available taxonomy names."""
        return list(TAXONOMIES.keys())
