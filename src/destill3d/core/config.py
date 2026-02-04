"""
Configuration system for Destill3D using Pydantic Settings.

Supports loading from environment variables and YAML files.
"""

from pathlib import Path
from typing import Literal, Optional
import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


def get_default_data_dir() -> Path:
    """Get the default data directory for Destill3D."""
    return Path.home() / ".destill3d"


def get_default_temp_dir() -> Path:
    """Get the default temp directory."""
    import tempfile
    return Path(tempfile.gettempdir()) / "destill3d"


class TessellationConfig(BaseSettings):
    """Configuration for CAD B-rep tessellation."""

    model_config = SettingsConfigDict(
        env_prefix="DESTILL3D_TESSELLATION_",
        extra="ignore",
    )

    linear_deflection: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.1,
        description="Linear deflection for tessellation (smaller = finer)",
    )
    angular_deflection: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Angular deflection in radians",
    )
    relative: bool = Field(
        default=False,
        description="Use relative deflection (scaled to model size)",
    )

    @classmethod
    def preview(cls) -> "TessellationConfig":
        """Quick preview quality settings."""
        return cls(linear_deflection=0.01, angular_deflection=0.8)

    @classmethod
    def standard(cls) -> "TessellationConfig":
        """Standard quality settings."""
        return cls(linear_deflection=0.001, angular_deflection=0.5)

    @classmethod
    def high_quality(cls) -> "TessellationConfig":
        """High quality settings for detailed models."""
        return cls(linear_deflection=0.0001, angular_deflection=0.2)


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DESTILL3D_DATABASE_",
        extra="ignore",
    )

    type: Literal["sqlite", "postgresql"] = Field(
        default="sqlite",
        description="Database type",
    )
    path: Path = Field(
        default_factory=lambda: get_default_data_dir() / "destill3d.db",
        description="Path to SQLite database file",
    )
    # PostgreSQL settings (for future use)
    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    database: str = Field(default="destill3d", description="PostgreSQL database name")
    user: str = Field(default="", description="PostgreSQL user")
    password: str = Field(default="", description="PostgreSQL password")

    @property
    def connection_string(self) -> str:
        """Get the database connection string."""
        if self.type == "sqlite":
            return f"sqlite:///{self.path}"
        elif self.type == "postgresql":
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        raise ValueError(f"Unknown database type: {self.type}")


class ExtractionConfig(BaseSettings):
    """Feature extraction configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DESTILL3D_EXTRACTION_",
        extra="ignore",
    )

    target_points: int = Field(
        default=2048,
        ge=256,
        le=16384,
        description="Target number of points in sampled point cloud",
    )
    sampling_strategy: Literal["uniform", "fps", "poisson", "hybrid"] = Field(
        default="hybrid",
        description="Point sampling strategy",
    )
    oversample_ratio: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Oversample ratio for FPS/hybrid sampling",
    )
    compute_views: bool = Field(
        default=False,
        description="Compute multi-view renderings (future feature)",
    )
    normal_estimation_k: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Number of neighbors for normal estimation",
    )
    curvature_estimation_k: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Number of neighbors for curvature estimation",
    )
    tessellation: TessellationConfig = Field(
        default_factory=TessellationConfig,
        description="CAD tessellation settings",
    )


class ClassificationConfig(BaseSettings):
    """Classification configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DESTILL3D_CLASSIFICATION_",
        extra="ignore",
    )

    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for inference",
    )
    device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        description="Device for inference (auto detects GPU)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=40,
        description="Number of top predictions to return",
    )
    default_model: str = Field(
        default="pointnet2_ssg_mn40",
        description="Default classification model",
    )
    compute_uncertainty: bool = Field(
        default=True,
        description="Compute uncertainty estimates",
    )
    uncertainty_method: Literal["entropy", "mc_dropout", "ensemble"] = Field(
        default="entropy",
        description="Method for uncertainty estimation",
    )


class AcquisitionConfig(BaseSettings):
    """Acquisition configuration (for future platform adapters)."""

    model_config = SettingsConfigDict(
        env_prefix="DESTILL3D_ACQUISITION_",
        extra="ignore",
    )

    concurrency: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of concurrent downloads",
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retry attempts for failed downloads",
    )
    retry_delay: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Delay between retry attempts in seconds",
    )
    # Platform API keys (loaded from environment)
    thingiverse_api_key: str = Field(
        default="",
        description="Thingiverse API key",
    )
    sketchfab_api_key: str = Field(
        default="",
        description="Sketchfab API key",
    )


class Destill3DConfig(BaseSettings):
    """Main configuration for Destill3D."""

    model_config = SettingsConfigDict(
        env_prefix="DESTILL3D_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)

    # Global settings
    models_dir: Path = Field(
        default_factory=lambda: get_default_data_dir() / "models",
        description="Directory for model weights",
    )
    temp_dir: Path = Field(
        default_factory=get_default_temp_dir,
        description="Directory for temporary files",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        if self.database.type == "sqlite":
            self.database.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: Path) -> "Destill3DConfig":
        """
        Load configuration from a YAML file.

        Environment variables take precedence over YAML values.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            yaml_data = yaml.safe_load(f) or {}

        # Expand environment variables in string values
        yaml_data = cls._expand_env_vars(yaml_data)

        return cls(**yaml_data)

    @classmethod
    def _expand_env_vars(cls, data: dict) -> dict:
        """Recursively expand environment variables in string values."""
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = cls._expand_env_vars(value)
            elif isinstance(value, str):
                result[key] = os.path.expandvars(value)
            else:
                result[key] = value
        return result

    def save_yaml(self, path: Path) -> None:
        """Save configuration to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, handling Path objects
        data = self._to_serializable_dict()

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _to_serializable_dict(self) -> dict:
        """Convert config to a YAML-serializable dictionary."""
        data = self.model_dump()
        return self._convert_paths(data)

    @classmethod
    def _convert_paths(cls, data: dict) -> dict:
        """Recursively convert Path objects to strings."""
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = cls._convert_paths(value)
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default configuration file path."""
        return get_default_data_dir() / "config.yaml"

    @classmethod
    def load(cls) -> "Destill3DConfig":
        """
        Load configuration from default locations.

        Priority:
        1. Environment variables
        2. ~/.destill3d/config.yaml (if exists)
        3. Default values
        """
        config_path = cls.get_default_config_path()
        if config_path.exists():
            return cls.from_yaml(config_path)
        return cls()
