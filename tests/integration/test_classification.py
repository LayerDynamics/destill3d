"""
Integration tests for classification pipeline.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from destill3d.classify.registry import TAXONOMIES, MODEL_REGISTRY
from destill3d.core.snapshot import Snapshot


class TestTaxonomyRegistry:
    """Test taxonomy registry integration."""

    def test_modelnet40_exists(self):
        assert "modelnet40" in TAXONOMIES

    def test_modelnet40_has_40_classes(self):
        assert TAXONOMIES["modelnet40"].num_classes == 40

    def test_shapenet55_exists(self):
        assert "shapenet55" in TAXONOMIES

    def test_shapenet55_has_55_classes(self):
        assert TAXONOMIES["shapenet55"].num_classes == 55

    def test_defcad_exists(self):
        assert "defcad" in TAXONOMIES

    def test_defcad_is_hierarchical(self):
        assert TAXONOMIES["defcad"].hierarchical is True

    def test_defcad_has_parent_map(self):
        assert TAXONOMIES["defcad"].parent_map is not None


class TestModelRegistry:
    """Test model registry integration."""

    def test_pointnet2_ssg_registered(self):
        assert "pointnet2_ssg_mn40" in MODEL_REGISTRY

    def test_openshape_registered(self):
        assert "openshape_pointbert" in MODEL_REGISTRY

    def test_model_has_required_fields(self):
        for model_id, info in MODEL_REGISTRY.items():
            assert info.name, f"{model_id} missing name"
            assert info.architecture, f"{model_id} missing architecture"
            assert info.taxonomy, f"{model_id} missing taxonomy"
            assert info.input_points > 0, f"{model_id} missing input_points"

    def test_model_taxonomy_exists(self):
        for model_id, info in MODEL_REGISTRY.items():
            assert info.taxonomy in TAXONOMIES, (
                f"{model_id} references unknown taxonomy: {info.taxonomy}"
            )


@pytest.mark.integration
class TestClassifierIntegration:
    """Test classifier initialization and registry integration."""

    def test_classifier_init(self):
        """Test Classifier can be instantiated."""
        from destill3d.classify.inference import Classifier
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            classifier = Classifier(models_dir=tmpdir, device="cpu")
            assert classifier.device == "cpu"

    def test_uncertainty_methods_exist(self):
        """Test uncertainty methods are available."""
        from destill3d.classify.inference import UncertaintyMethod

        assert hasattr(UncertaintyMethod, "ENTROPY")
        assert hasattr(UncertaintyMethod, "MC_DROPOUT")
        assert hasattr(UncertaintyMethod, "ENSEMBLE")
