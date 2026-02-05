"""
End-to-end CLI workflow tests for Destill3D.

Tests complete extract -> classify -> database workflows.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from destill3d.cli.main import app

runner = CliRunner()


class TestCLIWorkflow:
    """End-to-end CLI workflow tests."""

    @pytest.fixture
    def workflow_dir(self):
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    def test_extract_save_load(self, sample_stl_file, workflow_dir):
        """Test extract -> save -> load round-trip."""
        snapshot_path = workflow_dir / "output.d3d"

        # Extract (skip store to avoid needing DB)
        result = runner.invoke(
            app,
            [
                "extract",
                "file",
                str(sample_stl_file),
                "-o",
                str(snapshot_path),
                "--no-store",
            ],
        )
        assert result.exit_code == 0, f"Extract failed: {result.output}"
        assert snapshot_path.exists()

        # Verify snapshot can be loaded
        from destill3d.core.snapshot import Snapshot

        snapshot = Snapshot.load(snapshot_path)
        assert snapshot.geometry is not None
        assert snapshot.features is not None

    def test_batch_extract(self, temp_dir, workflow_dir):
        """Test extracting multiple files."""
        import trimesh

        # Create multiple test files
        for i in range(3):
            mesh = trimesh.creation.box(extents=[1.0 + i * 0.1, 1.0, 1.0])
            path = temp_dir / f"mesh_{i}.stl"
            mesh.export(path)

        # Extract directory (subcommand is "dir", not "directory")
        result = runner.invoke(
            app,
            [
                "extract",
                "dir",
                str(temp_dir),
                "-o",
                str(workflow_dir),
                "--no-store",
            ],
        )
        assert result.exit_code == 0, f"Batch extract failed: {result.output}"
