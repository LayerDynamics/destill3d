"""
End-to-end CLI tests for Destill3D.
"""

import pytest
from typer.testing import CliRunner

from destill3d.cli.main import app

runner = CliRunner()


class TestCLIBasic:
    """Basic CLI tests."""

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "destill3d" in result.output.lower() or "3D" in result.output

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_info_command(self):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Version" in result.output

    def test_extract_help(self):
        result = runner.invoke(app, ["extract", "--help"])
        assert result.exit_code == 0

    def test_classify_help(self):
        result = runner.invoke(app, ["classify", "--help"])
        assert result.exit_code == 0

    def test_db_help(self):
        result = runner.invoke(app, ["db", "--help"])
        assert result.exit_code == 0

    def test_acquire_help(self):
        result = runner.invoke(app, ["acquire", "--help"])
        assert result.exit_code == 0

    def test_models_help(self):
        result = runner.invoke(app, ["models", "--help"])
        assert result.exit_code == 0

    def test_config_help(self):
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0

    def test_server_help(self):
        result = runner.invoke(app, ["server", "--help"])
        assert result.exit_code == 0


class TestCLIExtract:
    """CLI extract command tests."""

    def test_extract_file(self, sample_stl_file, temp_dir):
        output_path = temp_dir / "output.d3d"
        result = runner.invoke(
            app,
            [
                "extract",
                "file",
                str(sample_stl_file),
                "-o",
                str(output_path),
                "--points",
                "512",
            ],
        )
        assert result.exit_code == 0
        assert output_path.exists()

    def test_extract_missing_file(self):
        result = runner.invoke(app, ["extract", "file", "/nonexistent/file.stl"])
        assert result.exit_code != 0

    def test_quick_command(self, sample_stl_file, temp_dir):
        output_path = temp_dir / "quick_output.d3d"
        result = runner.invoke(
            app,
            ["quick", str(sample_stl_file), "-o", str(output_path)],
        )
        assert result.exit_code == 0


class TestCLIClassify:
    """CLI classify command tests."""

    def test_taxonomies_command(self):
        result = runner.invoke(app, ["classify", "taxonomies"])
        assert result.exit_code == 0
        assert "modelnet40" in result.output

    def test_models_command(self):
        result = runner.invoke(app, ["classify", "models"])
        assert result.exit_code == 0


class TestCLIAcquire:
    """CLI acquire command tests."""

    def test_platforms_command(self):
        result = runner.invoke(app, ["acquire", "platforms"])
        assert result.exit_code == 0
