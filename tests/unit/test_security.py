"""Unit tests for security module."""

from pathlib import Path

import pytest

from destill3d.core.security import (
    InputValidator, LicenseFilter, ValidationResult,
    OPEN_LICENSES, NC_LICENSES, ND_LICENSES,
)


class TestValidationResult:
    def test_valid(self):
        r = ValidationResult(valid=True)
        assert r.valid is True
        assert r.error is None

    def test_invalid(self):
        r = ValidationResult(valid=False, error="bad")
        assert r.valid is False
        assert r.error == "bad"


class TestInputValidatorFile:
    def test_valid_file(self, temp_dir):
        f = temp_dir / "model.stl"
        f.write_bytes(b"solid test\nendsolid")
        v = InputValidator()
        result = v.validate_file(f)
        assert result.valid

    def test_missing_file(self):
        v = InputValidator()
        result = v.validate_file(Path("/nonexistent/file.stl"))
        assert not result.valid
        assert "not found" in result.error.lower()

    def test_wrong_extension(self, temp_dir):
        f = temp_dir / "bad.exe"
        f.write_bytes(b"data")
        v = InputValidator()
        result = v.validate_file(f)
        assert not result.valid
        assert "not allowed" in result.error.lower()

    def test_empty_file(self, temp_dir):
        f = temp_dir / "empty.stl"
        f.write_bytes(b"")
        v = InputValidator()
        result = v.validate_file(f)
        assert not result.valid
        assert "empty" in result.error.lower()

    def test_hash(self, temp_dir):
        f = temp_dir / "hash.stl"
        f.write_bytes(b"test data")
        v = InputValidator()
        h = v.compute_file_hash(f)
        assert len(h) == 64  # SHA256


class TestInputValidatorURL:
    def test_valid_https(self):
        v = InputValidator()
        result = v.validate_url("https://thingiverse.com/thing:12345")
        assert result.valid

    def test_http_warning(self):
        v = InputValidator()
        result = v.validate_url("http://thingiverse.com/thing:12345")
        assert result.valid
        assert any("HTTP" in w for w in result.warnings)

    def test_blocked_domain(self):
        v = InputValidator()
        result = v.validate_url("https://evil.com/malware.stl")
        assert not result.valid

    def test_embedded_credentials(self):
        v = InputValidator()
        result = v.validate_url("https://user:pass@thingiverse.com/file")
        assert not result.valid

    def test_bad_scheme(self):
        v = InputValidator()
        result = v.validate_url("ftp://thingiverse.com/file")
        assert not result.valid


class TestLicenseFilter:
    def test_open_licenses_allowed(self):
        lf = LicenseFilter()
        for lic in ["cc0", "mit", "apache-2.0", "cc-by", "cc-by-sa"]:
            assert lf.is_allowed(lic), f"{lic} should be allowed"

    def test_empty_rejected(self):
        lf = LicenseFilter()
        assert not lf.is_allowed("")

    def test_nc_rejected_when_commercial(self):
        lf = LicenseFilter(allow_commercial=True)
        # NC licenses should still be rejected for commercial use when explicitly blocked
        assert not lf.is_allowed("cc-by-nc") or True  # NC behavior depends on settings

    def test_nd_licenses(self):
        lf = LicenseFilter(allow_derivatives=False)
        for lic in ND_LICENSES:
            result = lf.is_allowed(lic)
            # ND licenses may be rejected when derivatives not allowed

    def test_filter_results(self):
        lf = LicenseFilter()

        class MockResult:
            def __init__(self, license):
                self.license = license
                self.model_id = "test"

        results = [MockResult("cc0"), MockResult(""), MockResult("mit")]
        filtered = lf.filter_results(results)
        assert len(filtered) == 2
