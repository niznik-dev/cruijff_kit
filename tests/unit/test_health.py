"""Tests for cruijff_kit.health — dep version sanity check and import helper."""

import warnings
from unittest.mock import patch

import pytest

from cruijff_kit import health


class TestParseExactPins:
    def test_picks_up_exact_pin(self):
        result = health._parse_exact_pins(["inspect-ai==0.3.209"])
        assert result == {"inspect-ai": "0.3.209"}

    def test_skips_range_pins(self):
        result = health._parse_exact_pins(["torch>=2.9.1", "transformers>=4.57.1,<5"])
        assert result == {}

    def test_skips_unpinned(self):
        result = health._parse_exact_pins(["datasets", "peft"])
        assert result == {}

    def test_normalizes_underscores_to_dashes(self):
        # PEP 503 normalizes; we follow.
        result = health._parse_exact_pins(["inspect_viz==0.3.5"])
        assert result == {"inspect-viz": "0.3.5"}

    def test_handles_empty_list(self):
        assert health._parse_exact_pins([]) == {}
        assert health._parse_exact_pins(None) == {}

    def test_mixed_specifiers_keeps_only_exact(self):
        result = health._parse_exact_pins(
            [
                "inspect-ai==0.3.209",
                "torch>=2.9.1",
                "inspect-viz==0.3.5",
                "transformers>=4.57.1,<5",
            ]
        )
        assert result == {"inspect-ai": "0.3.209", "inspect-viz": "0.3.5"}


class TestCheckVersions:
    def test_no_mismatch_when_versions_match(self):
        # Pick a pinned dep we know is installed; expect no mismatch when
        # we pass its installed version as the expected pin.
        from importlib.metadata import version

        installed_iv = version("inspect-viz")
        result = health.check_versions(pinned={"inspect-viz": installed_iv})
        assert result == []

    def test_mismatch_returned_when_version_differs(self):
        result = health.check_versions(pinned={"inspect-viz": "99.99.99"})
        assert len(result) == 1
        name, expected, installed = result[0]
        assert name == "inspect-viz"
        assert expected == "99.99.99"
        # installed is whatever is actually installed — just confirm it's not the fake
        assert installed != "99.99.99"

    def test_missing_package_silently_skipped(self):
        # check_versions only flags installed-but-different packages, not absent ones
        result = health.check_versions(pinned={"this-package-does-not-exist": "1.0.0"})
        assert result == []

    def test_empty_pinned_returns_empty(self):
        assert health.check_versions(pinned={}) == []


class TestFormatWarningMessage:
    def test_includes_each_mismatch(self):
        msg = health._format_warning_message([("inspect-viz", "0.3.5", "0.3.4")])
        assert "inspect-viz" in msg
        assert "expected 0.3.5" in msg
        assert "installed 0.3.4" in msg

    def test_includes_pip_fix_command(self):
        msg = health._format_warning_message(
            [("inspect-viz", "0.3.5", "0.3.4"), ("inspect-ai", "0.3.209", "0.3.200")]
        )
        assert "pip install" in msg
        assert "inspect-viz==0.3.5" in msg
        assert "inspect-ai==0.3.209" in msg


class TestWarnOnMismatch:
    def setup_method(self):
        # Reset the once-per-process flag so each test starts clean
        health._VERSION_WARNING_EMITTED = False

    def test_silent_when_no_mismatch(self):
        with patch.object(health, "check_versions", return_value=[]):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                health.warn_on_mismatch()
        version_warnings = [w for w in caught if "version mismatch" in str(w.message)]
        assert not version_warnings

    def test_warns_when_mismatch(self):
        with patch.object(
            health,
            "check_versions",
            return_value=[("inspect-viz", "0.3.5", "0.3.4")],
        ):
            with pytest.warns(UserWarning, match="version mismatch"):
                health.warn_on_mismatch()

    def test_only_warns_once_per_process(self):
        with patch.object(
            health,
            "check_versions",
            return_value=[("inspect-viz", "0.3.5", "0.3.4")],
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                health.warn_on_mismatch()
                health.warn_on_mismatch()
                health.warn_on_mismatch()
        version_warnings = [w for w in caught if "version mismatch" in str(w.message)]
        assert len(version_warnings) == 1

    def test_skipped_via_env_var(self, monkeypatch):
        monkeypatch.setenv("CK_SKIP_VERSION_CHECK", "1")
        with patch.object(
            health,
            "check_versions",
            return_value=[("inspect-viz", "0.3.5", "0.3.4")],
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                health.warn_on_mismatch()
        version_warnings = [w for w in caught if "version mismatch" in str(w.message)]
        assert not version_warnings
