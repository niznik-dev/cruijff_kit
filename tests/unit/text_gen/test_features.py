"""Tests for text_gen/lib/features.py — feature selection and validation."""

import pytest

from text_gen.lib.features import select_features, validate_features


class TestValidateFeatures:
    def test_valid_features(self, schema):
        warnings = validate_features(["AGEP", "ST"], schema)
        assert warnings == []

    def test_missing_feature_raises(self, schema):
        with pytest.raises(ValueError, match="Features not found in schema"):
            validate_features(["AGEP", "NONEXISTENT"], schema)

    def test_target_in_features_warns(self, schema):
        warnings = validate_features(["AGEP", "PINCP"], schema, target_column="PINCP")
        assert len(warnings) == 1
        assert "Target column" in warnings[0]

    def test_target_not_in_features_no_warning(self, schema):
        warnings = validate_features(["AGEP", "ST"], schema, target_column="PINCP")
        assert warnings == []


class TestSelectFeatures:
    def test_returns_ordered_pairs(self, schema):
        row = {"AGEP": "51", "ST": "New York", "OCCP": "Teacher", "PINCP": "75000"}
        result = select_features(row, ["OCCP", "AGEP"], schema)
        assert len(result) == 2
        assert result[0][0].key == "OCCP"
        assert result[0][1] == "Teacher"
        assert result[1][0].key == "AGEP"
        assert result[1][1] == "51"

    def test_missing_value_skipped(self, schema):
        row = {"AGEP": "51"}
        result = select_features(row, ["AGEP", "ST"], schema)
        assert len(result) == 1
        assert result[0][0].key == "AGEP"

    def test_nan_value_skipped(self, schema):
        row = {"AGEP": "51", "ST": float("nan")}
        result = select_features(row, ["AGEP", "ST"], schema)
        assert len(result) == 1
        assert result[0][0].key == "AGEP"

    def test_missing_value_included(self, schema):
        """When missing_value_handling='include', missing values use placeholder text."""
        row = {"AGEP": "51"}
        result = select_features(
            row,
            ["AGEP", "ST"],
            schema,
            missing_value_handling="include",
        )
        assert len(result) == 2
        assert result[1][0].key == "ST"
        assert result[1][1] == "missing"

    def test_nan_value_included(self, schema):
        """When missing_value_handling='include', NaN values use placeholder text."""
        row = {"AGEP": "51", "ST": float("nan")}
        result = select_features(
            row,
            ["AGEP", "ST"],
            schema,
            missing_value_handling="include",
            missing_value_text="not reported",
        )
        assert len(result) == 2
        assert result[1][0].key == "ST"
        assert result[1][1] == "not reported"

    def test_value_map_decodes(self):
        """value_map translates raw coded values to labels."""
        from text_gen.lib.schema import ColumnSchema, Schema

        col = ColumnSchema(
            key="SEX",
            display_name="sex",
            type="categorical",
            value_map={"1": "Male", "2": "Female"},
        )
        schema = Schema(name="t", description="", columns={"SEX": col})
        row = {"SEX": "1"}
        result = select_features(row, ["SEX"], schema)
        assert result[0][1] == "Male"

    def test_value_map_passthrough_unmapped(self):
        """Unmapped values pass through unchanged."""
        from text_gen.lib.schema import ColumnSchema, Schema

        col = ColumnSchema(
            key="SEX",
            display_name="sex",
            type="categorical",
            value_map={"1": "Male", "2": "Female"},
        )
        schema = Schema(name="t", description="", columns={"SEX": col})
        row = {"SEX": "9"}
        result = select_features(row, ["SEX"], schema)
        assert result[0][1] == "9"
