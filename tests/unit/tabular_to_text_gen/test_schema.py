"""Tests for tabular_to_text_gen/lib/schema.py — schema loading, validation, skeleton generation."""

import pytest
import yaml

from cruijff_kit.tabular_to_text_gen.lib.schema import Schema


class TestSchemaFromYaml:
    def test_loads_all_columns(self, schema_yaml):
        schema = Schema.from_yaml(schema_yaml)
        assert schema.name == "Test"
        assert set(schema.columns.keys()) == {"AGEP", "ST", "OCCP", "PINCP"}

    def test_column_fields(self, schema_yaml):
        schema = Schema.from_yaml(schema_yaml)
        agep = schema.columns["AGEP"]
        assert agep.display_name == "age"
        assert agep.type == "numeric"
        assert agep.unit == "years old"
        assert "years of age" in agep.synonyms

    def test_missing_display_name(self, tmp_path):
        bad = {"columns": {"X": {"type": "numeric"}}}
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(bad, f)
        with pytest.raises(ValueError, match="missing required field 'display_name'"):
            Schema.from_yaml(str(path))

    def test_missing_type(self, tmp_path):
        bad = {"columns": {"X": {"display_name": "x"}}}
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(bad, f)
        with pytest.raises(ValueError, match="missing required field 'type'"):
            Schema.from_yaml(str(path))

    def test_invalid_type(self, tmp_path):
        bad = {"columns": {"X": {"display_name": "x", "type": "boolean"}}}
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(bad, f)
        with pytest.raises(ValueError, match="invalid type 'boolean'"):
            Schema.from_yaml(str(path))

    def test_optional_fields_default(self, tmp_path):
        """Columns with only required fields get sensible defaults."""
        minimal = {"columns": {"X": {"display_name": "x", "type": "numeric"}}}
        path = tmp_path / "min.yaml"
        with open(path, "w") as f:
            yaml.dump(minimal, f)
        schema = Schema.from_yaml(str(path))
        col = schema.columns["X"]
        assert col.unit is None
        assert col.synonyms == []
        assert col.shorthand_map == {}
        assert col.restatements == []
        assert col.value_map == {}

    def test_value_map_loaded(self, tmp_path):
        """value_map is loaded and keys are stringified."""
        data = {
            "columns": {
                "SEX": {
                    "display_name": "sex",
                    "type": "categorical",
                    "value_map": {1: "Male", 2: "Female"},
                }
            }
        }
        path = tmp_path / "vm.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        schema = Schema.from_yaml(str(path))
        col = schema.columns["SEX"]
        assert col.value_map == {"1": "Male", "2": "Female"}


class TestSchemaGetColumn:
    def test_found(self, schema):
        col = schema.get_column("AGEP")
        assert col.key == "AGEP"

    def test_not_found(self, schema):
        with pytest.raises(KeyError, match="not found in schema"):
            schema.get_column("NONEXISTENT")


class TestGenerateSkeleton:
    def test_infers_types(self, sample_csv):
        skeleton = Schema.generate_skeleton(sample_csv, ["AGEP", "OCCP"])
        assert skeleton["columns"]["AGEP"]["type"] == "numeric"
        assert skeleton["columns"]["OCCP"]["type"] == "categorical"

    def test_generates_display_names(self, sample_csv):
        skeleton = Schema.generate_skeleton(sample_csv, ["AGEP"])
        assert skeleton["columns"]["AGEP"]["display_name"] == "agep"

    def test_missing_column_raises(self, sample_csv):
        with pytest.raises(ValueError, match="Columns not found"):
            Schema.generate_skeleton(sample_csv, ["NONEXISTENT"])

    def test_dataset_name_from_filename(self, sample_csv):
        skeleton = Schema.generate_skeleton(sample_csv, ["AGEP"])
        assert skeleton["dataset"]["name"] == "test_data"
