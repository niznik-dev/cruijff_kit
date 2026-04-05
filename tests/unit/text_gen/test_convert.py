"""Tests for text_gen/convert.py — CLI entrypoint and integration tests."""

import json

import pandas as pd
import pytest

from text_gen.convert import load_condition_from_file, main, split_dataframe


# ---------------------------------------------------------------------------
# split_dataframe
# ---------------------------------------------------------------------------


class TestSplitDataframe:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({"x": range(100)})

    def test_two_way_train(self, df):
        result = split_dataframe(
            df, seed=42, split="train", split_ratio=0.8, validation_ratio=None
        )
        assert len(result) == 80

    def test_two_way_test(self, df):
        result = split_dataframe(
            df, seed=42, split="test", split_ratio=0.8, validation_ratio=None
        )
        assert len(result) == 20

    def test_two_way_no_overlap(self, df):
        train = split_dataframe(
            df, seed=42, split="train", split_ratio=0.8, validation_ratio=None
        )
        test = split_dataframe(
            df, seed=42, split="test", split_ratio=0.8, validation_ratio=None
        )
        train_x = set(train["x"])
        test_x = set(test["x"])
        assert train_x & test_x == set()
        assert train_x | test_x == set(range(100))

    def test_three_way_split(self, df):
        train = split_dataframe(
            df, seed=42, split="train", split_ratio=0.7, validation_ratio=0.15
        )
        val = split_dataframe(
            df, seed=42, split="validation", split_ratio=0.7, validation_ratio=0.15
        )
        test = split_dataframe(
            df, seed=42, split="test", split_ratio=0.7, validation_ratio=0.15
        )
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        all_x = set(train["x"]) | set(val["x"]) | set(test["x"])
        assert all_x == set(range(100))

    def test_validation_without_ratio_raises(self, df):
        with pytest.raises(ValueError, match="validation-ratio not provided"):
            split_dataframe(
                df, seed=42, split="validation", split_ratio=0.8, validation_ratio=None
            )

    def test_deterministic(self, df):
        r1 = split_dataframe(
            df, seed=42, split="train", split_ratio=0.8, validation_ratio=None
        )
        r2 = split_dataframe(
            df, seed=42, split="train", split_ratio=0.8, validation_ratio=None
        )
        assert list(r1["x"]) == list(r2["x"])

    def test_different_seeds_different_splits(self, df):
        r1 = split_dataframe(
            df, seed=42, split="train", split_ratio=0.8, validation_ratio=None
        )
        r2 = split_dataframe(
            df, seed=99, split="train", split_ratio=0.8, validation_ratio=None
        )
        assert list(r1["x"]) != list(r2["x"])


# ---------------------------------------------------------------------------
# load_condition_from_file
# ---------------------------------------------------------------------------


class TestLoadConditionFromFile:
    def test_loads_condition(self, conditions_yaml):
        cond = load_condition_from_file(conditions_yaml, "dict_full")
        assert cond["features"] == ["AGEP", "ST", "OCCP"]
        assert cond["template"] == "dictionary"

    def test_missing_condition_raises(self, conditions_yaml):
        with pytest.raises(ValueError, match="not found"):
            load_condition_from_file(conditions_yaml, "nonexistent")


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def _run(self, tmp_path, sample_csv, schema_yaml, extra_args=None):
        """Helper to invoke main() and return parsed output + metadata."""
        output_path = str(tmp_path / "output.json")
        argv = [
            "--source",
            sample_csv,
            "--schema",
            schema_yaml,
            "--condition-name",
            "test_cond",
            "--features",
            "AGEP,ST,OCCP",
            "--template",
            "dictionary",
            "--target-column",
            "PINCP",
            "--target-threshold",
            "50000",
            "--context",
            "Context text.",
            "--question",
            "Is income > 50k?",
            "--split",
            "train",
            "--split-ratio",
            "0.8",
            "--seed",
            "42",
            "--output",
            output_path,
        ]
        if extra_args:
            argv.extend(extra_args)
        main(argv)

        with open(output_path) as f:
            data = json.load(f)

        meta_path = output_path.replace(".json", ".meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        return data, meta

    def test_dictionary_train(self, tmp_path, sample_csv, schema_yaml):
        data, meta = self._run(tmp_path, sample_csv, schema_yaml)
        assert "train" in data
        assert len(data["train"]) == 4  # 80% of 5 rows
        assert meta["condition_name"] == "test_cond"
        assert meta["row_count"] == 4

    def test_entry_structure(self, tmp_path, sample_csv, schema_yaml):
        data, _ = self._run(tmp_path, sample_csv, schema_yaml)
        entry = data["train"][0]
        assert "input" in entry
        assert "output" in entry
        assert entry["output"] in ("0", "1")
        assert "Context text." in entry["input"]
        assert "Is income > 50k?" in entry["input"]

    def test_test_split(self, tmp_path, sample_csv, schema_yaml):
        output_path = str(tmp_path / "test_output.json")
        main(
            [
                "--source",
                sample_csv,
                "--schema",
                schema_yaml,
                "--condition-name",
                "test_cond",
                "--features",
                "AGEP,ST,OCCP",
                "--template",
                "dictionary",
                "--target-column",
                "PINCP",
                "--target-threshold",
                "50000",
                "--split",
                "test",
                "--seed",
                "42",
                "--output",
                output_path,
            ]
        )
        with open(output_path) as f:
            data = json.load(f)
        assert "test" in data
        assert len(data["test"]) == 1  # 20% of 5 rows

    def test_with_perturbations(self, tmp_path, sample_csv, schema_yaml):
        data, meta = self._run(
            tmp_path, sample_csv, schema_yaml, extra_args=["--perturbations", "synonym"]
        )
        assert meta["perturbations"] == ["synonym"]
        # Entries should still have valid structure
        for entry in data["train"]:
            assert "input" in entry
            assert "output" in entry

    def test_system_prompt_placement(self, tmp_path, sample_csv, schema_yaml):
        """When context_placement is system_prompt, context is not in entries.

        The system prompt is an experiment-level setting, not per-entry data.
        """
        output_path = str(tmp_path / "sys_output.json")
        main(
            [
                "--source",
                sample_csv,
                "--schema",
                schema_yaml,
                "--condition-name",
                "sys_cond",
                "--features",
                "AGEP,ST",
                "--template",
                "dictionary",
                "--target-column",
                "PINCP",
                "--target-threshold",
                "50000",
                "--context",
                "You are a helpful assistant.",
                "--context-placement",
                "system_prompt",
                "--question",
                "Is income > 50k?",
                "--split",
                "train",
                "--seed",
                "42",
                "--output",
                output_path,
            ]
        )
        with open(output_path) as f:
            data = json.load(f)
        entry = data["train"][0]
        assert "system_prompt" not in entry
        assert "You are a helpful assistant." not in entry["input"]

    def test_conditions_file(self, tmp_path, sample_csv, schema_yaml, conditions_yaml):
        output_path = str(tmp_path / "cond_output.json")
        main(
            [
                "--source",
                sample_csv,
                "--schema",
                schema_yaml,
                "--condition-name",
                "dict_full",
                "--conditions-file",
                conditions_yaml,
                "--target-column",
                "PINCP",
                "--target-threshold",
                "50000",
                "--split",
                "train",
                "--seed",
                "42",
                "--output",
                output_path,
            ]
        )
        with open(output_path) as f:
            data = json.load(f)
        assert "train" in data
        assert len(data["train"]) == 4

    def test_narrative_template(self, tmp_path, sample_csv, schema_yaml):
        output_path = str(tmp_path / "narr_output.json")
        main(
            [
                "--source",
                sample_csv,
                "--schema",
                schema_yaml,
                "--condition-name",
                "narr_cond",
                "--features",
                "AGEP,ST,OCCP",
                "--template",
                "narrative",
                "--target-column",
                "PINCP",
                "--target-threshold",
                "50000",
                "--split",
                "train",
                "--seed",
                "42",
                "--output",
                output_path,
            ]
        )
        with open(output_path) as f:
            data = json.load(f)
        entry = data["train"][0]
        # Narrative should not have bullet-list format
        assert "- The" not in entry["input"]

    def test_subsampling_ratio(self, tmp_path, sample_csv, schema_yaml):
        """--subsampling-ratio subsamples source data before splitting."""
        data, meta = self._run(
            tmp_path, sample_csv, schema_yaml, extra_args=["--subsampling-ratio", "0.6"]
        )
        # 5 rows * 0.6 = 3 rows subsampled, 80% train → 2 entries
        assert len(data["train"]) == 2
        assert meta["source_rows_total"] == 5  # original count preserved

    def test_same_seed_same_rows(self, tmp_path, sample_csv, schema_yaml):
        """Two conditions with the same seed should process the same rows."""
        out1 = str(tmp_path / "cond1.json")
        out2 = str(tmp_path / "cond2.json")
        base_args = [
            "--source",
            sample_csv,
            "--schema",
            schema_yaml,
            "--features",
            "AGEP,ST,OCCP",
            "--template",
            "dictionary",
            "--target-column",
            "PINCP",
            "--target-threshold",
            "50000",
            "--split",
            "train",
            "--seed",
            "42",
        ]
        main(base_args + ["--condition-name", "c1", "--output", out1])
        main(
            base_args
            + ["--condition-name", "c2", "--output", out2, "--perturbations", "synonym"]
        )

        with open(out1) as f:
            d1 = json.load(f)
        with open(out2) as f:
            d2 = json.load(f)

        # Same number of entries, same labels (same rows, same target)
        assert len(d1["train"]) == len(d2["train"])
        labels1 = [e["output"] for e in d1["train"]]
        labels2 = [e["output"] for e in d2["train"]]
        assert labels1 == labels2
