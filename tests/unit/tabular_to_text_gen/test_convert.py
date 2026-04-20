"""Tests for tabular_to_text_gen/convert.py — CLI entrypoint and integration tests."""

import json

import pandas as pd
import pytest

from tabular_to_text_gen.convert import load_condition_from_file, main, split_dataframe


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

    def test_narrative_template(
        self, tmp_path, sample_csv, schema_yaml, narrative_template
    ):
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
                "--template-file",
                narrative_template,
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

    def test_one_to_many_cli(self, tmp_path, sample_csv, schema_yaml):
        """--one-to-many-copies expands each row into N entries."""
        data, meta = self._run(
            tmp_path,
            sample_csv,
            schema_yaml,
            extra_args=[
                "--one-to-many-copies",
                "3",
                "--one-to-many-perturbation",
                "reorder",
            ],
        )
        # 4 train rows * 3 copies = 12 entries
        assert len(data["train"]) == 12

        # Each group of 3 copies shares the same label
        for i in range(0, 12, 3):
            labels = {data["train"][j]["output"] for j in range(i, i + 3)}
            assert len(labels) == 1

        # At least some groups should have different input text across copies.
        # With 3 segments and 3 copies, a given group *might* collide by chance,
        # but across 4 groups it's extremely unlikely that all are identical.
        any_differ = False
        for i in range(0, 12, 3):
            inputs = [data["train"][j]["input"] for j in range(i, i + 3)]
            if len(set(inputs)) > 1:
                any_differ = True
                break
        assert any_differ, "Expected at least one group with different orderings"

        # Metadata records one_to_many config
        assert meta["one_to_many"] == {"copies": 3, "perturbation": "reorder"}

    def test_one_to_many_conditions_file(
        self, tmp_path, sample_csv, schema_yaml, conditions_yaml_otm
    ):
        """one_to_many works when loaded from a conditions YAML file."""
        output_path = str(tmp_path / "otm_cond_output.json")
        main(
            [
                "--source",
                sample_csv,
                "--schema",
                schema_yaml,
                "--condition-name",
                "dict_reordered_x3",
                "--conditions-file",
                conditions_yaml_otm,
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
        # 4 train rows * 3 copies = 12
        assert len(data["train"]) == 12

    def test_one_to_many_validation_overlap(self, tmp_path, sample_csv, schema_yaml):
        """Error when one_to_many perturbation overlaps with top-level."""
        output_path = str(tmp_path / "overlap_output.json")
        with pytest.raises(SystemExit):
            main(
                [
                    "--source",
                    sample_csv,
                    "--schema",
                    schema_yaml,
                    "--condition-name",
                    "overlap",
                    "--features",
                    "AGEP,ST,OCCP",
                    "--template",
                    "dictionary",
                    "--perturbations",
                    "reorder",
                    "--one-to-many-copies",
                    "2",
                    "--one-to-many-perturbation",
                    "reorder",
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

    def test_one_to_many_deterministic(self, tmp_path, sample_csv, schema_yaml):
        """Same seed produces identical one_to_many output."""
        out1 = str(tmp_path / "det1.json")
        out2 = str(tmp_path / "det2.json")
        extra = [
            "--one-to-many-copies",
            "2",
            "--one-to-many-perturbation",
            "reorder",
        ]
        main(
            [
                "--source",
                sample_csv,
                "--schema",
                schema_yaml,
                "--condition-name",
                "det1",
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
                "--output",
                out1,
            ]
            + extra
        )
        main(
            [
                "--source",
                sample_csv,
                "--schema",
                schema_yaml,
                "--condition-name",
                "det2",
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
                "--output",
                out2,
            ]
            + extra
        )
        with open(out1) as f:
            d1 = json.load(f)
        with open(out2) as f:
            d2 = json.load(f)
        assert d1 == d2

    def test_emit_source_parquet(self, tmp_path, sample_csv, schema_yaml):
        """--emit-source-parquet writes a parquet 1:1 with JSON entries."""
        parquet_path = str(tmp_path / "source_train.parquet")
        data, _ = self._run(
            tmp_path,
            sample_csv,
            schema_yaml,
            extra_args=["--emit-source-parquet", parquet_path],
        )

        import os as _os

        assert _os.path.exists(parquet_path)

        parquet_df = pd.read_parquet(parquet_path)

        # One parquet row per JSON entry
        assert len(parquet_df) == len(data["train"])

        # Original source columns are preserved (including the target)
        assert "PINCP" in parquet_df.columns
        assert "AGEP" in parquet_df.columns

    def test_bundled_train_and_validation(self, tmp_path, sample_csv, schema_yaml):
        """--split train + --validation-ratio bundles both splits in one JSON file.

        The output file should contain BOTH "train" and "validation" top-level
        keys.
        """
        output_path = str(tmp_path / "bundled.json")
        main(
            [
                "--source",
                sample_csv,
                "--schema",
                schema_yaml,
                "--condition-name",
                "bundled_cond",
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
                "--split-ratio",
                "0.6",
                "--validation-ratio",
                "0.2",
                "--seed",
                "42",
                "--output",
                output_path,
            ]
        )
        with open(output_path) as f:
            data = json.load(f)
        assert set(data.keys()) == {"train", "validation"}
        # 5 rows total * 0.6 train = 3, * 0.2 val = 1, so the file has 3 + 1 = 4 entries
        assert len(data["train"]) == 3
        assert len(data["validation"]) == 1
        for entry in data["train"] + data["validation"]:
            assert "input" in entry
            assert "output" in entry
            assert entry["output"] in ("0", "1")

        # Metadata should record the bundled layout
        with open(output_path.replace(".json", ".meta.json")) as f:
            meta = json.load(f)
        assert meta["split"] == "train"
        assert meta["row_count"] == 4  # train + validation combined
        assert meta["extra_splits"] == {"validation": 1}

    def test_test_split_not_bundled(self, tmp_path, sample_csv, schema_yaml):
        """--split test with --validation-ratio emits ONLY the test split.

        The test file is a separate artifact from the training+validation
        bundle, so the validation slice must not leak into the test file.
        """
        output_path = str(tmp_path / "test_only.json")
        main(
            [
                "--source",
                sample_csv,
                "--schema",
                schema_yaml,
                "--condition-name",
                "test_only",
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
                "--split-ratio",
                "0.6",
                "--validation-ratio",
                "0.2",
                "--seed",
                "42",
                "--output",
                output_path,
            ]
        )
        with open(output_path) as f:
            data = json.load(f)
        assert set(data.keys()) == {"test"}
        # 5 * (1 - 0.6 - 0.2) = 1 test row
        assert len(data["test"]) == 1

    def test_emit_source_parquet_deterministic(self, tmp_path, sample_csv, schema_yaml):
        """Same seed produces identical parquet rows across invocations."""
        p1 = str(tmp_path / "p1.parquet")
        p2 = str(tmp_path / "p2.parquet")
        self._run(
            tmp_path, sample_csv, schema_yaml, extra_args=["--emit-source-parquet", p1]
        )
        self._run(
            tmp_path, sample_csv, schema_yaml, extra_args=["--emit-source-parquet", p2]
        )
        df1 = pd.read_parquet(p1)
        df2 = pd.read_parquet(p2)
        pd.testing.assert_frame_equal(df1, df2)
