"""Tests for tabular_to_text_gen/convert.py — CLI entrypoint and integration tests."""

import glob
import json
import os

import pandas as pd
import pytest
import yaml

from tabular_to_text_gen.convert import load_condition_from_file, main, split_dataframe
from tabular_to_text_gen.lib.config_hash import resolve_dataset_path


def _find_output(tmp_path, condition: str, split: str) -> str:
    """Find the single {condition}_{split}_{hash8}.json in tmp_path."""
    pattern = os.path.join(str(tmp_path), f"{condition}_{split}_*.json")
    # Exclude the .meta.json and .parquet sidecars.
    matches = [p for p in glob.glob(pattern) if not p.endswith(".meta.json")]
    assert len(matches) == 1, (
        f"Expected exactly one output for {condition}/{split}, found: {matches}"
    )
    return matches[0]


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
    def _run(
        self,
        tmp_path,
        sample_csv,
        schema_yaml,
        extra_args=None,
        condition="test_cond",
        split="train",
    ):
        """Helper to invoke main() and return parsed output + metadata."""
        argv = [
            "--source",
            sample_csv,
            "--schema",
            schema_yaml,
            "--condition-name",
            condition,
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
            split,
            "--split-ratio",
            "0.8",
            "--seed",
            "42",
            "--output-dir",
            str(tmp_path),
        ]
        if extra_args:
            argv.extend(extra_args)
        main(argv)

        output_path = _find_output(tmp_path, condition, split)
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
                "--output-dir",
                str(tmp_path),
            ]
        )
        output_path = _find_output(tmp_path, "test_cond", "test")
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
                "--output-dir",
                str(tmp_path),
            ]
        )
        output_path = _find_output(tmp_path, "sys_cond", "train")
        with open(output_path) as f:
            data = json.load(f)
        entry = data["train"][0]
        assert "system_prompt" not in entry
        assert "You are a helpful assistant." not in entry["input"]

    def test_conditions_file(self, tmp_path, sample_csv, schema_yaml, conditions_yaml):
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
                "--output-dir",
                str(tmp_path),
            ]
        )
        output_path = _find_output(tmp_path, "dict_full", "train")
        with open(output_path) as f:
            data = json.load(f)
        assert "train" in data
        assert len(data["train"]) == 4

    def test_narrative_template(
        self, tmp_path, sample_csv, schema_yaml, narrative_template
    ):
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
                "--output-dir",
                str(tmp_path),
            ]
        )
        output_path = _find_output(tmp_path, "narr_cond", "train")
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
            "--output-dir",
            str(tmp_path),
        ]
        main(base_args + ["--condition-name", "c1"])
        main(base_args + ["--condition-name", "c2", "--perturbations", "synonym"])

        out1 = _find_output(tmp_path, "c1", "train")
        out2 = _find_output(tmp_path, "c2", "train")
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
                "--output-dir",
                str(tmp_path),
            ]
        )
        output_path = _find_output(tmp_path, "dict_reordered_x3", "train")
        with open(output_path) as f:
            data = json.load(f)
        # 4 train rows * 3 copies = 12
        assert len(data["train"]) == 12

    def test_one_to_many_validation_overlap(self, tmp_path, sample_csv, schema_yaml):
        """Error when one_to_many perturbation overlaps with top-level."""
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
                    "--output-dir",
                    str(tmp_path),
                ]
            )

    def test_one_to_many_deterministic(self, tmp_path, sample_csv, schema_yaml):
        """Same seed produces identical one_to_many output."""
        extra = [
            "--one-to-many-copies",
            "2",
            "--one-to-many-perturbation",
            "reorder",
        ]
        base = [
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
            "--output-dir",
            str(tmp_path),
        ]
        main(base + ["--condition-name", "det1"] + extra)
        main(base + ["--condition-name", "det2"] + extra)
        out1 = _find_output(tmp_path, "det1", "train")
        out2 = _find_output(tmp_path, "det2", "train")
        with open(out1) as f:
            d1 = json.load(f)
        with open(out2) as f:
            d2 = json.load(f)
        assert d1 == d2

    def test_emit_source_parquet(self, tmp_path, sample_csv, schema_yaml):
        """--emit-source-parquet writes a parquet 1:1 with JSON entries."""
        data, _ = self._run(
            tmp_path,
            sample_csv,
            schema_yaml,
            extra_args=["--emit-source-parquet"],
        )

        output_path = _find_output(tmp_path, "test_cond", "train")
        parquet_path = output_path.replace(".json", ".parquet")
        assert os.path.exists(parquet_path)

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
                "--output-dir",
                str(tmp_path),
            ]
        )
        output_path = _find_output(tmp_path, "bundled_cond", "train")
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
                "--output-dir",
                str(tmp_path),
            ]
        )
        output_path = _find_output(tmp_path, "test_only", "test")
        with open(output_path) as f:
            data = json.load(f)
        assert set(data.keys()) == {"test"}
        # 5 * (1 - 0.6 - 0.2) = 1 test row
        assert len(data["test"]) == 1

    def test_reuse_skips_existing(self, tmp_path, sample_csv, schema_yaml):
        """Second call with same config skips regeneration; --force overwrites."""
        self._run(tmp_path, sample_csv, schema_yaml)
        output_path = _find_output(tmp_path, "test_cond", "train")
        first_mtime = os.path.getmtime(output_path)

        # Second call with identical config: file should not be rewritten.
        self._run(tmp_path, sample_csv, schema_yaml)
        assert os.path.getmtime(output_path) == first_mtime

        # With --force, the file is rewritten (mtime advances).
        # sleep briefly so the filesystem mtime actually changes.
        import time

        time.sleep(0.02)
        self._run(tmp_path, sample_csv, schema_yaml, extra_args=["--force"])
        assert os.path.getmtime(output_path) > first_mtime

    def test_different_configs_produce_different_hashes(
        self, tmp_path, sample_csv, schema_yaml
    ):
        """Changing any hashed field yields a new filename, not a collision."""
        base = [
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
            "train",
            "--split-ratio",
            "0.8",
            "--output-dir",
            str(tmp_path),
        ]
        main(base + ["--seed", "42"])
        main(base + ["--seed", "99"])
        pattern = os.path.join(str(tmp_path), "test_cond_train_*.json")
        matches = [p for p in glob.glob(pattern) if not p.endswith(".meta.json")]
        assert len(matches) == 2

    def test_emit_source_parquet_deterministic(self, tmp_path, sample_csv, schema_yaml):
        """Same seed produces identical parquet rows across invocations."""
        self._run(
            tmp_path, sample_csv, schema_yaml, extra_args=["--emit-source-parquet"]
        )
        output_path = _find_output(tmp_path, "test_cond", "train")
        parquet_path = output_path.replace(".json", ".parquet")
        df1 = pd.read_parquet(parquet_path)
        # Regenerate with --force and confirm identical bytes.
        self._run(
            tmp_path,
            sample_csv,
            schema_yaml,
            extra_args=["--emit-source-parquet", "--force"],
        )
        df2 = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# --experiment-summary (YAML-driven invocation)
# ---------------------------------------------------------------------------


def _write_experiment_summary(
    tmp_path, sample_csv, schema_yaml, *, threshold=50000, question="Is income > 50k?"
):
    """Write a minimal experiment_summary.yaml and return its path."""
    doc = {
        "experiment": {
            "name": "t",
            "question": "q",
            "date": "2026-04-20",
            "directory": str(tmp_path),
        },
        "tools": {"preparation": "torchtune", "evaluation": "inspect-ai"},
        "data": {
            "data_generation": {
                "tool": "tabular_to_text_gen",
                "source": sample_csv,
                "schema": schema_yaml,
                "target": {"column": "PINCP", "threshold": threshold},
                "context": "Context text.",
                "context_placement": "preamble",
                "question": question,
                "split_ratio": 0.8,
                "validation_ratio": None,
                "seed": 42,
                "missing_value_handling": "skip",
                "conditions": {
                    "c_small": {
                        "features": ["AGEP", "ST"],
                        "template": "dictionary",
                        "perturbations": [],
                    },
                    "c_full": {
                        "features": ["AGEP", "ST", "OCCP"],
                        "template": "dictionary",
                        "perturbations": [],
                    },
                },
            }
        },
    }
    path = tmp_path / "experiment_summary.yaml"
    path.write_text(yaml.safe_dump(doc))
    return str(path)


class TestExperimentSummaryInvocation:
    """The YAML-driven path must produce the same filename convert.py does
    when invoked with equivalent CLI args. This protects the core invariant
    that scaffold agents (which go through resolve_dataset_path) and
    convert.py writes agree on filenames."""

    def test_yaml_driven_matches_cli(self, tmp_path, sample_csv, schema_yaml):
        es = _write_experiment_summary(tmp_path, sample_csv, schema_yaml)

        # Run via --experiment-summary
        yaml_dir = tmp_path / "yaml_out"
        yaml_dir.mkdir()
        main(
            [
                "--experiment-summary",
                es,
                "--condition-name",
                "c_full",
                "--split",
                "train",
                "--output-dir",
                str(yaml_dir),
            ]
        )
        yaml_output = _find_output(yaml_dir, "c_full", "train")

        # Run via direct CLI args matching the YAML content
        cli_dir = tmp_path / "cli_out"
        cli_dir.mkdir()
        main(
            [
                "--source",
                sample_csv,
                "--schema",
                schema_yaml,
                "--condition-name",
                "c_full",
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
                "--context-placement",
                "preamble",
                "--question",
                "Is income > 50k?",
                "--split",
                "train",
                "--split-ratio",
                "0.8",
                "--seed",
                "42",
                "--missing-value-handling",
                "skip",
                "--output-dir",
                str(cli_dir),
            ]
        )
        cli_output = _find_output(cli_dir, "c_full", "train")

        # Same hash8 in filenames (split is not part of the hash either)
        yaml_hash = yaml_output.rsplit("_", 1)[-1]
        cli_hash = cli_output.rsplit("_", 1)[-1]
        assert yaml_hash == cli_hash

    def test_resolve_dataset_path_agrees_with_convert(
        self, tmp_path, sample_csv, schema_yaml
    ):
        """The single most important invariant: the filename resolve_dataset_path
        predicts is the filename convert.py actually writes. If this ever
        drifts, scaffold agents will point at files that don't exist."""
        es = _write_experiment_summary(tmp_path, sample_csv, schema_yaml)
        with open(es) as f:
            dg = yaml.safe_load(f)["data"]["data_generation"]

        predicted = resolve_dataset_path(dg, "c_small", "train", tmp_path)

        main(
            [
                "--experiment-summary",
                es,
                "--condition-name",
                "c_small",
                "--split",
                "train",
                "--output-dir",
                str(tmp_path),
            ]
        )
        actual = _find_output(tmp_path, "c_small", "train")
        assert actual == predicted

    def test_int_yaml_threshold_matches_float_cli(
        self, tmp_path, sample_csv, schema_yaml
    ):
        """Regression: YAML-parsed `threshold: 50000` (int) and CLI
        `--target-threshold 50000` (argparse float) must agree on filename.
        This bug previously forked the hash and silently produced duplicate
        datasets."""
        es = _write_experiment_summary(
            tmp_path, sample_csv, schema_yaml, threshold=50000
        )

        yaml_dir = tmp_path / "y"
        yaml_dir.mkdir()
        main(
            [
                "--experiment-summary",
                es,
                "--condition-name",
                "c_small",
                "--split",
                "train",
                "--output-dir",
                str(yaml_dir),
            ]
        )
        yaml_hash = _find_output(yaml_dir, "c_small", "train").rsplit("_", 1)[-1]

        cli_dir = tmp_path / "c"
        cli_dir.mkdir()
        main(
            [
                "--source",
                sample_csv,
                "--schema",
                schema_yaml,
                "--condition-name",
                "c_small",
                "--features",
                "AGEP,ST",
                "--template",
                "dictionary",
                "--target-column",
                "PINCP",
                "--target-threshold",
                "50000",
                "--context",
                "Context text.",
                "--context-placement",
                "preamble",
                "--question",
                "Is income > 50k?",
                "--split",
                "train",
                "--split-ratio",
                "0.8",
                "--seed",
                "42",
                "--missing-value-handling",
                "skip",
                "--output-dir",
                str(cli_dir),
            ]
        )
        cli_hash = _find_output(cli_dir, "c_small", "train").rsplit("_", 1)[-1]
        assert yaml_hash == cli_hash

    def test_cli_overrides_ignored_with_warning(
        self, tmp_path, sample_csv, schema_yaml, caplog
    ):
        """If a user passes --question (or any managed field) alongside
        --experiment-summary, the YAML wins and a warning is logged. This
        preserves the single-source-of-truth invariant."""
        import logging

        es = _write_experiment_summary(
            tmp_path, sample_csv, schema_yaml, question="Is income > 50k?"
        )

        # Run once via YAML only
        yaml_dir = tmp_path / "y"
        yaml_dir.mkdir()
        main(
            [
                "--experiment-summary",
                es,
                "--condition-name",
                "c_small",
                "--split",
                "train",
                "--output-dir",
                str(yaml_dir),
            ]
        )
        yaml_hash = _find_output(yaml_dir, "c_small", "train").rsplit("_", 1)[-1]

        # Run again with a conflicting --question; expect it to be ignored
        # and a warning emitted. Hash must be identical.
        override_dir = tmp_path / "o"
        override_dir.mkdir()
        with caplog.at_level(logging.WARNING, logger="tabular_to_text_gen"):
            main(
                [
                    "--experiment-summary",
                    es,
                    "--condition-name",
                    "c_small",
                    "--split",
                    "train",
                    "--output-dir",
                    str(override_dir),
                    "--question",
                    "TOTALLY DIFFERENT QUESTION",
                ]
            )
        override_hash = _find_output(override_dir, "c_small", "train").rsplit("_", 1)[
            -1
        ]

        assert override_hash == yaml_hash
        assert any("YAML is authoritative" in r.getMessage() for r in caplog.records), (
            f"Expected override warning; got: {[r.getMessage() for r in caplog.records]}"
        )

    def test_missing_condition_raises(self, tmp_path, sample_csv, schema_yaml):
        es = _write_experiment_summary(tmp_path, sample_csv, schema_yaml)
        with pytest.raises(
            (KeyError, ValueError, SystemExit), match="not found|not present|c_missing"
        ):
            main(
                [
                    "--experiment-summary",
                    es,
                    "--condition-name",
                    "c_missing",
                    "--split",
                    "train",
                    "--output-dir",
                    str(tmp_path),
                ]
            )
