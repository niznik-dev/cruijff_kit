"""Tests for tabular_to_text_gen/lib/output.py — label computation, entry assembly, file writing."""

import json

import pytest

from tabular_to_text_gen.lib.output import (
    build_output_entry,
    compute_label,
    write_metadata,
    write_output,
)


# ---------------------------------------------------------------------------
# compute_label
# ---------------------------------------------------------------------------


class TestComputeLabel:
    def test_threshold_above(self):
        assert compute_label(75000, target_threshold=50000) == "1"

    def test_threshold_below(self):
        assert compute_label(42000, target_threshold=50000) == "0"

    def test_threshold_equal(self):
        # Equal to threshold → "0" (not strictly above)
        assert compute_label(50000, target_threshold=50000) == "0"

    def test_threshold_string_value(self):
        assert compute_label("75000", target_threshold=50000) == "1"

    def test_threshold_non_numeric_raises(self):
        with pytest.raises(ValueError, match="Value must be numeric"):
            compute_label("not_a_number", target_threshold=50000)

    def test_mapping(self):
        mapping = {"Employed": "1", "Unemployed": "0"}
        assert compute_label("Employed", target_mapping=mapping) == "1"
        assert compute_label("Unemployed", target_mapping=mapping) == "0"

    def test_mapping_missing_key_raises(self):
        mapping = {"Employed": "1"}
        with pytest.raises(ValueError, match="not found in mapping"):
            compute_label("Unknown", target_mapping=mapping)

    def test_no_threshold_or_mapping(self):
        assert compute_label(42) == "42"
        assert compute_label("yes") == "yes"

    def test_mapping_takes_priority_over_threshold(self):
        # When both are provided, mapping is checked first
        result = compute_label(
            "75000", target_threshold=50000, target_mapping={"75000": "high"}
        )
        assert result == "high"


# ---------------------------------------------------------------------------
# build_output_entry
# ---------------------------------------------------------------------------


class TestBuildOutputEntry:
    def test_preamble_with_all_parts(self):
        entry = build_output_entry(
            body_text="- age: 51",
            context="Context.",
            context_placement="preamble",
            question="Is income > 50k?",
            target_value=75000,
            target_threshold=50000,
        )
        assert entry["output"] == "1"
        assert entry["input"].startswith("Context.")
        assert "- age: 51" in entry["input"]
        assert entry["input"].endswith("Is income > 50k?")
        assert "system_prompt" not in entry

    def test_preamble_empty_context(self):
        entry = build_output_entry(
            body_text="body",
            context="",
            context_placement="preamble",
            question="question?",
            target_value=1,
        )
        # Should not start with double newline
        assert entry["input"].startswith("body")

    def test_system_prompt_placement(self):
        """When context_placement is system_prompt, context is ignored.

        The system prompt is an experiment-level setting (in
        experiment_summary.yaml), not baked into dataset entries.
        """
        entry = build_output_entry(
            body_text="body text",
            context="System context.",
            context_placement="system_prompt",
            question="question?",
            target_value=1,
        )
        assert "system_prompt" not in entry
        assert "System context." not in entry["input"]
        assert entry["input"].startswith("body text")
        assert entry["input"].endswith("question?")

    def test_system_prompt_no_context(self):
        entry = build_output_entry(
            body_text="body",
            context="",
            context_placement="system_prompt",
            question="q?",
            target_value=1,
        )
        assert "system_prompt" not in entry

    def test_invalid_placement_raises(self):
        with pytest.raises(ValueError, match="Invalid context_placement"):
            build_output_entry("body", "ctx", "invalid", "q?", 1)


# ---------------------------------------------------------------------------
# write_output
# ---------------------------------------------------------------------------


class TestWriteOutput:
    def test_writes_json_with_split_key(self, tmp_path):
        entries = [{"input": "hello", "output": "1"}]
        path = str(tmp_path / "out.json")
        write_output(entries, path, "train")
        with open(path) as f:
            data = json.load(f)
        assert "train" in data
        assert data["train"] == entries

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "out.json")
        write_output([], path, "test")
        with open(path) as f:
            data = json.load(f)
        assert data == {"test": []}


# ---------------------------------------------------------------------------
# write_metadata
# ---------------------------------------------------------------------------


class TestWriteMetadata:
    def test_writes_sidecar(self, tmp_path):
        out_path = str(tmp_path / "output.json")
        # Write a dummy output file first (so size_bytes is nonzero)
        with open(out_path, "w") as f:
            json.dump({"train": [{"input": "x", "output": "1"}]}, f)

        write_metadata(
            output_path=out_path,
            condition_name="test_cond",
            split="train",
            seed=42,
            split_ratio=0.8,
            row_count=1,
            source_path="/tmp/source.csv",
            source_rows_total=100,
            schema_path="/tmp/schema.yaml",
            features=["AGEP", "ST"],
            template="dictionary",
            perturbations=["synonym"],
            target_config={"column": "PINCP", "threshold": 50000},
            context="Context.",
            context_placement="preamble",
            question="Q?",
        )

        meta_path = out_path.replace(".json", ".meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["condition_name"] == "test_cond"
        assert meta["seed"] == 42
        assert meta["row_count"] == 1
        assert meta["features"] == ["AGEP", "ST"]
        assert meta["size_bytes"] > 0
        assert "generated_at" in meta
        assert "one_to_many" not in meta  # not set, should be absent

    def test_writes_one_to_many_metadata(self, tmp_path):
        out_path = str(tmp_path / "output_otm.json")
        with open(out_path, "w") as f:
            json.dump({"train": [{"input": "x", "output": "1"}]}, f)

        otm_config = {"copies": 3, "perturbation": "reorder"}
        write_metadata(
            output_path=out_path,
            condition_name="otm_cond",
            split="train",
            seed=42,
            split_ratio=0.8,
            row_count=3,
            source_path="/tmp/source.csv",
            source_rows_total=100,
            schema_path="/tmp/schema.yaml",
            features=["AGEP", "ST"],
            template="dictionary",
            perturbations=[],
            target_config={"column": "PINCP", "threshold": 50000},
            context="Context.",
            context_placement="preamble",
            question="Q?",
            one_to_many=otm_config,
        )

        meta_path = out_path.replace(".json", ".meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["one_to_many"] == {"copies": 3, "perturbation": "reorder"}
