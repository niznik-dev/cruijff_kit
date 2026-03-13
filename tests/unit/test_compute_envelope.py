"""Unit tests for tools/slurm/compute_envelope.py

Run with:
    pytest tests/unit/test_compute_envelope.py -v

Tests use fixture data — no cluster or GPU required.
"""

import json

import pytest
import yaml

from cruijff_kit.tools.slurm.compute_envelope import (
    build_envelope,
    load_envelope,
    save_envelope,
)


# Minimal experiment_summary.yaml content for testing
SAMPLE_EXPERIMENT_SUMMARY = {
    "experiment": {
        "name": "cap_test_2025-10-22",
        "date": "2025-10-22",
        "type": "sanity_check",
        "question": "Test question",
    },
    "models": {
        "base": [
            {
                "name": "Llama-3.2-1B-Instruct",
                "path": "/path/to/model",
            }
        ]
    },
    "data": {
        "training": {
            "path": "/path/to/data.json",
            "splits": {"train": 800, "validation": 100, "test": 100},
        }
    },
    "controls": {
        "epochs": 2,
        "batch_size": 4,
    },
}

SAMPLE_JOBS = [
    {
        "run_name": "1B_rank4",
        "job_type": "finetune",
        "wall_time": "0:05:23",
        "gpus": 1,
    },
    {
        "run_name": "1B_rank4",
        "job_type": "eval",
        "wall_time": "0:00:45",
        "gpus": 1,
    },
]


class TestBuildEnvelope:
    def test_basic(self, tmp_path):
        yaml_path = tmp_path / "experiment_summary.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(SAMPLE_EXPERIMENT_SUMMARY, f)

        envelope = build_envelope(SAMPLE_JOBS, yaml_path)
        assert envelope["experiment_name"] == "cap_test_2025-10-22"
        assert envelope["model"] == "Llama-3.2-1B-Instruct"
        assert envelope["dataset_size"] == 800
        assert envelope["epochs"] == 2
        assert envelope["batch_size"] == 4
        assert envelope["date"] == "2025-10-22"
        assert len(envelope["jobs"]) == 2

    def test_missing_yaml(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_envelope(SAMPLE_JOBS, tmp_path / "nonexistent.yaml")

    def test_no_base_models(self, tmp_path):
        """If no base models listed, model should be None."""
        config = {**SAMPLE_EXPERIMENT_SUMMARY}
        config["models"] = {"base": []}
        yaml_path = tmp_path / "experiment_summary.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        envelope = build_envelope(SAMPLE_JOBS, yaml_path)
        assert envelope["model"] is None


class TestSaveAndLoadEnvelope:
    def test_roundtrip(self, tmp_path):
        envelope = {
            "experiment_name": "test",
            "model": "Llama-3.2-1B-Instruct",
            "dataset_size": 800,
            "epochs": 2,
            "batch_size": 4,
            "date": "2025-10-22",
            "jobs": SAMPLE_JOBS,
        }
        out_path = tmp_path / "analysis" / "compute_metrics.json"
        save_envelope(envelope, out_path)
        loaded = load_envelope(out_path)
        assert loaded == envelope

    def test_creates_parent_dirs(self, tmp_path):
        envelope = {"experiment_name": "test", "jobs": []}
        out_path = tmp_path / "deep" / "nested" / "compute_metrics.json"
        result = save_envelope(envelope, out_path)
        assert result.exists()

    def test_rejects_old_format(self, tmp_path):
        """Bare list (old format) should raise ValueError."""
        old_path = tmp_path / "compute_metrics.json"
        with open(old_path, "w") as f:
            json.dump(SAMPLE_JOBS, f)

        with pytest.raises(ValueError, match="bare job list"):
            load_envelope(old_path)
