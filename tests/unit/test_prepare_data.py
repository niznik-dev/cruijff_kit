"""Unit tests for tools/experiment/prepare_data.py (issue #418, chunk 5).

Run with:
    pytest tests/unit/test_prepare_data.py -v
"""

import json

import pytest
import yaml

from cruijff_kit.tools.experiment.prepare_data import prepare


def _write_yaml(exp_dir, config):
    (exp_dir / "experiment_summary.yaml").write_text(yaml.dump(config))


def _model_organism(**kwargs):
    """Build a data.data_generation block with tool=model_organism."""
    block = {"tool": "model_organism"}
    block.update(kwargs)
    return {"data": {"data_generation": block}}


@pytest.fixture
def experiment_dir(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    return exp_dir


class TestNoOp:
    def test_missing_yaml_returns_error(self, tmp_path):
        assert prepare(tmp_path / "does_not_exist") == 1

    def test_no_data_generation_block(self, experiment_dir):
        _write_yaml(experiment_dir, {"experiment": {"name": "t"}})
        assert prepare(experiment_dir) == 0

    def test_data_block_without_data_generation(self, experiment_dir):
        _write_yaml(experiment_dir, {"data": {"training": {"path": "x"}}})
        assert prepare(experiment_dir) == 0


class TestSingleDataset:
    def test_memorization_dataset_written(self, experiment_dir):
        _write_yaml(
            experiment_dir,
            _model_organism(
                name="bits_parity",
                input_type="bits",
                rule="parity",
                k=4,
                N=8,
                seed=1,
                design="memorization",
                output_path="data/bits_parity.json",
            ),
        )
        assert prepare(experiment_dir) == 0

        out = experiment_dir / "data" / "bits_parity.json"
        assert out.exists()
        ds = json.loads(out.read_text())
        assert len(ds["train"]) == 8
        assert len(ds["validation"]) == 8  # memorization: train == validation
        assert ds["metadata"]["rule"] == "parity"
        assert ds["metadata"]["design"] == "memorization"

    def test_in_distribution_dataset_split(self, experiment_dir):
        _write_yaml(
            experiment_dir,
            _model_organism(
                name="digits_length",
                input_type="digits",
                rule="length",
                k=5,
                N=20,
                seed=2,
                design="in_distribution",
                split=0.75,
                output_path="data/dl.json",
            ),
        )
        assert prepare(experiment_dir) == 0
        ds = json.loads((experiment_dir / "data" / "dl.json").read_text())
        assert len(ds["train"]) == 15
        assert len(ds["validation"]) == 5


class TestLogging:
    def test_log_file_created_with_provenance(self, experiment_dir):
        _write_yaml(
            experiment_dir,
            _model_organism(
                name="prov_test",
                input_type="bits",
                rule="parity",
                k=4,
                N=8,
                seed=42,
                design="memorization",
                output_path="data/p.json",
            ),
        )
        prepare(experiment_dir)

        log_path = experiment_dir / "logs" / "scaffold-prepare-data.log"
        assert log_path.exists()
        text = log_path.read_text()
        assert "GENERATED: prov_test" in text
        assert "equivalent CLI:" in text
        # Provenance command should include all critical params
        assert "--input_type bits" in text
        assert "--rule parity" in text
        assert "--seed 42" in text


class TestErrors:
    def test_invalid_rule_fails(self, experiment_dir):
        _write_yaml(
            experiment_dir,
            _model_organism(
                name="bad",
                input_type="bits",
                rule="nonexistent_rule",
                k=4,
                N=8,
                seed=1,
                design="memorization",
                output_path="data/bad.json",
            ),
        )
        assert prepare(experiment_dir) == 1

    def test_missing_required_field_fails(self, experiment_dir):
        _write_yaml(
            experiment_dir,
            _model_organism(
                name="incomplete",
                input_type="bits",
                rule="parity",
                # missing k, N, seed, design, output_path
            ),
        )
        assert prepare(experiment_dir) == 1

    def test_missing_tool_field_fails(self, experiment_dir):
        # data.data_generation is present but has no `tool:` key.
        _write_yaml(
            experiment_dir,
            {"data": {"data_generation": {"name": "x", "rule": "parity"}}},
        )
        assert prepare(experiment_dir) == 1

    def test_unsupported_tool_fails(self, experiment_dir):
        _write_yaml(
            experiment_dir,
            {"data": {"data_generation": {"tool": "mystery_generator"}}},
        )
        assert prepare(experiment_dir) == 1


class TestAbsolutePath:
    def test_absolute_output_path_respected(self, experiment_dir, tmp_path):
        target = tmp_path / "elsewhere" / "abs.json"
        _write_yaml(
            experiment_dir,
            _model_organism(
                name="abs",
                input_type="bits",
                rule="parity",
                k=4,
                N=8,
                seed=1,
                design="memorization",
                output_path=str(target),
            ),
        )
        assert prepare(experiment_dir) == 0
        assert target.exists()
