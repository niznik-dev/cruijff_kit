"""Unit tests for tools/experiment/prepare_data.py (issue #418, chunk 5).

Run with:
    pytest tests/unit/test_prepare_data.py -v
"""

import json

import pytest
import yaml

from cruijff_kit.tools.experiment.prepare_data import prepare


def _write_yaml(exp_directory, config):
    (exp_directory / "experiment_summary.yaml").write_text(yaml.dump(config))


def _model_organism(**kwargs):
    """Build a data.data_generation block with tool=model_organism."""
    block = {"tool": "model_organism"}
    block.update(kwargs)
    return {"data": {"data_generation": block}}


@pytest.fixture
def experiment_directory(tmp_path):
    exp_directory = tmp_path / "exp"
    exp_directory.mkdir()
    return exp_directory


class TestNoOp:
    def test_missing_yaml_returns_error(self, tmp_path):
        assert prepare(tmp_path / "does_not_exist") == 1

    def test_no_data_generation_block(self, experiment_directory):
        _write_yaml(experiment_directory, {"experiment": {"name": "t"}})
        assert prepare(experiment_directory) == 0

    def test_data_block_without_data_generation(self, experiment_directory):
        _write_yaml(experiment_directory, {"data": {"training": {"path": "x"}}})
        assert prepare(experiment_directory) == 0


class TestSingleDataset:
    def test_memorization_dataset_written(self, experiment_directory):
        _write_yaml(
            experiment_directory,
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
        assert prepare(experiment_directory) == 0

        out = experiment_directory / "data" / "bits_parity.json"
        assert out.exists()
        ds = json.loads(out.read_text())
        assert len(ds["train"]) == 8
        assert len(ds["validation"]) == 8  # memorization: train == validation
        assert ds["metadata"]["rule"] == "parity"
        assert ds["metadata"]["design"] == "memorization"

    def test_in_distribution_dataset_split(self, experiment_directory):
        _write_yaml(
            experiment_directory,
            _model_organism(
                name="digits_length",
                input_type="digits",
                rule="length",
                k=5,
                N=20,
                seed=2,
                design="in_distribution",
                split_ratio=0.75,
                output_path="data/dl.json",
            ),
        )
        assert prepare(experiment_directory) == 0
        ds = json.loads((experiment_directory / "data" / "dl.json").read_text())
        assert len(ds["train"]) == 15
        assert len(ds["validation"]) == 5

    def test_legacy_split_key_ignored_and_warns(self, experiment_directory):
        """Regression: the pre-rename `split` key is ignored (the train fraction
        falls back to the 0.8 default, NOT the stale value) and prepare_data
        logs a migration warning rather than silently changing the split (#372).
        """
        _write_yaml(
            experiment_directory,
            _model_organism(
                name="legacy_split",
                input_type="digits",
                rule="length",
                k=5,
                N=20,
                seed=2,
                split=0.5,  # pre-rename key — must NOT be honored
                design="in_distribution",
                output_path="data/ls.json",
            ),
        )
        assert prepare(experiment_directory) == 0
        ds = json.loads((experiment_directory / "data" / "ls.json").read_text())
        # Stale split=0.5 ignored -> default split_ratio=0.8 -> 16/4, not 10/10.
        assert len(ds["train"]) == 16
        assert len(ds["validation"]) == 4
        assert ds["metadata"]["split_ratio"] == 0.8
        # And the migration tripwire fired.
        log_text = (
            experiment_directory / "logs" / "scaffold-prepare-data.log"
        ).read_text()
        assert "'split' was renamed to 'split_ratio'" in log_text


class TestLogging:
    def test_log_file_created_with_provenance(self, experiment_directory):
        _write_yaml(
            experiment_directory,
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
        prepare(experiment_directory)

        log_path = experiment_directory / "logs" / "scaffold-prepare-data.log"
        assert log_path.exists()
        text = log_path.read_text()
        assert "GENERATED: prov_test" in text
        assert "equivalent CLI:" in text
        # Provenance command should include all critical params
        assert "--input_type bits" in text
        assert "--rule parity" in text
        assert "--seed 42" in text


class TestErrors:
    def test_invalid_rule_fails(self, experiment_directory):
        _write_yaml(
            experiment_directory,
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
        assert prepare(experiment_directory) == 1

    def test_missing_required_field_fails(self, experiment_directory):
        _write_yaml(
            experiment_directory,
            _model_organism(
                name="incomplete",
                input_type="bits",
                rule="parity",
                # missing k, N, seed, design, output_path
            ),
        )
        assert prepare(experiment_directory) == 1

    def test_missing_tool_field_fails(self, experiment_directory):
        # data.data_generation is present but has no `tool:` key.
        _write_yaml(
            experiment_directory,
            {"data": {"data_generation": {"name": "x", "rule": "parity"}}},
        )
        assert prepare(experiment_directory) == 1

    def test_unsupported_tool_fails(self, experiment_directory):
        _write_yaml(
            experiment_directory,
            {"data": {"data_generation": {"tool": "mystery_generator"}}},
        )
        assert prepare(experiment_directory) == 1


class TestAbsolutePath:
    def test_absolute_output_path_respected(self, experiment_directory, tmp_path):
        target = tmp_path / "elsewhere" / "abs.json"
        _write_yaml(
            experiment_directory,
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
        assert prepare(experiment_directory) == 0
        assert target.exists()
