"""Unit tests for the unified model-organism inspect task (issue #418).

Run with:
    pytest tests/unit/test_model_organism_inspect_task.py -v
"""

import json

import pytest
from inspect_ai._util.registry import registry_info

from cruijff_kit.tools.model_organisms.generate import generate
from cruijff_kit.tools.model_organisms.inspect_task import model_organism


def _scorer_names(scorers):
    """Return the inspect-ai registry name for each scorer in the list.

    Strips the ``cruijff_kit/`` package prefix so assertions work across
    inspect-ai versions (bare pre-0.3.200, prefixed after).
    """
    return [registry_info(s).name.removeprefix("cruijff_kit/") for s in scorers]


@pytest.fixture
def tiny_dataset(tmp_path):
    """Write a small bits/parity memorization dataset to disk."""
    ds = generate(
        input_type="bits",
        rule="parity",
        k=4,
        N=8,
        seed=1,
        design="memorization",
    )
    path = tmp_path / "tiny.json"
    with open(path, "w") as f:
        json.dump(ds, f)
    return str(path)


class TestDefaults:
    def test_returns_task(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset)
        assert t.name == "model_organism"

    def test_no_calibration_no_logprobs(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset)
        assert t.config.logprobs is None
        assert t.config.top_logprobs is None

    def test_default_scorers_are_match_and_includes(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset)
        names = _scorer_names(t.scorer)
        assert names == ["inspect_ai/match", "inspect_ai/includes"]


class TestCalibration:
    def test_calibration_enables_logprobs(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset, calibration=True)
        assert t.config.logprobs is True
        assert t.config.top_logprobs == 20

    def test_calibration_appends_risk_scorer(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset, calibration=True)
        names = _scorer_names(t.scorer)
        assert "risk_scorer" in names
        assert "inspect_ai/match" in names
        assert "inspect_ai/includes" in names

    def test_calibration_does_not_duplicate_risk_scorer(self, tiny_dataset, tmp_path):
        config_path = tmp_path / "eval_config.yaml"
        config_path.write_text(
            "prompt: '{input}'\n"
            "system_prompt: ''\n"
            "scorer:\n"
            "  - name: match\n"
            "    params:\n"
            "      location: exact\n"
            "  - name: risk_scorer\n"
        )
        t = model_organism(
            data_path=tiny_dataset,
            config_path=str(config_path),
            calibration=True,
        )
        names = _scorer_names(t.scorer)
        assert names.count("risk_scorer") == 1


class TestVisLabel:
    def test_vis_label_suffixes_task_name(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset, vis_label="foo")
        assert t.name == "model_organism_foo"

    def test_empty_vis_label_uses_bare_name(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset, vis_label="")
        assert t.name == "model_organism"


class TestSolver:
    def test_chat_template_prepends_system_message(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset, use_chat_template=True)
        assert len(t.solver) == 2

    def test_no_chat_template_is_generate_only(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset, use_chat_template=False)
        assert len(t.solver) == 1
