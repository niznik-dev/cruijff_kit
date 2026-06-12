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


def _write_config(tmp_path, *, scorers=None):
    """Write a minimal eval_config.yaml. ``scorers`` is a list of name strings."""
    config_path = tmp_path / "eval_config.yaml"
    lines = ["prompt: '{input}'", "system_prompt: ''"]
    if scorers is not None:
        lines.append("scorers:")
        for name in scorers:
            lines.append(f"  - name: {name}")
    config_path.write_text("\n".join(lines) + "\n")
    return str(config_path)


class TestDefaults:
    def test_returns_task(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset)
        assert t.name == "model_organism"

    def test_default_no_logprobs(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset)
        assert t.config.logprobs is None
        assert t.config.top_logprobs is None

    def test_default_scorers_are_match_and_includes(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset)
        names = _scorer_names(t.scorer)
        assert names == ["inspect_ai/match", "inspect_ai/includes"]


class TestLogprobs:
    def test_explicit_true_enables_logprobs(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset, logprobs=True)
        assert t.config.logprobs is True
        assert t.config.top_logprobs == 20

    def test_top_logprobs_is_configurable(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset, logprobs=True, top_logprobs=50)
        assert t.config.logprobs is True
        assert t.config.top_logprobs == 50

    def test_top_logprobs_ignored_when_disabled(self, tiny_dataset):
        t = model_organism(data_path=tiny_dataset, logprobs=False, top_logprobs=50)
        assert t.config.logprobs is None
        assert t.config.top_logprobs is None


class TestScorerDrivenLogprobs:
    def test_risk_scorer_auto_enables_logprobs(self, tiny_dataset, tmp_path):
        config_path = _write_config(tmp_path, scorers=["risk_scorer"])
        t = model_organism(data_path=tiny_dataset, config_path=config_path)
        assert t.config.logprobs is True
        assert t.config.top_logprobs == 20

    def test_risk_scorer_present_no_duplicate(self, tiny_dataset, tmp_path):
        config_path = _write_config(tmp_path, scorers=["match", "risk_scorer"])
        t = model_organism(data_path=tiny_dataset, config_path=config_path)
        names = _scorer_names(t.scorer)
        assert names.count("risk_scorer") == 1

    def test_match_only_does_not_enable_logprobs(self, tiny_dataset, tmp_path):
        config_path = _write_config(tmp_path, scorers=["match"])
        t = model_organism(data_path=tiny_dataset, config_path=config_path)
        assert t.config.logprobs is None

    def test_numeric_risk_scorer_does_not_enable_logprobs(self, tiny_dataset, tmp_path):
        # numeric_risk_scorer parses a float from text completion — no logprobs
        # needed. A name-based heuristic would over-include it; an attribute
        # check correctly leaves it alone.
        config_path = _write_config(tmp_path, scorers=["numeric_risk_scorer"])
        t = model_organism(data_path=tiny_dataset, config_path=config_path)
        assert t.config.logprobs is None


class TestConflictSemantics:
    def test_explicit_false_with_risk_scorer_raises(self, tiny_dataset, tmp_path):
        config_path = _write_config(tmp_path, scorers=["risk_scorer"])
        with pytest.raises(ValueError, match="risk_scorer.*logprobs=False"):
            model_organism(
                data_path=tiny_dataset,
                config_path=config_path,
                logprobs=False,
            )

    def test_explicit_true_with_risk_scorer_works(self, tiny_dataset, tmp_path):
        config_path = _write_config(tmp_path, scorers=["risk_scorer"])
        t = model_organism(
            data_path=tiny_dataset,
            config_path=config_path,
            logprobs=True,
        )
        assert t.config.logprobs is True

    def test_explicit_false_with_match_only_disables_logprobs(
        self, tiny_dataset, tmp_path
    ):
        config_path = _write_config(tmp_path, scorers=["match"])
        t = model_organism(
            data_path=tiny_dataset,
            config_path=config_path,
            logprobs=False,
        )
        assert t.config.logprobs is None


class TestRemovedCalibrationFlag:
    def test_calibration_kwarg_rejected(self, tiny_dataset):
        # The old `calibration` flag was removed; passing it must fail loud.
        with pytest.raises(TypeError):
            model_organism(data_path=tiny_dataset, calibration=True)


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
