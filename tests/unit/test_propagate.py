"""Tests for cruijff_kit.tools.experiment.propagate.

The parameterized `test_no_*_field_can_silently_drop` suites are the
regression lock for #470 (#502): every entry in EVAL_FIELDS / TRAIN_FIELDS
must propagate from experiment_summary.yaml into the intermediate YAML, with
no quiet drops. Adding a new propagated field requires adding a sample value
below so the test fails loudly if the field is forgotten.
"""

import pytest

from cruijff_kit.tools.experiment.propagate import (
    EVAL_FIELDS,
    TRAIN_FIELDS,
    propagate_eval_fields,
    propagate_train_fields,
)


def _set_dotted(d: dict, path: str, value) -> dict:
    """Set `value` at a dotted path inside `d` (creates intermediate dicts)."""
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value
    return d


# Representative values per intermediate-YAML key. Shared by EVAL_FIELDS and
# TRAIN_FIELDS (system_prompt lives in both).
_SAMPLE_VALUES: dict = {
    # eval
    "system_prompt": "Be helpful and concise.",
    "temperature": 0.7,
    "do_sample": True,
    "max_tokens": 5,
    "max_connections": 16,
    "scorer": [{"name": "match"}],
    # train
    "epochs": 3,
    "batch_size": 4,
    "batch_size_val": 8,
    "gradient_accumulation_steps": 2,
    "weight_decay": 0.01,
    "lora_dropout": 0.1,
    "prompt": "Capitalize: {input}\n",
}


# -----------------------------------------------------------------------------
# Parameterized regression tests — the #470 silent-drop lock
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("source_path,target_key", list(EVAL_FIELDS.items()))
def test_no_eval_field_can_silently_drop(source_path, target_key):
    """Each EVAL_FIELDS entry must propagate (the #470 pattern)."""
    summary = _set_dotted({}, source_path, _SAMPLE_VALUES[target_key])
    eval_config: dict = {}
    propagate_eval_fields(summary, eval_config)
    assert eval_config[target_key] == _SAMPLE_VALUES[target_key]


@pytest.mark.parametrize("source_path,target_key", list(TRAIN_FIELDS.items()))
def test_no_train_field_can_silently_drop(source_path, target_key):
    """Each TRAIN_FIELDS entry must propagate (symmetric to eval)."""
    summary = _set_dotted({}, source_path, _SAMPLE_VALUES[target_key])
    setup_finetune: dict = {}
    propagate_train_fields(summary, setup_finetune)
    assert setup_finetune[target_key] == _SAMPLE_VALUES[target_key]


# -----------------------------------------------------------------------------
# Round-trip with a representative full experiment_summary
# -----------------------------------------------------------------------------


@pytest.fixture
def representative_summary() -> dict:
    return {
        "evaluation": {
            "system_prompt": _SAMPLE_VALUES["system_prompt"],
            "temperature": _SAMPLE_VALUES["temperature"],
            "do_sample": _SAMPLE_VALUES["do_sample"],
            "max_tokens": _SAMPLE_VALUES["max_tokens"],
            "max_connections": _SAMPLE_VALUES["max_connections"],
            "scorer": _SAMPLE_VALUES["scorer"],
        },
        "controls": {
            "epochs": _SAMPLE_VALUES["epochs"],
            "batch_size": _SAMPLE_VALUES["batch_size"],
            "batch_size_val": _SAMPLE_VALUES["batch_size_val"],
            "gradient_accumulation_steps": _SAMPLE_VALUES[
                "gradient_accumulation_steps"
            ],
            "weight_decay": _SAMPLE_VALUES["weight_decay"],
            "lora_dropout": _SAMPLE_VALUES["lora_dropout"],
            "prompt": _SAMPLE_VALUES["prompt"],
            "system_prompt": _SAMPLE_VALUES["system_prompt"],
        },
    }


def test_eval_roundtrip_full(representative_summary):
    eval_config: dict = {}
    propagate_eval_fields(representative_summary, eval_config)
    for target_key in EVAL_FIELDS.values():
        assert eval_config[target_key] == _SAMPLE_VALUES[target_key]


def test_train_roundtrip_full(representative_summary):
    setup_finetune: dict = {}
    propagate_train_fields(representative_summary, setup_finetune)
    for target_key in TRAIN_FIELDS.values():
        assert setup_finetune[target_key] == _SAMPLE_VALUES[target_key]


# -----------------------------------------------------------------------------
# Idempotence: agent-set per-cell / per-run values survive propagation
# -----------------------------------------------------------------------------


def test_eval_idempotence_preserves_per_cell_value(representative_summary):
    """Per-cell system_prompt override (agent-written) survives propagation."""
    eval_config = {"system_prompt": "Per-cell custom prompt"}
    propagate_eval_fields(representative_summary, eval_config)
    assert eval_config["system_prompt"] == "Per-cell custom prompt"


def test_train_idempotence_preserves_per_run_value(representative_summary):
    """Per-run batch_size override (agent-written) survives propagation."""
    setup_finetune = {"batch_size": 999}
    propagate_train_fields(representative_summary, setup_finetune)
    assert setup_finetune["batch_size"] == 999


# -----------------------------------------------------------------------------
# Source-side skip rules
# -----------------------------------------------------------------------------


def test_source_none_value_is_skipped():
    summary = {"evaluation": {"temperature": None}}
    eval_config: dict = {}
    propagate_eval_fields(summary, eval_config)
    assert "temperature" not in eval_config


def test_source_missing_root_is_skipped():
    eval_config: dict = {}
    propagate_eval_fields({}, eval_config)
    assert eval_config == {}


def test_source_partial_missing_path_is_skipped():
    summary = {"evaluation": {}}  # no temperature key
    eval_config: dict = {}
    propagate_eval_fields(summary, eval_config)
    assert eval_config == {}


# -----------------------------------------------------------------------------
# Target-side merge rules
# -----------------------------------------------------------------------------


def test_target_none_placeholder_is_overwritten():
    """YAML null placeholder in target should receive the propagated value."""
    summary = {"evaluation": {"temperature": 0.7}}
    eval_config = {"temperature": None}
    propagate_eval_fields(summary, eval_config)
    assert eval_config["temperature"] == 0.7


# -----------------------------------------------------------------------------
# Falsy-but-not-None values must still propagate
# -----------------------------------------------------------------------------


def test_zero_temperature_propagates():
    """0.0 is falsy but not None — should still be copied."""
    summary = {"evaluation": {"temperature": 0.0}}
    eval_config: dict = {}
    propagate_eval_fields(summary, eval_config)
    assert eval_config["temperature"] == 0.0


def test_false_do_sample_propagates():
    """False is falsy but not None — should still be copied."""
    summary = {"evaluation": {"do_sample": False}}
    eval_config: dict = {}
    propagate_eval_fields(summary, eval_config)
    assert eval_config["do_sample"] is False


# -----------------------------------------------------------------------------
# In-place semantics
# -----------------------------------------------------------------------------


def test_propagate_eval_returns_same_dict():
    eval_config: dict = {}
    result = propagate_eval_fields({"evaluation": {"temperature": 0.7}}, eval_config)
    assert result is eval_config


def test_propagate_train_returns_same_dict():
    setup_finetune: dict = {}
    result = propagate_train_fields({"controls": {"epochs": 3}}, setup_finetune)
    assert result is setup_finetune
