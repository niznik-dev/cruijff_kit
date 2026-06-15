"""Tests for cruijff_kit.tools.experiment.propagate.

The parameterized `test_no_*_field_can_silently_drop` suites are the
regression lock for #470 (#502): every entry in EVAL_FIELDS / TRAIN_FIELDS
must propagate from experiment_summary.yaml into the intermediate YAML, with
no quiet drops. Adding a new propagated field requires adding a sample value
below so the test fails loudly if the field is forgotten.
"""

import pytest

from cruijff_kit.tools.experiment.propagate import (
    DEFAULT_SEED,
    EVAL_FIELDS,
    TRAIN_FIELDS,
    propagate_eval_fields,
    propagate_train_fields,
    resolve_seed,
)


def _set_dotted(d: dict, path: str, value) -> dict:
    """Set `value` at a dotted path inside `d` (creates intermediate dicts)."""
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value
    return d


# Representative values per intermediate-YAML key. `system_prompt` is a single
# source in `controls` (propagated to both eval and train) — not duplicated
# into `evaluation`.
_SAMPLE_VALUES: dict = {
    # eval
    "system_prompt": "Be helpful and concise.",
    "temperature": 0.7,
    "do_sample": True,
    "max_tokens": 5,
    "max_connections": 16,
    "scorers": [{"name": "match"}],
    "dataset_type": "chat_completion",
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
            "temperature": _SAMPLE_VALUES["temperature"],
            "do_sample": _SAMPLE_VALUES["do_sample"],
            "max_tokens": _SAMPLE_VALUES["max_tokens"],
            "max_connections": _SAMPLE_VALUES["max_connections"],
            "scorers": _SAMPLE_VALUES["scorers"],
        },
        "controls": {
            "dataset_type": _SAMPLE_VALUES["dataset_type"],
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


def test_eval_system_prompt_sourced_from_controls(representative_summary):
    """`system_prompt` is single-sourced in `controls`, not `evaluation` (#562).

    Guards the de-duplication: eval must derive `system_prompt` from
    `controls.system_prompt`. A stray `evaluation.system_prompt` is not a source
    and must be ignored, so it can never silently override the single source.
    """
    summary = dict(representative_summary)
    summary["evaluation"] = {
        **summary["evaluation"],
        "system_prompt": "STALE eval-side value",
    }
    eval_config: dict = {}
    propagate_eval_fields(summary, eval_config)
    assert eval_config["system_prompt"] == _SAMPLE_VALUES["system_prompt"]
    assert eval_config["system_prompt"] != "STALE eval-side value"


def test_double_propagation_is_stable(representative_summary):
    """f(f(x)) == f(x): a second pass must not clobber the first.

    The scaffold agents may re-run propagation on the same config; the
    docstring promises idempotence. Lock it across *two* calls (not just one),
    and confirm a pre-existing per-cell override survives both passes.
    """
    eval_config = {"system_prompt": "Per-cell custom prompt"}
    propagate_eval_fields(representative_summary, eval_config)
    after_first = dict(eval_config)
    propagate_eval_fields(representative_summary, eval_config)
    assert eval_config == after_first  # second pass is a no-op
    assert eval_config["system_prompt"] == "Per-cell custom prompt"  # override intact


# -----------------------------------------------------------------------------
# Source-side skip rules
# -----------------------------------------------------------------------------


def test_source_none_value_is_skipped():
    summary = {"evaluation": {"temperature": None}}
    eval_config: dict = {}
    propagate_eval_fields(summary, eval_config)
    assert "temperature" not in eval_config


def test_source_missing_root_is_skipped():
    # The flat EVAL_FIELDS copies skip when absent; only the always-resolved
    # seed lands (the default, since the stage seed is unset).
    eval_config: dict = {}
    propagate_eval_fields({}, eval_config)
    assert eval_config == {"seed": DEFAULT_SEED}


def test_source_partial_missing_path_is_skipped():
    summary = {"evaluation": {}}  # no temperature key
    eval_config: dict = {}
    propagate_eval_fields(summary, eval_config)
    assert eval_config == {"seed": DEFAULT_SEED}


def test_malformed_section_is_skipped_not_crashed():
    """A non-dict section (hand-edited YAML garbage) must skip, not crash.

    _get_dotted's `isinstance(cur, dict)` guard handles this gracefully; lock
    it so a future refactor can't silently turn the graceful skip into an
    AttributeError when `evaluation` is a string/list instead of a mapping.
    """
    summary = {"evaluation": "oops not a dict"}
    eval_config: dict = {}
    result = propagate_eval_fields(summary, eval_config)
    # Malformed section skips every flat copy; seed still resolves to default.
    assert result == {"seed": DEFAULT_SEED}


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


# -----------------------------------------------------------------------------
# Eval-only decoupling: prompt + dataset_type reach eval_config WITHOUT
# setup_finetune.yaml (#478)
# -----------------------------------------------------------------------------


def test_eval_only_gets_prompt_and_dataset_type_from_summary():
    """An eval-only experiment has no setup_finetune.yaml, so the inspect task's
    `prompt` and the chat-template-driving `dataset_type` must reach eval_config
    straight from experiment_summary (#478). Before they joined EVAL_FIELDS,
    `prompt` silently fell back to the bare "{input}" default at eval time and
    `dataset_type` was guessed from the model name — both corrupting evals.
    """
    summary = {
        "controls": {
            "prompt": "Capitalize: {input}\n",
            "dataset_type": "text_completion",
        },
    }
    eval_config: dict = {}
    propagate_eval_fields(summary, eval_config)
    assert eval_config["prompt"] == "Capitalize: {input}\n"
    assert eval_config["dataset_type"] == "text_completion"


# -----------------------------------------------------------------------------
# Seed resolution: independent train + eval seeds, both defaulting to 14
# -----------------------------------------------------------------------------


def test_resolve_seed_uses_stage_value_when_set():
    summary = {"evaluation": {"seed": 7}, "controls": {"seed": 9}}
    assert resolve_seed(summary, "evaluation.seed") == 7
    assert resolve_seed(summary, "controls.seed") == 9


def test_resolve_seed_defaults_when_unset():
    assert resolve_seed({}, "evaluation.seed") == DEFAULT_SEED
    assert resolve_seed({}, "controls.seed") == DEFAULT_SEED


def test_resolve_seed_zero_is_a_valid_distinct_value():
    """0 is falsy but a legitimate seed — must not collapse to the default."""
    summary = {"controls": {"seed": 0}}
    assert resolve_seed(summary, "controls.seed") == 0


@pytest.mark.parametrize(
    "bad_seed",
    ["14", "14abc", 3.7, [1, 2], True],
    ids=["quoted-int", "garbage-str", "float", "list", "bool"],
)
def test_resolve_seed_rejects_non_int(bad_seed):
    """A non-int seed must fail loudly at scaffold time, not late in a SLURM
    job (or silently as a float reaching torchtune). bool is rejected too —
    `seed: true` is a mistake, not the seed 1."""
    summary = {"evaluation": {"seed": bad_seed}}
    with pytest.raises(ValueError, match="evaluation.seed must be an integer"):
        resolve_seed(summary, "evaluation.seed")


def test_both_seeds_match_by_default():
    """With nothing set, train and eval both resolve to DEFAULT_SEED — they
    match for free, no shared knob required."""
    eval_config: dict = {}
    setup_finetune: dict = {}
    propagate_eval_fields({}, eval_config)
    propagate_train_fields({}, setup_finetune)
    assert eval_config["seed"] == DEFAULT_SEED
    assert setup_finetune["seed"] == DEFAULT_SEED


def test_stage_seeds_are_independent():
    """evaluation.seed moves only eval; controls.seed moves only train.
    Neither leaks into the other — divergence is per-stage and deliberate."""
    summary = {"evaluation": {"seed": 7}, "controls": {"seed": 9}}
    eval_config: dict = {}
    setup_finetune: dict = {}
    propagate_eval_fields(summary, eval_config)
    propagate_train_fields(summary, setup_finetune)
    assert eval_config["seed"] == 7
    assert setup_finetune["seed"] == 9


def test_agent_per_run_seed_wins_over_resolution():
    """A seed the scaffold agent already wrote (e.g. a per-run sweep value)
    survives propagation — same idempotence rule as every other field."""
    eval_config = {"seed": 4242}
    setup_finetune = {"seed": 0}  # 0 is a real per-run value, not "unset"
    propagate_eval_fields({}, eval_config)
    propagate_train_fields({}, setup_finetune)
    assert eval_config["seed"] == 4242
    assert setup_finetune["seed"] == 0
