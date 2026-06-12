"""Golden-fixture oracle for the eval-only scaffolding path (#478).

`tests/fixtures/eval_only/eval_config.golden.yaml` was minted by a real
`scaffold-experiment` run on an experiment with no fine-tuned runs and no
`setup_finetune.yaml`. These assertions pin the invariants that the eval /
torchtune decoupling must preserve, so a regression in propagation or in the
scaffold-inspect agent doc is caught here rather than at eval time.

Complements `test_propagate.py` (which checks the field-level propagation in
isolation): this checks the *shape of the generated artifact* for an eval-only
run end-to-end.
"""

from pathlib import Path

import yaml

GOLDEN = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "eval_only"
    / "eval_config.golden.yaml"
)


def _load() -> dict:
    with open(GOLDEN) as f:
        return yaml.safe_load(f)


def test_golden_parses():
    assert isinstance(_load(), dict)


def test_prompt_present_and_not_bare_default():
    """`prompt` must be carried from experiment_summary, not silently left to the
    @task's bare "{input}" fallback (the failure mode behind reversing Bram's
    "drop prompt" call)."""
    cfg = _load()
    assert "prompt" in cfg
    assert cfg["prompt"].strip() != "{input}"
    assert "{input}" in cfg["prompt"]  # still a template, just a richer one


def test_dataset_type_drives_use_chat_template():
    """use_chat_template is derived from the propagated dataset_type, with no
    model-name guessing."""
    cfg = _load()
    assert cfg["dataset_type"] in {"chat_completion", "text_completion"}
    assert cfg["use_chat_template"] is (cfg["dataset_type"] == "chat_completion")


def test_eval_only_model_path_is_checkpoint_verbatim():
    """An eval-only run points at its pre-existing checkpoint, used verbatim —
    not a path computed under this experiment's own directory."""
    cfg = _load()
    assert cfg["model_path"] == "{CHECKPOINT_DIR}"
    assert cfg["is_finetuned"] is True  # the checkpoint IS fine-tuned, just elsewhere


def test_no_setup_finetune_dependency():
    """The generated config must stand alone — no value should reference a
    setup_finetune.yaml (eval-only runs do not have one). Checks the parsed
    values, not the file's explanatory header comment."""
    cfg = _load()
    for value in cfg.values():
        assert "setup_finetune" not in str(value)
