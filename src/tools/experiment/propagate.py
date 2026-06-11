"""Deterministic propagation of fields from experiment_summary.yaml into the
intermediate YAMLs that scaffold-* agents write (eval_config.yaml,
setup_finetune.yaml).

Background (#502): the scaffold-inspect and scaffold-torchtune agents
historically hand-copied fields from experiment_summary.yaml into their
intermediate YAMLs by following prose bullets in their agent docs. If a bullet
was missing or quietly removed, the field could be silently dropped. That is
exactly how `evaluation.temperature` vanished in #470.

Moving the dumb copies into code makes the map the single source of truth: add
or remove an entry here and both agents update behavior in lockstep. Agents
still own judgment work (per-cell system_prompt overrides, path composition,
branching on dataset type) but no longer touch flat field copies.
"""

from typing import Any


# Canonical run-time seed when neither a stage override nor experiment.seed is
# set. 14 is Johan Cruijff's kit number and matches setup_finetune.py's
# historical --seed default.
DEFAULT_SEED = 14


EVAL_FIELDS: dict[str, str] = {
    "evaluation.system_prompt": "system_prompt",
    "evaluation.temperature": "temperature",
    "evaluation.do_sample": "do_sample",
    "evaluation.max_tokens": "max_tokens",
    "evaluation.max_connections": "max_connections",
    "evaluation.scorer": "scorer",
    # The @task reads `prompt` from config_path at runtime (default "{input}")
    # and the agent derives `use_chat_template` from `dataset_type`. Both lived
    # only in setup_finetune.yaml before; carrying them here lets an eval-only
    # run (no fine-tuning, no setup_finetune.yaml) source them from the
    # experiment_summary single source of truth instead of silently defaulting.
    "controls.prompt": "prompt",
    "controls.dataset_type": "dataset_type",
}


TRAIN_FIELDS: dict[str, str] = {
    "controls.epochs": "epochs",
    "controls.batch_size": "batch_size",
    "controls.batch_size_val": "batch_size_val",
    "controls.gradient_accumulation_steps": "gradient_accumulation_steps",
    "controls.weight_decay": "weight_decay",
    "controls.lora_dropout": "lora_dropout",
    "controls.prompt": "prompt",
    "controls.system_prompt": "system_prompt",
}


def _get_dotted(d: dict, path: str) -> Any:
    """Walk `path` through nested dicts; return None if any step is missing."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _propagate(source: dict, target: dict, fields: dict[str, str]) -> dict:
    """Copy each (source_path, target_key) pair from source to target.

    Rules (idempotent merge):
    - Source value of None or missing path: skip.
    - Target key already present with a non-None value: skip (agent override wins).
    - Otherwise: copy.

    Returns `target` modified in place.
    """
    for source_path, target_key in fields.items():
        value = _get_dotted(source, source_path)
        if value is None:
            continue
        if target.get(target_key) is not None:
            continue
        target[target_key] = value
    return target


def resolve_seed(experiment_summary: dict, override_path: str) -> int:
    """Resolve the run-time seed for a stage.

    Precedence: stage override (`override_path`, e.g. "evaluation.seed" or
    "controls.seed") > shared `experiment.seed` > DEFAULT_SEED. Always returns
    an int, so the resolved value can be recorded downstream for provenance and
    no stage ever silently falls back to a random seed.

    A seed of 0 is a valid, distinct value (checked with `is not None`, not
    truthiness), so it survives every hop intact.
    """
    override = _get_dotted(experiment_summary, override_path)
    if override is not None:
        return override
    shared = _get_dotted(experiment_summary, "experiment.seed")
    if shared is not None:
        return shared
    return DEFAULT_SEED


def propagate_eval_fields(experiment_summary: dict, eval_config: dict) -> dict:
    """Copy EVAL_FIELDS from experiment_summary into eval_config in place.

    Idempotent: existing non-None values in eval_config take precedence so the
    agent's per-cell decisions (e.g. per-task system_prompt overrides) survive
    propagation.

    The eval seed is resolved (override > experiment.seed > default) and written
    unless the agent already set a per-cell seed, which wins.
    """
    _propagate(experiment_summary, eval_config, EVAL_FIELDS)
    if eval_config.get("seed") is None:
        eval_config["seed"] = resolve_seed(experiment_summary, "evaluation.seed")
    return eval_config


def propagate_train_fields(experiment_summary: dict, setup_finetune: dict) -> dict:
    """Copy TRAIN_FIELDS from experiment_summary into setup_finetune in place.

    Idempotent (same semantics as propagate_eval_fields).

    Note on `controls.system_prompt`: this is propagated by default. The
    scaffold-torchtune agent is responsible for removing it after propagation
    when the run's `dataset_type` is `text_completion` (base / non-instruct
    models have no chat template to hold a system prompt).

    The training seed is resolved (override > experiment.seed > default) and
    written unless the agent already set a per-run seed, which wins. This is
    what makes training deterministic by default — absent any seed, training
    resolves to DEFAULT_SEED rather than the recipe's historical null.
    """
    _propagate(experiment_summary, setup_finetune, TRAIN_FIELDS)
    if setup_finetune.get("seed") is None:
        setup_finetune["seed"] = resolve_seed(experiment_summary, "controls.seed")
    return setup_finetune
