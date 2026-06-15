"""Deterministic propagation of fields from experiment_summary.yaml into the
intermediate YAMLs that scaffold-* agents write (eval.yaml,
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

import warnings
from typing import Any


# Run-time seed a stage falls back to when it doesn't set its own. 14 is Johan
# Cruijff's kit number and matches setup_finetune.py's historical --seed default.
DEFAULT_SEED = 14


EVAL_FIELDS: dict[str, str] = {
    "evaluation.temperature": "temperature",
    "evaluation.do_sample": "do_sample",
    "evaluation.max_tokens": "max_tokens",
    "evaluation.max_connections": "max_connections",
    "evaluation.scorers": "scorers",
    # Task-framing invariants source from `controls` (shared by train and eval),
    # so an eval-only run with no setup_finetune.yaml still gets them.
    "controls.prompt": "prompt",
    "controls.system_prompt": "system_prompt",
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


def resolve_seed(experiment_summary: dict, seed_path: str) -> int:
    """Resolve a stage's seed (its value, else DEFAULT_SEED); reject non-ints."""
    seed = _get_dotted(experiment_summary, seed_path)
    if seed is None:
        return DEFAULT_SEED
    # reject bool: True is an int subclass, but `seed: true` is a mistake
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"{seed_path} must be an integer, got {seed!r} "
            f"({type(seed).__name__}). Unquote numbers in experiment_summary.yaml."
        )
    return seed


def propagate_eval_fields(experiment_summary: dict, eval_config: dict) -> dict:
    """Copy EVAL_FIELDS from experiment_summary into eval_config in place.

    Idempotent: existing non-None values in eval_config take precedence so the
    agent's per-cell decisions (e.g. per-task prompt / system_prompt overrides,
    or a per-run prompt carried in for train/eval parity) survive propagation.

    The eval seed is resolved (evaluation.seed, else DEFAULT_SEED) and written
    unless the agent already set a per-cell seed, which wins.

    Warns on a stray ``evaluation.system_prompt`` (no longer a source) so an
    un-migrated file can't silently mismatch training.
    """
    if _get_dotted(experiment_summary, "evaluation.system_prompt") is not None:
        warnings.warn(
            "evaluation.system_prompt is no longer read — system_prompt is "
            "single-sourced at controls.system_prompt (propagated to eval). "
            "Remove evaluation.system_prompt; for per-task variation set "
            "evaluation.tasks[].system_prompt.",
            stacklevel=2,
        )
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

    The training seed is resolved (controls.seed, else DEFAULT_SEED) and
    written unless the agent already set a per-run seed, which wins.
    """
    _propagate(experiment_summary, setup_finetune, TRAIN_FIELDS)
    if setup_finetune.get("seed") is None:
        setup_finetune["seed"] = resolve_seed(experiment_summary, "controls.seed")
    return setup_finetune
