"""Content-hash helper for dataset filenames.

The hash is computed over a canonicalized config dict so that two generations
with identical inputs produce the same filename (`{condition}_{split}_{hash8}.json`)
and can be reused across experiments. See TABULAR_DATASET_NAMING.md for details.

Public API:
    canonicalize(config: dict) -> dict
    hash_config(config: dict) -> str           # full hex SHA-256
    hash_config_short(config: dict, n=8) -> str
    file_sha256(path: str) -> str              # for source/schema content hashing
    build_generation_config(data_generation, condition_name) -> dict
    resolve_dataset_path(data_generation, condition_name, split, output_dir) -> str
"""

import hashlib
import json
import os
from pathlib import Path


# Default values used to normalize an absent field to its default before hashing.
# A field present with its default value must hash the same as the field omitted.
_DEFAULTS = {
    "perturbations": [],
    "subsampling_ratio": 1.0,
    "missing_value_handling": "skip",
    "context": "",
    "context_placement": "preamble",
    "question": "",
    "one_to_many": None,
    "template_file_sha256": None,
    "style_guidance": None,
}


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's content as a lowercase hex string."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def canonicalize(config: dict) -> dict:
    """Normalize a raw config dict into the stable form that gets hashed.

    - Fills in defaults so absent fields and explicit-default fields hash identically.
    - Drops `missing_value_text` when handling != "include" (only relevant in that case).
    - Drops `style_guidance` for templates where it has no effect on output.
    - Preserves `features` ordering (ordering affects rendered output).

    The returned dict is JSON-serializable and intended to be passed to hash_config().
    """
    c = dict(config)

    for key, default in _DEFAULTS.items():
        c.setdefault(key, default)

    if c.get("missing_value_handling") != "include":
        c.pop("missing_value_text", None)
    else:
        c.setdefault("missing_value_text", "missing")

    # style_guidance only affects llm_narrative output and custom-narrative
    # template authoring. For dictionary and generic narrative it is inert.
    template = c.get("template")
    if template not in {"llm_narrative", "narrative"}:
        c["style_guidance"] = None

    # Normalize numeric target threshold to float
    target = c.get("target")
    if isinstance(target, dict) and target.get("threshold") is not None:
        target = dict(target)
        target["threshold"] = float(target["threshold"])
        c["target"] = target

    return c


def hash_config(config: dict) -> str:
    """Full hex SHA-256 over canonicalize(config) serialized with sorted keys."""
    canonical = canonicalize(config)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_config_short(config: dict, n: int = 8) -> str:
    """First `n` chars of hash_config(config). Default 8 (~4B-space)."""
    return hash_config(config)[:n]


def build_generation_config(data_generation: dict, condition_name: str) -> dict:
    """Build the canonical generation-config dict for one condition.

    Reads the shared ``data_generation`` block (source, schema, target, context,
    question, splits, seed, subsampling, missing-value handling) and merges the
    per-condition settings (``features``, ``template``, ``template_file``,
    ``perturbations``, ``style_guidance``, ``one_to_many``) into the single
    dict that ``hash_config`` / ``hash_config_short`` operate on.

    This is the one shared place where the mapping from an experiment's
    ``data_generation`` YAML to the canonical hash input lives. Callers
    (convert.py, scaffold-torchtune, scaffold-inspect) all go through it so a
    single YAML config always resolves to the same filename everywhere.

    Parameters
    ----------
    data_generation : dict
        The ``data.data_generation`` block from an ``experiment_summary.yaml``.
    condition_name : str
        Key into ``data_generation["conditions"]``.

    Raises
    ------
    KeyError if ``condition_name`` is not present in the conditions mapping.
    """
    conditions = data_generation.get("conditions") or {}
    if condition_name not in conditions:
        available = ", ".join(sorted(conditions.keys())) or "(none)"
        raise KeyError(
            f"condition '{condition_name}' not found in data_generation.conditions. "
            f"Available: {available}"
        )
    cond = conditions[condition_name]

    target: dict = {}
    raw_target = data_generation.get("target") or {}
    if "column" in raw_target:
        target["column"] = raw_target["column"]
    if raw_target.get("threshold") is not None:
        target["threshold"] = raw_target["threshold"]
    if raw_target.get("mapping") is not None:
        target["mapping"] = raw_target["mapping"]

    template_file = cond.get("template_file")
    template_file_sha = file_sha256(template_file) if template_file else None

    threshold = raw_target.get("threshold")
    threshold = float(threshold) if threshold is not None else None

    return {
        "source_sha256": file_sha256(data_generation["source"]),
        "schema_sha256": file_sha256(data_generation["schema"]),
        "features": list(cond.get("features") or []),
        "template": cond.get("template", "dictionary"),
        "template_file_sha256": template_file_sha,
        "perturbations": list(cond.get("perturbations") or []),
        "target": target,
        "context": data_generation.get("context", ""),
        "context_placement": data_generation.get("context_placement", "preamble"),
        "question": data_generation.get("question", ""),
        "split_ratio": data_generation.get("split_ratio", 0.8),
        "validation_ratio": data_generation.get("validation_ratio"),
        "seed": data_generation.get("seed", 42),
        "subsampling_ratio": data_generation.get("subsampling_ratio"),
        "missing_value_handling": data_generation.get("missing_value_handling", "skip"),
        "missing_value_text": data_generation.get("missing_value_text", "missing"),
        "one_to_many": cond.get("one_to_many"),
        "style_guidance": cond.get("style_guidance"),
    }


def resolve_dataset_path(
    data_generation: dict,
    condition_name: str,
    split: str,
    output_dir: str | Path,
) -> str:
    """Return the absolute path convert.py would write for ``(condition, split)``.

    Combines ``build_generation_config`` + ``hash_config_short`` with the
    canonical filename format ``{condition}_{split}_{hash8}.json``. Scaffold
    agents call this to discover which file to point each generated torchtune
    / inspect config at, without duplicating the hashing logic.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"split must be one of train|validation|test, got {split!r}")
    cfg = build_generation_config(data_generation, condition_name)
    short_hash = hash_config_short(cfg)
    filename = f"{condition_name}_{split}_{short_hash}.json"
    return os.path.join(str(output_dir), filename)
