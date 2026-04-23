"""Output assembly and writing.

Assembles final JSON entries from processed rows, handles context
placement (preamble vs system_prompt), computes target labels, and
writes output JSON files with .meta.json sidecar metadata.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def compute_label(
    target_value,
    target_threshold: float | None = None,
    target_mapping: dict | None = None,
) -> str:
    """Compute the output label from a raw target value.

    For numeric targets with threshold: "1" if value > threshold, else "0"
    For categorical targets with mapping: mapping[value]
    """
    if target_mapping is not None:
        str_value = str(target_value)
        if str_value not in target_mapping:
            raise ValueError(
                f"Target value '{str_value}' not found in mapping. "
                f"Available keys: {list(target_mapping.keys())}"
            )
        return target_mapping[str_value]

    if target_threshold is not None:
        try:
            numeric_value = float(target_value)
        except (ValueError, TypeError):
            raise ValueError(
                f"Cannot compare target value '{target_value}' against "
                f"threshold {target_threshold}. Value must be numeric."
            )
        return "1" if numeric_value > target_threshold else "0"

    # No threshold or mapping — return raw value as string
    return str(target_value)


def build_output_entry(
    body_text: str,
    context: str,
    context_placement: str,
    question: str,
    target_value,
    target_threshold: float | None = None,
    target_mapping: dict | None = None,
) -> dict:
    """Build a single output entry.

    When context_placement == "preamble":
        Returns {"input": context + body + question, "output": label}

    When context_placement == "system_prompt":
        Context is ignored here — it is an experiment-level setting
        managed in experiment_summary.yaml, not baked into dataset entries.
        Returns {"input": body + question, "output": label}
    """
    label = compute_label(target_value, target_threshold, target_mapping)

    if context_placement == "preamble":
        parts = []
        if context:
            parts.append(context)
        parts.append(body_text)
        if question:
            parts.append(question)
        input_text = "\n\n".join(parts)
        return {"input": input_text, "output": label}

    elif context_placement == "system_prompt":
        parts = [body_text]
        if question:
            parts.append(question)
        input_text = "\n\n".join(parts)
        return {"input": input_text, "output": label}

    else:
        raise ValueError(
            f"Invalid context_placement '{context_placement}'. "
            f"Must be 'preamble' or 'system_prompt'"
        )


def write_output(
    entries: list[dict],
    output_path: str,
    split: str,
    extra_splits: dict[str, list[dict]] | None = None,
) -> None:
    """Write the final JSON file.

    Wraps entries under the split key, e.g.:
        {"train": [{...}, {...}]}

    If extra_splits is provided, its entries are merged into the same file
    under additional top-level keys. This is used to bundle training and validation slices
    into a single file: {"train": [...], "validation": [...]}
    """
    data = {split: entries}
    if extra_splits:
        for extra_split_name, extra_entries in extra_splits.items():
            if extra_split_name == split:
                raise ValueError(
                    f"extra_splits key {extra_split_name!r} collides with "
                    f"primary split {split!r}"
                )
            data[extra_split_name] = extra_entries
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def write_metadata(
    output_path: str,
    condition_name: str,
    split: str,
    seed: int,
    split_ratio: float,
    row_count: int,
    source_path: str,
    source_rows_total: int,
    schema_path: str,
    features: list[str],
    template: str,
    perturbations: list[str],
    target_config: dict,
    context: str,
    context_placement: str,
    question: str,
    template_file: str | None = None,
    one_to_many: dict | None = None,
    extra_splits: dict[str, int] | None = None,
    source_sha256: str | None = None,
    schema_sha256: str | None = None,
    template_file_sha256: str | None = None,
    config: dict | None = None,
    config_hash: str | None = None,
    non_deterministic: bool = False,
) -> None:
    """Write the .meta.json sidecar file alongside the output."""
    meta_path = str(Path(output_path).with_suffix(".meta.json"))

    size_bytes = os.path.getsize(output_path) if os.path.exists(output_path) else 0

    metadata = {
        "condition_name": condition_name,
        "split": split,
        "seed": seed,
        "split_ratio": split_ratio,
        "row_count": row_count,
        "size_bytes": size_bytes,
        "source": {"path": source_path, "sha256": source_sha256}
        if source_sha256
        else source_path,
        "source_rows_total": source_rows_total,
        "schema": {"path": schema_path, "sha256": schema_sha256}
        if schema_sha256
        else schema_path,
        "features": features,
        "template": template,
        "perturbations": perturbations,
        "target": target_config,
        "context": context,
        "context_placement": context_placement,
        "question": question,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "non_deterministic": non_deterministic,
    }

    if config_hash:
        metadata["config_hash"] = config_hash
        metadata["config_hash_short"] = config_hash[:8]
    if config is not None:
        metadata["config"] = config

    if template_file:
        metadata["template_file"] = (
            {
                "path": os.path.abspath(template_file),
                "sha256": template_file_sha256,
            }
            if template_file_sha256
            else os.path.abspath(template_file)
        )

    if one_to_many:
        metadata["one_to_many"] = one_to_many

    if extra_splits:
        # When the output JSON bundles multiple splits (e.g. train+validation),
        # record the row count of each additional split here. The top-level
        # row_count field reflects the total across all splits in the file.
        metadata["extra_splits"] = extra_splits

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
