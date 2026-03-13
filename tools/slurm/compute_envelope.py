"""Envelope wrapper for compute_metrics.json.

Wraps the raw job metrics list with experiment-level metadata,
producing the envelope format consumed by design-experiment for
compute estimation.

The envelope schema adds context (model, dataset size, epochs, etc.)
that allows downstream scaling logic to compare prior runs against
new experiment parameters.
"""

import json
from pathlib import Path
import yaml


def build_envelope(
    jobs: list[dict],
    experiment_summary_path: str | Path,
) -> dict:
    """Wrap a job metrics list with experiment metadata.

    Reads experiment_summary.yaml to extract the metadata fields
    needed for downstream compute estimation (model name, dataset
    size, epochs, batch size, date).

    Args:
        jobs: List of job metric dicts (same format as
            ``format_compute_table``).
        experiment_summary_path: Path to experiment_summary.yaml.

    Returns:
        Envelope dict with ``experiment_name``, ``model``,
        ``dataset_size``, ``epochs``, ``batch_size``, ``date``,
        and ``jobs`` keys.

    Raises:
        FileNotFoundError: If experiment_summary.yaml does not exist.
        KeyError: If required fields are missing from the YAML.
    """
    path = Path(experiment_summary_path)
    with open(path) as f:
        config = yaml.safe_load(f)

    experiment = config["experiment"]
    models = config["models"]
    data = config["data"]
    controls = config["controls"]

    # Primary model is the first base model listed
    base_models = models.get("base", [])
    model_name = base_models[0]["name"] if base_models else None

    return {
        "experiment_name": experiment["name"],
        "model": model_name,
        "dataset_size": data["training"]["splits"]["train"],
        "epochs": controls["epochs"],
        "batch_size": controls["batch_size"],
        "date": experiment.get("date", None),
        "jobs": jobs,
    }


def save_envelope(
    envelope: dict,
    output_path: str | Path,
) -> Path:
    """Write envelope to a JSON file.

    Args:
        envelope: Envelope dict from ``build_envelope``.
        output_path: Destination file path.

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(envelope, f, indent=2, default=str)
    return output_path


def load_envelope(path: str | Path) -> dict:
    """Load and validate a compute_metrics.json envelope.

    Args:
        path: Path to compute_metrics.json.

    Returns:
        Parsed envelope dict.

    Raises:
        ValueError: If the file contains a bare list (old format)
            instead of the envelope dict.
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        raise ValueError(
            f"{path} contains a bare job list (old format). "
            "Re-run analyze-experiment to generate the envelope format."
        )

    return data
