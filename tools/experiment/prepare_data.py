"""Generate the dataset declared in experiment_summary.yaml's data.data_generation block.

Runs as the first step of scaffold-experiment. Dispatches on the ``tool:``
field under ``data.data_generation``. Currently supports:

- ``model_organism`` — cheap, deterministic sequence datasets from
  ``sanity_checks/model_organisms/``.

Expensive/flaky generators (e.g. Sarah's tabular_to_text_gen) have their own
dedicated skills and do NOT route through this tool.

One generator, one dataset per experiment. Multi-dataset support is a
tracked follow-up.

Usage::

    python -m cruijff_kit.tools.experiment.prepare_data <experiment_dir>

Exit code is 0 on success, 1 on any failure. Output is logged to
``{experiment_dir}/logs/scaffold-prepare-data.log``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from cruijff_kit.sanity_checks.model_organisms.generate import generate


LOG_NAME = "scaffold-prepare-data.log"


def _resolve_output(output_path: str, experiment_dir: Path) -> Path:
    out = Path(output_path)
    return out if out.is_absolute() else experiment_dir / out


def _equivalent_cli(params: dict, out_path: Path) -> str:
    """Build the equivalent generate.py CLI invocation for audit logging."""
    parts = [
        "python -m cruijff_kit.sanity_checks.model_organisms.generate",
        f"--input_type {params['input_type']}",
        f"--rule {params['rule']}",
        f"--k {params['k']}",
        f"--N {params['N']}",
        f"--seed {params['seed']}",
        f"--design {params['design']}",
        f"--format {params['fmt']}",
        f"--rule_kwargs '{json.dumps(params['rule_kwargs'])}'",
        f"--split {params['split']}",
        f"--ood_tests '{json.dumps(params['ood_tests'] or [])}'",
        f"--output {out_path.name}",
        f"--output_dir {out_path.parent}",
    ]
    return " ".join(parts)


def generate_model_organism(
    spec: dict, experiment_dir: Path, logger: logging.Logger
) -> Path:
    """Generate one model-organism dataset per ``spec``; return the output path."""
    required = ("input_type", "rule", "k", "N", "seed", "design", "output_path")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(
            f"data.data_generation (tool=model_organism) missing required fields: {missing}"
        )

    out_path = _resolve_output(spec["output_path"], experiment_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "input_type": spec["input_type"],
        "rule": spec["rule"],
        "k": spec["k"],
        "N": spec["N"],
        "seed": spec["seed"],
        "design": spec["design"],
        "fmt": spec.get("fmt", "spaced"),
        "rule_kwargs": spec.get("rule_kwargs") or {},
        "split": spec.get("split", 0.8),
        "ood_tests": spec.get("ood_tests") or None,
    }

    dataset = generate(**params)

    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    name = spec.get("name", "<unnamed>")
    n_train = len(dataset["train"])
    n_val = len(dataset["validation"])
    logger.info(f"GENERATED: {name} ({n_train} train + {n_val} val rows) -> {out_path}")
    logger.info(f"  equivalent CLI: {_equivalent_cli(params, out_path)}")
    return out_path


# Registered generators, keyed by `tool` field.
_GENERATORS = {
    "model_organism": generate_model_organism,
}


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"prepare_data:{log_path}")
    logger.setLevel(logging.INFO)
    # Clean any handlers left over from a previous test run against the same path.
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    )
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(stream_handler)

    return logger


def prepare(experiment_dir: Path) -> int:
    """Generate the dataset declared in experiment_summary.yaml, if any.

    Returns 0 on success (including the "no data_generation block" case),
    1 on any failure.
    """
    yaml_path = experiment_dir / "experiment_summary.yaml"
    if not yaml_path.exists():
        print(
            f"ERROR: experiment_summary.yaml not found at {yaml_path}",
            file=sys.stderr,
        )
        return 1

    with open(yaml_path) as f:
        config = yaml.safe_load(f) or {}

    data_gen = (config.get("data") or {}).get("data_generation")
    if not data_gen:
        print("No data.data_generation block; nothing to prepare.")
        return 0

    tool = data_gen.get("tool")
    if not tool:
        print(
            "ERROR: data.data_generation is missing required 'tool' field.",
            file=sys.stderr,
        )
        return 1
    if tool not in _GENERATORS:
        print(
            f"ERROR: data.data_generation tool={tool!r} is not supported. "
            f"Known: {sorted(_GENERATORS)}",
            file=sys.stderr,
        )
        return 1

    log_path = experiment_dir / "logs" / LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(log_path)
    logger.info(f"START: prepare_data for {experiment_dir}")

    try:
        _GENERATORS[tool](data_gen, experiment_dir, logger)
    except Exception as exc:
        logger.error(
            f"FAILED: tool={tool} ({data_gen.get('name', '<unnamed>')}): {exc}"
        )
        return 1

    logger.info("END: dataset generated successfully")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the dataset declared in experiment_summary.yaml's "
        "data.data_generation block."
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to experiment directory containing experiment_summary.yaml",
    )
    args = parser.parse_args()
    raise SystemExit(prepare(args.experiment_dir))


if __name__ == "__main__":
    main()
