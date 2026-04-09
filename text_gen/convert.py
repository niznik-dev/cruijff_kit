"""CLI entrypoint for text_gen.

Reads a source tabular file, schema, and condition configuration,
then produces a single output JSON file for one condition and one split.

Usage:
    python -m text_gen.convert \
        --source /path/to/data.csv \
        --schema /path/to/schema.yaml \
        --condition-name dict_full \
        --features AGEP,COW,SCHL \
        --template dictionary \
        --target-column PINCP \
        --target-threshold 50000 \
        --context "The following data corresponds to..." \
        --context-placement preamble \
        --question "Is this person's income above $50,000?" \
        --split train \
        --split-ratio 0.8 \
        --seed 42 \
        --output /path/to/output.json
"""

import argparse
import json
import logging
import os
import random
import sys

import yaml

from text_gen.lib.features import select_features, validate_features
from text_gen.lib.output import build_output_entry, write_metadata, write_output
from text_gen.lib.perturbations.engine import (
    apply_perturbations,
    build_perturbation_chain,
)
from text_gen.lib.readers import read_tabular
from text_gen.lib.schema import Schema
from text_gen.lib.segments import render_segments
from text_gen.lib.templates import get_template

logger = logging.getLogger("text_gen")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert tabular data to textual representations for LLM experiments."
    )

    # Source data
    parser.add_argument("--source", required=True, help="Path to source tabular file")
    parser.add_argument("--schema", required=True, help="Path to schema YAML file")

    # Condition configuration (direct CLI args)
    parser.add_argument("--condition-name", required=True, help="Condition identifier")
    parser.add_argument(
        "--features",
        help="Comma-separated list of feature column keys",
    )
    parser.add_argument(
        "--template",
        default="dictionary",
        choices=["dictionary", "narrative", "llm_narrative"],
        help="Template type (default: dictionary)",
    )
    parser.add_argument(
        "--template-file",
        help="Path to Jinja2 template file (narrative mode only)",
    )
    parser.add_argument(
        "--perturbations",
        default="",
        help="Comma-separated list of perturbation types",
    )

    # Condition configuration (from file, alternative to CLI)
    parser.add_argument(
        "--conditions-file",
        help="Path to conditions YAML file (alternative to --features/--template/--perturbations)",
    )

    # Target
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument(
        "--target-threshold",
        type=float,
        help="Threshold for binary classification",
    )
    parser.add_argument(
        "--target-mapping",
        help="JSON string mapping target values to labels",
    )

    # Context
    parser.add_argument("--context", default="", help="Context/preamble text")
    parser.add_argument(
        "--context-placement",
        default="preamble",
        choices=["preamble", "system_prompt"],
        help="Where to place context (default: preamble)",
    )
    parser.add_argument(
        "--question", default="", help="Question text appended to input"
    )

    # Split
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "validation", "test"],
        help="Which split to generate",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for train split (default: 0.8)",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=None,
        help="Fraction of data for validation split. When provided, "
        "enables three-way split (train/validation/test).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--subsampling-ratio",
        type=float,
        default=None,
        help="Fraction of source data to use (e.g., 0.33 for 33%%). "
        "Applied before splitting. Uses the seed for deterministic sampling.",
    )

    # Output
    parser.add_argument("--output", required=True, help="Output JSON file path")

    # Missing value handling
    parser.add_argument(
        "--missing-value-handling",
        default="skip",
        choices=["skip", "include"],
        help="How to handle missing/NaN feature values: "
        "'skip' omits them (default), 'include' represents them with --missing-value-text",
    )
    parser.add_argument(
        "--missing-value-text",
        default="missing",
        help="Text to use for missing values when --missing-value-handling is 'include' "
        "(default: 'missing')",
    )

    # LLM narrative options
    parser.add_argument(
        "--cache-path",
        help="Path to LLM response cache file (llm_narrative only)",
    )
    parser.add_argument(
        "--style-guidance",
        help="Style instructions for LLM narrative generation (llm_narrative only)",
    )

    return parser.parse_args(argv)


def split_dataframe(
    df, seed: int, split: str, split_ratio: float, validation_ratio: float | None
):
    """Deterministically split a DataFrame by shuffling row indices.

    Returns the rows for the requested split.
    """
    indices = list(range(len(df)))
    random.Random(seed).shuffle(indices)

    n = len(indices)
    train_end = int(n * split_ratio)

    if validation_ratio is not None:
        val_end = train_end + int(n * validation_ratio)
        if split == "train":
            selected = indices[:train_end]
        elif split == "validation":
            selected = indices[train_end:val_end]
        else:  # test
            selected = indices[val_end:]
    else:
        if split == "train":
            selected = indices[:train_end]
        elif split == "validation":
            raise ValueError(
                "Validation split requested but --validation-ratio not provided. "
                "Use --validation-ratio to enable three-way split."
            )
        else:  # test
            selected = indices[train_end:]

    return df.iloc[selected].reset_index(drop=True)


def load_condition_from_file(conditions_file: str, condition_name: str) -> dict:
    """Load a single condition's config from a conditions YAML file."""
    with open(conditions_file) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "conditions" not in raw:
        raise ValueError(
            f"Conditions file {conditions_file} must have a top-level "
            f"'conditions:' key."
        )

    conditions = raw["conditions"]
    if condition_name not in conditions:
        available = list(conditions.keys())
        raise ValueError(
            f"Condition '{condition_name}' not found in {conditions_file}. "
            f"Available: {available}"
        )

    return conditions[condition_name]


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Resolve condition configuration
    if args.conditions_file:
        condition = load_condition_from_file(args.conditions_file, args.condition_name)
        features = condition.get("features", [])
        template_type = condition.get("template", "dictionary")
        perturbation_names = condition.get("perturbations", [])
    else:
        if not args.features:
            logger.error("Either --features or --conditions-file is required")
            sys.exit(1)
        features = [f.strip() for f in args.features.split(",")]
        template_type = args.template
        perturbation_names = [
            p.strip() for p in args.perturbations.split(",") if p.strip()
        ]

    # Guardrail: perturbations are not supported with llm_narrative
    if template_type == "llm_narrative" and perturbation_names:
        logger.error(
            "Perturbations cannot be applied with llm_narrative template. "
            "LLM-generated text does not produce per-feature segments."
        )
        sys.exit(1)

    # Parse target mapping if provided
    target_mapping = None
    if args.target_mapping:
        target_mapping = json.loads(args.target_mapping)

    # Load source data
    logger.info("Reading source data: %s", args.source)
    df = read_tabular(args.source)
    source_rows_total = len(df)
    logger.info("Loaded %d rows", source_rows_total)

    # Subsample if requested
    if args.subsampling_ratio is not None:
        if not 0 < args.subsampling_ratio <= 1:
            logger.error(
                "--subsampling-ratio must be between 0 (exclusive) and 1 (inclusive)"
            )
            sys.exit(1)
        n = int(len(df) * args.subsampling_ratio)
        df = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
        logger.info(
            "Subsampled to %d rows (%.0f%% of %d, seed=%d)",
            len(df),
            args.subsampling_ratio * 100,
            source_rows_total,
            args.seed,
        )

    # Load schema
    logger.info("Loading schema: %s", args.schema)
    schema = Schema.from_yaml(args.schema)

    # Validate features
    warnings = validate_features(features, schema, args.target_column)
    for w in warnings:
        logger.warning(w)

    # Split data
    split_df = split_dataframe(
        df, args.seed, args.split, args.split_ratio, args.validation_ratio
    )
    logger.info("Split '%s': %d rows", args.split, len(split_df))

    # Initialize template and perturbation chain
    template = get_template(
        template_type,
        template_file=args.template_file,
        cache_path=args.cache_path,
        style_guidance=args.style_guidance,
    )
    chain = build_perturbation_chain(perturbation_names, seed=args.seed)

    # Process each row
    entries = []
    for row_idx, (_, row) in enumerate(split_df.iterrows()):
        row_dict = row.to_dict()

        # Select features
        feature_pairs = select_features(
            row_dict,
            features,
            schema,
            missing_value_handling=args.missing_value_handling,
            missing_value_text=args.missing_value_text,
        )

        # Render to segments via template
        segments = template.render_row(feature_pairs, schema)

        # Apply perturbations
        if perturbation_names:
            segments = apply_perturbations(segments, chain, row_idx)

        # Render segments to text
        body_text = render_segments(segments, template_type)

        # Get target value
        target_value = row_dict.get(args.target_column)
        if target_value is None:
            raise ValueError(
                f"Target column '{args.target_column}' not found in row {row_idx}"
            )

        # Build output entry
        entry = build_output_entry(
            body_text=body_text,
            context=args.context,
            context_placement=args.context_placement,
            question=args.question,
            target_value=target_value,
            target_threshold=args.target_threshold,
            target_mapping=target_mapping,
        )
        entries.append(entry)

    # Write output
    logger.info("Writing %d entries to %s", len(entries), args.output)
    write_output(entries, args.output, args.split)

    # Build target config for metadata
    target_config = {"column": args.target_column}
    if args.target_threshold is not None:
        target_config["threshold"] = args.target_threshold
    if target_mapping is not None:
        target_config["mapping"] = target_mapping

    # Write metadata sidecar
    write_metadata(
        output_path=args.output,
        condition_name=args.condition_name,
        split=args.split,
        seed=args.seed,
        split_ratio=args.split_ratio,
        row_count=len(entries),
        source_path=os.path.abspath(args.source),
        source_rows_total=source_rows_total,
        schema_path=os.path.abspath(args.schema),
        features=features,
        template=template_type,
        perturbations=perturbation_names,
        target_config=target_config,
        context=args.context,
        context_placement=args.context_placement,
        question=args.question,
        template_file=args.template_file,
    )
    logger.info("Done. Output: %s", args.output)


if __name__ == "__main__":
    main()
