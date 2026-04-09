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
    parser.add_argument(
        "--emit-source-parquet",
        help="Optional path to also write the post-subsample, post-split source "
        "DataFrame as a parquet file. Rows are in 1:1 correspondence with the "
        "JSON entries (same seed/subsample/split). All original source columns "
        "are preserved, including the target column, which is useful for "
        "training downstream comparison models (e.g., logistic regression) on "
        "the same rows.",
    )

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

    # One-to-many expansion
    parser.add_argument(
        "--one-to-many-copies",
        type=int,
        default=None,
        help="Number of copies per row (each with a different application "
        "of --one-to-many-perturbation)",
    )
    parser.add_argument(
        "--one-to-many-perturbation",
        default=None,
        help="Perturbation to apply differently per copy (e.g., 'reorder')",
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


def render_entries(
    split_df,
    *,
    features,
    schema,
    template,
    template_type,
    perturbation_names,
    chain,
    one_to_many,
    otm_chain,
    target_column: str,
    target_threshold,
    target_mapping,
    context: str,
    context_placement: str,
    question: str,
    missing_value_handling: str,
    missing_value_text: str,
) -> list[dict]:
    """Render a DataFrame slice into JSON entries.

    Walks every row of split_df, selects features, applies perturbations,
    expands one-to-many copies, and builds the final output entries.
    """
    n_copies = one_to_many["copies"] if one_to_many else 1
    entries: list[dict] = []
    for row_idx, (_, row) in enumerate(split_df.iterrows()):
        row_dict = row.to_dict()

        feature_pairs = select_features(
            row_dict,
            features,
            schema,
            missing_value_handling=missing_value_handling,
            missing_value_text=missing_value_text,
        )

        segments = template.render_row(feature_pairs, schema)

        if perturbation_names:
            segments = apply_perturbations(segments, chain, row_idx)

        target_value = row_dict.get(target_column)
        if target_value is None:
            raise ValueError(
                f"Target column '{target_column}' not found in row {row_idx}"
            )

        for copy_idx in range(n_copies):
            if otm_chain:
                copy_segments = apply_perturbations(
                    segments, otm_chain, row_idx * n_copies + copy_idx
                )
            else:
                copy_segments = segments

            body_text = render_segments(copy_segments, template_type)

            entry = build_output_entry(
                body_text=body_text,
                context=context,
                context_placement=context_placement,
                question=question,
                target_value=target_value,
                target_threshold=target_threshold,
                target_mapping=target_mapping,
            )
            entries.append(entry)

    return entries


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
        one_to_many = condition.get("one_to_many")
    else:
        if not args.features:
            logger.error("Either --features or --conditions-file is required")
            sys.exit(1)
        features = [f.strip() for f in args.features.split(",")]
        template_type = args.template
        perturbation_names = [
            p.strip() for p in args.perturbations.split(",") if p.strip()
        ]
        # Build one_to_many config from CLI args
        if (
            args.one_to_many_copies is not None
            or args.one_to_many_perturbation is not None
        ):
            one_to_many = {
                "copies": args.one_to_many_copies,
                "perturbation": args.one_to_many_perturbation,
            }
        else:
            one_to_many = None

    # Guardrail: perturbations are not supported with llm_narrative
    if template_type == "llm_narrative" and perturbation_names:
        logger.error(
            "Perturbations cannot be applied with llm_narrative template. "
            "LLM-generated text does not produce per-feature segments."
        )
        sys.exit(1)

    # Validate one_to_many configuration
    if one_to_many:
        otm_copies = one_to_many.get("copies")
        otm_perturbation = one_to_many.get("perturbation")

        if otm_copies is None or otm_perturbation is None:
            logger.error(
                "one_to_many requires both 'copies' and 'perturbation' to be set."
            )
            sys.exit(1)

        if otm_copies < 1:
            logger.error("one_to_many.copies must be >= 1, got %d", otm_copies)
            sys.exit(1)

        from text_gen.lib.perturbations.engine import PERTURBATION_REGISTRY

        if otm_perturbation not in PERTURBATION_REGISTRY:
            available = ", ".join(sorted(PERTURBATION_REGISTRY.keys()))
            logger.error(
                "Unknown one_to_many perturbation: '%s'. Available: %s",
                otm_perturbation,
                available,
            )
            sys.exit(1)

        if otm_perturbation in perturbation_names:
            logger.error(
                "one_to_many perturbation '%s' must not also appear in "
                "top-level perturbations.",
                otm_perturbation,
            )
            sys.exit(1)

        if template_type == "llm_narrative":
            logger.error("one_to_many is not supported with llm_narrative template.")
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

    # Drop rows with a missing target value. This must happen BEFORE
    # subsampling so that (a) the subsample draws only from usable rows and
    # (b) JSON entries and the optional source-parquet sidecar stay in 1:1
    # correspondence (no row-dropping happens in the render loop below).
    if args.target_column not in df.columns:
        logger.error(
            "Target column '%s' not found in source data. "
            "Available columns (first 20): %s",
            args.target_column,
            list(df.columns)[:20],
        )
        sys.exit(1)
    pre_drop = len(df)
    df = df.dropna(subset=[args.target_column]).reset_index(drop=True)
    dropped_target = pre_drop - len(df)
    if dropped_target > 0:
        logger.info(
            "Dropped %d rows with missing target '%s' (%.2f%% of source)",
            dropped_target,
            args.target_column,
            100 * dropped_target / pre_drop,
        )

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

    # Optionally emit the source parquet for this split
    if args.emit_source_parquet:
        parquet_path = args.emit_source_parquet
        os.makedirs(os.path.dirname(os.path.abspath(parquet_path)), exist_ok=True)
        split_df.to_parquet(parquet_path, index=False)
        logger.info(
            "Wrote source parquet for split '%s' (%d rows): %s",
            args.split,
            len(split_df),
            parquet_path,
        )

    # Initialize template and perturbation chain
    template = get_template(
        template_type,
        template_file=args.template_file,
        cache_path=args.cache_path,
        style_guidance=args.style_guidance,
    )
    chain = build_perturbation_chain(perturbation_names, seed=args.seed)

    # One-to-many setup (n_copies is read inside render_entries from one_to_many)
    otm_chain = None
    if one_to_many:
        otm_chain = build_perturbation_chain(
            [one_to_many["perturbation"]], seed=args.seed
        )

    # Shared per-row rendering kwargs.
    render_kwargs = dict(
        features=features,
        schema=schema,
        template=template,
        template_type=template_type,
        perturbation_names=perturbation_names,
        chain=chain,
        one_to_many=one_to_many,
        otm_chain=otm_chain,
        target_column=args.target_column,
        target_threshold=args.target_threshold,
        target_mapping=target_mapping,
        context=args.context,
        context_placement=args.context_placement,
        question=args.question,
        missing_value_handling=args.missing_value_handling,
        missing_value_text=args.missing_value_text,
    )

    # Render the primary split
    entries = render_entries(split_df, **render_kwargs)

    # Train-evaluation bundling: when --split train is requested alongside
    # --validation-ratio, emit a single JSON file containing BOTH the train
    # and validation slices under their respective top-level keys:
    #     {"train": [...], "validation": [...]}
    extra_splits: dict[str, list[dict]] = {}
    if args.split == "train" and args.validation_ratio is not None:
        val_df = split_dataframe(
            df, args.seed, "validation", args.split_ratio, args.validation_ratio
        )
        logger.info(
            "Bundling validation split into train file: %d validation rows",
            len(val_df),
        )
        extra_splits["validation"] = render_entries(val_df, **render_kwargs)

    # Write output
    total_entries = len(entries) + sum(len(v) for v in extra_splits.values())
    logger.info("Writing %d entries to %s", total_entries, args.output)
    write_output(entries, args.output, args.split, extra_splits=extra_splits or None)

    # Build target config for metadata
    target_config = {"column": args.target_column}
    if args.target_threshold is not None:
        target_config["threshold"] = args.target_threshold
    if target_mapping is not None:
        target_config["mapping"] = target_mapping

    # Write metadata sidecar. row_count is the total entry count across all
    # splits in the file, so a bundled train+validation file reports the sum.
    write_metadata(
        output_path=args.output,
        condition_name=args.condition_name,
        split=args.split,
        seed=args.seed,
        split_ratio=args.split_ratio,
        row_count=total_entries,
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
        one_to_many=one_to_many,
        extra_splits={k: len(v) for k, v in extra_splits.items()} or None,
    )
    logger.info("Done. Output: %s", args.output)


if __name__ == "__main__":
    main()
