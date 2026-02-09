#!/usr/bin/env python3
"""
Extract state-specific ACS prediction data from HuggingFace.

Filters folktexts ACS datasets by state FIPS code and creates
equal-sized per-state datasets in cruijff_kit-compatible JSON format.

Usage:
    # Extract 6 focus states for PublicCoverage
    python extract_acs_by_state.py --task ACSPublicCoverage \
        --states CA TX OH GA MS MT \
        --train-size 2000 --val-size 250 --test-size 250 \
        --output-dir ../../data/green/acs/state/

    # Extract a single state
    python extract_acs_by_state.py --task ACSPublicCoverage \
        --states CA --train-size 5000

    # Extract with multiple-choice A/B format (for base model evaluation)
    python extract_acs_by_state.py --task ACSPublicCoverage \
        --states CA TX OH GA MS MT --mc \
        --output-dir ../../data/green/acs/state/mc/
"""

import json
import argparse
from pathlib import Path

from datasets import load_dataset

# Task configurations (same as extract_acs_verbose.py)
ACS_TASKS = {
    "ACSIncome": "Is this person's income above $50,000?",
    "ACSEmployment": "Is this person employed as a civilian?",
    "ACSMobility": "Did this person move in the last year?",
    "ACSPublicCoverage": "Does this person have public health insurance?",
    "ACSTravelTime": "Is this person's commute longer than 20 minutes?",
}

# Tasks that have the ST (state) column in the HuggingFace dataset
TASKS_WITH_STATE = {"ACSPublicCoverage", "ACSMobility", "ACSTravelTime"}

# State abbreviation -> FIPS code mapping
STATE_FIPS = {
    "AL": 1, "AK": 2, "AZ": 4, "AR": 5, "CA": 6, "CO": 8, "CT": 9,
    "DE": 10, "DC": 11, "FL": 12, "GA": 13, "HI": 15, "ID": 16, "IL": 17,
    "IN": 18, "IA": 19, "KS": 20, "KY": 21, "LA": 22, "ME": 23, "MD": 24,
    "MA": 25, "MI": 26, "MN": 27, "MS": 28, "MO": 29, "MT": 30, "NE": 31,
    "NV": 32, "NH": 33, "NJ": 34, "NM": 35, "NY": 36, "NC": 37, "ND": 38,
    "OH": 39, "OK": 40, "OR": 41, "PA": 42, "RI": 44, "SC": 45, "SD": 46,
    "TN": 47, "TX": 48, "UT": 49, "VT": 50, "VA": 51, "WA": 53, "WV": 54,
    "WI": 55, "WY": 56, "PR": 72,
}

# Reverse lookup: FIPS -> full state name (for filenames and metadata)
FIPS_TO_NAME = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "District of Columbia",
    12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois",
    18: "Indiana", 19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana",
    23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan",
    27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana",
    31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
    35: "New Mexico", 36: "New York", 37: "North Carolina", 38: "North Dakota",
    39: "Ohio", 40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania",
    44: "Rhode Island", 45: "South Carolina", 46: "South Dakota",
    47: "Tennessee", 48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia",
    53: "Washington", 54: "West Virginia", 55: "Wisconsin", 56: "Wyoming",
    72: "Puerto Rico",
}


def extract_by_state(
    task: str,
    states: list[str],
    output_dir: Path,
    train_size: int = 2000,
    val_size: int = 250,
    test_size: int = 250,
    random_seed: int = 42,
    mc: bool = False,
):
    """
    Extract state-filtered ACS data from HuggingFace.

    Creates one JSON file per state with equal sample sizes.
    """
    if task not in ACS_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid: {list(ACS_TASKS.keys())}")
    if task not in TASKS_WITH_STATE:
        raise ValueError(
            f"Task {task} does not have a state (ST) column. "
            f"Tasks with state: {TASKS_WITH_STATE}"
        )

    # Resolve state abbreviations to FIPS codes
    state_codes = {}
    for abbrev in states:
        abbrev_upper = abbrev.upper()
        if abbrev_upper not in STATE_FIPS:
            raise ValueError(f"Unknown state abbreviation: {abbrev}")
        state_codes[abbrev_upper] = STATE_FIPS[abbrev_upper]

    binary_question = ACS_TASKS[task]
    format_label = "mc" if mc else "verbose"

    print(f"Loading {task} from HuggingFace...")
    print(f"  Format: {format_label}")
    dataset = load_dataset("acruz/folktexts", task)

    print(f"  Train: {len(dataset['train']):,}")
    print(f"  Val:   {len(dataset['validation']):,}")
    print(f"  Test:  {len(dataset['test']):,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    task_lower = task.lower().replace("acs", "acs_")

    for abbrev, fips in state_codes.items():
        state_name = FIPS_TO_NAME[fips]
        print(f"\n{'='*60}")
        print(f"Extracting {state_name} ({abbrev}, FIPS={fips})")
        print(f"{'='*60}")

        output_data = {}
        for split_name, size in [("train", train_size), ("validation", val_size), ("test", test_size)]:
            split = dataset[split_name]
            filtered = split.filter(lambda x: x["ST"] == fips)
            available = len(filtered)

            if available < size:
                raise ValueError(
                    f"Not enough data for {state_name} in {split_name}: "
                    f"need {size}, have {available}"
                )

            sampled = filtered.shuffle(seed=random_seed).select(range(size))

            converted = []
            for example in sampled:
                if mc:
                    full_input = (
                        f"{example['instruction']}\n"
                        f"{example['description']}\n\n"
                        f"{example['choice_question_prompt']}"
                    )
                    output = example["answer_key"]  # "A" or "B"
                else:
                    full_input = (
                        f"{example['instruction']}\n"
                        f"{example['description']}\n\n"
                        f"{binary_question}"
                    )
                    output = str(example["label"])  # "0" or "1"

                converted.append({
                    "input": full_input,
                    "output": output,
                    "metadata": {
                        "state": state_name,
                        "state_abbrev": abbrev,
                        "state_fips": fips,
                    },
                })

            pos_key = "A" if mc else "1"
            pos = sum(1 for r in converted if r["output"] == pos_key)
            print(f"  {split_name}: {len(converted)} samples "
                  f"(base rate: {pos}/{len(converted)} = {pos/len(converted)*100:.1f}%)")

            output_data[split_name] = converted

        # Save
        suffix = "_mc" if mc else ""
        filename = f"{task_lower}_{abbrev.lower()}_2500{suffix}.json"
        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        total = sum(len(v) for v in output_data.values())
        print(f"  Saved: {output_path} ({total} total)")

    print(f"\nDone! {len(state_codes)} state datasets written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract state-specific ACS data from HuggingFace"
    )
    parser.add_argument(
        "--task", type=str, required=True, choices=list(ACS_TASKS.keys()),
        help="ACS task to extract",
    )
    parser.add_argument(
        "--states", type=str, nargs="+", required=True,
        help="State abbreviations (e.g., CA TX OH GA MS MT)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: data/green/acs/state/)",
    )
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=250)
    parser.add_argument("--test-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mc", action="store_true",
        help="Use multiple-choice A/B format instead of binary 0/1",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[2] / "data" / "green" / "acs" / "state"

    extract_by_state(
        task=args.task,
        states=args.states,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_seed=args.seed,
        mc=args.mc,
    )


if __name__ == "__main__":
    main()
