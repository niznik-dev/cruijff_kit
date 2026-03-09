#!/usr/bin/env python3
"""
Convert verbose ACS data to condensed formats.

Supports all 5 ACS tasks - auto-detects task from filename or use --task flag.

Usage:
    python convert_to_condensed.py --input INPUT.json --output-dir OUTPUT_DIR [--task TASK]

Examples:
    # Auto-detect task from filename
    python convert_to_condensed.py --input acs_employment_verbose_50000_80P.json --output-dir ./output

    # Explicit task override
    python convert_to_condensed.py --input verbose.json --output-dir ./output --task ACSIncome
"""

import argparse
import json
import re
from pathlib import Path

# Task-specific questions (condensed and terse versions)
TASK_QUESTIONS = {
    "income": ("Income >$50k?", ">50k?"),
    "employment": ("Employed as civilian?", "employed?"),
    "mobility": ("Moved in last year?", "moved?"),
    "publiccoverage": ("Has public health insurance?", "pub_ins?"),
    "traveltime": ("Commute >20 min?", ">20min?"),
}


def detect_task_from_filename(filename: str) -> str:
    """Detect ACS task from filename. Returns lowercase task name."""
    filename_lower = filename.lower()
    for task in TASK_QUESTIONS:
        if task in filename_lower:
            return task
    return None


# Mapping from verbose field names to condensed keys
# Covers all fields across all ACS tasks
FIELD_MAP = {
    # Common fields
    "age": "AGE",
    "sex": "SEX",
    "race": "RACE",
    "marital status": "MARITAL",
    "highest educational attainment": "EDUCATION",
    "relationship to the reference person in the survey": "RELATIONSHIP",
    # Income-specific
    "class of worker": "WORKER_CLASS",
    "occupation": "OCCUPATION",
    "place of birth": "BIRTHPLACE",
    "usual number of hours worked per week": "HOURS_WEEK",
    # Employment-specific
    "disability status": "DISABILITY",
    "employment status of parents": "PARENT_EMP",
    "citizenship status": "CITIZENSHIP",
    "mobility status over the last year": "MOBILITY",
    "military service status": "MILITARY",
    "ancestry": "ANCESTRY",
    "nativity": "NATIVITY",
    "hearing status": "HEARING",
    "vision status": "VISION",
    "cognition status": "COGNITION",
    # Mobility/PublicCoverage/TravelTime fields
    "employment status": "EMP_STATUS",
    "commute time": "COMMUTE",
    "yearly income": "INCOME",
    "resident state": "STATE",
    "means of transportation to work": "TRANSPORT",
    "income-to-poverty ratio": "POVERTY_RATIO",
    "Public Use Microdata Area (PUMA) code": "PUMA",
    "Public Use Microdata Area (PUMA) code for the place of work": "WORK_PUMA",
}

# Condensed value mappings for ultra-terse format
TERSE_VALUES = {
    # Worker class
    "Working for a for-profit private company or organization": "priv_profit",
    "Working for a non-profit organization": "nonprofit",
    "Owner of incorporated business, professional practice, or farm": "owner_inc",
    "Owner of non-incorporated business, professional practice, or farm": "owner_noninc",
    "Working for the state government": "state_gov",
    "Working for the local government": "local_gov",
    "Working for the federal government": "fed_gov",
    "Self-employed, not incorporated": "self_emp",
    # Education
    "No schooling completed": "none",
    "Nursery school, preschool": "preschool",
    "Kindergarten": "kindergarten",
    "Grade 1": "grade1",
    "Grade 2": "grade2",
    "Grade 3": "grade3",
    "Grade 4": "grade4",
    "Grade 5": "grade5",
    "Grade 6": "grade6",
    "Grade 7": "grade7",
    "Grade 8": "grade8",
    "Grade 9": "grade9",
    "Grade 10": "grade10",
    "Grade 11": "grade11",
    "12th grade, no diploma": "grade12_nodip",
    "Regular high school diploma": "hs_diploma",
    "GED or alternative credential": "ged",
    "Some college, less than 1 year": "some_college_lt1",
    "Some college, 1 or more years, no degree": "some_college_1plus",
    "Associate's degree": "associates",
    "Bachelor's degree": "bachelors",
    "Master's degree": "masters",
    "Professional school degree": "professional",
    "Doctorate degree": "doctorate",
    # Marital status
    "Married": "married",
    "Never married": "never_married",
    "Divorced": "divorced",
    "Widowed": "widowed",
    "Separated": "separated",
    # Relationship
    "The reference person itself": "self",
    "Husband/wife": "spouse",
    "Biological son or daughter": "bio_child",
    "Adopted son or daughter": "adopted_child",
    "Stepson or stepdaughter": "stepchild",
    "Brother or sister": "sibling",
    "Father or mother": "parent",
    "Grandchild": "grandchild",
    "Parent-in-law": "parent_in_law",
    "Son-in-law or daughter-in-law": "child_in_law",
    "Other relative": "other_relative",
    "Roomer or boarder": "roomer",
    "Housemate or roommate": "roommate",
    "Unmarried partner": "partner",
    "Foster child": "foster_child",
    "Other nonrelative": "other_nonrel",
    # Sex
    "Male": "M",
    "Female": "F",
    # Race
    "White": "white",
    "Black or African American": "black",
    "Asian": "asian",
    "American Indian or Alaska Native": "native",
    "Native Hawaiian or Pacific Islander": "pacific",
    "Some other race alone (non-White)": "other",
    "Two or more races": "multiracial",
}


def parse_verbose_input(text: str) -> dict:
    """Extract field values from verbose natural language format.

    Uses generic pattern to capture any "The X is: Y." field.
    """
    fields = {}

    # Generic pattern: "The <field> is: <value>."
    # Handles both "X years old." and "X hours." endings, plus standard "X."
    pattern = r"- The ([^:]+) is: ([^.]+(?:years old|hours|\.?))"

    for match in re.finditer(pattern, text):
        field_name = match.group(1).strip()
        value = match.group(2).strip().rstrip(".")
        # Clean up "X years old" -> just the number for age
        if value.endswith("years old"):
            value = value.replace(" years old", "")
        # Clean up "X hours" -> just the number for hours
        if value.endswith("hours"):
            value = value.replace(" hours", "")
        fields[field_name] = value

    return fields


def get_short_key(field_name: str) -> str:
    """Get short key for a field, or generate one if not in FIELD_MAP."""
    if field_name in FIELD_MAP:
        return FIELD_MAP[field_name]
    # Generate a short key: uppercase, replace spaces with underscores
    return field_name.upper().replace(" ", "_")[:15]


def to_condensed_readable(fields: dict, task: str) -> str:
    """Convert to readable condensed format with full values."""
    question, _ = TASK_QUESTIONS[task]
    parts = []
    for field_name, value in fields.items():
        short_key = get_short_key(field_name)
        parts.append(f"{short_key}: {value}")

    return " | ".join(parts) + f"\n\n{question}"


def to_ultra_terse(fields: dict, task: str) -> str:
    """Convert to ultra-terse format with coded values."""
    _, terse_question = TASK_QUESTIONS[task]
    parts = []
    for field_name, value in fields.items():
        short_key = get_short_key(field_name)
        # Use terse mapping if available, otherwise keep original
        terse_value = TERSE_VALUES.get(value, value)
        # For long text fields, truncate/simplify
        if field_name == "occupation":
            terse_value = value[:25].replace(" ", "_").replace(",", "").lower()
        elif len(str(terse_value)) > 30:
            # Truncate other long values
            terse_value = str(terse_value)[:25].replace(" ", "_").lower()
        parts.append(f"{short_key}:{terse_value}")

    return "|".join(parts) + f"\n{terse_question}"


def convert_sample(sample: dict, format_type: str, task: str) -> dict:
    """Convert a single sample to the specified format."""
    fields = parse_verbose_input(sample["input"])

    if format_type == "condensed":
        new_input = to_condensed_readable(fields, task)
    elif format_type == "terse":
        new_input = to_ultra_terse(fields, task)
    else:
        raise ValueError(f"Unknown format: {format_type}")

    return {"input": new_input, "output": sample["output"]}


def convert_dataset(input_path: Path, output_path: Path, format_type: str, task: str):
    """Convert entire dataset to new format."""
    with open(input_path) as f:
        data = json.load(f)

    converted = {}
    for split_name, samples in data.items():
        converted[split_name] = [convert_sample(s, format_type, task) for s in samples]
        print(f"Converted {len(samples)} {split_name} samples")

    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Saved to {output_path}")

    # Print example
    print(f"\n--- Example ({format_type}) ---")
    print(converted["train"][0]["input"])
    print(f"Output: {converted['train'][0]['output']}")


def derive_output_names(input_path: Path, output_dir: Path, prefix: str = None):
    """
    Derive output filenames from input filename.

    If input is 'acs_income_verbose_50000_80P.json', outputs will be:
    - acs_income_condensed_50000_80P.json
    - acs_income_terse_50000_80P.json
    """
    stem = input_path.stem  # e.g., "acs_income_verbose_50000_80P"

    # Try to replace 'verbose' with format name
    if "verbose" in stem:
        condensed_name = stem.replace("verbose", "condensed") + ".json"
        terse_name = stem.replace("verbose", "terse") + ".json"
    elif prefix:
        # Use prefix if provided
        condensed_name = f"{prefix}_condensed.json"
        terse_name = f"{prefix}_terse.json"
    else:
        # Fallback: append format to stem
        condensed_name = f"{stem}_condensed.json"
        terse_name = f"{stem}_terse.json"

    return output_dir / condensed_name, output_dir / terse_name


def main():
    parser = argparse.ArgumentParser(
        description="Convert verbose ACS data to condensed and terse formats"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to verbose format JSON file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=list(TASK_QUESTIONS.keys()),
        help=f"ACS task (default: auto-detect from filename). Choices: {list(TASK_QUESTIONS.keys())}",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional prefix for output filenames (default: derive from input)",
    )

    args = parser.parse_args()

    # Validate input exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Detect or validate task
    if args.task:
        task = args.task.lower().replace("acs", "")
    else:
        task = detect_task_from_filename(args.input.name)
        if task is None:
            print(f"Error: Could not detect task from filename '{args.input.name}'")
            print(f"Use --task to specify: {list(TASK_QUESTIONS.keys())}")
            return 1
        print(f"Auto-detected task: {task}")

    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output filenames
    condensed_path, terse_path = derive_output_names(
        args.input, args.output_dir, args.prefix
    )

    # Generate both formats
    print("=" * 60)
    print(f"CONDENSED READABLE FORMAT (task: {task})")
    print("=" * 60)
    convert_dataset(args.input, condensed_path, "condensed", task)

    print("\n" + "=" * 60)
    print(f"ULTRA-TERSE FORMAT (task: {task})")
    print("=" * 60)
    convert_dataset(args.input, terse_path, "terse", task)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Condensed: {condensed_path}")
    print(f"Terse: {terse_path}")

    return 0


if __name__ == "__main__":
    exit(main())
