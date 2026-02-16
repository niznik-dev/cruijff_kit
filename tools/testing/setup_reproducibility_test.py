#!/usr/bin/env python3
"""Set up A/B directories for scaffold reproducibility testing.

Takes any experiment_summary.yaml and creates two copies (_A, _B) with
modified experiment names, directories, and output paths. This allows
scaffolding both independently and comparing outputs for reproducibility.

Usage:
    python tools/testing/setup_reproducibility_test.py <experiment_summary.yaml> [--clean]

Output (JSON to stdout):
    {"dir_a": "...", "dir_b": "...", "base_name": "..."}
"""

import argparse
import json
import os
import shutil
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "yaml_path", help="Path to experiment_summary.yaml"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing _A/_B directories before creating new ones",
    )
    args = parser.parse_args()

    yaml_path = os.path.abspath(args.yaml_path)
    if not os.path.isfile(yaml_path):
        sys.exit(f"File not found: {yaml_path}")

    # Load YAML to extract key fields
    with open(yaml_path) as f:
        raw_text = f.read()

    data = yaml.safe_load(raw_text)

    exp_name = data["experiment"]["name"]
    exp_dir = data["experiment"]["directory"]
    out_dir = data["output"]["base_directory"]

    # Determine parent directories
    exp_parent = os.path.dirname(exp_dir.rstrip("/"))
    out_parent = os.path.dirname(out_dir.rstrip("/"))

    dir_a = os.path.join(exp_parent, exp_name + "_A")
    dir_b = os.path.join(exp_parent, exp_name + "_B")

    # Clean if requested
    if args.clean:
        for d in (dir_a, dir_b):
            if os.path.exists(d):
                shutil.rmtree(d)
                print(f"Cleaned: {d}", file=sys.stderr)
        # Also clean output directories
        out_a = os.path.join(out_parent, exp_name + "_A")
        out_b = os.path.join(out_parent, exp_name + "_B")
        for d in (out_a, out_b):
            if os.path.exists(d):
                shutil.rmtree(d)
                print(f"Cleaned: {d}", file=sys.stderr)

    # Create _A and _B by string-replacing the experiment name in raw YAML.
    # The experiment name appears in: experiment.name, experiment.directory,
    # output.base_directory. Since it's a unique string, global replace is safe.
    for suffix, target_dir in [("_A", dir_a), ("_B", dir_b)]:
        new_name = exp_name + suffix
        modified = raw_text.replace(exp_name, new_name)

        os.makedirs(target_dir, exist_ok=True)
        out_path = os.path.join(target_dir, "experiment_summary.yaml")
        with open(out_path, "w") as f:
            f.write(modified)
        print(f"Created: {out_path}", file=sys.stderr)

    # Print JSON result to stdout for caller to parse
    result = {
        "dir_a": dir_a,
        "dir_b": dir_b,
        "base_name": exp_name,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
