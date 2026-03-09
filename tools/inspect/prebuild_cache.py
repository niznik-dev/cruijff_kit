#!/usr/bin/env python3
"""
Pre-build HuggingFace datasets cache for inspect-ai evaluations.

Usage:
    python prebuild_cache.py <experiment_summary.yaml>

Pre-builds Arrow cache files on the login node so parallel SLURM eval jobs
don't race to build the same cache simultaneously.

Output:
    JSON to stdout with cache build results:
    {
        "status": "success",
        "datasets_cached": 2,
        "datasets_failed": 0,
        "paths_cached": ["/path/to/data.json"],
        "paths_failed": []
    }
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    import yaml
except ImportError:
    yaml = None


def prebuild_cache(summary_path: str) -> dict:
    """
    Pre-build HF datasets cache for all unique datasets in an experiment.

    Args:
        summary_path: Path to experiment_summary.yaml

    Returns:
        Dictionary with cache build results or error information
    """
    if yaml is None:
        return {
            "status": "error",
            "message": "PyYAML not installed. Run: pip install pyyaml",
        }

    path = Path(summary_path)
    if not path.exists():
        return {
            "status": "error",
            "message": f"File not found: {summary_path}",
        }

    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to parse YAML: {e}",
        }

    # Extract unique dataset paths
    dataset_paths = set()
    for task in config.get("evaluation", {}).get("tasks", []):
        dataset = task.get("dataset")
        if dataset:
            dataset_paths.add(dataset)

    if not dataset_paths:
        print("CACHE_PREBUILD_COMPLETE: 0 datasets cached (none specified)")
        return {
            "status": "success",
            "datasets_cached": 0,
            "datasets_failed": 0,
            "paths_cached": [],
            "paths_failed": [],
        }

    if load_dataset is None:
        return {
            "status": "error",
            "message": "datasets not installed. Run: pip install datasets",
        }

    paths_cached = []
    paths_failed = []

    for dataset_path in sorted(dataset_paths):
        if not Path(dataset_path).exists():
            print(f"CACHE_FAILED: {dataset_path} (file not found)")
            paths_failed.append(dataset_path)
            continue

        try:
            load_dataset("json", data_files=dataset_path, field="test", split="train")
            print(f"CACHE_BUILT: {dataset_path}")
            paths_cached.append(dataset_path)
        except Exception as e:
            print(f"CACHE_FAILED: {dataset_path} ({e})")
            paths_failed.append(dataset_path)

    print(f"CACHE_PREBUILD_COMPLETE: {len(paths_cached)} datasets cached")

    return {
        "status": "success",
        "datasets_cached": len(paths_cached),
        "datasets_failed": len(paths_failed),
        "paths_cached": paths_cached,
        "paths_failed": paths_failed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pre-build HuggingFace datasets cache for inspect-ai evaluations."
    )
    parser.add_argument("summary_path", help="Path to experiment_summary.yaml")
    parser.add_argument(
        "--pretty", "-p", action="store_true", help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    result = prebuild_cache(args.summary_path)

    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))

    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
