#!/usr/bin/env python3
"""
Parse an inspect-ai evaluation log file and extract key metrics.

Usage:
    python parse_eval_log.py <path_to_eval_file>

Output:
    JSON to stdout with extracted metrics:
    {
        "status": "success",
        "task": "capitalization",
        "accuracy": 0.85,
        "samples": 100,
        "scorer": "exact_match",
        "model": "meta-llama/Llama-3.2-1B-Instruct"
    }

    On error:
    {
        "status": "error",
        "message": "Error description",
        "path": "/path/to/file.eval"
    }
"""

import argparse
import json
import sys
from pathlib import Path


def parse_eval_log(eval_path: str) -> dict:
    """
    Parse an inspect-ai .eval log file and extract key metrics.

    Args:
        eval_path: Path to the .eval file

    Returns:
        Dictionary with extracted metrics or error information
    """
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        return {
            "status": "error",
            "message": "inspect_ai not installed. Run: pip install inspect-ai",
            "path": eval_path,
        }

    path = Path(eval_path)
    if not path.exists():
        return {
            "status": "error",
            "message": f"File not found: {eval_path}",
            "path": eval_path,
        }

    try:
        log = read_eval_log(str(path))

        # Extract task name from the log
        task_name = log.eval.task if hasattr(log.eval, "task") else "unknown"

        # Extract model name
        model_name = log.eval.model if hasattr(log.eval, "model") else "unknown"

        # Get sample count
        samples = len(log.samples) if log.samples else 0

        # Extract metrics from results
        # The structure is: log.results.scores[i].metrics
        result = {
            "status": "success",
            "task": task_name,
            "model": model_name,
            "samples": samples,
            "path": eval_path,
            "metrics": {},
        }

        if log.results and log.results.scores:
            # Get the first scorer (primary)
            primary_score = log.results.scores[0]
            result["scorer"] = primary_score.name

            # Extract all metrics from the primary scorer
            if primary_score.metrics:
                for metric_name, metric_value in primary_score.metrics.items():
                    # metric_value is a Metric object with a .value attribute
                    if hasattr(metric_value, "value"):
                        result["metrics"][metric_name] = metric_value.value
                    else:
                        result["metrics"][metric_name] = metric_value

            # For convenience, pull accuracy to top level if it exists
            if "accuracy" in result["metrics"]:
                result["accuracy"] = result["metrics"]["accuracy"]

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "path": eval_path,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Parse an inspect-ai evaluation log file and extract key metrics."
    )
    parser.add_argument("eval_path", help="Path to the .eval file to parse")
    parser.add_argument(
        "--pretty", "-p", action="store_true", help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    result = parse_eval_log(args.eval_path)

    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))

    # Exit with error code if parsing failed
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
