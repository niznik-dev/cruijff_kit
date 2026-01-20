#!/usr/bin/env python3
"""
Quick summary of binary classification results from inspect-ai .eval files.

Like summary() in R - just a quick glance at the confusion matrix and key metrics.
Not a robust analysis tool, just a sanity check.

Metrics computed:
- Accuracy: (TP + TN) / n
- Balanced Accuracy: (Recall_0 + Recall_1) / 2 - robust to class imbalance
- F1 Score: harmonic mean of precision and recall - penalizes imbalance
- Precision(1), Recall(1), Recall(0): per-class metrics

Inspect-ai stores logs as zip archives with samples/*.json containing results.

Usage:
    python summary_binary.py /path/to/log.eval
    python summary_binary.py /path/to/logs/  # all .eval files in directory
    python summary_binary.py /path/to/log.eval --json  # machine-readable output
"""

import argparse
import json
import zipfile
from pathlib import Path
from collections import defaultdict


def load_samples(path: Path) -> list:
    """Load samples from an inspect-ai .eval file (zip archive)."""
    samples = []
    with zipfile.ZipFile(path, 'r') as z:
        for name in z.namelist():
            if name.startswith('samples/') and name.endswith('.json'):
                with z.open(name) as f:
                    samples.append(json.load(f))
    return samples


def get_prediction(sample: dict) -> str:
    """Extract model's prediction from sample (last assistant message)."""
    for msg in reversed(sample.get('messages', [])):
        if msg.get('role') == 'assistant':
            return msg.get('content', '').strip()
    return ''


def compute_metrics(path: Path) -> dict:
    """Compute binary classification metrics from an eval file.

    Returns dict with metrics or error status.
    """
    samples = load_samples(path)
    if not samples:
        return {"status": "error", "message": f"No samples in {path}", "path": str(path)}

    # Build confusion matrix for binary (0/1) classification
    matrix = {'0': {'0': 0, '1': 0, 'other': 0},
              '1': {'0': 0, '1': 0, 'other': 0}}

    for s in samples:
        actual = s['target']
        pred = get_prediction(s)
        if pred not in ('0', '1'):
            pred = 'other'
        if actual in matrix:
            matrix[actual][pred] += 1

    # Compute metrics
    tp = matrix['1']['1']
    tn = matrix['0']['0']
    fp = matrix['0']['1']
    fn = matrix['1']['0']
    other = matrix['0']['other'] + matrix['1']['other']

    accuracy = (tp + tn) / len(samples) if samples else 0
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall_0 + recall_1) / 2
    f1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    return {
        "status": "success",
        "path": str(path),
        "samples": len(samples),
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "f1": round(f1, 4),
        "precision_1": round(precision_1, 4),
        "recall_1": round(recall_1, 4),
        "recall_0": round(recall_0, 4),
        "confusion_matrix": {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "other_0": matrix['0']['other'],
            "other_1": matrix['1']['other'],
            "other": other
        }
    }


def print_summary(path: Path) -> dict:
    """Print human-readable summary and return metrics dict."""
    result = compute_metrics(path)

    if result["status"] == "error":
        print(result["message"])
        return result

    matrix = result["confusion_matrix"]

    print(f"\n{'='*60}")
    print(f"{path.name}")
    print(f"{'='*60}")
    print(f"n = {result['samples']}")

    print(f"\n                    Predicted")
    print(f"                 0      1      other")
    print(f"           +------+------+------+")
    print(f"Actual  0  | {matrix['tn']:4} | {matrix['fp']:4} | {result['confusion_matrix'].get('other_0', 0):4} |")
    print(f"        1  | {matrix['fn']:4} | {matrix['tp']:4} | {result['confusion_matrix'].get('other_1', 0):4} |")
    print(f"           +------+------+------+")

    print(f"\nAccuracy:          {result['accuracy']:.1%}")
    print(f"Balanced Accuracy: {result['balanced_accuracy']:.1%}")
    print(f"F1 Score:          {result['f1']:.1%}")
    print(f"Precision(1):      {result['precision_1']:.1%}")
    print(f"Recall(1):         {result['recall_1']:.1%}")
    print(f"Recall(0):         {result['recall_0']:.1%}")

    # Prediction distribution
    total_0 = matrix['tn'] + matrix['fn']
    total_1 = matrix['fp'] + matrix['tp']
    other = matrix['other']
    print(f"\nPredicted 0: {total_0} ({total_0/result['samples']*100:.1f}%)")
    print(f"Predicted 1: {total_1} ({total_1/result['samples']*100:.1f}%)")
    if other > 0:
        print(f"Other:       {other} ({other/result['samples']*100:.1f}%)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Quick summary of binary classification eval results')
    parser.add_argument('path', type=Path, help='.eval file or directory')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output JSON instead of human-readable format')
    args = parser.parse_args()

    if args.json:
        # JSON output mode - single file only
        if args.path.is_dir():
            results = []
            for f in sorted(args.path.glob('**/*.eval')):
                results.append(compute_metrics(f))
            print(json.dumps(results, indent=2))
        else:
            result = compute_metrics(args.path)
            print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        if args.path.is_dir():
            for f in sorted(args.path.glob('**/*.eval')):
                print_summary(f)
        else:
            print_summary(args.path)


if __name__ == '__main__':
    main()
