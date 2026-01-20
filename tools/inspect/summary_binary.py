#!/usr/bin/env python3
"""
Quick summary of binary classification results from inspect-ai .eval files.

Like summary() in R - just a quick glance at the confusion matrix and key metrics.
Not a robust analysis tool, just a sanity check.

Inspect-ai stores logs as zip archives with samples/*.json containing results.

Usage:
    python summary_binary.py /path/to/log.eval
    python summary_binary.py /path/to/logs/  # all .eval files in directory
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


def summary(path: Path) -> None:
    """Print quick confusion matrix and metrics for binary classification."""
    samples = load_samples(path)
    if not samples:
        print(f"No samples in {path}")
        return

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

    # Print it
    print(f"\n{'='*60}")
    print(f"{path.name}")
    print(f"{'='*60}")
    print(f"n = {len(samples)}")

    print(f"\n                    Predicted")
    print(f"                 0      1      other")
    print(f"           +------+------+------+")
    print(f"Actual  0  | {matrix['0']['0']:4} | {matrix['0']['1']:4} | {matrix['0']['other']:4} |")
    print(f"        1  | {matrix['1']['0']:4} | {matrix['1']['1']:4} | {matrix['1']['other']:4} |")
    print(f"           +------+------+------+")

    # Quick metrics
    tp = matrix['1']['1']
    tn = matrix['0']['0']
    fp = matrix['0']['1']
    fn = matrix['1']['0']
    other = matrix['0']['other'] + matrix['1']['other']

    accuracy = (tp + tn) / len(samples) if samples else 0
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\nAccuracy:      {accuracy:.1%}")
    print(f"Precision(1):  {precision_1:.1%}")
    print(f"Recall(1):     {recall_1:.1%}")

    # Prediction distribution
    total_0 = matrix['0']['0'] + matrix['1']['0']
    total_1 = matrix['0']['1'] + matrix['1']['1']
    print(f"\nPredicted 0: {total_0} ({total_0/len(samples)*100:.1f}%)")
    print(f"Predicted 1: {total_1} ({total_1/len(samples)*100:.1f}%)")
    if other > 0:
        print(f"Other:       {other} ({other/len(samples)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Quick summary of binary classification eval results')
    parser.add_argument('path', type=Path, help='.eval file or directory')
    args = parser.parse_args()

    if args.path.is_dir():
        for f in sorted(args.path.glob('**/*.eval')):
            summary(f)
    else:
        summary(args.path)


if __name__ == '__main__':
    main()
