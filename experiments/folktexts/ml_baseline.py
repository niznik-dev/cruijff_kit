#!/usr/bin/env python3
"""
ML baseline for ACS prediction tasks using VERBOSE format data.

Trains CatBoost and XGBoost on the same data used for LLM evaluation.
Parses the verbose natural language format.

Usage:
    python ml_baseline_verbose.py <data_path>
    python ml_baseline_verbose.py <data_path> --no-catboost   # XGBoost only
    python ml_baseline_verbose.py <data_path> --no-xgboost    # CatBoost only
    python ml_baseline_verbose.py <data_path> --json          # Output JSON summary

Examples:
    python ml_baseline_verbose.py data/green/acs/acs_income_verbose_1000_80P.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder


# Map verbose field names to short names
FIELD_MAPPING = {
    "age": "AGE",
    "class of worker": "WORKER_CLASS",
    "highest educational attainment": "EDUCATION",
    "marital status": "MARITAL",
    "occupation": "OCCUPATION",
    "place of birth": "BIRTHPLACE",
    "relationship to the reference person": "RELATIONSHIP",
    "usual number of hours worked per week": "HOURS_WEEK",
    "sex": "SEX",
    "race": "RACE",
    "employment status": "EMPLOYMENT",
    "travel time to work": "COMMUTE",
    "means of transportation to work": "TRANSPORT",
    "health insurance coverage": "HEALTH_COVERAGE",
}

# Features that are numeric
NUMERIC_FEATURES = {"AGE", "HOURS_WEEK"}


def parse_verbose_example(text: str) -> dict:
    """Parse verbose format text into feature dict."""
    features = {}

    # Find all "- The X is: Y" patterns
    pattern = r'- The ([^:]+) is: ([^\n]+)'
    matches = re.findall(pattern, text)

    for field_name, value in matches:
        field_name = field_name.strip().lower()
        value = value.strip().rstrip('.')

        # Map to short name
        short_name = FIELD_MAPPING.get(field_name, field_name.upper().replace(' ', '_'))

        # Parse numeric values
        if short_name == "AGE":
            # "37 years old" -> 37
            match = re.search(r'(\d+)', value)
            if match:
                value = match.group(1)
        elif short_name == "HOURS_WEEK":
            # "45 hours" -> 45
            match = re.search(r'(\d+)', value)
            if match:
                value = match.group(1)
        elif short_name == "COMMUTE":
            # "25 minutes" -> 25
            match = re.search(r'(\d+)', value)
            if match:
                value = match.group(1)

        features[short_name] = value

    return features


def detect_feature_schema(data: dict) -> tuple[list[str], list[str]]:
    """Detect feature columns and their types from the data."""
    # Get first example to detect columns
    first_example = data["train"][0]
    features = parse_verbose_example(first_example["input"])

    numeric_cols = []
    categorical_cols = []

    for key in sorted(features.keys()):
        if key in NUMERIC_FEATURES:
            numeric_cols.append(key)
        else:
            categorical_cols.append(key)

    return numeric_cols, categorical_cols


def load_data(data_path: str):
    """Load JSON data and parse into raw features."""
    with open(data_path) as f:
        data = json.load(f)

    # Detect schema
    numeric_cols, categorical_cols = detect_feature_schema(data)

    # Parse all examples
    train_raw = [parse_verbose_example(ex["input"]) for ex in data["train"]]
    test_raw = [parse_verbose_example(ex["input"]) for ex in data["test"]]

    y_train = [int(ex["output"]) for ex in data["train"]]
    y_test = [int(ex["output"]) for ex in data["test"]]

    return train_raw, test_raw, y_train, y_test, numeric_cols, categorical_cols


def prepare_catboost_data(train_raw, test_raw, numeric_cols, categorical_cols):
    """Prepare data for CatBoost (handles categoricals natively)."""
    all_cols = numeric_cols + categorical_cols

    def to_feature_list(raw_data):
        X = []
        for ex in raw_data:
            row = []
            for col in numeric_cols:
                val = ex.get(col, "0")
                try:
                    row.append(int(val))
                except ValueError:
                    row.append(0)
            for col in categorical_cols:
                row.append(ex.get(col, ""))
            X.append(row)
        return X

    X_train = to_feature_list(train_raw)
    X_test = to_feature_list(test_raw)
    cat_indices = list(range(len(numeric_cols), len(all_cols)))

    return X_train, X_test, cat_indices


def prepare_xgboost_data(train_raw, test_raw, numeric_cols, categorical_cols):
    """Prepare data for XGBoost (needs label encoding)."""
    encoders = {}
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        all_values = [ex.get(col, "") for ex in train_raw + test_raw]
        encoders[col].fit(all_values)

    def to_numeric_array(raw_data):
        X = []
        for ex in raw_data:
            row = []
            for col in numeric_cols:
                val = ex.get(col, "0")
                try:
                    row.append(int(val))
                except ValueError:
                    row.append(0)
            for col in categorical_cols:
                row.append(encoders[col].transform([ex.get(col, "")])[0])
            X.append(row)
        return np.array(X)

    X_train = to_numeric_array(train_raw)
    X_test = to_numeric_array(test_raw)

    return X_train, X_test


def train_catboost(X_train, y_train, cat_indices):
    """Train CatBoost classifier."""
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        cat_features=cat_indices,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost classifier."""
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def compute_metrics(y_test, y_pred):
    """Compute all metrics and return as dict."""
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": cm,
    }


def print_metrics(name: str, metrics: dict):
    """Print metrics for a single model."""
    print(f"\n{name}:")
    print(f"  Accuracy:          {metrics['accuracy']:.1%}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.1%}")
    print(f"  Precision:         {metrics['precision']:.1%}")
    print(f"  Recall:            {metrics['recall']:.1%}")
    print(f"  F1:                {metrics['f1']:.2f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix:  TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Train ML baselines on ACS data (verbose format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("data_path", help="Path to JSON data file")
    parser.add_argument("--no-catboost", action="store_true", help="Skip CatBoost")
    parser.add_argument("--no-xgboost", action="store_true", help="Skip XGBoost")
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    args = parser.parse_args()

    run_catboost = not args.no_catboost
    run_xgboost = not args.no_xgboost

    if not run_catboost and not run_xgboost:
        print("Error: Cannot skip both models", file=sys.stderr)
        return 1

    # Infer task/sample info from filename
    filename = Path(args.data_path).stem
    parts = filename.split("_")
    task_name = parts[1] if len(parts) > 1 else "unknown"

    if not args.json:
        print("=" * 60)
        print(f"ML Baselines: ACS {task_name.title()} Prediction (Verbose Format)")
        print("=" * 60)

    # Load data
    if not args.json:
        print("\nLoading data...")
    train_raw, test_raw, y_train, y_test, numeric_cols, categorical_cols = load_data(args.data_path)

    if not args.json:
        print(f"  Train: {len(train_raw)} samples")
        print(f"  Test:  {len(test_raw)} samples")
        print(f"  Numeric features ({len(numeric_cols)}): {', '.join(numeric_cols)}")
        print(f"  Categorical features ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}")

    results = {}

    # CatBoost
    if run_catboost:
        if not args.json:
            print("\nTraining CatBoost...")
        X_train_cb, X_test_cb, cat_indices = prepare_catboost_data(
            train_raw, test_raw, numeric_cols, categorical_cols
        )
        model_cb = train_catboost(X_train_cb, y_train, cat_indices)
        y_pred_cb = model_cb.predict(X_test_cb).flatten()
        results["CatBoost"] = compute_metrics(y_test, y_pred_cb)

    # XGBoost
    if run_xgboost:
        if not args.json:
            print("Training XGBoost...")
        X_train_xgb, X_test_xgb = prepare_xgboost_data(
            train_raw, test_raw, numeric_cols, categorical_cols
        )
        model_xgb = train_xgboost(X_train_xgb, np.array(y_train))
        y_pred_xgb = model_xgb.predict(X_test_xgb)
        results["XGBoost"] = compute_metrics(y_test, y_pred_xgb)

    # JSON output mode
    if args.json:
        output = {
            "data_path": args.data_path,
            "train_samples": len(train_raw),
            "test_samples": len(test_raw),
            "results": {}
        }
        for name, metrics in results.items():
            output["results"][name] = {
                "accuracy": round(metrics["accuracy"], 4),
                "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
                "f1": round(metrics["f1"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
            }
        print(json.dumps(output, indent=2))
        return 0

    # Human-readable output
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for name, metrics in results.items():
        print_metrics(name, metrics)

    # Comparison table if both ran
    if len(results) == 2:
        print("\n" + "-" * 40)
        print("COMPARISON")
        print("-" * 40)
        print(f"{'Metric':<20} {'CatBoost':>10} {'XGBoost':>10}")
        print("-" * 40)
        print(f"{'Accuracy':<20} {results['CatBoost']['accuracy']:>10.1%} {results['XGBoost']['accuracy']:>10.1%}")
        print(f"{'Balanced Accuracy':<20} {results['CatBoost']['balanced_accuracy']:>10.1%} {results['XGBoost']['balanced_accuracy']:>10.1%}")
        print(f"{'F1':<20} {results['CatBoost']['f1']:>10.2f} {results['XGBoost']['f1']:>10.2f}")

    # Class distribution
    y_test_arr = np.array(y_test)
    print(f"\nClass distribution (test):")
    print(f"  Class 0: {sum(y_test_arr == 0)} ({sum(y_test_arr == 0)/len(y_test_arr):.1%})")
    print(f"  Class 1: {sum(y_test_arr == 1)} ({sum(y_test_arr == 1)/len(y_test_arr):.1%})")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
