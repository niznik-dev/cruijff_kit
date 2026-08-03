#!/usr/bin/env python3
"""
CatBoost ML baseline for GGS books-of-life prediction tasks.

Trains CatBoost on the SAME books-of-life JSON the LLM eval consumes
({train,validation,test} -> [{input, output}]), so the AUC is directly
comparable to the LLM's. Parses the pipe-joined "KEY: val | KEY: val" input
format produced by ggs-hh-dk-synthetic/src/to_books_of_life.py.

CatBoost was the strongest tabular baseline on the past ACS-Income experiments
(beat XGBoost on every metric), and it ingests the GGS coded categoricals
natively — no label-encoding information loss.

Non-features are dropped before training: unique-per-row IDs (e.g. RESPID,
ARID) and constant columns (e.g. COUNTRY, YEAR_S). The LLM sees these in its
prompt but treats them as inert text; for tabular ML a high-cardinality ID is
actively harmful, so a fair baseline excludes them. Dropped columns are
reported.

Usage:
    python baseline.py <books_of_life.json>
    python baseline.py <books_of_life.json> --json
    python baseline.py <books_of_life.json> --id-threshold 0.5

Run in the `xgboost` conda env (has catboost):
    module load anaconda3/2025.6 && conda activate xgboost
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def parse_input(text: str) -> dict:
    """Pipe-joined 'KEY: val | KEY: val' -> {KEY: val}. Splits on the first
    ': ' per pair so coded values like '2. Female' survive intact."""
    out = {}
    for pair in text.split(" | "):
        if ": " in pair:
            key, val = pair.split(": ", 1)
            out[key.strip()] = val.strip()
    return out


def load_split(records: list) -> tuple[list[dict], np.ndarray]:
    feats = [parse_input(r["input"]) for r in records]
    y = np.array([int(r["output"]) for r in records])
    return feats, y


def classify_columns(rows: list[dict], n: int, id_threshold: float):
    """Decide, from the training rows, which columns to drop and which of the
    survivors are numeric vs categorical.

    - drop: constant (1 unique value) or near-unique (>= id_threshold * n
      distinct values -> an identifier).
    - numeric: every surviving value parses as a float; else categorical.
    """
    cols = sorted({k for r in rows for k in r})
    dropped, numeric, categorical = {}, [], []
    for c in cols:
        vals = [r.get(c, "") for r in rows]
        nuniq = len(set(vals))
        if nuniq <= 1:
            dropped[c] = "constant"
            continue
        if nuniq >= id_threshold * n:
            dropped[c] = f"id-like ({nuniq}/{n} unique)"
            continue
        is_numeric = True
        for v in vals:
            try:
                float(v)
            except ValueError:
                is_numeric = False
                break
        (numeric if is_numeric else categorical).append(c)
    return dropped, numeric, categorical


def build_matrix(rows, numeric, categorical):
    """Ordered feature matrix: numeric cols first (as float), categorical after
    (as str so CatBoost reads them as categories). Returns (X, cat_indices)."""
    order = numeric + categorical
    cat_idx = list(range(len(numeric), len(order)))
    X = []
    for r in rows:
        row = []
        for c in numeric:
            try:
                row.append(float(r.get(c, "nan")))
            except ValueError:
                row.append(float("nan"))
        for c in categorical:
            row.append(str(r.get(c, "__missing__")))
        X.append(row)
    return X, cat_idx


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("data_path", help="books-of-life JSON ({train,validation,test})")
    ap.add_argument("--json", action="store_true", help="emit a JSON summary")
    ap.add_argument(
        "--id-threshold",
        type=float,
        default=0.5,
        help="drop a column if its distinct-value fraction over the "
        "train split reaches this (default 0.5 -> identifiers)",
    )
    args = ap.parse_args()

    from catboost import CatBoostClassifier

    data = json.load(open(args.data_path))
    train_rows, y_train = load_split(data["train"])
    test_rows, y_test = load_split(data["test"])

    dropped, numeric, categorical = classify_columns(
        train_rows, len(train_rows), args.id_threshold
    )

    X_train, cat_idx = build_matrix(train_rows, numeric, categorical)
    X_test, _ = build_matrix(test_rows, numeric, categorical)

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        cat_features=cat_idx,
        verbose=False,
        train_dir=str(Path(args.data_path).parent / "catboost_info"),
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "auc": round(roc_auc_score(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }

    if args.json:
        print(
            json.dumps(
                {
                    "data_path": args.data_path,
                    "model": "CatBoost",
                    "train_samples": len(train_rows),
                    "test_samples": len(test_rows),
                    "n_features_used": len(numeric) + len(categorical),
                    "numeric_features": numeric,
                    "categorical_features": categorical,
                    "dropped_features": dropped,
                    "test_positive_rate": round(float(np.mean(y_test)), 4),
                    "metrics": metrics,
                },
                indent=2,
            )
        )
        return 0

    print("=" * 60)
    print(f"CatBoost baseline — {Path(args.data_path).name}")
    print("=" * 60)
    print(
        f"  train={len(train_rows)}  test={len(test_rows)}  "
        f"test positive rate={np.mean(y_test):.1%}"
    )
    print(f"  features used: {len(numeric)} numeric + {len(categorical)} categorical")
    print(
        f"  dropped ({len(dropped)}): "
        + ", ".join(f"{k} [{v}]" for k, v in dropped.items())
    )
    print("-" * 60)
    print(f"  AUC:               {metrics['auc']:.4f}")
    print(f"  Accuracy:          {metrics['accuracy']:.1%}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.1%}")
    print(f"  F1:                {metrics['f1']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
