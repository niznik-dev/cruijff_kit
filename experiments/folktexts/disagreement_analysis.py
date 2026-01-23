#!/usr/bin/env python3
"""
Disagreement Analysis: LLM vs ML on ACS Traveltime

Identifies cases where LLM and CatBoost disagree, then analyzes
what features distinguish these disagreement groups.

Usage:
    python disagreement_analysis.py

Outputs:
    - disagreement_report.md: Human-readable analysis
    - disagreement_data.csv: Full data for further analysis
"""

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================
# Configuration
# ============================================================================

# Paths (hardcoded for this analysis)
EVAL_FILE = Path(
    "/scratch/gpfs/MSALGANIK/niznik/ck-experiments/"
    "acs_income_sample_size_2026-01-21/1B_50K/eval/logs/"
    "2026-01-21T21-13-20-05-00_acs-income_MKqbigjA7Sfu9nGApseu5U.eval"
)
DATA_FILE = Path(
    "/scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/"
    "data/green/acs/acs_income_verbose_50000_80P.json"
)
OUTPUT_DIR = Path(
    "/scratch/gpfs/MSALGANIK/niznik/ck-experiments/"
    "acs_income_sample_size_2026-01-21/disagreement_analysis_1B"
)

# Feature parsing config (from ml_baseline.py)
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
    "citizenship status": "CITIZENSHIP",
    "disability status": "DISABILITY",
    "employment status of parents": "PARENT_EMPLOYMENT",
    "mobility status": "MOBILITY",
    "puma code": "PUMA",
    "resident state": "STATE",
    "income-to-poverty ratio": "POVERTY_RATIO",
}

NUMERIC_FEATURES = {"AGE", "HOURS_WEEK", "POVERTY_RATIO"}


# ============================================================================
# LLM Prediction Extraction
# ============================================================================

def load_llm_predictions(eval_path: Path) -> pd.DataFrame:
    """Load per-sample LLM predictions from .eval ZIP file."""
    samples = []
    with zipfile.ZipFile(eval_path, "r") as z:
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                with z.open(name) as f:
                    samples.append(json.load(f))

    results = []
    for s in samples:
        sample_id = s["id"]
        target = s["target"]
        # Get prediction from last assistant message
        pred = ""
        for msg in reversed(s.get("messages", [])):
            if msg.get("role") == "assistant":
                pred = msg.get("content", "").strip()
                break
        results.append({
            "sample_id": sample_id,
            "target": int(target),
            "llm_pred": int(pred) if pred in ("0", "1") else -1,
        })

    df = pd.DataFrame(results)
    df["llm_correct"] = (df["target"] == df["llm_pred"]).astype(int)
    return df.sort_values("sample_id").reset_index(drop=True)


# ============================================================================
# ML Baseline Training & Prediction
# ============================================================================

def parse_verbose_example(text: str) -> dict:
    """Parse verbose format text into feature dict."""
    features = {}
    pattern = r"- The ([^:]+) is: ([^\n]+)"
    matches = re.findall(pattern, text)

    for field_name, value in matches:
        field_name = field_name.strip().lower()
        value = value.strip().rstrip(".")

        short_name = FIELD_MAPPING.get(
            field_name, field_name.upper().replace(" ", "_")
        )

        # Parse numeric values
        if short_name == "AGE":
            match = re.search(r"(\d+)", value)
            if match:
                value = int(match.group(1))
        elif short_name == "HOURS_WEEK":
            match = re.search(r"(\d+)", value)
            if match:
                value = int(match.group(1))
        elif short_name == "POVERTY_RATIO":
            match = re.search(r"([\d.]+)", value)
            if match:
                value = float(match.group(1))

        features[short_name] = value

    return features


def train_catboost_and_predict(data_path: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Train CatBoost on training data and return test predictions + features."""
    from catboost import CatBoostClassifier

    with open(data_path) as f:
        data = json.load(f)

    # Parse all examples
    train_raw = [parse_verbose_example(ex["input"]) for ex in data["train"]]
    test_raw = [parse_verbose_example(ex["input"]) for ex in data["test"]]

    y_train = [int(ex["output"]) for ex in data["train"]]
    y_test = [int(ex["output"]) for ex in data["test"]]

    # Detect columns from first example
    first = train_raw[0]
    numeric_cols = [k for k in sorted(first.keys()) if k in NUMERIC_FEATURES]
    categorical_cols = [k for k in sorted(first.keys()) if k not in NUMERIC_FEATURES]

    # Prepare CatBoost format
    def to_feature_list(raw_data):
        X = []
        for ex in raw_data:
            row = []
            for col in numeric_cols:
                val = ex.get(col, 0)
                try:
                    row.append(float(val) if isinstance(val, (int, float)) else 0)
                except (ValueError, TypeError):
                    row.append(0)
            for col in categorical_cols:
                row.append(str(ex.get(col, "")))
            X.append(row)
        return X

    X_train = to_feature_list(train_raw)
    X_test = to_feature_list(test_raw)
    cat_indices = list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))

    # Train
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        cat_features=cat_indices,
        verbose=False,
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test).flatten().astype(int)

    # Build results DataFrame
    results = pd.DataFrame({
        "sample_id": range(1, len(y_test) + 1),  # 1-indexed to match eval
        "target": y_test,
        "ml_pred": y_pred,
        "ml_correct": (y_pred == np.array(y_test)).astype(int),
    })

    return results, test_raw


# ============================================================================
# Disagreement Analysis
# ============================================================================

def categorize_disagreement(row: pd.Series) -> str:
    """Categorize each sample by agreement/disagreement."""
    llm_correct = row["llm_correct"]
    ml_correct = row["ml_correct"]

    if llm_correct and ml_correct:
        return "both_correct"
    elif not llm_correct and not ml_correct:
        return "both_wrong"
    elif llm_correct and not ml_correct:
        return "llm_only_correct"
    else:
        return "ml_only_correct"


def analyze_categorical_feature(
    df: pd.DataFrame, feature: str, category_col: str = "category"
) -> pd.DataFrame:
    """Analyze distribution of a categorical feature across disagreement groups."""
    # Get counts per category per group
    cross = pd.crosstab(df[feature], df[category_col], normalize="columns") * 100
    return cross.round(1)


def analyze_numeric_feature(
    df: pd.DataFrame, feature: str, category_col: str = "category"
) -> pd.DataFrame:
    """Analyze distribution of a numeric feature across disagreement groups."""
    # Convert to numeric, coercing errors
    numeric_col = pd.to_numeric(df[feature], errors="coerce")
    stats_df = numeric_col.groupby(df[category_col]).agg(["mean", "std", "median", "count"])
    return stats_df.round(2)


def chi_square_test(df: pd.DataFrame, feature: str, group1: str, group2: str) -> float:
    """Run chi-square test comparing two groups on a categorical feature."""
    subset = df[df["category"].isin([group1, group2])]
    contingency = pd.crosstab(subset[feature], subset["category"])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 1.0  # Can't compute
    _, p_value, _, _ = stats.chi2_contingency(contingency)
    return p_value


def ttest_groups(df: pd.DataFrame, feature: str, group1: str, group2: str) -> float:
    """Run t-test comparing two groups on a numeric feature."""
    g1 = df[df["category"] == group1][feature].dropna()
    g2 = df[df["category"] == group2][feature].dropna()
    if len(g1) < 2 or len(g2) < 2:
        return 1.0
    _, p_value = stats.ttest_ind(g1, g2)
    return p_value


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(
    merged_df: pd.DataFrame,
    features_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate the markdown report."""
    lines = []
    lines.append("# Disagreement Analysis: LLM vs CatBoost on Traveltime\n")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Eval file:** `{EVAL_FILE.name}`\n")
    lines.append(f"**Data:** `{DATA_FILE.name}`\n")
    lines.append("")

    # Summary counts
    lines.append("## Summary\n")
    counts = merged_df["category"].value_counts()
    total = len(merged_df)
    lines.append("| Category | Count | Percent |")
    lines.append("|----------|-------|---------|")
    for cat in ["both_correct", "both_wrong", "llm_only_correct", "ml_only_correct"]:
        n = counts.get(cat, 0)
        pct = n / total * 100
        lines.append(f"| {cat} | {n} | {pct:.1f}% |")
    lines.append(f"| **Total** | **{total}** | **100%** |")
    lines.append("")

    # Overall accuracy
    llm_acc = merged_df["llm_correct"].mean() * 100
    ml_acc = merged_df["ml_correct"].mean() * 100
    lines.append(f"**LLM Accuracy:** {llm_acc:.1f}%\n")
    lines.append(f"**ML Accuracy:** {ml_acc:.1f}%\n")
    lines.append("")

    # Merge features into main df
    full_df = pd.concat([merged_df.reset_index(drop=True), features_df], axis=1)

    # Identify numeric vs categorical columns
    numeric_cols = [c for c in features_df.columns if c in NUMERIC_FEATURES]
    categorical_cols = [c for c in features_df.columns if c not in NUMERIC_FEATURES]

    # Feature analysis - focus on LLM vs ML differences
    lines.append("## Feature Analysis: LLM-Only-Correct vs ML-Only-Correct\n")
    lines.append("These are the cases where the models disagree. ")
    lines.append("We compare features to understand what distinguishes them.\n")
    lines.append("")

    # Numeric features
    lines.append("### Numeric Features\n")
    lines.append("| Feature | LLM-Only Mean | ML-Only Mean | p-value |")
    lines.append("|---------|---------------|--------------|---------|")
    for col in numeric_cols:
        if col not in full_df.columns:
            continue
        # Convert to numeric, coercing errors
        full_df[col] = pd.to_numeric(full_df[col], errors="coerce")
        llm_mean = full_df[full_df["category"] == "llm_only_correct"][col].mean()
        ml_mean = full_df[full_df["category"] == "ml_only_correct"][col].mean()
        p = ttest_groups(full_df, col, "llm_only_correct", "ml_only_correct")
        sig = "**" if p < 0.05 else ""
        lines.append(f"| {col} | {llm_mean:.1f} | {ml_mean:.1f} | {sig}{p:.4f}{sig} |")
    lines.append("")

    # Categorical features - find most different
    lines.append("### Categorical Features (Top Differences)\n")
    significant_cats = []
    for col in categorical_cols:
        if col not in full_df.columns:
            continue
        p = chi_square_test(full_df, col, "llm_only_correct", "ml_only_correct")
        if p < 0.1:  # Include marginally significant
            significant_cats.append((col, p))

    significant_cats.sort(key=lambda x: x[1])

    if significant_cats:
        for col, p in significant_cats[:5]:  # Top 5
            lines.append(f"#### {col} (p={p:.4f})\n")
            cross = pd.crosstab(
                full_df[col],
                full_df["category"],
                normalize="columns"
            ) * 100
            if "llm_only_correct" in cross.columns and "ml_only_correct" in cross.columns:
                # Show top values by difference
                cross["diff"] = cross["llm_only_correct"] - cross["ml_only_correct"]
                cross_sorted = cross.sort_values("diff", ascending=False)
                lines.append("| Value | LLM-Only % | ML-Only % | Diff |")
                lines.append("|-------|------------|-----------|------|")
                for idx in cross_sorted.head(5).index:
                    llm_pct = cross_sorted.loc[idx, "llm_only_correct"]
                    ml_pct = cross_sorted.loc[idx, "ml_only_correct"]
                    diff = cross_sorted.loc[idx, "diff"]
                    lines.append(f"| {idx[:40]} | {llm_pct:.1f}% | {ml_pct:.1f}% | {diff:+.1f} |")
                lines.append("")
    else:
        lines.append("No statistically significant categorical differences found.\n")

    # Example samples
    lines.append("## Example Samples\n")

    for cat, label in [
        ("llm_only_correct", "LLM Correct, ML Wrong"),
        ("ml_only_correct", "ML Correct, LLM Wrong"),
    ]:
        lines.append(f"### {label}\n")
        subset = full_df[full_df["category"] == cat].head(3)
        for i, row in subset.iterrows():
            lines.append(f"**Sample {row['sample_id']}** (target={row['target']}, llm={row['llm_pred']}, ml={row['ml_pred']})\n")
            # Show key features
            for col in ["AGE", "EDUCATION", "OCCUPATION", "TRANSPORT", "STATE"]:
                if col in row:
                    lines.append(f"- {col}: {row[col]}")
            lines.append("")

    # Key findings
    lines.append("## Key Findings\n")
    lines.append("*(To be filled in based on analysis above)*\n")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report written to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Disagreement Analysis: LLM vs CatBoost on Traveltime")
    print("=" * 60)

    # Step 1: Load LLM predictions
    print("\n1. Loading LLM predictions...")
    llm_df = load_llm_predictions(EVAL_FILE)
    print(f"   Loaded {len(llm_df)} samples")
    print(f"   LLM accuracy: {llm_df['llm_correct'].mean():.1%}")

    # Step 2: Train ML and get predictions
    print("\n2. Training CatBoost and getting predictions...")
    ml_df, test_features = train_catboost_and_predict(DATA_FILE)
    print(f"   ML accuracy: {ml_df['ml_correct'].mean():.1%}")

    # Step 3: Merge predictions
    print("\n3. Merging predictions...")
    merged = llm_df.merge(ml_df[["sample_id", "ml_pred", "ml_correct"]], on="sample_id")
    merged["category"] = merged.apply(categorize_disagreement, axis=1)

    print("\n   Disagreement breakdown:")
    for cat, count in merged["category"].value_counts().items():
        print(f"   - {cat}: {count} ({count/len(merged):.1%})")

    # Step 4: Parse features
    print("\n4. Parsing features...")
    features_df = pd.DataFrame(test_features)
    print(f"   Features: {list(features_df.columns)}")

    # Step 5: Generate report
    print("\n5. Generating report...")
    report_path = OUTPUT_DIR / "disagreement_report.md"
    generate_report(merged, features_df, report_path)

    # Step 6: Save full data
    csv_path = OUTPUT_DIR / "disagreement_data.csv"
    full_df = pd.concat([merged.reset_index(drop=True), features_df], axis=1)
    full_df.to_csv(csv_path, index=False)
    print(f"   Data saved to: {csv_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
