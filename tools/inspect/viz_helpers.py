"""
Helper functions for generating inspect-viz visualizations from evaluation logs.

This module provides utilities for loading experiment data and preparing it
for visualization with inspect-viz pre-built views.

Example usage:
    from tools.inspect.viz_helpers import evals_df_prep, parse_eval_metadata, detect_metrics

    # Load and prepare evaluation logs
    logs_df = evals_df_prep(eval_files)
    logs_df = parse_eval_metadata(logs_df)

    # Detect available metrics dynamically
    metrics = detect_metrics(logs_df)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log
from inspect_ai.analysis import (
    evals_df,
    EvalModel,
    EvalResults,
    EvalScores,
    EvalInfo,
    EvalTask,
)

logger = logging.getLogger(__name__)


def deduplicate_eval_files(eval_files: list[str]) -> tuple[list[str], list[str]]:
    """
    Keep most recent eval file per model+epoch combination.

    When multiple evaluations exist for the same model and epoch (e.g., from
    re-runs), this keeps only the most recent one based on timestamp in filename.

    Args:
        eval_files: List of paths to .eval log files

    Returns:
        Tuple of (kept_files, skipped_files) where:
        - kept_files: List of paths to keep (most recent per model+epoch)
        - skipped_files: List of paths that were duplicates

    Example:
        kept, skipped = deduplicate_eval_files(eval_files)
        print(f"Kept {len(kept)} files, skipped {len(skipped)} duplicates")
    """
    # Read model and epoch info from each file
    file_details = []
    for filepath in eval_files:
        try:
            log = read_eval_log(filepath)
            # Extract timestamp from filename (format: YYYYMMDDTHHMMSS_...)
            filename = os.path.basename(filepath)
            timestamp_str = filename.split('_')[0]

            # Get epoch from metadata if available
            metadata = {}
            if hasattr(log.eval, 'metadata') and log.eval.metadata:
                metadata = log.eval.metadata
            epoch = metadata.get('epoch', 'unknown')

            file_details.append({
                'path': filepath,
                'timestamp': timestamp_str,
                'model': log.eval.model,
                'epoch': epoch,
            })
        except Exception as e:
            # If we can't read the file, skip it
            print(f"Warning: Could not read {filepath}: {e}")
            continue

    # Sort by timestamp descending (most recent first)
    file_details.sort(key=lambda x: x['timestamp'], reverse=True)

    # Keep most recent per model+epoch combination
    seen_keys = set()
    kept = []
    skipped = []

    for fd in file_details:
        key = (fd['model'], fd['epoch'])
        if key not in seen_keys:
            seen_keys.add(key)
            kept.append(fd['path'])
        else:
            skipped.append(fd['path'])

    return kept, skipped


def parse_eval_metadata(logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse JSON metadata column into separate columns for epoch, finetuned, source_model.

    Extracts structured metadata from the 'metadata' column (which contains JSON strings)
    and adds them as proper DataFrame columns. Also uses task_arg_vis_label for task_name.

    Args:
        logs_df: DataFrame from evals_df() with 'metadata' and 'task_arg_vis_label' columns

    Returns:
        DataFrame with additional columns:
        - epoch: Training epoch number (from metadata)
        - finetuned: Boolean indicating if model was fine-tuned (from metadata)
        - source_model: Original model name before fine-tuning (from metadata)
        - task_name: Visualization label for the task (from task_arg_vis_label)

    Example:
        logs_df = evals_df_prep(eval_files)
        logs_df = parse_eval_metadata(logs_df)
        print(logs_df['epoch'].unique())  # [1, 2, 3]
    """
    def _parse_metadata(meta_str):
        """Parse JSON metadata string into dict."""
        # Handle pandas NA/None values
        if pd.isna(meta_str):
            return {}
        if isinstance(meta_str, str):
            try:
                return json.loads(meta_str)
            except json.JSONDecodeError:
                return {}
        elif isinstance(meta_str, dict):
            return meta_str
        return {}

    # Parse metadata columns
    if 'metadata' in logs_df.columns:
        logs_df['epoch'] = logs_df['metadata'].apply(
            lambda x: _parse_metadata(x).get('epoch', None)
        )
        logs_df['finetuned'] = logs_df['metadata'].apply(
            lambda x: _parse_metadata(x).get('finetuned', None)
        )
        logs_df['source_model'] = logs_df['metadata'].apply(
            lambda x: _parse_metadata(x).get('source_model', None)
        )

    # Use vis_label for task_name
    if 'task_arg_vis_label' in logs_df.columns:
        logs_df['task_name'] = logs_df['task_arg_vis_label']

    return logs_df


METRIC_DISPLAY_NAMES: dict[str, str] = {
    "risk_scorer_cruijff_kit/ece": "C-ECE",
    "risk_scorer_cruijff_kit/expected_calibration_error": "C-ECE",
    "risk_scorer_cruijff_kit/risk_calibration_error": "R-ECE",
    "risk_scorer_cruijff_kit/brier_score": "Brier Score",
    "risk_scorer_cruijff_kit/auc_score": "AUC",
    "risk_scorer_cruijff_kit/mean_risk_score": "Mean Risk Score",
}


def display_name(metric_name: str) -> str:
    """Return human-readable display name for a metric, or title-case the raw name."""
    return METRIC_DISPLAY_NAMES.get(
        metric_name,
        metric_name.rsplit("/", 1)[-1].replace("_", " ").title(),
    )


@dataclass
class DetectedMetrics:
    """Metrics detected from score columns in an evaluation dataframe.

    Attributes:
        accuracy: Accuracy metric names (e.g., ['match', 'includes']).
            Use as ``f"score_{name}_accuracy"`` to get the column name.
        supplementary: Non-accuracy score column names with the ``score_`` prefix
            stripped (e.g., ['risk_scorer_cruijff_kit/auc_score']).
            Use as ``f"score_{name}"`` to get the column name.
    """

    accuracy: list[str] = field(default_factory=list)
    supplementary: list[str] = field(default_factory=list)

    @property
    def has_risk_scorer(self) -> bool:
        """True when the evaluation used risk_scorer (has AUC metric)."""
        return any("auc_score" in m for m in self.supplementary)


def detect_metrics(logs_df: pd.DataFrame) -> DetectedMetrics:
    """
    Detect available metrics from score columns in the dataframe.

    Args:
        logs_df: DataFrame with score columns (e.g., score_match_accuracy)

    Returns:
        DetectedMetrics with ``.accuracy`` and ``.supplementary`` lists.

    Example:
        detected = detect_metrics(logs_df)
        for metric in detected.accuracy:
            score_col = f"score_{metric}_accuracy"
            # ... generate accuracy plot

        for metric in detected.supplementary:
            score_col = f"score_{metric}"
            label = display_name(metric)
            # ... generate supplementary plot
    """
    accuracy_cols = [
        c for c in logs_df.columns
        if c.startswith('score_') and c.endswith('_accuracy')
    ]
    accuracy_names = [
        c.replace('score_', '').replace('_accuracy', '') for c in accuracy_cols
    ]

    supplementary_names = [
        c[len('score_'):] for c in logs_df.columns
        if c.startswith('score_')
        and not c.endswith('_accuracy')
        and not c.endswith('_stderr')
        and not c.startswith('score_headline_')
    ]

    # Sort into a deliberate presentation order:
    #   AUC (discrimination) → Brier (composite) → C-ECE → R-ECE → Mean Risk Score
    # Unknown metrics fall to the end in their original order.
    _METRIC_ORDER = {
        "auc_score": 0,
        "brier_score": 1,
        "expected_calibration_error": 2,  # C-ECE
        "ece": 2,                         # C-ECE alias
        "risk_calibration_error": 3,      # R-ECE
        "mean_risk_score": 4,
    }

    def _sort_key(name: str) -> int:
        suffix = name.rsplit("/", 1)[-1]
        return _METRIC_ORDER.get(suffix, 100)

    supplementary_names.sort(key=_sort_key)

    return DetectedMetrics(accuracy=accuracy_names, supplementary=supplementary_names)


def sanitize_columns_for_viz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *df* with column names safe for inspect-viz.

    inspect-viz uses DuckDB internally for data queries. DuckDB interprets
    "/" in unquoted column names as the division operator, which causes
    Binder Errors for columns like ``score_risk_scorer_cruijff_kit/auc_score``
    (parsed as ``score_risk_scorer_cruijff_kit`` ÷ ``auc_score``).

    This is an upstream quoting bug in inspect-viz. Until it's fixed, we
    work around it by replacing "/" with "__" in column names before passing
    the DataFrame to ``Data.from_dataframe()``.

    Important: only use the sanitized DataFrame for inspect-viz rendering.
    The report generator should receive the *original* DataFrame so that
    ``detect_metrics()`` and ``display_name()`` see the canonical names.

    Example:
        viz_df = sanitize_columns_for_viz(logs_df)
        data = Data.from_dataframe(viz_df)  # safe for inspect-viz views
    """
    rename_map = {
        col: col.replace("/", "__")
        for col in df.columns
        if "/" in col
    }
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def evals_df_prep(logs: list[str]) -> pd.DataFrame:
    """
    Prepare evaluation-level data for plotting.

    Args:
        logs: List of paths to .eval log files

    Returns:
        DataFrame with evaluation-level metrics and metadata
    """
    logs_df = evals_df(
        logs=logs,
        columns=(EvalInfo + EvalTask + EvalModel + EvalResults + EvalScores)
    )
    return logs_df


# =============================================================================
# Per-sample risk data extraction and plotting
# =============================================================================

@dataclass
class PerSampleRiskData:
    """Per-sample binary prediction data extracted from a single eval log.

    Attributes:
        model_name: Display label for this model (from vis_label or model id).
        y_true: Binary ground-truth labels (1.0 = positive class, 0.0 = negative).
        y_score: Predicted probability of the positive class (risk_score).
        n_total: Total samples in the eval log (including those without risk data).
        n_valid: Samples with usable risk data.
    """

    model_name: str
    y_true: list[float]
    y_score: list[float]
    n_total: int
    n_valid: int


def _extract_risk_from_log(log) -> PerSampleRiskData | None:
    """Extract per-sample (y_true, y_score) from a single eval log.

    This is the inner loop factored out for testability. Returns None when the
    log has fewer than 2 valid samples or only one class present.
    """
    # Determine display name
    task_args = {}
    if hasattr(log.eval, 'task_args') and log.eval.task_args:
        task_args = log.eval.task_args
    model_name = task_args.get('vis_label', log.eval.model)

    y_true: list[float] = []
    y_score: list[float] = []
    n_total = len(log.samples) if log.samples else 0

    for sample in (log.samples or []):
        scores = sample.scores or {}
        risk_score_obj = scores.get('risk_scorer')
        if risk_score_obj is None:
            continue
        meta = risk_score_obj.metadata or {}
        risk = meta.get('risk_score')
        target = meta.get('target')
        option_probs = meta.get('option_probs')
        if risk is None or target is None or option_probs is None:
            continue

        positive_token = next(iter(option_probs))
        y_true.append(1.0 if target == positive_token else 0.0)
        y_score.append(risk)

    n_valid = len(y_true)
    if n_valid < 2 or len(set(y_true)) < 2:
        logger.info(
            "Skipping %s: %d valid samples, %d classes",
            model_name, n_valid, len(set(y_true)),
        )
        return None

    return PerSampleRiskData(
        model_name=model_name,
        y_true=y_true,
        y_score=y_score,
        n_total=n_total,
        n_valid=n_valid,
    )


def extract_per_sample_risk_data(eval_files: list[str]) -> list[PerSampleRiskData]:
    """Read eval files and return per-sample risk data for each model.

    Models with <2 valid samples or only one class are silently skipped.

    Args:
        eval_files: Paths to .eval log files.

    Returns:
        List of PerSampleRiskData (one per model with usable data).
    """
    results: list[PerSampleRiskData] = []
    for filepath in eval_files:
        try:
            log = read_eval_log(filepath)
        except Exception as e:
            logger.warning("Could not read %s: %s", filepath, e)
            continue
        data = _extract_risk_from_log(log)
        if data is not None:
            results.append(data)
    return results


def generate_roc_overlay(
    risk_data: list[PerSampleRiskData],
    output_path: str | Path,
) -> Path | None:
    """Generate a single ROC-curve figure overlaying all models.

    Args:
        risk_data: Per-sample data for each model (from extract_per_sample_risk_data).
        output_path: Where to save the PNG.

    Returns:
        Path to the saved PNG, or None if risk_data is empty.
    """
    if not risk_data:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(7, 6))
    for rd in risk_data:
        fpr, tpr, _ = roc_curve(rd.y_true, rd.y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{rd.model_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize="small")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()

    out = Path(output_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_prediction_histogram(
    risk_data: list[PerSampleRiskData],
    output_path: str | Path,
    n_bins: int = 30,
) -> Path | None:
    """Generate prediction histograms split by true class, one subplot per model.

    Each subplot shows overlapping histograms of risk_score for the positive
    and negative classes, revealing how well the model separates them.

    Args:
        risk_data: Per-sample data for each model.
        output_path: Where to save the PNG.
        n_bins: Number of histogram bins.

    Returns:
        Path to the saved PNG, or None if risk_data is empty.
    """
    if not risk_data:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_models = len(risk_data)
    fig, axes = plt.subplots(n_models, 1, figsize=(7, 3.5 * n_models), squeeze=False)

    for i, rd in enumerate(risk_data):
        ax = axes[i, 0]
        y_true = np.array(rd.y_true)
        y_score = np.array(rd.y_score)

        neg_scores = y_score[y_true == 0.0]
        pos_scores = y_score[y_true == 1.0]
        base_rate = len(pos_scores) / len(y_score)

        bins = np.linspace(0, 1, n_bins + 1)
        ax.hist(neg_scores, bins=bins, alpha=0.5, label=f"Negative (n={len(neg_scores)})", color="tab:blue")
        ax.hist(pos_scores, bins=bins, alpha=0.5, label=f"Positive (n={len(pos_scores)})", color="tab:orange")
        ax.set_xlabel("Predicted P(positive)")
        ax.set_ylabel("Count")
        ax.set_title(f"{rd.model_name}  (base rate = {base_rate:.1%})")
        ax.legend(fontsize="small")
        ax.set_xlim([0, 1])

    fig.suptitle("Prediction Distributions by True Class", fontsize=13, y=1.01)
    fig.tight_layout()

    out = Path(output_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out

def generate_calibration_overlay(
    risk_data: list[PerSampleRiskData],
    output_path: str | Path,
    n_bins: int = 10,
) -> Path | None:
    """Generate a single calibration-curve figure overlaying all models.

    Args:
        risk_data: Per-sample data for each model.
        output_path: Where to save the PNG.
        n_bins: Number of calibration bins.

    Returns:
        Path to the saved PNG, or None if risk_data is empty.
    """
    if not risk_data:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    for rd in risk_data:
        fraction_pos, mean_predicted = calibration_curve(
            rd.y_true, rd.y_score, n_bins=n_bins, strategy="uniform",
        )
        ax.plot(
            mean_predicted, fraction_pos, "s-", linewidth=2,
            label=f"{rd.model_name} (n={rd.n_valid})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves")
    ax.legend(loc="best", fontsize="small")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()

    out = Path(output_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
