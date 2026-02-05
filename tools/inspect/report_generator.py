"""
Generate markdown reports from experiment evaluation results.

This module produces structured markdown reports with metrics, confidence intervals,
and model comparisons for stakeholders.

Example usage:
    from tools.inspect.report_generator import generate_report

    # Generate report from evaluation dataframe
    report = generate_report(
        df=logs_df,
        experiment_name="acs_income_2026-01-29",
        output_path=Path("experiments/acs_income/analysis/report.md")
    )
"""

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ModelMetrics:
    """Metrics for a single model evaluation."""

    name: str
    accuracy: float
    ci_lower: float
    ci_upper: float
    sample_size: int
    epoch: Optional[int] = None
    task_name: Optional[str] = None
    is_baseline: bool = False
    is_synthetic: bool = False  # True for synthetic baselines (not actual runs)
    balanced_accuracy: Optional[float] = None
    f1: Optional[float] = None


@dataclass
class Comparison:
    """Comparison between two models."""

    model_a: str
    model_b: str
    absolute_diff: float
    relative_diff: float  # as percentage
    direction: str  # "improvement" or "decline"


def compute_wilson_ci(
    accuracy: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.

    The Wilson interval is preferred over the normal approximation because it:
    - Never produces intervals outside [0, 1]
    - Performs better with small samples or extreme proportions

    Args:
        accuracy: Observed proportion (0.0 to 1.0)
        n: Sample size
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)

    Example:
        lower, upper = compute_wilson_ci(0.75, 100)
        # Returns approximately (0.656, 0.826)
    """
    if n == 0:
        return (0.0, 0.0)

    # Z-score for confidence level (1.96 for 95%)
    z = 1.96 if confidence == 0.95 else _z_score(confidence)

    # Wilson score interval formula
    denominator = 1 + z**2 / n
    center = (accuracy + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(
        (accuracy * (1 - accuracy) / n) + (z**2 / (4 * n**2))
    )

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


def _z_score(confidence: float) -> float:
    """Approximate z-score for common confidence levels."""
    # Common z-scores
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    return z_scores.get(confidence, 1.96)


@dataclass
class BaselineInfo:
    """Information about the baseline for comparison."""

    model_name: Optional[str] = None  # Model name if baseline is in experiment
    accuracy: Optional[float] = None  # Known accuracy if synthetic baseline
    source: Optional[str] = None  # Description of where baseline comes from


def identify_baseline(
    df: pd.DataFrame, config: Optional[dict] = None
) -> BaselineInfo:
    """
    Identify baseline model or value from dataframe or config.

    Priority order:
    1. finetuned == False in metadata
    2. type == "control" in experiment_summary.yaml config
    3. "base" in model name (case-insensitive)
    4. evaluation.baseline.accuracy in config (known value like random chance)

    Args:
        df: DataFrame with model column and optional finetuned column
        config: Optional experiment_summary.yaml config dict

    Returns:
        BaselineInfo with either model_name or accuracy/source populated
    """
    # Priority 1: Look for finetuned == False
    if "finetuned" in df.columns:
        not_finetuned = df[df["finetuned"] == False]  # noqa: E712
        if not not_finetuned.empty:
            return BaselineInfo(model_name=not_finetuned["model"].iloc[0])

    # Priority 2: Check config for type == "control"
    if config and "runs" in config:
        for run in config["runs"]:
            if run.get("type") == "control":
                # Try to match run name to model in df
                run_name = run.get("name", "")
                matching = df[df["model"].str.contains(run_name, case=False, na=False)]
                if not matching.empty:
                    return BaselineInfo(model_name=matching["model"].iloc[0])

    # Priority 3: Look for "base" in model name
    model_col = "model" if "model" in df.columns else None
    if model_col:
        base_models = df[df[model_col].str.contains("base", case=False, na=False)]
        if not base_models.empty:
            return BaselineInfo(model_name=base_models[model_col].iloc[0])

    # Priority 4: Check config for evaluation.baseline with known accuracy
    if config:
        eval_config = config.get("evaluation", {})
        baseline_config = eval_config.get("baseline", {})
        if baseline_config.get("accuracy") is not None:
            return BaselineInfo(
                accuracy=baseline_config["accuracy"],
                source=baseline_config.get("source", "specified in config"),
            )

    return BaselineInfo()


def extract_metrics(
    df: pd.DataFrame, config: Optional[dict] = None
) -> list[ModelMetrics]:
    """
    Extract ModelMetrics from evaluation dataframe.

    Args:
        df: DataFrame with columns: model, sample_size (or results_total_samples),
            and score columns (e.g., score_match_accuracy)
        config: Optional experiment config for baseline identification

    Returns:
        List of ModelMetrics dataclasses (includes synthetic baseline if configured)
    """
    metrics = []
    baseline_info = identify_baseline(df, config)

    # Determine sample size column
    sample_col = None
    for col in ["sample_size", "results_total_samples", "total_samples", "n"]:
        if col in df.columns:
            sample_col = col
            break

    # Find accuracy column(s)
    accuracy_cols = [c for c in df.columns if c.endswith("_accuracy")]
    if not accuracy_cols:
        # Try generic 'accuracy' column
        if "accuracy" in df.columns:
            accuracy_cols = ["accuracy"]

    # Get primary accuracy column
    primary_acc_col = accuracy_cols[0] if accuracy_cols else None

    # Group by model (and task if present)
    group_cols = ["model"]
    if "task_name" in df.columns:
        group_cols.append("task_name")
    if "epoch" in df.columns:
        group_cols.append("epoch")

    for group_key, group_df in df.groupby(group_cols, dropna=False):
        # Handle single vs multiple groupby columns
        if len(group_cols) == 1:
            model_name = group_key
            task_name = None
            epoch = None
        else:
            model_name = group_key[0]
            task_name = group_key[1] if len(group_cols) > 1 else None
            epoch = group_key[2] if len(group_cols) > 2 else None

        # Get accuracy
        if primary_acc_col and primary_acc_col in group_df.columns:
            acc = group_df[primary_acc_col].iloc[0]
        else:
            continue  # Skip if no accuracy available

        # Get sample size
        n = int(group_df[sample_col].iloc[0]) if sample_col else 100

        # Compute confidence interval
        ci_lower, ci_upper = compute_wilson_ci(acc, n)

        # Check if baseline (model in experiment)
        is_baseline = (
            baseline_info.model_name is not None
            and model_name == baseline_info.model_name
        )

        # Get optional metrics
        balanced_acc = None
        f1 = None
        if "balanced_accuracy" in group_df.columns:
            balanced_acc = group_df["balanced_accuracy"].iloc[0]
        if "f1" in group_df.columns:
            f1 = group_df["f1"].iloc[0]

        metrics.append(
            ModelMetrics(
                name=model_name,
                accuracy=acc,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                sample_size=n,
                epoch=epoch,
                task_name=task_name,
                is_baseline=is_baseline,
                balanced_accuracy=balanced_acc,
                f1=f1,
            )
        )

    # Add synthetic baseline if configured (known accuracy value)
    if baseline_info.accuracy is not None:
        # Use median sample size from actual data for CI calculation
        median_n = 1000  # Conservative default for synthetic baseline
        if metrics:
            median_n = int(sorted(m.sample_size for m in metrics)[len(metrics) // 2])

        ci_lower, ci_upper = compute_wilson_ci(baseline_info.accuracy, median_n)
        baseline_name = baseline_info.source or "baseline"

        metrics.append(
            ModelMetrics(
                name=baseline_name,
                accuracy=baseline_info.accuracy,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                sample_size=median_n,
                epoch=None,
                task_name=None,
                is_baseline=True,
                is_synthetic=True,
            )
        )

    return metrics


def compute_comparisons(
    metrics: list[ModelMetrics], baseline_name: Optional[str] = None
) -> list[Comparison]:
    """
    Compute comparisons between models, relative to baseline if available.

    Args:
        metrics: List of ModelMetrics to compare
        baseline_name: Optional explicit baseline model name

    Returns:
        List of Comparison dataclasses
    """
    comparisons = []

    # Find baseline
    baseline = None
    if baseline_name:
        baseline = next((m for m in metrics if m.name == baseline_name), None)
    else:
        baseline = next((m for m in metrics if m.is_baseline), None)

    if not baseline:
        # No baseline - compare all pairs? Or skip comparisons
        return comparisons

    for m in metrics:
        if m.name == baseline.name:
            continue

        abs_diff = m.accuracy - baseline.accuracy
        rel_diff = (abs_diff / baseline.accuracy * 100) if baseline.accuracy > 0 else 0
        direction = "improvement" if abs_diff > 0 else "decline"

        comparisons.append(
            Comparison(
                model_a=m.name,
                model_b=baseline.name,
                absolute_diff=abs_diff,
                relative_diff=rel_diff,
                direction=direction,
            )
        )

    return comparisons


def generate_narrative(
    metrics: list[ModelMetrics], comparisons: list[Comparison]
) -> str:
    """
    Generate executive summary narrative.

    Args:
        metrics: List of model metrics
        comparisons: List of model comparisons

    Returns:
        Markdown-formatted narrative string
    """
    # Separate actual runs from synthetic baselines
    actual_runs = [m for m in metrics if not m.is_synthetic]

    if not actual_runs:
        return "No model metrics available for analysis."

    # Find best performer (among actual runs only)
    best = max(actual_runs, key=lambda m: m.accuracy)

    # Find baseline (could be synthetic or actual)
    baseline = next((m for m in metrics if m.is_baseline), None)

    lines = []

    # Best performer statement
    lines.append(
        f"**{best.name}** achieved the highest accuracy at "
        f"**{best.accuracy:.1%}** (95% CI: {best.ci_lower:.1%}-{best.ci_upper:.1%})."
    )

    # Comparison to baseline if available
    if baseline and baseline.name != best.name:
        improvement = best.accuracy - baseline.accuracy
        rel_improvement = (
            (improvement / baseline.accuracy * 100) if baseline.accuracy > 0 else 0
        )
        lines.append(
            f"This represents a **{improvement:.1%} absolute improvement** "
            f"({rel_improvement:.1f}% relative) over the baseline ({baseline.name})."
        )
    elif baseline and not baseline.is_synthetic:
        lines.append("The baseline model was the best performer.")

    # Model count (actual runs only)
    lines.append(f"\n{len(actual_runs)} model configurations were evaluated.")

    return "\n".join(lines)


def _format_model_table(metrics: list[ModelMetrics]) -> str:
    """Format model comparison table."""
    lines = [
        "| Model | Epoch | Accuracy | 95% CI | Sample Size |",
        "|-------|-------|----------|--------|-------------|",
    ]

    # Sort by accuracy descending, with synthetic baselines last
    sorted_metrics = sorted(
        metrics,
        key=lambda x: (x.is_synthetic, -x.accuracy)
    )

    seen_synthetic = False
    for m in sorted_metrics:
        # Add separator before first synthetic baseline
        if m.is_synthetic and not seen_synthetic:
            lines.append("| | | | | |")
            seen_synthetic = True

        epoch_str = str(m.epoch) if m.epoch is not None else "-"
        ci_str = f"{m.ci_lower:.1%}-{m.ci_upper:.1%}"
        sample_str = "-" if m.is_synthetic else str(m.sample_size)
        baseline_marker = " *" if m.is_baseline else ""
        lines.append(
            f"| {m.name}{baseline_marker} | {epoch_str} | "
            f"{m.accuracy:.1%} | {ci_str} | {sample_str} |"
        )

    return "\n".join(lines)


def _format_improvement_table(comparisons: list[Comparison]) -> str:
    """Format improvement vs baseline table."""
    if not comparisons:
        return "*No baseline identified for comparison.*"

    lines = [
        "| Model | Absolute | Relative |",
        "|-------|----------|----------|",
    ]

    for c in sorted(comparisons, key=lambda x: x.absolute_diff, reverse=True):
        sign = "+" if c.absolute_diff > 0 else ""
        lines.append(
            f"| {c.model_a} | {sign}{c.absolute_diff:.1%} | "
            f"{sign}{c.relative_diff:.1f}% |"
        )

    return "\n".join(lines)


def _format_per_task_breakdown(metrics: list[ModelMetrics]) -> str:
    """Format per-task breakdown if multiple tasks exist."""
    # Group by task (exclude synthetic baselines)
    tasks = {}
    for m in metrics:
        if m.is_synthetic:
            continue
        task = m.task_name or "default"
        if task not in tasks:
            tasks[task] = []
        tasks[task].append(m)

    if len(tasks) <= 1:
        return ""

    # Skip if not informative (each task has only 1 model)
    if all(len(task_metrics) == 1 for task_metrics in tasks.values()):
        return ""

    lines = ["## Per-Task Breakdown\n"]

    for task_name, task_metrics in sorted(tasks.items()):
        best = max(task_metrics, key=lambda x: x.accuracy)
        lines.append(f"### {task_name}")
        lines.append(f"Best: **{best.name}** ({best.accuracy:.1%})\n")

    return "\n".join(lines)


def _format_visualizations(output_path: Path) -> str:
    """Embed any PNG visualizations found in the analysis directory."""
    analysis_dir = output_path.parent
    png_files = sorted(analysis_dir.glob("*.png"))

    if not png_files:
        return ""

    lines = ["## Visualizations\n"]

    for png_path in png_files:
        # Use relative path from report.md location
        rel_path = png_path.name
        # Create a readable title from filename
        title = png_path.stem.replace("_", " ").title()
        lines.append(f"### {title}\n")
        lines.append(f"![{title}]({rel_path})\n")

    return "\n".join(lines)


def generate_report(
    df: pd.DataFrame,
    experiment_name: str,
    output_path: Path,
    config: Optional[dict] = None,
    future_directions: Optional[str] = None,
) -> str:
    """
    Generate complete markdown report from evaluation data.

    Args:
        df: DataFrame with evaluation metrics (from evals_df_prep + parse_eval_metadata)
        experiment_name: Name of the experiment
        output_path: Path where report will be saved
        config: Optional experiment_summary.yaml config dict
        future_directions: Optional skill-generated analysis and suggestions

    Returns:
        Generated markdown report content

    Example:
        report = generate_report(
            df=logs_df,
            experiment_name="acs_income_2026-01-29",
            output_path=Path("experiments/acs_income/analysis/report.md"),
            future_directions="Based on the results, consider..."
        )
    """
    # Extract metrics
    metrics = extract_metrics(df, config)

    # Compute comparisons
    comparisons = compute_comparisons(metrics)

    # Generate narrative
    narrative = generate_narrative(metrics, comparisons)

    # Build report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    model_count = len(set(m.name for m in metrics if not m.is_synthetic))

    report_lines = [
        "# Experiment Analysis Report\n",
        f"**Experiment:** {experiment_name} | "
        f"**Generated:** {timestamp} | "
        f"**Models:** {model_count}\n",
        "## Executive Summary\n",
        narrative + "\n",
        "## Model Comparison\n",
        _format_model_table(metrics) + "\n",
        "*\\* indicates baseline model*\n",
        "## Improvement vs Baseline\n",
        _format_improvement_table(comparisons) + "\n",
    ]

    # Add per-task breakdown if applicable
    task_breakdown = _format_per_task_breakdown(metrics)
    if task_breakdown:
        report_lines.append(task_breakdown)

    # Add visualizations if PNGs exist
    visualizations = _format_visualizations(output_path)
    if visualizations:
        report_lines.append(visualizations)

    # Add future directions if provided by skill
    if future_directions:
        report_lines.append("## Future Directions\n")
        report_lines.append(future_directions + "\n")

    report_lines.append("---\n")
    report_lines.append("*Generated by analyze-experiment skill*\n")

    report_content = "\n".join(report_lines)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content)

    return report_content
