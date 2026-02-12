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

from cruijff_kit.tools.inspect.viz_helpers import DetectedMetrics, detect_metrics, display_name


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
    balanced_accuracy: Optional[float] = None
    f1: Optional[float] = None


@dataclass
class CalibrationResult:
    """Calibration / risk metrics for a single model evaluation."""

    model_name: str
    metrics: dict[str, Optional[float]]  # metric_name -> value (None if NA)
    sample_size: int
    epoch: Optional[int] = None


def extract_calibration_metrics(
    df: pd.DataFrame,
    supplementary_metrics: list[str],
) -> list[CalibrationResult]:
    """
    Extract calibration / risk metrics from evaluation dataframe.

    Args:
        df: DataFrame with score columns (from evals_df_prep + parse_eval_metadata)
        supplementary_metrics: List of supplementary metric names (from DetectedMetrics)

    Returns:
        List of CalibrationResult per model/epoch combination
    """
    if not supplementary_metrics:
        return []

    # Determine sample size column
    sample_col = None
    for col in ["sample_size", "results_total_samples", "total_samples", "n"]:
        if col in df.columns:
            sample_col = col
            break

    # Group by model (and epoch if present)
    group_cols = ["model"]
    if "epoch" in df.columns:
        group_cols.append("epoch")

    results = []
    for group_key, group_df in df.groupby(group_cols, dropna=False):
        # groupby with a list always returns tuple keys
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        model_name = group_key[0]
        epoch = group_key[1] if len(group_key) > 1 else None

        n = int(group_df[sample_col].iloc[0]) if sample_col else 0

        metric_values: dict[str, Optional[float]] = {}
        for metric_name in supplementary_metrics:
            col_name = f"score_{metric_name}"
            if col_name in group_df.columns:
                val = group_df[col_name].iloc[0]
                metric_values[metric_name] = None if pd.isna(val) else float(val)
            else:
                metric_values[metric_name] = None

        # Skip if all values are None
        if all(v is None for v in metric_values.values()):
            continue

        results.append(
            CalibrationResult(
                model_name=model_name,
                metrics=metric_values,
                sample_size=n,
                epoch=epoch,
            )
        )

    return results


def _format_calibration_table(results: list[CalibrationResult]) -> str:
    """Format calibration metrics as a markdown table.

    Args:
        results: List of CalibrationResult to display

    Returns:
        Markdown table string
    """
    if not results:
        return "*No calibration metrics available.*"

    # Collect all metric names across results (preserving order)
    all_metric_names: list[str] = []
    for r in results:
        for name in r.metrics:
            if name not in all_metric_names:
                all_metric_names.append(name)

    # Build header
    headers = ["Model", "Epoch"]
    headers.extend(display_name(m) for m in all_metric_names)
    headers.append("Sample Size")

    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"

    lines = [header_line, separator]

    # Sort by model name
    sorted_results = sorted(results, key=lambda r: r.model_name)

    for r in sorted_results:
        epoch_str = str(r.epoch) if r.epoch is not None else "-"

        cells = [r.model_name, epoch_str]
        for metric_name in all_metric_names:
            val = r.metrics.get(metric_name)
            cells.append(f"{val:.3f}" if val is not None else "-")
        cells.append(str(r.sample_size))

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


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


def extract_metrics(
    df: pd.DataFrame,
) -> list[ModelMetrics]:
    """
    Extract ModelMetrics from evaluation dataframe.

    Args:
        df: DataFrame with columns: model, sample_size (or results_total_samples),
            and score columns (e.g., score_match_accuracy)

    Returns:
        List of ModelMetrics dataclasses
    """
    metrics = []

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
        # groupby with a list always returns tuple keys
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        model_name = group_key[0]
        task_name = group_key[1] if len(group_key) > 1 else None
        epoch = group_key[2] if len(group_key) > 2 else None

        # Get accuracy
        if primary_acc_col and primary_acc_col in group_df.columns:
            acc = group_df[primary_acc_col].iloc[0]
        else:
            continue  # Skip if no accuracy available

        # Get sample size
        n = int(group_df[sample_col].iloc[0]) if sample_col else 100

        # Compute confidence interval
        ci_lower, ci_upper = compute_wilson_ci(acc, n)

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
                balanced_accuracy=balanced_acc,
                f1=f1,
            )
        )

    return metrics


def generate_narrative(metrics: list[ModelMetrics]) -> str:
    """
    Generate executive summary narrative.

    Args:
        metrics: List of model metrics

    Returns:
        Markdown-formatted narrative string
    """
    if not metrics:
        return "No model metrics available for analysis."

    best = max(metrics, key=lambda m: m.accuracy)

    lines = []

    # Best performer statement
    lines.append(
        f"**{best.name}** achieved the highest accuracy at "
        f"**{best.accuracy:.1%}** (95% CI: {best.ci_lower:.1%}-{best.ci_upper:.1%})."
    )

    # Model count
    lines.append(f"\n{len(metrics)} model configurations were evaluated.")

    return "\n".join(lines)


def _format_model_table(
    metrics: list[ModelMetrics],
    calibration: Optional[list[CalibrationResult]] = None,
) -> tuple[str, list[str]]:
    """Format unified model comparison table.

    When *calibration* is provided, supplementary metric columns are appended
    after the accuracy column.

    Returns:
        Tuple of (table_markdown, footnotes) where footnotes is a list of
        strings to render below the table.
    """
    # Collect supplementary column names (preserving order across results)
    supp_names: list[str] = []
    cal_lookup: dict[tuple[str, Optional[int]], CalibrationResult] = {}
    if calibration:
        for r in calibration:
            # Build lookup by (model, epoch) — epoch may be int or float
            epoch_key = int(r.epoch) if r.epoch is not None and not pd.isna(r.epoch) else None
            cal_lookup[(r.model_name, epoch_key)] = r
            for name in r.metrics:
                if name not in supp_names:
                    supp_names.append(name)

    # Check if sample sizes are uniform
    actual_sizes = set(m.sample_size for m in metrics)
    uniform_sample_size = actual_sizes.pop() if len(actual_sizes) == 1 else None

    # Header
    headers = ["Model", "Epoch", "Accuracy"]
    headers.extend(display_name(s) for s in supp_names)
    if uniform_sample_size is None:
        headers.append("Sample Size")

    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    lines = [header_line, separator]

    # Sort by accuracy descending
    sorted_metrics = sorted(metrics, key=lambda x: -x.accuracy)

    for m in sorted_metrics:
        epoch_str = str(int(m.epoch)) if m.epoch is not None and not pd.isna(m.epoch) else "-"

        cells = [
            m.name,
            epoch_str,
            f"{m.accuracy:.3f}",
        ]

        # Supplementary metric cells
        if supp_names:
            epoch_key = int(m.epoch) if m.epoch is not None and not pd.isna(m.epoch) else None
            cal = cal_lookup.get((m.name, epoch_key))
            for s in supp_names:
                val = cal.metrics.get(s) if cal else None
                cells.append(f"{val:.3f}" if val is not None else "-")

        if uniform_sample_size is None:
            cells.append(str(m.sample_size))

        lines.append("| " + " | ".join(cells) + " |")

    # Build footnotes
    footnotes = []
    if uniform_sample_size is not None:
        footnotes.append(f"*Sample size: {uniform_sample_size} per model*")
    if calibration:
        footnotes.append(
            "*ECE and Brier Score: lower is better. AUC: higher is better.*"
        )

    return "\n".join(lines), footnotes


def _format_per_task_breakdown(metrics: list[ModelMetrics]) -> str:
    """Format per-task breakdown if multiple tasks exist."""
    tasks = {}
    for m in metrics:
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


def _format_visualizations(
    output_path: Path, generated_pngs: Optional[list[Path]] = None
) -> str:
    """Embed PNG visualizations in the report.

    Args:
        output_path: Path to report.md (used to compute relative paths)
        generated_pngs: Optional list of PNG paths to embed. If None, embeds
            all PNGs found in the analysis directory (legacy behavior).
    """
    if generated_pngs is not None:
        png_files = sorted(generated_pngs)
    else:
        # Legacy fallback: glob all PNGs in directory
        analysis_dir = output_path.parent
        png_files = sorted(analysis_dir.glob("*.png"))

    if not png_files:
        return ""

    lines = ["## Visualizations\n"]

    for png_path in png_files:
        # Use relative path from report.md location
        rel_path = png_path.name if isinstance(png_path, Path) else Path(png_path).name
        # Create a readable title from filename
        stem = png_path.stem if isinstance(png_path, Path) else Path(png_path).stem
        title = stem.replace("_", " ").title()
        lines.append(f"### {title}\n")
        lines.append(f"![{title}]({rel_path})\n")

    return "\n".join(lines)


def _format_research_question(config: Optional[dict]) -> str:
    """Format research question section from experiment config.

    Pulls question, hypothesis, and purpose from experiment_summary.yaml
    to provide context for interpreting results.
    """
    if not config:
        return ""

    experiment = config.get("experiment", {})
    question = experiment.get("question")
    hypothesis = experiment.get("hypothesis")
    purpose = experiment.get("purpose")

    # Need at least a question to show this section
    if not question:
        return ""

    lines = ["## Research Question\n"]

    lines.append(f"**Question:** {question}\n")

    if hypothesis:
        lines.append(f"**Hypothesis:** {hypothesis}\n")

    if purpose:
        lines.append(f"**Purpose:** {purpose}\n")

    return "\n".join(lines)


def _format_inspect_view_commands(
    eval_log_paths: list[Path], max_commands: int = 20
) -> str:
    """Format inspect view commands for eval log directories.

    Generates copy-paste-ready ``inspect view start`` commands, one per unique
    log directory.  When the number of directories exceeds *max_commands*, a
    single template command with a ``<LOG_DIR>`` placeholder is shown instead.

    Args:
        eval_log_paths: List of ``.eval`` file paths.
        max_commands: Threshold above which commands collapse to a template.

    Returns:
        Markdown string (collapsible ``<details>`` block), or empty string if
        *eval_log_paths* is empty.
    """
    if not eval_log_paths:
        return ""

    log_dirs = sorted(set(str(p.parent) for p in eval_log_paths))

    lines = ["<details>", "<summary>Inspect view commands</summary>", ""]

    if len(log_dirs) > max_commands:
        lines.append("<pre><code>inspect view start --log-dir &lt;LOG_DIR&gt;</code></pre>")
        lines.append("")
        lines.append(
            f"Replace <code>&lt;LOG_DIR&gt;</code> with one of the {len(log_dirs)} log directories listed above."
        )
    else:
        commands = "\n".join(
            f"inspect view start --log-dir {d}" for d in log_dirs
        )
        lines.append(f"<pre><code>{commands}</code></pre>")

    lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def generate_report(
    df: pd.DataFrame,
    experiment_name: str,
    output_path: Path,
    config: Optional[dict] = None,
    future_directions: Optional[str] = None,
    generated_pngs: Optional[list[Path]] = None,
    eval_log_paths: Optional[list[Path]] = None,
    generated_by: Optional[str] = None,
) -> str:
    """
    Generate complete markdown report from evaluation data.

    Args:
        df: DataFrame with evaluation metrics (from evals_df_prep + parse_eval_metadata)
        experiment_name: Name of the experiment
        output_path: Path where report will be saved
        config: Optional experiment_summary.yaml config dict
        future_directions: Optional skill-generated analysis and suggestions
        generated_pngs: Optional list of PNG paths to embed. If None, embeds
            all PNGs found in the analysis directory (legacy behavior).
        eval_log_paths: Optional list of .eval file paths that contributed data.
        generated_by: Optional attribution string (e.g., "Claude Opus 4.6").

    Returns:
        Generated markdown report content

    Example:
        report = generate_report(
            df=logs_df,
            experiment_name="acs_income_2026-01-29",
            output_path=Path("experiments/acs_income/analysis/report.md"),
            future_directions="Based on the results, consider...",
            generated_pngs=[Path("analysis/scores_by_task.png")],
            eval_log_paths=[Path("logs/log1.eval"), Path("logs/log2.eval")],
            generated_by="Claude Opus 4.6",
        )
    """
    # Extract metrics
    metrics = extract_metrics(df)

    # Generate narrative
    narrative = generate_narrative(metrics)

    # Build report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    model_count = len(set(m.name for m in metrics))

    header_parts = [
        f"**Experiment:** {experiment_name}",
        f"**Report generated:** {timestamp}",
        f"**Models evaluated:** {model_count}",
    ]
    if generated_by:
        header_parts.append(f"**Generated by:** {generated_by}")

    report_lines = [
        "# Experiment Analysis Report\n",
        "  \n".join(header_parts) + "\n",
    ]

    # Add research question section if config has experiment metadata
    research_question = _format_research_question(config)
    if research_question:
        report_lines.append(research_question)

    # Detect supplementary metrics and extract calibration results
    detected = detect_metrics(df)
    calibration_results = None
    if detected.supplementary:
        calibration_results = extract_calibration_metrics(
            df, detected.supplementary
        ) or None

    model_table, footnotes = _format_model_table(metrics, calibration_results)

    report_lines.extend([
        "## Executive Summary\n",
        narrative + "\n",
        "## Model Comparison\n",
        model_table + "\n",
    ])

    if footnotes:
        report_lines.append("  \n".join(footnotes) + "\n")

    # Add per-task breakdown if applicable
    task_breakdown = _format_per_task_breakdown(metrics)
    if task_breakdown:
        report_lines.append(task_breakdown)

    # Add visualizations if PNGs exist
    visualizations = _format_visualizations(output_path, generated_pngs)
    if visualizations:
        report_lines.append(visualizations)

    # Add skill-generated analysis if provided
    if future_directions:
        report_lines.append("## Analysis & Interpretation\n")
        report_lines.append("*This section is generated by the analyze-experiment skill based on the results above.*\n\n")
        report_lines.append(future_directions + "\n")

    report_lines.append("---\n")

    # Provenance footer
    attribution = f"Generated by {generated_by} via analyze-experiment skill" if generated_by else "Generated by analyze-experiment skill"
    report_lines.append(f"*{attribution}*\n")

    if eval_log_paths:
        report_lines.append("<details>\n<summary>Source eval logs</summary>\n")
        for p in sorted(eval_log_paths):
            report_lines.append(f"- `{p}`")
        report_lines.append("\n</details>\n")

        view_commands = _format_inspect_view_commands(eval_log_paths)
        if view_commands:
            report_lines.append(view_commands)

    report_content = "\n".join(report_lines)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content)

    return report_content
