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
import os

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
        if meta_str and isinstance(meta_str, str):
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


def detect_metrics(logs_df: pd.DataFrame) -> list[str]:
    """
    Detect available metrics from score columns in the dataframe.

    Args:
        logs_df: DataFrame with score columns (e.g., score_match_accuracy)

    Returns:
        List of metric names (e.g., ['match', 'includes'])

    Example:
        metrics = detect_metrics(logs_df)
        for metric in metrics:
            score_col = f"score_{metric}_accuracy"
            # ... generate plot
    """
    score_cols = [
        c for c in logs_df.columns
        if c.startswith('score_') and c.endswith('_accuracy')
    ]
    return [c.replace('score_', '').replace('_accuracy', '') for c in score_cols]


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
