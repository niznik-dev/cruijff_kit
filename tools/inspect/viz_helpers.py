"""
Helper functions for generating inspect-viz visualizations from evaluation logs.

This module provides utilities for loading experiment data and preparing it
for visualization with inspect-viz pre-built views.

Example usage:
    from tools.inspect.viz_helpers import load_experiment_logs

    logs_df = load_experiment_logs(
        experiment_path="/path/to/experiment",
        subdirs=["run1", "run2"],
        log_viewer_url="http://localhost:8000/logs/",
        metadata_extractors={
            "model": lambda df: df['model'].str.extract(r'(\d+B)', expand=False)
        }
    )
"""

import os
from typing import Callable

import pandas as pd
from inspect_ai.log import read_eval_log
from inspect_ai.analysis import (
    log_viewer,
    prepare,
    evals_df,
    EvalModel,
    EvalResults,
    EvalScores,
    EvalInfo,
    EvalTask,
)


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


def load_experiment_logs(
    experiment_path: str,
    subdirs: list[str],
    log_viewer_url: str,
    metadata_extractors: dict[str, Callable[[pd.DataFrame], pd.Series]],
) -> pd.DataFrame:
    """
    Load evaluation logs from an experiment with multiple subdirectories.

    This function collects .eval files from experiment subdirectories,
    extracts model paths and custom metadata, and prepares the data
    for visualization with inspect-viz.

    Args:
        experiment_path: Path to main experiment directory
        subdirs: List of subdirectory names containing eval/logs
        log_viewer_url: URL for log viewer (e.g., "http://localhost:8000/logs/")
        metadata_extractors: Dict mapping column names to lambda functions that
            extract values from the dataframe. Each function receives the full
            dataframe and should return a Series.
            E.g., {"model": lambda df: df['model'].str.extract(r'(\d+B)', expand=False)}

    Returns:
        Prepared DataFrame ready for visualization with inspect-viz.
        Includes log_viewer links for interactive drill-down.

    Example:
        logs_df = load_experiment_logs(
            experiment_path="/scratch/experiments/cap_wordlen_2026-01-12",
            subdirs=["Llama-3.2-1B-Instruct_5L", "Llama-3.2-3B-Instruct_5L"],
            log_viewer_url="http://localhost:8000/cap_wordlen_logs/",
            metadata_extractors={
                "model": lambda df: df['model'].str.extract(r'Llama-3\.2-(\d+B)', expand=False),
                "task_name": lambda df: df['task_arg_data_path'].str.extract(r'words_(\d+L)', expand=False)
            }
        )
    """
    # Build paths and LOG_DIRS mapping
    log_paths = [
        os.path.join(experiment_path, subdir, "eval", "logs")
        for subdir in subdirs
    ]
    log_dirs = {path: log_viewer_url for path in log_paths}

    # Collect all log files
    logs = []
    for path in log_paths:
        logs.extend([os.path.join(path, f) for f in os.listdir(path)])

    # Extract model paths from each log
    model_paths = [
        read_eval_log(log).eval.model_args['model_path']
        for log in logs
    ]

    # Read into eval-level dataframe
    logs_df = evals_df_prep(logs)

    # Add model_path column
    logs_df['model_path'] = model_paths

    # Apply custom metadata extractors
    for col_name, extractor in metadata_extractors.items():
        logs_df[col_name] = extractor(logs_df)

    # Prepare with log viewer for interactive links
    return prepare(logs_df, [log_viewer("eval", log_dirs)])
