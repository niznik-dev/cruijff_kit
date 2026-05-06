"""Compute estimation from prior experiment data.

Provides deterministic, testable scaling logic for estimating SLURM
resource requirements (time, GPUs, memory) for new experiments based
on metrics from prior runs.

This replaces the prose-based scaling instructions previously embedded
in the design-experiment skill documentation.
"""

from __future__ import annotations

import math

# Approximate parameter counts (billions) for supported models.
# Used for cross-model scaling when prior data is from a different model size.
MODEL_PARAMS_B: dict[str, float] = {
    "Llama-3.2-1B": 1.24,
    "Llama-3.2-1B-Instruct": 1.24,
    "Llama-3.2-3B-Instruct": 3.21,
    "Llama-3.1-8B-Instruct": 8.03,
    "Llama-3.3-70B-Instruct": 70.6,
    "Mistral-7B-v0.1": 7.24,
    "Mistral-7B-Instruct-v0.1": 7.24,
    "Qwen2.5-3B": 3.09,
    "Qwen2.5-3B-Instruct": 3.09,
}

# Default safety multiplier applied to scaled time estimates.
DEFAULT_SAFETY_MARGIN = 1.5

# Minimum SLURM time allocation (seconds). Prevents estimates that
# are too short to even start up.
MIN_TIME_SECONDS = 300  # 5 minutes


def parse_wall_time(time_str: str) -> int:
    """Parse HH:MM:SS or H:MM:SS to total seconds.

    Args:
        time_str: Wall time string (e.g., "0:05:23" or "00:05:23").

    Returns:
        Total seconds as integer.

    Raises:
        ValueError: If format is invalid.
    """
    parts = time_str.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS format, got: {time_str!r}")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + s


def format_wall_time(seconds: int) -> str:
    """Format seconds as H:MM:SS (SLURM-compatible).

    Args:
        seconds: Total seconds (non-negative).

    Returns:
        Time string in H:MM:SS format.
    """
    seconds = max(0, seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


def round_up_to_increment(seconds: int, increment_minutes: int = 5) -> int:
    """Round seconds up to the nearest N-minute increment.

    Args:
        seconds: Time in seconds.
        increment_minutes: Rounding increment in minutes.

    Returns:
        Rounded-up time in seconds.
    """
    increment_s = increment_minutes * 60
    return math.ceil(seconds / increment_s) * increment_s


def scale_finetune_time(
    prior_wall_seconds: int,
    prior_epochs: int,
    prior_dataset_size: int,
    new_epochs: int,
    new_dataset_size: int,
    prior_model: str | None = None,
    new_model: str | None = None,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> int:
    """Estimate fine-tuning wall time for a new experiment.

    Scales linearly with epochs and dataset size, then applies a
    model-size correction factor and safety margin.

    The linear scaling is approximate — it does not account for:
    - Per-epoch checkpoint overhead (constant, not proportional)
    - Dataset packing efficiency changes
    - CUDA/model startup time (fixed cost)

    The safety margin (default 1.5x) is intended to absorb these
    effects for typical workloads.

    Args:
        prior_wall_seconds: Actual wall time from prior run (seconds).
        prior_epochs: Number of epochs in prior run.
        prior_dataset_size: Training samples in prior run.
        new_epochs: Number of epochs in new experiment.
        new_dataset_size: Training samples in new experiment.
        prior_model: Model name from prior run (for cross-model scaling).
        new_model: Model name for new experiment.
        safety_margin: Multiplier for safety (default 1.5).

    Returns:
        Estimated wall time in seconds, rounded up to nearest
        5-minute increment, with minimum of MIN_TIME_SECONDS.
    """
    if prior_epochs <= 0 or prior_dataset_size <= 0:
        raise ValueError(
            f"Prior epochs ({prior_epochs}) and dataset size "
            f"({prior_dataset_size}) must be positive."
        )

    # Linear scaling by epochs and dataset size
    epoch_ratio = new_epochs / prior_epochs
    dataset_ratio = new_dataset_size / prior_dataset_size
    scaled = prior_wall_seconds * epoch_ratio * dataset_ratio

    # Cross-model scaling: time per step scales roughly linearly with
    # parameter count for LoRA fine-tuning (forward + backward pass).
    model_ratio = _model_size_ratio(prior_model, new_model)
    scaled *= model_ratio

    # Apply safety margin and round up
    estimated = int(scaled * safety_margin)
    estimated = round_up_to_increment(estimated)
    return max(estimated, MIN_TIME_SECONDS)


def scale_eval_time(
    prior_wall_seconds: int,
    prior_dataset_size: int,
    new_dataset_size: int,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> int:
    """Estimate evaluation wall time for a new experiment.

    Simpler than fine-tuning: scales only by dataset size.
    Epoch count is irrelevant for evaluation.

    Args:
        prior_wall_seconds: Actual eval wall time from prior run (seconds).
        prior_dataset_size: Eval samples in prior run.
        new_dataset_size: Eval samples in new experiment.
        safety_margin: Multiplier for safety (default 1.5).

    Returns:
        Estimated wall time in seconds, rounded up to nearest
        5-minute increment, with minimum of MIN_TIME_SECONDS.
    """
    if prior_dataset_size <= 0:
        raise ValueError(f"Prior dataset size ({prior_dataset_size}) must be positive.")

    ratio = new_dataset_size / prior_dataset_size
    estimated = int(prior_wall_seconds * ratio * safety_margin)
    estimated = round_up_to_increment(estimated)
    return max(estimated, MIN_TIME_SECONDS)


def recommend_batch_size(
    prior_gpu_mem_used_gb: float,
    prior_gpu_mem_total_gb: float,
    prior_batch_size: int,
    prior_model: str | None = None,
    new_model: str | None = None,
    prior_gradient_accumulation_steps: int = 1,
) -> dict:
    """Recommend batch size and gradient accumulation based on GPU memory.

    Memory is driven primarily by model size (weights dominate in LoRA).
    Batch size may increase within the memory envelope, but only
    conservatively — larger effective batch sizes should be achieved
    via ``gradient_accumulation_steps`` once batch size hits the memory
    ceiling. When batch size must decrease (e.g., scaling to a larger
    model), gradient accumulation is increased to preserve the prior
    effective batch size.

    The memory envelope target is 85% GPU utilization. Activation
    memory (which scales with batch size) is estimated at ~15% of
    total GPU memory for LoRA workloads, with model weights making
    up the rest.

    Args:
        prior_gpu_mem_used_gb: Mean GPU memory used in prior run.
        prior_gpu_mem_total_gb: Total GPU memory available.
        prior_batch_size: Batch size used in prior run.
        prior_model: Model name from prior run.
        new_model: Model name for new experiment.
        prior_gradient_accumulation_steps: Gradient accumulation steps
            from prior run (default 1).

    Returns:
        Dict with keys:
        - ``batch_size``: Recommended batch size (int).
        - ``gradient_accumulation_steps``: Recommended grad accum steps.
        - ``effective_batch_size``: batch_size × gradient_accumulation_steps.
        - ``reason``: Human-readable explanation.
        - ``mem_ratio``: Prior memory utilization ratio.
        - ``estimated_mem_gb``: Estimated GPU memory for new config.
    """
    prior_effective = prior_batch_size * prior_gradient_accumulation_steps

    if prior_gpu_mem_total_gb <= 0:
        return {
            "batch_size": prior_batch_size,
            "gradient_accumulation_steps": prior_gradient_accumulation_steps,
            "effective_batch_size": prior_effective,
            "reason": "No GPU memory data available; using prior settings.",
            "mem_ratio": None,
            "estimated_mem_gb": None,
        }

    mem_ratio = prior_gpu_mem_used_gb / prior_gpu_mem_total_gb
    model_ratio = _model_size_ratio(prior_model, new_model)

    # Estimate memory for new model at prior batch size.
    # Model weights scale with model_ratio; activation memory stays
    # roughly constant at the same batch size.
    estimated_mem = prior_gpu_mem_used_gb * model_ratio
    estimated_ratio = estimated_mem / prior_gpu_mem_total_gb

    recommended_bs = prior_batch_size
    reason_parts = []

    if model_ratio != 1.0:
        reason_parts.append(
            f"Prior: {prior_model}, batch_size={prior_batch_size}, "
            f"used {prior_gpu_mem_used_gb:.1f}/{prior_gpu_mem_total_gb:.0f} GB "
            f"({mem_ratio:.0%}). "
            f"Estimated for {new_model}: ~{estimated_mem:.0f} GB "
            f"({estimated_ratio:.0%})."
        )
    else:
        reason_parts.append(
            f"GPU memory at {mem_ratio:.0%} ({prior_gpu_mem_used_gb:.1f}/"
            f"{prior_gpu_mem_total_gb:.0f} GB)."
        )

    # --- Fit within memory envelope (target <85% utilization) ---

    if estimated_ratio > 0.85:
        # Over budget — reduce batch size until it fits
        while estimated_ratio > 0.85 and recommended_bs > 1:
            recommended_bs = max(1, recommended_bs // 2)
            # Halving batch reduces activation memory but model weights
            # stay constant. Assume ~15% total reduction per halving
            # (conservative for LoRA where weights dominate).
            estimated_mem *= 0.85
            estimated_ratio = estimated_mem / prior_gpu_mem_total_gb

        if recommended_bs < prior_batch_size:
            reason_parts.append(
                f"Reduced batch_size to {recommended_bs} to fit within "
                f"GPU memory (target <85% utilization)."
            )
        else:
            # batch_size is already 1 (or was 1) and still over budget
            reason_parts.append(
                f"Near GPU limit — keeping batch_size={recommended_bs}. Watch for OOM."
            )
    elif estimated_ratio < 0.5:
        # Significant headroom — try doubling batch size if it still
        # fits under the 85% ceiling.
        # Estimate: doubling batch increases activation memory (~15%
        # of total), so total increases by ~15%.
        doubled_mem = estimated_mem * 1.15
        doubled_ratio = doubled_mem / prior_gpu_mem_total_gb
        if doubled_ratio < 0.85:
            recommended_bs = prior_batch_size * 2
            estimated_mem = doubled_mem
            estimated_ratio = doubled_ratio
            reason_parts.append(
                f"Significant headroom — increased batch_size to "
                f"{recommended_bs} (doubled from {prior_batch_size})."
            )
        else:
            reason_parts.append(
                f"Significant headroom but doubling batch_size would "
                f"approach GPU limit. Keeping batch_size={recommended_bs}."
            )
    elif estimated_ratio < 0.7:
        # Moderate headroom — try a modest increase (+50%) if it fits.
        candidate = prior_batch_size + max(1, prior_batch_size // 2)
        # Rough scale: increasing by 50% adds ~7.5% total memory
        increased_mem = estimated_mem * 1.075
        increased_ratio = increased_mem / prior_gpu_mem_total_gb
        if increased_ratio < 0.85:
            recommended_bs = candidate
            estimated_mem = increased_mem
            estimated_ratio = increased_ratio
            reason_parts.append(
                f"Some headroom — increased batch_size to "
                f"{recommended_bs} (up from {prior_batch_size})."
            )
        else:
            reason_parts.append(
                f"Some headroom but increasing batch_size would "
                f"approach GPU limit. Keeping batch_size={recommended_bs}."
            )
    elif estimated_ratio >= 0.9:
        reason_parts.append(
            f"Near GPU limit — keeping batch_size={recommended_bs}. Watch for OOM."
        )
    else:
        # 70-85%: well-fitted
        reason_parts.append(f"Well-fitted — keeping batch_size={recommended_bs}.")

    # --- Compute gradient accumulation to preserve effective batch size ---
    recommended_gas = max(1, prior_effective // recommended_bs)
    effective = recommended_bs * recommended_gas

    if recommended_gas != prior_gradient_accumulation_steps:
        reason_parts.append(
            f"Set gradient_accumulation_steps={recommended_gas} "
            f"to maintain effective batch size of {effective}."
        )

    return {
        "batch_size": recommended_bs,
        "gradient_accumulation_steps": recommended_gas,
        "effective_batch_size": effective,
        "reason": " ".join(reason_parts),
        "mem_ratio": round(mem_ratio, 3),
        "estimated_mem_gb": round(estimated_mem, 1),
    }


def estimate_from_prior(
    prior_summary: dict,
    new_model: str,
    new_dataset_size: int,
    new_epochs: int,
    new_eval_dataset_size: int | None = None,
) -> dict:
    """Top-level estimation from a prior compute_metrics.json summary.

    Finds the best-matching finetune and eval jobs in the prior data,
    scales their wall times, and returns structured estimates.

    Args:
        prior_summary: Parsed summary dict from ``load_summary``.
        new_model: Model name for the new experiment.
        new_dataset_size: Training samples in new experiment.
        new_epochs: Number of training epochs.
        new_eval_dataset_size: Eval samples (defaults to
            ``new_dataset_size`` if not provided).

    Returns:
        Dict with keys:
        - ``finetune``: Dict with ``time`` (str), ``gpus`` (int),
          ``mem`` (str), ``time_seconds`` (int).
        - ``eval``: Same structure for eval jobs.
        - ``batch_size``: Result from ``recommend_batch_size``
          (or None if no GPU memory data).
        - ``prior_experiment``: Name of the prior experiment used.
        - ``scaling_details``: Human-readable summary of what
          was scaled and by how much.
    """
    if new_eval_dataset_size is None:
        new_eval_dataset_size = new_dataset_size

    prior_model = prior_summary.get("model")
    prior_dataset_size = prior_summary.get("dataset_size", 0)
    prior_epochs = prior_summary.get("epochs", 1)
    prior_batch_size = prior_summary.get("batch_size")
    prior_gas = prior_summary.get("gradient_accumulation_steps", 1)
    jobs = prior_summary.get("jobs", [])

    # Split by job type
    finetune_jobs = [j for j in jobs if j.get("job_type") == "finetune"]
    eval_jobs = [j for j in jobs if j.get("job_type") == "eval"]

    result: dict = {
        "finetune": None,
        "eval": None,
        "batch_size": None,
        "prior_experiment": prior_summary.get("experiment_name"),
        "scaling_details": [],
    }

    # --- Fine-tuning estimate ---
    if finetune_jobs and prior_dataset_size > 0:
        # Average wall time across finetune jobs
        wall_times = [
            parse_wall_time(j["wall_time"]) for j in finetune_jobs if j.get("wall_time")
        ]
        if wall_times:
            avg_wall = sum(wall_times) // len(wall_times)
            estimated_seconds = scale_finetune_time(
                prior_wall_seconds=avg_wall,
                prior_epochs=prior_epochs,
                prior_dataset_size=prior_dataset_size,
                new_epochs=new_epochs,
                new_dataset_size=new_dataset_size,
                prior_model=prior_model,
                new_model=new_model,
            )

            # GPU count and memory from prior (adjust for model size if needed)
            prior_gpus = finetune_jobs[0].get("gpus", 1)
            prior_mem = finetune_jobs[0].get("cpu_mem_allocated_gb")

            result["finetune"] = {
                "time": format_wall_time(estimated_seconds),
                "time_seconds": estimated_seconds,
                "gpus": prior_gpus,
                "mem": f"{int(prior_mem)}G" if prior_mem else "80G",
            }

            result["scaling_details"].append(
                f"Finetune: {format_wall_time(avg_wall)} (prior avg) "
                f"× {new_epochs}/{prior_epochs} epochs "
                f"× {new_dataset_size}/{prior_dataset_size} samples "
                f"× {_model_size_ratio(prior_model, new_model):.2f} model ratio "
                f"× {DEFAULT_SAFETY_MARGIN} safety "
                f"→ {format_wall_time(estimated_seconds)}"
            )

    # --- Eval estimate ---
    if eval_jobs:
        wall_times = [
            parse_wall_time(j["wall_time"]) for j in eval_jobs if j.get("wall_time")
        ]
        if wall_times:
            # Use the prior test split size for scaling base. If not in
            # summary, fall back to average eval wall time with dataset ratio.
            avg_wall = sum(wall_times) // len(wall_times)
            # Eval dataset size in prior: not always in summary.
            # Use the test split if available, otherwise assume same as new.
            prior_eval_size = new_eval_dataset_size  # conservative fallback

            estimated_seconds = scale_eval_time(
                prior_wall_seconds=avg_wall,
                prior_dataset_size=prior_eval_size,
                new_dataset_size=new_eval_dataset_size,
            )

            prior_gpus = eval_jobs[0].get("gpus", 1)
            prior_mem = eval_jobs[0].get("cpu_mem_allocated_gb")

            result["eval"] = {
                "time": format_wall_time(estimated_seconds),
                "time_seconds": estimated_seconds,
                "gpus": prior_gpus,
                "mem": f"{int(prior_mem)}G" if prior_mem else "80G",
            }

            result["scaling_details"].append(
                f"Eval: {format_wall_time(avg_wall)} (prior avg) "
                f"× {new_eval_dataset_size}/{prior_eval_size} samples "
                f"× {DEFAULT_SAFETY_MARGIN} safety "
                f"→ {format_wall_time(estimated_seconds)}"
            )

    # --- Batch size recommendation ---
    if finetune_jobs and prior_batch_size:
        # Find the first finetune job with GPU memory data
        for job in finetune_jobs:
            gpu_mem = job.get("gpu_mem_used_mean_gb")
            gpu_total = job.get("gpu_mem_total_gb")
            if gpu_mem is not None and gpu_total is not None:
                result["batch_size"] = recommend_batch_size(
                    prior_gpu_mem_used_gb=gpu_mem,
                    prior_gpu_mem_total_gb=gpu_total,
                    prior_batch_size=prior_batch_size,
                    prior_model=prior_model,
                    new_model=new_model,
                    prior_gradient_accumulation_steps=prior_gas,
                )
                break

    return result


def _model_size_ratio(
    prior_model: str | None,
    new_model: str | None,
) -> float:
    """Compute the parameter count ratio between two models.

    Returns 1.0 if either model is unknown or they're the same.
    """
    if not prior_model or not new_model:
        return 1.0
    if prior_model == new_model:
        return 1.0

    prior_params = MODEL_PARAMS_B.get(prior_model)
    new_params = MODEL_PARAMS_B.get(new_model)

    if prior_params is None or new_params is None:
        return 1.0

    return new_params / prior_params
