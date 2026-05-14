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
    prior_tps_gpu: float,
    new_total_tokens: int,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> int:
    """Estimate fine-tuning wall time using throughput math.

    Predicts wall time as ``new_total_tokens / prior_tps_gpu``, then
    applies a safety margin and rounds up. Because tps is in
    tokens/sec/GPU (a per-token rate), this naturally absorbs changes
    in seq_len, batch_size, epochs, and dataset size — anything that
    affects total token count is captured by ``new_total_tokens``.

    The safety margin (default 1.5x) absorbs fixed startup costs
    (CUDA init, model load, checkpointer overhead) and end-of-run
    overhead (final checkpoint write), which are not proportional to
    token count.

    Args:
        prior_tps_gpu: Mean tokens/sec/GPU from prior run (typically
            ``tps_gpu_train_mean`` from the wandb end-of-run summary).
        new_total_tokens: Total training tokens for new run, usually
            ``epochs × dataset_size × seq_len``.
        safety_margin: Multiplier for safety (default 1.5).

    Returns:
        Estimated wall time in seconds, rounded up to nearest
        5-minute increment, with minimum of MIN_TIME_SECONDS.

    Raises:
        ValueError: If prior_tps_gpu or new_total_tokens is non-positive.
    """
    if prior_tps_gpu <= 0:
        raise ValueError(f"prior_tps_gpu must be positive, got {prior_tps_gpu}")
    if new_total_tokens <= 0:
        raise ValueError(f"new_total_tokens must be positive, got {new_total_tokens}")

    estimated = int((new_total_tokens / prior_tps_gpu) * safety_margin)
    estimated = round_up_to_increment(estimated)
    return max(estimated, MIN_TIME_SECONDS)


def scale_eval_time(
    prior_tps_gpu: float,
    new_total_tokens: int,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> int:
    """Estimate evaluation wall time using throughput math.

    Predicts wall time as ``new_total_tokens / prior_tps_gpu`` × safety
    margin. For inspect-ai evals, ``prior_tps_gpu`` is end-to-end
    (``tps_gpu_eval_e2e`` from parser): total tokens (input + output)
    divided by total wall time.

    Args:
        prior_tps_gpu: End-to-end tokens/sec/GPU from prior eval run.
        new_total_tokens: Estimated total tokens for new eval, usually
            ``prior_total_tokens × new_dataset_size / prior_dataset_size``
            (assuming similar tokens-per-sample distribution).
        safety_margin: Multiplier for safety (default 1.5).

    Returns:
        Estimated wall time in seconds, rounded up to nearest
        5-minute increment, with minimum of MIN_TIME_SECONDS.

    Raises:
        ValueError: If prior_tps_gpu or new_total_tokens is non-positive.
    """
    if prior_tps_gpu <= 0:
        raise ValueError(f"prior_tps_gpu must be positive, got {prior_tps_gpu}")
    if new_total_tokens <= 0:
        raise ValueError(f"new_total_tokens must be positive, got {new_total_tokens}")

    estimated = int((new_total_tokens / prior_tps_gpu) * safety_margin)
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


def _mean_tps(jobs: list[dict], field: str) -> float | None:
    """Return mean tps across jobs that have ``field``, or None if none do."""
    values = [j[field] for j in jobs if isinstance(j.get(field), int | float)]
    if not values:
        return None
    return sum(values) / len(values)


def estimate_from_prior(
    prior_summary: dict,
    new_model: str,
    new_dataset_size: int,
    new_epochs: int,
    new_seq_len: int,
    new_eval_dataset_size: int | None = None,
) -> dict:
    """Top-level estimation from a prior compute_metrics.json summary.

    Uses throughput-based scaling: prior tokens-per-second-per-GPU
    (parsed from slurm-out and stored on each job dict via
    ``throughput_parsers.enrich_job_with_throughput``) divided into
    the new run's total token count yields predicted wall time.

    Args:
        prior_summary: Parsed summary dict from ``load_summary``.
        new_model: Model name for the new experiment (used only for
            batch-size memory scaling).
        new_dataset_size: Training samples in new experiment.
        new_epochs: Number of training epochs.
        new_seq_len: Max sequence length for new experiment. Combined
            with epochs × dataset_size to compute total training
            tokens.
        new_eval_dataset_size: Eval samples (defaults to
            ``new_dataset_size`` if not provided).

    Returns:
        Dict with keys:
        - ``finetune``: Dict with ``time`` (str), ``gpus`` (int),
          ``mem`` (str), ``time_seconds`` (int). None if no prior
          finetune job had a parseable tps value.
        - ``eval``: Same structure for eval jobs. None if no prior
          eval job had a parseable tps value.
        - ``batch_size``: Result from ``recommend_batch_size``
          (or None if no GPU memory data).
        - ``prior_experiment``: Name of the prior experiment used.
        - ``scaling_details``: Human-readable summary of what
          was scaled and by how much.
    """
    if new_eval_dataset_size is None:
        new_eval_dataset_size = new_dataset_size

    prior_model = prior_summary.get("model")
    prior_eval_size = prior_summary.get("eval_dataset_size")
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

    # --- Cross-model warning ---
    # tps cannot extrapolate reliably across model sizes: in IO-bound regimes
    # (short seq_len, small batch) the slowdown ratio between models is far
    # smaller than the parameter-count ratio (e.g. 8B/1B ≈ 2x at seq=128, not
    # the 6.5x param-count would predict); in compute-bound regimes (long
    # seq_len, large batch) the slowdown approaches the parameter-count ratio.
    # No single correction factor handles both. See issue #490 for a planned
    # empirical calibration table.
    if prior_model and prior_model != new_model:
        result["scaling_details"].append(
            f"CROSS-MODEL WARNING: prior used {prior_model}, new will use "
            f"{new_model}. tps does not extrapolate reliably across model "
            "sizes — the slowdown ratio depends on whether the run is IO- or "
            "compute-bound. Estimate will under-predict if the new model is "
            "compute-bound; over-predict if IO-bound. Prefer a same-model "
            "prior when possible. See cruijff_kit#490 for empirical "
            "calibration plans."
        )

    # --- Fine-tuning estimate ---
    prior_train_tps = _mean_tps(finetune_jobs, "tps_gpu_train_mean")
    if finetune_jobs and prior_train_tps:
        new_total_tokens = new_dataset_size * new_seq_len * new_epochs
        estimated_seconds = scale_finetune_time(
            prior_tps_gpu=prior_train_tps,
            new_total_tokens=new_total_tokens,
        )

        prior_gpus = finetune_jobs[0].get("gpus", 1)
        prior_mem = finetune_jobs[0].get("cpu_mem_allocated_gb")

        result["finetune"] = {
            "time": format_wall_time(estimated_seconds),
            "time_seconds": estimated_seconds,
            "gpus": prior_gpus,
            "mem": f"{int(prior_mem)}G" if prior_mem else "80G",
        }

        result["scaling_details"].append(
            f"Finetune: prior tps={prior_train_tps:.1f} tok/s/gpu "
            f"× new tokens={new_total_tokens:,} "
            f"({new_dataset_size:,} samples × {new_seq_len} seq_len × {new_epochs} epochs) "
            f"× {DEFAULT_SAFETY_MARGIN} safety "
            f"→ {format_wall_time(estimated_seconds)}"
        )

    # --- Eval estimate ---
    prior_eval_tps = _mean_tps(eval_jobs, "tps_gpu_eval_e2e")
    prior_eval_total_tokens = _mean_tps(eval_jobs, "total_tokens")
    if eval_jobs and prior_eval_tps and prior_eval_total_tokens:
        if prior_eval_size and prior_eval_size > 0:
            tokens_per_sample = prior_eval_total_tokens / prior_eval_size
            new_total_tokens = int(tokens_per_sample * new_eval_dataset_size)
            scale_note = (
                f"({new_eval_dataset_size:,} samples "
                f"× {tokens_per_sample:.0f} tokens/sample, "
                f"derived from {int(prior_eval_total_tokens):,} prior tokens / "
                f"{prior_eval_size:,} prior samples)"
            )
        else:
            new_total_tokens = int(prior_eval_total_tokens)
            scale_note = (
                f"({int(prior_eval_total_tokens):,} prior tokens, "
                "prior eval dataset size unknown — no scaling applied)"
            )

        estimated_seconds = scale_eval_time(
            prior_tps_gpu=prior_eval_tps,
            new_total_tokens=new_total_tokens,
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
            f"Eval: prior tps={prior_eval_tps:.1f} tok/s/gpu "
            f"× new tokens={new_total_tokens:,} {scale_note} "
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
