"""Throughput parsers for SLURM job stdout.

Extracts tokens-per-second-per-GPU and related quantities from slurm-out
files produced by torchtune fine-tunes (via the wandb end-of-run summary)
and inspect-ai evaluations (via the run footer).

These values feed estimate_compute.scale_finetune_time /
scale_eval_time, which predict wall-time as total_tokens / tps.

v1 supports single-GPU runs only; the torchtune parser hard-fails on
distributed recipes (see issue #473 for multi-GPU follow-up).
"""

import re
import sys
from pathlib import Path

# Torchtune wandb end-of-run summary line, e.g.
#   "wandb: tokens_per_second_per_gpu 186.75362"
_WANDB_TPS_RE = re.compile(
    r"^wandb:\s+tokens_per_second_per_gpu\s+([\d.]+)\s*$",
    re.MULTILINE,
)

# Wandb summary global_step line, e.g. "wandb:               global_step 2000"
_WANDB_STEP_RE = re.compile(
    r"^wandb:\s+global_step\s+(\d+)\s*$",
    re.MULTILINE,
)

# Torchtune recipe banner at top of slurm-out, e.g.
#   "Running LoRAFinetuneRecipeSingleDevice with resolved config:"
_RECIPE_BANNER_RE = re.compile(r"^Running\s+(\S+)\s+with resolved config", re.MULTILINE)

# Inspect-ai footer total time, e.g. "total time:                              0:05:00"
_INSPECT_TOTAL_TIME_RE = re.compile(
    r"^total time:\s+(\d+):(\d{2}):(\d{2})\s*$",
    re.MULTILINE,
)

# Inspect-ai footer tokens line, e.g.
#   "hf/Llama-3.2-3B-Instruct_mc256_hifreq    302,904 tokens [I: 297,904, O: 5,000]"
_INSPECT_TOKENS_RE = re.compile(
    r"^\S+\s+([\d,]+)\s+tokens\s+\[I:\s*([\d,]+),\s*O:\s*([\d,]+)\]\s*$",
    re.MULTILINE,
)


def parse_torchtune_throughput(slurm_out_text: str) -> dict:
    """Extract throughput from a torchtune fine-tune slurm-out file.

    Parses the wandb end-of-run summary block. The wandb
    ``tokens_per_second_per_gpu`` value is the mean over the run
    (including warmup), which is the correct quantity for predicting
    wall-time of a similar run via ``total_tokens / tps``.

    Args:
        slurm_out_text: Full text of a slurm-%j.out file.

    Returns:
        Dict with keys:
            tps_gpu_train_mean: float, tokens/sec/GPU, mean over run
            global_step: int, total optimizer steps from wandb summary

    Raises:
        ValueError: If the wandb tps line is missing (run did not complete,
            wandb format changed, or a non-wandb metric_logger was used),
            or if the recipe banner indicates a distributed recipe (v1
            supports single-GPU only).
    """
    recipe_match = _RECIPE_BANNER_RE.search(slurm_out_text)
    if recipe_match:
        recipe_name = recipe_match.group(1)
        if "Distributed" in recipe_name or "distributed" in recipe_name.lower():
            raise ValueError(
                f"Distributed recipe {recipe_name!r} is not supported in v1 "
                "(single-GPU only). See issue #473 for multi-GPU follow-up."
            )

    tps_match = _WANDB_TPS_RE.search(slurm_out_text)
    if not tps_match:
        raise ValueError(
            "Could not find 'wandb: tokens_per_second_per_gpu <value>' in slurm-out. "
            "The run may not have completed, wandb may not be the configured "
            "metric_logger, or the wandb summary format may have changed."
        )

    step_match = _WANDB_STEP_RE.search(slurm_out_text)
    if not step_match:
        raise ValueError(
            "Found tps line but not 'wandb: global_step <N>' — unexpected wandb "
            "summary format."
        )

    return {
        "tps_gpu_train_mean": float(tps_match.group(1)),
        "global_step": int(step_match.group(1)),
    }


def parse_inspect_throughput(slurm_out_text: str) -> dict:
    """Extract throughput from an inspect-ai eval slurm-out file.

    Parses the inspect-ai run footer for total wall time and total
    tokens, then derives end-to-end tokens-per-second-per-GPU.

    Args:
        slurm_out_text: Full text of a slurm-%j.out file.

    Returns:
        Dict with keys:
            total_seconds: int, eval wall time in seconds
            total_tokens: int, sum of input + output tokens
            input_tokens: int
            output_tokens: int
            tps_gpu_eval_e2e: float, total_tokens / total_seconds

    Raises:
        ValueError: If the inspect footer is missing the total time line
            or the tokens line (eval may not have completed, or footer
            format changed upstream).
    """
    time_match = _INSPECT_TOTAL_TIME_RE.search(slurm_out_text)
    if not time_match:
        raise ValueError(
            "Could not find 'total time: H:MM:SS' line in slurm-out. "
            "The eval may not have completed, or the inspect-ai footer "
            "format may have changed."
        )

    hours, minutes, seconds = (int(x) for x in time_match.groups())
    total_seconds = hours * 3600 + minutes * 60 + seconds
    if total_seconds == 0:
        raise ValueError(
            "Inspect-ai reported total time of 0 — cannot compute throughput."
        )

    tokens_match = _INSPECT_TOKENS_RE.search(slurm_out_text)
    if not tokens_match:
        raise ValueError(
            "Could not find inspect-ai tokens footer line (expected format: "
            "'<model> N,NNN tokens [I: ..., O: ...]'). Footer format may have changed."
        )

    total_tokens = int(tokens_match.group(1).replace(",", ""))
    input_tokens = int(tokens_match.group(2).replace(",", ""))
    output_tokens = int(tokens_match.group(3).replace(",", ""))

    return {
        "total_seconds": total_seconds,
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tps_gpu_eval_e2e": total_tokens / total_seconds,
    }


def enrich_job_with_throughput(
    job: dict,
    slurm_out_path: str | Path,
) -> dict:
    """Merge throughput fields into a job dict by parsing its slurm-out.

    Routes to ``parse_torchtune_throughput`` or ``parse_inspect_throughput``
    based on ``job["job_type"]``. On any parse failure (missing line,
    malformed output, file not found, unsupported recipe) a warning is
    emitted to stderr and the job dict is returned unmodified. The
    downstream estimator raises a clear error if it later tries to scale
    from a job that lacks tps fields, so silent omission here is safe.

    Args:
        job: Job dict with at least ``job_type`` (``"finetune"`` or
            ``"eval"``) and ``run_name`` (used for warning messages).
        slurm_out_path: Path to the slurm-%j.out file for this job.

    Returns:
        The job dict with throughput fields merged in on success, or
        unmodified on failure.
    """
    path = Path(slurm_out_path)
    run_name = job.get("run_name", "unknown")
    job_type = job.get("job_type", "")

    if not path.exists():
        print(
            f"WARNING: slurm-out not found for {run_name} at {path} — "
            "skipping throughput enrichment.",
            file=sys.stderr,
        )
        return job

    try:
        text = path.read_text()
    except OSError as exc:
        print(
            f"WARNING: could not read slurm-out for {run_name}: {exc} — "
            "skipping throughput enrichment.",
            file=sys.stderr,
        )
        return job

    try:
        if job_type == "finetune":
            fields = parse_torchtune_throughput(text)
        elif job_type == "eval":
            fields = parse_inspect_throughput(text)
        else:
            print(
                f"WARNING: unknown job_type {job_type!r} for {run_name} — "
                "skipping throughput enrichment.",
                file=sys.stderr,
            )
            return job
    except ValueError as exc:
        print(
            f"WARNING: could not parse throughput for {run_name}: {exc}",
            file=sys.stderr,
        )
        return job

    return {**job, **fields}
