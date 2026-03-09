# Artifact Locations

This document describes the canonical file layout for cruijff_kit experiments. Use it to find where artifacts are created and where to look when debugging.

## Experiment Directory

Each experiment lives in a single directory. The root contains the experiment plan, results summary, and subdirectories for logs, runs, and analysis.

```
{experiment_dir}/
├── experiment_summary.yaml      # Experiment design (from design-experiment)
├── summary.md                   # Post-run results summary (from summarize-experiment)
├── logs/                        # All skill pipeline logs
│   ├── design-experiment.log
│   ├── scaffold-experiment.log
│   ├── scaffold-torchtune.log
│   ├── scaffold-inspect.log
│   ├── run-torchtune.log
│   ├── run-inspect.log
│   ├── analyze-experiment.log
│   └── summarize-experiment.log
├── {run_name}/                  # Per-run directory (one per run)
│   ├── setup_finetune.yaml      # Fine-tuning configuration input
│   ├── finetune.yaml            # Generated torchtune config
│   ├── finetune.slurm           # Generated SLURM script
│   ├── slurm-*.out              # SLURM stdout (stays in run dir)
│   └── eval/                    # Evaluation configs and results
│       ├── eval_config.yaml     # Evaluation configuration
│       ├── {task}_epoch{N}.slurm
│       ├── slurm-*.out          # Eval SLURM stdout
│       └── logs/                # inspect-ai evaluation logs
│           └── *.eval
└── analysis/                    # Visualizations and reports
    ├── report.md                # Markdown report with metrics
    ├── *.html                   # Interactive HTML plots
    └── *.png                    # Static plot exports
```

## Output Directory

Fine-tuning outputs (model checkpoints, WandB logs, GPU metrics) are written to a separate output directory to keep experiment configs and large model artifacts on different storage paths.

```
{output_dir_base}/ck-out-{run_name}/
├── epoch_0/                     # Checkpoint for epoch 0
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── ...
├── epoch_1/                     # Checkpoint for epoch 1 (if >1 epoch)
│   └── ...
├── slurm-*.out                  # Training SLURM stdout
├── gpu_metrics.csv              # GPU utilization from nvidia-smi
└── logs/
    └── wandb/                   # Weights & Biases run data
```

The `output_dir_base` is configured in `experiment_summary.yaml` under `output.base_directory`.

## Per-Stage Artifacts

| Stage | Creates | Location |
|-------|---------|----------|
| design-experiment | `experiment_summary.yaml`, `logs/design-experiment.log` | Experiment dir |
| scaffold-experiment | Run directories, configs, SLURM scripts, `logs/scaffold-*.log` | Experiment dir |
| run-experiment | SLURM outputs, checkpoints, eval logs, `logs/run-*.log` | Experiment dir + output dir |
| summarize-experiment | `summary.md`, `logs/summarize-experiment.log` | Experiment dir |
| analyze-experiment | `analysis/` directory, `logs/analyze-experiment.log` | Experiment dir |

## Where SLURM `.out` Files Land

SLURM writes `.out` files to the directory from which `sbatch` is invoked:

| Job Type | Submitted From | `.out` Location |
|----------|---------------|-----------------|
| Fine-tuning | `{run_dir}/` | `{run_dir}/slurm-*.out` |
| Evaluation | `{run_dir}/eval/` | `{run_dir}/eval/slurm-*.out` |

Training SLURM outputs may also appear in the output directory (`ck-out-{run_name}/slurm-*.out`) depending on the job's working directory configuration.
