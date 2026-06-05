# Artifact Locations

This document describes the canonical file layout for cruijff_kit experiments. Use it to find where artifacts are created and where to look when debugging.

## Experiment Directory

Each experiment lives in a single self-contained directory. The root contains the experiment plan, the generated dataset, results summary, and subdirectories for skill logs, per-run config, per-run training artifacts, and cross-run analysis.

```
{experiment_dir}/
├── experiment_summary.yaml      # Experiment design (from design-experiment)
├── summary.md                   # Post-run results summary (from summarize-experiment)
├── {dataset}.json               # Generated dataset (e.g. last_digits_k10_N2000.json)
├── logs/                        # Skill pipeline logs
│   ├── design-experiment.log
│   ├── scaffold-experiment.log
│   ├── scaffold-prepare-data.log
│   ├── scaffold-torchtune.log
│   ├── scaffold-inspect.log
│   ├── run-experiment.log
│   ├── run-torchtune.log
│   ├── run-torchtune.state.json # Submitter resume state (JSON)
│   ├── run-inspect.log
│   ├── run-inspect.state.json   # Submitter resume state (JSON)
│   ├── .detach                  # Optional sentinel — touch to detach the watcher
│   ├── monitor.json             # Optional live settings (poll_sec / stagger_sec / max_submit)
│   ├── summarize-experiment.log
│   └── explore-experiment.log
├── {run_name}/                  # Self-contained per-run directory (one per run)
│   ├── setup_finetune.yaml      # Fine-tuning configuration input
│   ├── finetune.yaml            # Generated torchtune config
│   ├── finetune.slurm           # Generated SLURM script
│   ├── eval/                    # Evaluation configs and results, organized
│   │   │                        # as one directory per cell — a (task, epoch)
│   │   │                        # pair. Per-cell layout lets two cells in the
│   │   │                        # same run carry different overrides
│   │   │                        # (e.g. system_prompt) without colliding. See
│   │   │                        # issue #498.
│   │   └── {task}_epoch{N}/     # One cell per (task, epoch) — name omits
│   │       │                    # `_epoch{N}` for base evals
│   │       ├── eval_config.yaml # Per-cell evaluation configuration
│   │       ├── cell.slurm       # Per-cell SLURM script (always named cell.slurm)
│   │       └── logs/            # inspect-ai evaluation logs
│   │           └── *.eval
│   └── artifacts/               # Training artifacts (checkpoints + W&B + GPU metrics)
│       ├── slurm-*.out          # Training SLURM stdout
│       ├── gpu_metrics.csv      # GPU utilization from nvidia-smi
│       ├── torchtune_config.yaml # Resolved torchtune config
│       ├── logs/
│       │   └── wandb/           # Weights & Biases run data
│       ├── epoch_0/             # Checkpoint for epoch 0
│       │   ├── adapter_model/   # HF-loadable adapter directory
│       │   ├── adapter_weights/ # Raw adapter weight files
│       │   ├── original/        # Base model snapshot
│       │   ├── model.safetensors # Merged model
│       │   ├── config.json
│       │   ├── generation_config.json
│       │   ├── torchtune_config.yaml
│       │   ├── gpu_metrics.csv
│       │   ├── slurm-*.out
│       │   ├── tokenizer.json
│       │   ├── tokenizer_config.json
│       │   ├── special_tokens_map.json
│       │   ├── LICENSE.txt
│       │   ├── README.md
│       │   └── USE_POLICY.md
│       └── epoch_N/             # Additional epoch checkpoints (if multi-epoch)
│           └── ...
└── analysis/                    # explore-experiment output (dir name predates the rename — see note)
    ├── report.md                # "Claude's Exploration" report (dir/file names retained; see note)
    ├── compute_metrics.json     # Raw compute metrics (JSON, see explore-experiment/generation.md for schema)
    ├── *.html                   # Interactive HTML plots
    └── *.png                    # Static plot exports
```

> **Note on the `analysis/` name.** The skill that writes this directory was renamed `analyze-experiment` → `explore-experiment`, and the report's title is now "Claude's Exploration." The directory and file names (`analysis/`, `report.md`) are deliberately retained: they are hardcoded by ~19 downstream readers (`archive_experiment.py`, `create-quiz`, `analyze-to-pdf`, the workflow tests, and tests), so renaming them would be wide churn for no functional gain. This is a documented decision, not a missed rename.

For a multi-run experiment (e.g. comparing two models), each run gets its own self-contained `{run_name}/` directory at the experiment root, with configs, `eval/`, and `artifacts/` all nested inside. A run can be copied as a unit (`cp -r {run_name}/ elsewhere/`).

## Training Artifact Directory

Training artifacts live at `{run_name}/artifacts/` inside the experiment directory — they are *not* written to a separate base path. The relevant fields in `experiment_summary.yaml`:

```yaml
output:
  base_directory: "{experiment_dir}"          # Set to the experiment dir itself
  checkpoint_pattern: "{run_name}/artifacts/epoch_{N}"
```

So the resolved checkpoint path for a given run is `{experiment_dir}/{run_name}/artifacts/epoch_{N}/`. See the tree above for the full contents of `{run_name}/artifacts/`.

> **Note:** A few files (`gpu_metrics.csv`, `torchtune_config.yaml`, `slurm-*.out`) appear both at the top of `{run_name}/artifacts/` and inside `epoch_0/`. The top-level copies are the live job's record; the per-epoch copies are snapshotted alongside the checkpoint.

## Per-Stage Artifacts

| Stage | Creates | Location |
|-------|---------|----------|
| design-experiment | `experiment_summary.yaml`, `logs/design-experiment.log` | Experiment dir |
| scaffold-experiment | Run directories, configs, SLURM scripts, `logs/scaffold-*.log` | Experiment dir |
| run-experiment | SLURM outputs, checkpoints, eval logs, `logs/run-*.log` | Experiment dir |
| summarize-experiment | `summary.md`, `logs/summarize-experiment.log` | Experiment dir |
| explore-experiment | `analysis/` directory, `logs/explore-experiment.log` | Experiment dir |
| archive-experiment | `archive.log`, mirrored experiment dir minus `*/artifacts/` | Archive dir (originals deleted) |

## Archive Directory

After archiving with `archive-experiment`, the experiment is mirrored under its project, with per-run `artifacts/` directories deleted as the only large items:

```
{archive_base}/{project}/{experiment_name}/
├── experiment_summary.yaml      # Reproduces the experiment via scaffold + run
├── findings.md                  # What was learned (only if user wrote one)
├── summary.md                   # Quick results reference (if produced)
├── {dataset}.json               # Generated dataset
├── logs/                        # Skill pipeline logs
│   ├── design-experiment.log
│   └── ...
├── {run_name}/                  # Per-run dir, mirroring the experiment layout
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   ├── finetune.slurm
│   └── eval/
│       └── {task}_epoch{N}/     # One cell per (task, epoch)
│           ├── eval_config.yaml
│           ├── cell.slurm
│           └── logs/
│               └── *.eval
├── analysis/                    # Reports and visualizations
│   ├── report.md
│   └── *.html
└── archive.log                  # Archive process log
```

The default `{archive_base}` is `ck-archive/` as a sibling of the experiment's grandparent dir. For an experiment at `__SCRATCH__/ck-projects/{project}/{experiment_name}/`, the archive lands at `__SCRATCH__/ck-archive/{project}/{experiment_name}/`. The `{project}` layer is required — `experiment.project` must be set in `experiment_summary.yaml`.

Symlinks are not archived. Per-run `artifacts/` directories are not archived (the only large items, regenerable by re-running fine-tuning).

## Where SLURM `.out` Files Land

| Job Type | `.out` Location |
|----------|-----------------|
| Fine-tuning (training) | `{experiment_dir}/{run_name}/artifacts/slurm-*.out`, plus a per-epoch copy at `{run_name}/artifacts/epoch_N/slurm-*.out` |
| Evaluation (fine-tuned cell) | `{run_dir}/artifacts/epoch_N/slurm-*.out` (eval GPU metrics land alongside the checkpoint) |
| Evaluation (base/control cell) | `{run_dir}/artifacts/slurm-*.out` |

Training jobs write `.out` files into the run's `artifacts/` directory. Eval `.out` files follow the GPU-metrics destination — for fine-tuned cells that's the per-epoch checkpoint directory, so the eval's `gpu_metrics.csv` lands next to the checkpoint it was measured against. The cell directory itself (`{run}/eval/{task}_epoch{N}/`) holds the configs, the slurm script, and the inspect-ai `.eval` logs, but not the SLURM stdout.
