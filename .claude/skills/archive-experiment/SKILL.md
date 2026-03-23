---
name: archive-experiment
description: Archive a completed experiment, preserving metadata and evaluation results while deleting bulk artifacts (checkpoints, configs, SLURM scripts). Use after summarize-experiment or analyze-experiment when an experiment is complete and results have been reviewed.
---

# Archive Experiment

Archive a completed experiment down to its irreplaceable metadata, freeing storage occupied by reproducible artifacts (model checkpoints, generated configs, SLURM scripts).

## Your Task

Archive an experiment by:

1. Locating the experiment and validating completeness
2. Running a dry-run inventory showing what will be kept vs deleted
3. Resolving findings.md (prompting user if needed)
4. Getting user confirmation
5. Executing the archive (copy → verify → delete)
6. Logging the process

## Prerequisites

- experiment_summary.yaml exists in the experiment directory
- Experiment is complete (all runs evaluated) — or user explicitly uses `--force`
- **Conda environment activated** — the script requires PyYAML

## Workflow

### 1. Locate Experiment

Find the experiment directory:
- If user provided a path: use it
- If in an experiment directory (contains experiment_summary.yaml): use current directory
- Otherwise: ask user for the path

Read `experiment_summary.yaml` to determine:
- `experiment.name` — the experiment name
- `output.base_directory` — where model checkpoints live
- `runs` — list of run names
- `evaluation.matrix` — expected evaluations

### 2. Dry Run Preview

Run the archive script in dry-run mode:

```bash
python tools/experiment/archive_experiment.py <experiment_dir> --dry-run --pretty
```

If the default archive location is wrong, the user can specify `--archive-base <path>`.

**Default archive location:** `ck-archive/` as sibling of the experiment's grandparent directory. For an experiment at `__SCRATCH__/ck-experiments/my_experiment/`, the archive lands at `__SCRATCH__/ck-archive/my_experiment/`.

Show the user:
- Number of files and total size to KEEP
- Number of files and total size to DELETE
- Archive destination path
- Any incomplete runs (if `--force` would be needed)

### 3. Resolve findings.md

Check the dry-run output for `findings_source`:

1. **If a source was found** — tell the user which file will be copied as `findings.md`
2. **If no source was found** (`findings_source: null`) — ask the user:
   - "Would you like to write a brief findings.md before archiving? This captures what you learned from the experiment."
   - If yes: create `{experiment_dir}/findings.md` with user's content, then re-run dry-run
   - If no: proceed without findings.md

### 4. Confirm with User

Present the plan clearly:

```
Archive plan for: {experiment_name}
  Keep:   {N} files ({X} MB) → {archive_path}
  Delete: {M} files ({Y} MB)

Proceed? (y/n)
```

Use AskUserQuestion to get explicit confirmation before proceeding.

### 5. Execute Archive

Run the archive script without dry-run:

```bash
python tools/experiment/archive_experiment.py <experiment_dir> [--force] [--archive-base <path>] --pretty
```

The script will:
1. Copy all KEEP files to the archive directory
2. Verify the archive (file existence + size check)
3. Delete original experiment directory and output directories
4. Report results

### 6. Report Results

Show the user:
- Archive location
- Files preserved
- Storage freed
- Any warnings (incomplete runs, copy errors)

### 7. Create Log

Write `archive-experiment.log` to the **archive directory** (since the experiment directory is deleted):

```
{archive_dir}/archive.log
```

See [logging.md](logging.md) for action types and format.

## What Gets Kept

| Artifact | Archive Path | Why |
|----------|-------------|-----|
| `experiment_summary.yaml` | `experiment_summary.yaml` | Reproduces the entire experiment |
| `findings.md` | `findings.md` | What we learned |
| `summary.md` | `summary.md` | Quick results reference |
| `logs/*.log` | `logs/` | Audit trail of all skill operations |
| `{run}/eval/logs/*.eval` | `eval_logs/{run}/` | Actual evaluation results |
| `analysis/*` | `analysis/` | Reports, visualizations |

## What Gets Deleted

| Artifact | Why Safe |
|----------|---------|
| Model checkpoints (`ck-out-*/`) | Reproducible via fine-tuning |
| `finetune.yaml`, `finetune.slurm` | Reproducible via scaffold-torchtune |
| `setup_finetune.yaml` | Reproducible via scaffold-torchtune |
| Eval SLURM scripts (`*.slurm`) | Reproducible via scaffold-inspect |
| `eval_config.yaml` | Reproducible via scaffold-inspect |
| SLURM output logs (`slurm-*.out`) | Debugging artifacts, not archival |

## Error Handling

### Incomplete experiment (no --force)
- Show which runs are missing eval logs
- Refuse to archive
- Suggest: complete the runs, or use `--force` if results aren't needed

### Archive directory already exists
- Refuse to overwrite
- Suggest: remove the existing archive first, or choose a different `--archive-base`

### Verification failure
- Do NOT delete originals
- Report which files are missing or mismatched
- The archive directory may be in a partial state — user should investigate

### experiment_summary.yaml missing or invalid
- Refuse to proceed
- This is the one file we absolutely need

## Output Files

```
{archive_base}/{experiment_name}/
├── experiment_summary.yaml
├── findings.md
├── summary.md
├── logs/
│   ├── design-experiment.log
│   ├── scaffold-torchtune.log
│   └── ...
├── eval_logs/
│   ├── {run_name}/
│   │   └── *.eval
│   └── ...
├── analysis/
│   ├── report.md
│   └── *.html
└── archive.log
```

## Relationship to Other Skills

- **After:** summarize-experiment, analyze-experiment (when experiment is complete)
- **Reproducing:** Feed `experiment_summary.yaml` back through scaffold-experiment + run-experiment
- **Related:** #398 (round-trip unarchive test), #399 (batch archiving)
