# Cache Pre-building - HuggingFace Datasets Cache

**CRITICAL:** Pre-build HuggingFace datasets cache before submitting parallel eval jobs.

When multiple eval SLURM jobs launch simultaneously, they race to build the same HF datasets cache (`~/.cache/huggingface/datasets/`). This causes `FileNotFoundError` crashes and incomplete eval logs. Pre-building the cache once on the login node prevents this.

## When This Runs

After dependency checking (checkpoints verified), before job submission. This step takes seconds on the login node and only needs to run once per unique dataset.

## Process

### 1. Extract Unique Dataset Paths

Parse `experiment_summary.yaml` and collect unique dataset paths from `evaluation.tasks[].dataset`:

```python
import yaml

with open("experiment_summary.yaml") as f:
    config = yaml.safe_load(f)

dataset_paths = set()
for task in config.get("evaluation", {}).get("tasks", []):
    dataset = task.get("dataset")
    if dataset:
        dataset_paths.add(dataset)
```

Each unique `data_path` produces a different cache entry. Only unique paths need pre-building.

### 2. Pre-build Cache for Each Dataset

For each unique dataset path, run:

```python
from datasets import load_dataset

for path in dataset_paths:
    load_dataset("json", data_files=path, field="test", split="train")
```

This builds the Arrow cache files that all eval jobs will share. Run this in a Python one-liner or short script on the login node:

```bash
python -c "
from datasets import load_dataset
load_dataset('json', data_files='DATA_PATH', field='test', split='train')
print('Cache built for: DATA_PATH')
"
```

Replace `DATA_PATH` with each unique dataset path.

### 3. Verify Cache Built

After pre-building, the cache directory should contain Arrow files:

```bash
ls ~/.cache/huggingface/datasets/json/
```

Verify that entries exist (the exact directory names are hash-based).

## Logging

Log each cache pre-build operation:

```
[YYYY-MM-DD HH:MM:SS] CACHE_PREBUILD: Pre-building HF datasets cache
Unique datasets: 1
[YYYY-MM-DD HH:MM:SS] CACHE_BUILT: /path/to/dataset.json
[YYYY-MM-DD HH:MM:SS] CACHE_PREBUILD_COMPLETE: All dataset caches ready
```

## Error Handling

**If cache pre-build fails:**
- Log the error with the dataset path
- Warn the user: cache pre-building failed, parallel evals may encounter race conditions
- Ask: "Continue with eval submission anyway?"
- If yes: proceed (jobs may fail from cache races)
- If no: stop and investigate

**If dataset path doesn't exist:**
- Log warning for the specific path
- Continue pre-building other datasets
- Report missing datasets in summary

## Why This Matters

Without pre-building, the first eval job to run triggers cache construction. If multiple jobs start simultaneously (common with SLURM), they all try to build the same cache concurrently, causing file system conflicts. Building once beforehand eliminates the race entirely.

## Next Stage

Pass to evaluation_selection.md to determine which evaluations need submission.
