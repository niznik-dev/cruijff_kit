# Cache Pre-building - HuggingFace Datasets Cache

**CRITICAL:** Pre-build HuggingFace datasets cache before submitting parallel eval jobs.

When multiple eval SLURM jobs launch simultaneously, they race to build the same HF datasets cache (`~/.cache/huggingface/datasets/`). This causes `FileNotFoundError` crashes and incomplete eval logs. Pre-building the cache once on the login node prevents this.

## When This Runs

After dependency checking (checkpoints verified), before job submission. This step takes seconds on the login node and only needs to run once per unique dataset.

## Process

Run the cache pre-build script from the experiment directory:

```bash
python src/tools/inspect/prebuild_cache.py experiment_summary.yaml
```

The script extracts unique dataset paths from `evaluation.tasks[].dataset`, validates each file exists, and builds the Arrow cache. It prints status per dataset and a summary on completion.

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

## Next Stage

Pass to evaluation_selection.md to determine which evaluations need submission.
