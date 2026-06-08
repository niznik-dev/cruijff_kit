---
name: summarize-experiment
description: Create the standard summary of experiment results from a completed (fine-tuned and evaluated) experiment. Run this right after run-experiment as the required post-run step — captures key metrics (loss, accuracy, regression metrics) into summary.md.
---

# Summarize Experiment

Generate a `summary.md` file capturing key metrics from a completed experiment. Think R's `summary()` for experiment results.

## Your Task

Create the standard post-run summary of experiment results:

1. Parse run status from experiment_summary.yaml
2. Extract final training loss from SLURM stdout
3. Extract accuracy from inspect-ai .eval files
4. Generate summary.md in experiment directory
5. Log the process in logs/summarize-experiment.log

## Prerequisites

- experiment_summary.yaml exists
- At least some runs have completed (partial results acceptable)
- run-experiment has been executed (or manual SLURM jobs run)
- **Conda environment activated** - The `parse_eval_log.py` script requires inspect-ai. Activate the conda environment from `claude.local.md` before running extraction commands.

## Workflow

### 1. Locate Experiment

Find the experiment directory:
- If in an experiment directory (contains experiment_summary.yaml): use current directory
- Otherwise: ask user for path

### 2. Parse Run Status

Read experiment_summary.yaml to identify runs:

**From `runs:` section:**
- `name`: Run identifier
- `type`: "fine-tuned" or "control"
- `model`: Model name
- `parameters`: Dict of hyperparameters (empty for control runs)

**From `evaluation.matrix:` section:**
- `run`: Run name
- `tasks`: List of evaluation task names
- `epochs`: List of epochs to evaluate (null for control runs)

**Determine status by checking filesystem:**
- Fine-tuning: Check for `{output_base}/{run_name}/artifacts/` and SLURM outputs
- Evaluation: Check for `{run_dir}/eval/*/logs/*.eval` files

### 3. Extract Training Loss

For each COMPLETED fine-tuning run:

1. Find SLURM stdout in the **output directory**:
   - Parse experiment_summary.yaml "Output" section for `output_dir_base`
   - Look in: `{output_dir_base}/{run_name}/artifacts/slurm-*.out`
   - If multiple files, use most recent by modification time
2. Extract final loss using `cruijff_kit.tools.torchtune.extract_loss`:
   ```python
   from cruijff_kit.tools.torchtune.extract_loss import final_loss
   result = final_loss(slurm_text)  # returns (epoch, step, loss) or None
   ```
   - The canonical regex and helpers live in `src/tools/torchtune/extract_loss.py`
   - `final_loss()` returns the last match (epoch, step, loss) or None
   - `extract_losses()` returns all matches as a list
   - The step number from the last match is the total training steps
3. Record: run_name, final_loss, total_steps, epoch, step

**Note:** Training SLURM outputs are in the output directory, NOT the run directory. (Missing stdout → record `N/A`, continue; see Error Handling.)

### 4. Extract Evaluation Accuracy

For each COMPLETED evaluation:

1. Find .eval files: `{run_dir}/eval/*/logs/*.eval`
2. For each .eval file, run:
   ```bash
   python -m cruijff_kit.tools.inspect.parse_eval_log {path}
   ```
3. Parse JSON output for the scorer's metrics (the `scorer` field tells you the type)
4. **Map to epoch using SLURM job names** (see below)
5. **Dispatch on scorer type:**
   - **Binary classification** (`match`/`exact_match` on 0/1 targets): also run `summary_binary.py` for balanced accuracy and F1 (below).
   - **`continuous_scorer`** (regression targets): run `summary_continuous.py` for mae/rmse/r²/parse_rate and a prediction-distribution glance (below). Accuracy is not meaningful here — do not report it.
   - **`risk_scorer`** (binary 0/1, but its CORRECT/INCORRECT is *exact-string-match* and `summary_binary.py` crashes on its `.eval` archives): take `accuracy`/`samples` from `parse_eval_log` only — do **not** run `summary_binary.py` on it. Argmax accuracy and format-compliance % are explore-experiment's job, not summarize's.
6. Record the metrics appropriate to the scorer — classification → accuracy, balanced_accuracy, f1; regression → mae, rmse, r_squared, parse_rate. Always record run_name, task, epoch, samples.

**Returns** (JSON): consume `task`, `accuracy`, `samples`, `scorer`, `model`. The `scorer` field drives the dispatch in step 5; on failure it returns `{"status": "error", "message": ...}` (see Error Handling).

#### Mapping Epochs via SLURM Job Names

The `.eval` files don't currently store epoch information directly. To reliably map each evaluation to its epoch:

1. **Find SLURM output files** in the eval directory: `{run_dir}/eval/slurm-*.out`
2. **Extract job IDs** from filenames (e.g., `slurm-2773062.out` → job ID 2773062)
3. **Query job names via sacct:**
   ```bash
   sacct -j {job_ids} --format=JobID,JobName%50
   ```
4. **Parse epoch from the job name** — scaffold-inspect names eval jobs `eval-{task}-ep{N}` (or `eval-{task}` for a base run with no epoch). The *run* is identified by which `{run_dir}/eval/` the slurm file sits in, not by the job name:
   - `eval-acs_income-ep0` → epoch 0
   - `eval-acs_income-ep9` → epoch 9

Reliable regardless of submission order, timing, or resubmissions — the epoch comes from the scaffold-set job name, **never** from file order or timestamp.

(Extraction failure → record `ERROR` for accuracy, continue; see Error Handling.)

#### Computing Balanced Accuracy and F1 (Binary Classification)

For binary (0/1) classification, run `summary_binary.py` for the imbalance-aware metrics:

```bash
python -m cruijff_kit.tools.inspect.summary_binary {path_to_eval_file} --json
```

Consume `accuracy`, `balanced_accuracy`, `f1`, and `class_balance` (`{frac_1, frac_0, n_1, n_0}`). Balanced accuracy and F1 are the imbalance-robust reads; record `class_balance` in **Datasets Used** as provenance (the "not a floor" rule lives there).

**Multiclass:** report accuracy only (Bal. Acc and F1 as "-"). **Regression** (`continuous_scorer`): use the path below, never accuracy.

#### Computing Regression Metrics (Continuous Targets)

For `continuous_scorer` (regression) tasks, run `summary_continuous.py`:

```bash
python -m cruijff_kit.tools.inspect.summary_continuous {path_to_eval_file} --json
```

Consume `metrics` (`mae`, `rmse`, `r_squared`, `parse_rate`) and the `prediction`/`target`/`residual` distributions. Two operational flags to always surface: **`parse_rate` below 1.0** means the model isn't really doing the task (non-numeric output — common on base models, see `references/scorers.md`), and a **prediction std of 0** means it emitted a constant (the low R² then reflects that, not noise).

### 5. Generate summary.md

Create `{experiment_dir}/summary.md` with the following structure:

```markdown
# Experiment Summary

**Experiment:** `{experiment_name}` | **Generated:** {timestamp} | **Status:** {X}/{Y} complete

## Run Status

| Run | Type | Fine-tuning | Evaluation |
|-----|------|-------------|------------|
| rank4_lr1e-5 | Fine-tuned | COMPLETED | COMPLETED |
| rank8_lr1e-5 | Fine-tuned | COMPLETED | COMPLETED |
| base_model | Control | N/A | COMPLETED |

## Training Results

| Run | Final Loss | Total Steps | Epochs | Duration |
|-----|------------|-------------|--------|----------|
| rank4_lr1e-5 | 0.234 | 250 | 2 | 8m 15s |
| rank8_lr1e-5 | 0.198 | 250 | 2 | 9m 02s |

**Notes:**
- Base model runs have no training loss (control)
- Duration from SLURM elapsed time (if available)

## Evaluation Results

| Run | Task | Epoch | Accuracy | Bal. Acc | F1 | Sat. | Samples |
|-----|------|-------|----------|----------|------|------|---------|
| rank4_lr1e-5 | capitalization | 0 | 0.85 | 0.83 | 0.82 |   | 100 |
| rank4_lr1e-5 | capitalization | 1 | 0.88 | 0.86 | 0.85 |   | 100 |
| rank8_lr1e-5 | capitalization | 0 | 0.82 | 0.80 | 0.78 |   | 100 |
| rank8_lr1e-5 | capitalization | 1 | 0.96 | 0.95 | 0.95 | ✓ | 100 |
| base_model | capitalization | - | 0.45 | 0.50 | 0.31 |   | 100 |

**Sat.** = saturated, `✓` when accuracy ≥ 0.95 (a maxed-out cell, no headroom left to measure). summarize flags it here; explore-experiment reads it from this column rather than recomputing.

**Best performing:** rank8_lr1e-5 (epoch 1) with 95% balanced accuracy

## Datasets Used

Record the literal file that backed each training run and each evaluation. For generated-text experiments these are the `{condition}_{split}_{hash8}.json` files scaffold resolved from `data.data_generation`; for standard experiments this is `data.training.path` and any per-task `dataset`/`eval_condition`.

Read from (in order of preference):

1. `{experiment_dir}/{run_name}/setup_finetune.yaml` → `input_dir_base` + `dataset_label` + `dataset_ext` = training dataset path for that run. This is authoritative — it's what torchtune actually loaded.
2. `{experiment_dir}/{run_name}/eval/*/eval_config.yaml` (per-cell — one config per (task, epoch); see issue #498) → test dataset path for that evaluation. Equivalently, the inspect `cell.slurm` script's `DATA_PATH`.
3. `logs/scaffold-torchtune.log` and `logs/scaffold-inspect.log` — include `resolve_dataset_path` entries recording the exact path scaffold chose.

| Run | Training dataset | Eval dataset(s) |
|-----|------------------|-----------------|
| rank4_lr1e-5 | `.../ck-data/generated/dict_subset_train_f18f10eb.json` | `.../dict_subset_test_f18f10eb.json` |
| rank8_lr1e-5 | `.../ck-data/generated/dict_full_train_d064ec15.json`   | `.../dict_full_test_d064ec15.json`   |

If the dataset file has a `.meta.json` sidecar (generated by `convert-tabular-to-text`), the sidecar contains the full canonical `data_generation` config that produced it — cite its `config_hash_short` in the table so the entire generation provenance is captured.

**Eval-set class balance (binary classification).** For binary tasks, also record the actual label split of each *distinct* eval dataset, taken from `summary_binary.py`'s `class_balance` field — e.g. `dict_full_test: 62% / 38% (n=100)`. Report it once per distinct eval set, not once per run (runs evaluated on the same task share the split). This is provenance — it lets a reader spot a skewed split at a glance. Do **not** present it as a "floor" or "the bar to beat": whether the split implies a meaningful accuracy baseline depends on experiment intent (often the base-model eval is the truer baseline), and that judgment belongs to explore-experiment.

## Incomplete Runs

| Run | Stage | Status | Notes |
|-----|-------|--------|-------|
| rank16_lr1e-5 | Fine-tuning | FAILED | Check slurm-12345.out |

## Next Steps

1. View detailed evaluation results: `inspect view --port=$(get_free_port)`
2. Export raw data: `inspect log export {run_dir}/eval/*/logs/*.eval --format csv`
3. Full analysis: `explore-experiment`

---
*Generated by summarize-experiment skill*
```

**Continuous/regression experiments:** the Evaluation Results section uses regression columns instead of accuracy — populate from `summary_continuous.py`:

| Run | Task | Epoch | MAE | RMSE | R² | Parse Rate | Samples |
|-----|------|-------|-----|------|-----|------------|---------|
| rank4_lr1e-5 | acs_age | 0 | 8.42 | 11.30 | 0.21 | 1.00 | 100 |
| base_model | acs_age | - | 14.10 | 18.05 | -0.12 | 0.62 | 100 |

Pick the best run by lowest RMSE (or highest R²), and always surface `parse_rate` — a run well below 1.0 isn't really doing the task.

### 6. Create Log

Document the process in `{experiment_dir}/logs/summarize-experiment.log`.

See [logging.md](logging.md) for action types and format.

## Error Handling

### If SLURM stdout missing
- Log warning with action type `EXTRACT_LOSS`
- Record "N/A" for loss in summary
- Continue with other runs

### If .eval file cannot be parsed
- Log error with file path
- Record "ERROR" for accuracy in summary
- Continue with other evaluations

### If all runs failed
- Generate summary noting all failures
- Include failure states in "Incomplete Runs" section
- Suggest troubleshooting steps

### If partial results
- Generate summary with available data
- Clearly indicate which runs are missing in "Incomplete Runs" section
- Still identify best performing run from available data

## Idempotency

Re-running overwrites `summary.md` — intentional, so the summary always reflects current state (e.g. after fixing and re-running failed runs).

## Output Files

Writes `summary.md` and `logs/summarize-experiment.log` in the experiment directory.

## Relationship to Other Skills

- **After:** run-experiment — this is the required post-run step once an experiment completes (or manual execution)
- **Before:** explore-experiment is optional and can be run any time afterward
- **Invoked by:** run-experiment runs this as its standard completion step (part of the skill flow, not a configured hook)

