# Summarize Experiments

You are helping the user create comprehensive markdown reports that summarize training experiments, including checkpoint verification, training metrics, and results comparison.

## Your Task

Generate a detailed markdown report that documents completed fine-tuning experiments with training metrics, checkpoint verification, and comparative analysis.

## Report Structure

### 1. Title and Overview

Start with:
- Experiment set name (from directory name)
- Date of experiments
- Brief description of what was tested (e.g., "LoRA rank comparison on Llama models")
- Overall completion status

Example:
```markdown
# Experiment Report: cap_8L_llama32_lora_comparison_2025-10-18

**Date:** October 18, 2025
**Experiment:** LoRA rank comparison for capitalization task
**Models:** Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct
**Variants:** Rank 4 vs Rank 64
**Status:** ✓ All 4 experiments completed successfully

## Overview

This experiment compared different LoRA ranks (4 vs 64) across two model sizes (1B and 3B)
for the capitalization task using 8-letter words with 10,000 training samples.
```

### 2. Checkpoint Verification

Verify all experiment checkpoints exist and document:
- Which epochs were saved
- Checkpoint completeness (model weights, adapter weights, configs)
- File sizes
- Any missing or corrupted checkpoints

**Process:**
1. List all experiment directories
2. Check each epoch directory for required files:
   - `model-*.safetensors` (full model weights)
   - `adapter_model.pt` and `adapter_model.safetensors` (adapter weights)
   - `adapter_config.json` (adapter configuration)
   - Tokenizer files
3. Verify file sizes are reasonable (not 0 bytes)
4. Report checkpoint status

Example:
```markdown
## Checkpoint Verification

All experiments saved checkpoints for **epochs 0, 1, and 2**.

| Experiment  | Epoch 0 | Epoch 1 | Epoch 2 | Notes |
|-------------|---------|---------|---------|-------|
| 1B_rank4    | ✓       | ✓       | ✓       | Complete |
| 1B_rank64   | ✓       | ✓       | ✓       | Complete |
| 3B_rank4    | ✓       | ✓       | ✓       | Complete |
| 3B_rank64   | ✓       | ✓       | ✓       | Complete |

### Checkpoint Contents

Each epoch directory contains:
- Full model weights (`model-*.safetensors`)
- Adapter weights (`adapter_model.pt`, `adapter_model.safetensors`)
- Adapter configuration (`adapter_config.json`)
- Tokenizer files and configs
- Model configuration files
```

### 3. Training Results Summary

Extract and present key training metrics:
- Final training loss
- Final validation loss (if available)
- Total steps completed
- Training duration
- Throughput (tokens/sec/GPU)
- Memory usage

**How to extract metrics:**
1. Find the most recent SLURM output file: `slurm-*.out`
2. Search for "Run summary:" section
3. Extract: `global_step`, `loss`, `val_loss`, `tokens_per_second_per_gpu`, `peak_memory_*`
4. Parse training progress bars for epoch/step info

Example:
```markdown
## Training Results Summary

| Experiment  | Model | LoRA Rank | Epochs | Steps | Final Loss | Val Loss | Tokens/sec/GPU |
|-------------|-------|-----------|--------|-------|------------|----------|----------------|
| 1B_rank4    | 1B    | 4         | 3      | 1500  | 0.00154    | 0.02345  | 1224           |
| 1B_rank64   | 1B    | 64        | 3      | 1500  | 0.00036    | 0.02336  | 1252           |
| 3B_rank4    | 3B    | 4         | 3      | 3000  | 0.00031    | 0.01656  | 321            |
| 3B_rank64   | 3B    | 64        | 3      | 3000  | 0.00027    | 0.01830  | 486            |

### Training Duration

- **1B_rank4**: ~4.7 minutes (4:41)
- **1B_rank64**: ~4.9 minutes (4:54)
- **3B_rank4**: ~22.6 minutes (22:34)
- **3B_rank64**: ~15.8 minutes (15:46)
```

### 4. Key Findings

Provide analysis and insights:
- Performance comparison across configurations
- Impact of LoRA rank on training/validation loss
- Model size effects
- Training efficiency observations
- Unexpected results or anomalies

Example:
```markdown
## Key Findings

### 1. LoRA Rank Impact

**Training Loss:**
- Higher rank (64) consistently achieved lower final training loss
- 1B models: rank 64 was 4.3× lower than rank 4 (0.00036 vs 0.00154)
- 3B models: rank 64 was 1.1× lower than rank 4 (0.00027 vs 0.00031)

**Validation Loss:**
- Minimal difference between ranks within same model size
- 1B models: ~0.023 (both ranks)
- 3B models: ~0.017-0.018 (both ranks)

### 2. Model Size Impact

- 3B models achieved ~26% better validation loss than 1B models
- 3B models showed better generalization (smaller gap between train/val loss)
- Training time increased ~4-5× for 3B vs 1B models

### 3. Training Efficiency

- Rank 64 trained faster than rank 4 on 3B model (15.8 min vs 22.6 min)
- Likely due to better GPU utilization with larger adapter
- Throughput: 3B rank 64 (486 tok/s) > 3B rank 4 (321 tok/s)
```

### 5. Storage Summary

Document checkpoint sizes:
- Base model size
- Adapter sizes for different ranks
- Total storage per experiment
- Storage recommendations

Example:
```markdown
## Storage Summary

### Model and Adapter Sizes

| Component          | 1B Model | 3B Model |
|--------------------|----------|----------|
| Base Model         | 2.4 GB   | 6.1 GB   |
| Adapter (rank 4)   | 5.2 MB   | 11 MB    |
| Adapter (rank 64)  | 82 MB    | 172 MB   |

### Total Storage per Experiment

| Experiment  | 3 Epochs (Full Model) | 3 Epochs (Adapter Only) |
|-------------|-----------------------|-------------------------|
| 1B_rank4    | ~7.2 GB               | ~16 MB                  |
| 1B_rank64   | ~7.5 GB               | ~246 MB                 |
| 3B_rank4    | ~18.3 GB              | ~33 MB                  |
| 3B_rank64   | ~19.3 GB              | ~516 MB                 |

**Note:** Adapter-only storage is significantly more efficient if base model is shared.
```

### 6. Issues and Notes

Document any issues encountered:
- Failed jobs (even if training succeeded)
- SLURM errors
- Missing data
- Warnings or anomalies

Example:
```markdown
## Issues and Notes

### SLURM Exit Code Issue

All jobs were marked as `FAILED` (exit code 1:0) despite successful training completion.

**Cause:** SLURM script attempted to move output file to a location where it already existed:
```bash
mv: 'slurm-123.out' and '/path/to/slurm-123.out' are the same file
```

**Resolution:** Fixed in template (`tools/torchtune/templates/finetune_template.slurm`)
**Impact:** None - all training completed successfully, only affected exit code reporting

### Validation Tracking

Validation loss tracked every 50 steps as configured (`run_val_every_n_steps: 50`).
All experiments showed consistent validation loss convergence.
```

### 7. Next Steps

Suggest follow-up actions:
- Evaluation recommendations
- Additional experiments to run
- Analysis to perform
- Checkpoints to test

Example:
```markdown
## Next Steps

### 1. Run Evaluations

All experiments have standalone evaluation scripts ready:
- `1B_rank4/eval.py`
- `1B_rank64/eval.py`
- `3B_rank4/eval.py`
- `3B_rank64/eval.py`

**Recommended evaluation order:**
1. Evaluate final epoch (epoch_2) for all experiments
2. Compare exact match scores across configurations
3. Evaluate intermediate epochs if needed

### 2. Suggested Analysis

- Compare evaluation accuracy vs validation loss
- Test if higher LoRA rank improves task performance
- Analyze per-epoch performance progression
- Compare 1B vs 3B model capabilities

### 3. Potential Follow-up Experiments

- Test intermediate ranks (8, 16, 32) to find optimal rank
- Longer training (5-10 epochs) to see if validation loss continues improving
- Larger dataset (50K samples) to test scaling behavior
- Different learning rate schedules
```

## Implementation Guide

### Step-by-Step Process

1. **Find experiment directory:**
   ```bash
   ls -d /scratch/gpfs/MSALGANIK/mjs3/*_comparison_*
   ```

2. **List all experiments:**
   ```bash
   ls /path/to/experiment_set/
   ```

3. **For each experiment, check:**
   - Epoch directories exist: `ls experiment/epoch_*/`
   - Required files present: `ls experiment/epoch_0/adapter_model.safetensors`
   - Get file sizes: `ls -lh experiment/epoch_0/`

4. **Extract training metrics:**
   ```bash
   # Find most recent SLURM output
   ls -t experiment/slurm-*.out | head -1

   # Extract final metrics
   grep "Run summary:" -A 10 experiment/slurm-*.out
   ```

5. **Parse key metrics:**
   - `global_step`: Total training steps
   - `loss`: Final training loss
   - `val_loss`: Final validation loss
   - `tokens_per_second_per_gpu`: Throughput
   - `peak_memory_active`: GPU memory usage

6. **Check for issues:**
   - Look at job exit codes: `sacct -j <job_ids> --format=JobID,State,ExitCode`
   - Check for errors in SLURM output: `tail -50 experiment/slurm-*.out`

7. **Write report:**
   - Create `RESULTS.md` or `SUMMARY.md` in experiment set directory
   - Use markdown tables for structured data
   - Include emojis for status indicators (✓, ✗, ⚠️)
   - Add code blocks for commands and examples

## Output Format

Save the report as `RESULTS.md` or `EXPERIMENT_SUMMARY.md` in the experiment directory:

```
/scratch/gpfs/MSALGANIK/mjs3/cap_8L_llama32_lora_comparison_2025-10-18/
├── 1B_rank4/
├── 1B_rank64/
├── 3B_rank4/
├── 3B_rank64/
├── README.md              (if exists, describes experiment setup)
└── RESULTS.md            (your generated summary report)
```

## Best Practices

1. **Be Comprehensive:** Include all relevant metrics and observations
2. **Be Accurate:** Verify all numbers and file paths
3. **Be Clear:** Use tables, headings, and formatting for readability
4. **Be Helpful:** Provide actionable insights and next steps
5. **Document Issues:** Note any problems encountered, even if resolved
6. **Include Commands:** Show commands for reproducing analysis
7. **Cross-reference:** Link to relevant files (configs, logs, eval scripts)
8. **Version Control:** Consider committing reports to git for tracking

## Example Commands for User

Include these at the end of the report:

```markdown
## Appendix: Useful Commands

### View Training Logs
```bash
# View specific experiment log
cat /path/to/experiment/slurm-<job_id>.out

# View most recent log
ls -t /path/to/experiment/slurm-*.out | head -1 | xargs cat
```

### Check Checkpoint Contents
```bash
# List all checkpoints
find /path/to/experiment -name "adapter_model.safetensors"

# Check checkpoint sizes
du -sh /path/to/experiment/epoch_*
```

### Sync Wandb Logs
```bash
# Sync specific run
wandb sync /path/to/experiment/logs/wandb/offline-run-*

# Sync all runs in experiment set
for exp in 1B_rank4 1B_rank64 3B_rank4 3B_rank64; do
    wandb sync /path/to/$exp/logs/wandb/offline-run-*
done
```

### Run Evaluations
```bash
# Evaluate final checkpoint
inspect eval /path/to/experiment/eval.py --model /path/to/experiment/epoch_2

# Or use setup_inspect.py
python tools/inspect/setup_inspect.py --finetune_epoch_dir /path/to/experiment/epoch_2
sbatch inspect.slurm
```
```

## Template Variables

When creating reports, use these placeholders that you'll fill in:

- `{EXPERIMENT_SET_NAME}`: Name of the experiment directory
- `{DATE}`: Date of experiments
- `{TASK_DESCRIPTION}`: What task was being fine-tuned
- `{MODEL_NAMES}`: List of models tested
- `{NUM_EXPERIMENTS}`: Total number of experiments
- `{COMPLETION_STATUS}`: How many succeeded/failed

## Notes

- **Always verify data:** Don't report metrics without checking the actual files
- **Handle missing data gracefully:** Note if some metrics aren't available
- **Provide context:** Explain what the numbers mean, not just raw data
- **Be objective:** Report results accurately, even if unexpected
- **Make it actionable:** Users should know what to do next after reading
