# Plan Runs

You are helping the user plan a fine-tuning runs workflow for LLMs.

## Your Task

Guide the user through defining the structure of their experiments by asking questions, verifying resources, and clarifying their goals.

## Questions to Ask

1. **What do you want to vary in these runs?**
   - Different model sizes? (e.g., 1B-Instruct, 3B-Instruct, 8B-Instruct, 70B-Instruct)
   - Different LoRA ranks?
   - Different datasets?
   - Different hyperparameters?
   - Any combination of the above?

2. **Resource constraints:**
   - How many GPUs per job? (Default: 1 GPU = single-device recipe, >1 = distributed recipe)

3. **Validation during training:**
   - Do you want to track validation loss during training? (requires `_val` recipe variant)

4. **Control condition:**
   - Should we include control runs (evaluating base models without fine-tuning)?
   - Note: Controls help measure the effect of fine-tuning vs. base model performance

5. **Hyperparameters** (if not specified, ask the user or calculate from prior runs):
   - Epochs? (Suggest: 1-3, depends on task complexity)
   - Batch size? (Calculate from prior runs and GPU memory - see Batch Size Estimation section below)
   - LoRA alpha? (Default: 2 × rank, auto-set by setup_finetune.py)

## Required Ingredients

For each run, verify and identify:

### 1. Base Models
- **Location**: `/scratch/gpfs/MSALGANIK/pretrained-llms/`
- **Action**: Check that model paths exist using `ls` before proceeding
- **If missing**: Suggest using `download-model-from-hf` skill

### 2. Dataset
- **Check**: Does the dataset exist?
- **Verify**: Does it have train/validation/test splits?
- **If missing**: Offer to create it (e.g., for capitalization tasks)
- **Show**: Dataset size and split information

Example for capitalization task:
```bash
# Check if exists
ls tasks/capitalization/input/words_{word_len}L_80P_{num_words}.json

# If not, create it
cd tasks/capitalization/input
python sample_words.py --word-len 8 --num-words 10000
```

### 3. Evaluation Script
- **Location**: `tasks/{task_name}/inspect.py`
- **Action**: Verify it exists and matches the task

## Recipe Selection Logic

**Automatically determine the recipe based on:**

1. **GPU count:**
   - 1 GPU → `lora_finetune_single_device.py`
   - >1 GPU → `lora_finetune_distributed_v1.py`

2. **Validation preference:**
   - If user wants validation during training → use `_val` variant
   - Example: `lora_finetune_single_device_val.py`

## Resource Estimation

**IMPORTANT**: Use actual prior run logs to estimate time, not assumptions!

### Steps to Estimate Time:

1. **Find similar prior runs:**
   ```bash
   # Look for prior runs in output directory
   ls /scratch/gpfs/MSALGANIK/mjs3/ | grep "ck-out"

   # Find SLURM output logs
   find /scratch/gpfs/MSALGANIK/mjs3/ck-out-*/slurm-*.out -type f
   ```

2. **Extract training speed from logs:**
   ```bash
   # Look for iteration speed (e.g., "4.34it/s" or "8.38s/it")
   grep -E "[0-9.]+it/s|[0-9.]+s/it" /path/to/slurm-*.out | tail -20

   # Check model, batch size, dataset size, epochs
   grep -E "model:|batch_size:|epochs:|dataset_label:" /path/to/slurm-*.out | head -10
   ```

3. **Calculate time estimate:**
   - **Steps per epoch** = (dataset_size × train_fraction) ÷ batch_size
   - **Seconds per epoch** = steps_per_epoch ÷ iterations_per_second
   - **Total time** = seconds_per_epoch × epochs

4. **Scale for model size differences:**
   - If no prior run for exact model, use scaling factors:
     - 3B is ~2x slower than 1B
     - 7B is ~4-5x slower than 1B
     - 8B is ~5-6x slower than 1B

### Example Calculation:

**Prior run data** (from actual log):
- Model: Llama-3.2-1B
- Dataset: 5,000 words
- Batch size: 1
- Speed: 4.34 it/s (after warmup)
- Steps: 109
- Time: ~25 seconds for 1 epoch

**New experiment**:
- Model: Llama-3.2-1B
- Dataset: 10,000 words × 0.8 = 8,000 training samples
- Batch size: 4
- Epochs: 2
- Steps per epoch: 8,000 ÷ 4 = 2,000
- Time per epoch: 2,000 ÷ 4.34 ≈ 460s ≈ 8 minutes
- Total time: 8 min × 2 epochs = **16 minutes**

**DO NOT use generic assumptions** like "1B = 2 hours" - always base estimates on actual logs!

## Batch Size Estimation

**IMPORTANT**: Estimate maximum batch size from prior runs and GPU memory!

### Steps to Estimate Batch Size:

1. **Find GPU memory usage from prior runs:**
   ```bash
   # Look for GPU memory info in logs
   grep -E "GPU peak memory" /scratch/gpfs/MSALGANIK/mjs3/ck-out-*/slurm-*.out | head -5

   # Example output:
   # GPU peak memory allocation: 2.41 GiB
   # GPU peak memory reserved: 2.42 GiB
   ```

2. **Check configuration of that run:**
   ```bash
   grep -E "batch_size:|lora_rank:|model:" /path/to/slurm-*.out | head -10
   ```

3. **Calculate maximum batch size:**
   - **Formula**: max_batch_size ≈ (GPU_memory_available ÷ peak_memory_per_batch) × safety_factor
   - **Safety factor**: Use 0.6-0.7 to leave headroom

4. **Scale for different models:**
   - If changing batch size: memory scales roughly linearly
   - If changing model size:
     - 3B uses ~2-3x memory of 1B
     - 7B uses ~4-5x memory of 1B

### Example Calculation:

**Prior run data:**
- Model: Llama-3.2-1B
- Batch size: 1
- LoRA rank: 64
- GPU peak memory: 2.4 GB

**For 80GB GPU:**
- Headroom: 80 ÷ 2.4 ≈ 33x
- Conservative max (70% utilization): 33 × 0.7 ≈ 23
- **Recommended for 1B**: batch_size = 16-20

**For 3B model (estimated 2.5x more memory):**
- Base memory: 2.4 × 2.5 = 6 GB per batch
- Headroom: 80 ÷ 6 ≈ 13x
- Conservative max (70% utilization): 13 × 0.7 ≈ 9
- **Recommended for 3B**: batch_size = 8

### Important Notes:

- **LoRA rank has minimal impact** on memory (LoRA adds very little compared to base model)
- **Sequence length matters**: Longer sequences use more memory
- **Always test**: Start conservative, monitor actual usage, adjust if needed
- **Different batch sizes per model**: It's fine to use different batch sizes for different model sizes (e.g., 16 for 1B, 8 for 3B)

## Run Summary

Once you understand the run structure, create a summary table:

### Example Output Format

```markdown
## Run Plan Summary

### Run Design
**Type**: 2×2 Factorial Design
**Total Runs**: 6 (4 fine-tuned + 2 controls)

### Variables
| Factor | Levels |
|--------|--------|
| Model | 1B-Instruct, 3B-Instruct |
| LoRA Rank | 4, 64 |

### All Runs
| # | Model | LoRA Rank | Type | Est. Time |
|---|-------|-----------|------|-----------|
| 1 | 1B | 4 | Fine-tuned | 2h |
| 2 | 1B | 64 | Fine-tuned | 2h |
| 3 | 3B | 4 | Fine-tuned | 4h |
| 4 | 3B | 64 | Fine-tuned | 4h |
| 5 | 1B | - | Control | eval only |
| 6 | 3B | - | Control | eval only |

### Resources Verified ✓
- Models: `/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-{1B,3B}-Instruct`
- Dataset: `tasks/capitalization/input/words_8L_80P_10000.json` (655KB)
- Evaluation: `tasks/capitalization/inspect.py`

### Configuration
- Recipe: `lora_finetune_single_device.py` (1 GPU)
- Epochs: 1
- Batch size: 4 (1B), 2 (3B)
- Total GPU hours: ~12 hours
```

## Naming Convention Preview

Based on run factors, suggest a naming convention:

**Pattern**: `{model_size}_rank{lora_rank}` for fine-tuned runs
**Examples**:
- `1B_rank4`
- `1B_rank64`
- `3B_rank4`
- `3B_rank64`
- `1B_base` (control)
- `3B_base` (control)

## SLURM Job Organization Strategy

Once runs are planned, they will be organized as **individual scripts with sequential submission**:

### Directory Structure
Each run gets its own directory with complete configuration:
```
{run_group_name}/
  1B_rank4/
    setup_finetune.yaml    # Input config for this run
    finetune.yaml          # Generated torchtune config
    finetune.slurm         # Generated SLURM script
  1B_rank64/
    setup_finetune.yaml
    finetune.yaml
    finetune.slurm
  3B_rank4/
    ...
  3B_rank64/
    ...
  submit_all.sh            # Master submission script
  README.md                # Run documentation
```

### Master Submission Script
The `submit_all.sh` script submits all jobs sequentially:
```bash
#!/bin/bash
# Submit all fine-tuning jobs

cd 1B_rank4 && sbatch finetune.slurm && cd ..
cd 1B_rank64 && sbatch finetune.slurm && cd ..
cd 3B_rank4 && sbatch finetune.slurm && cd ..
cd 3B_rank64 && sbatch finetune.slurm && cd ..

echo "All jobs submitted!"
echo "Check status: squeue -u \$USER"
```

### Benefits
- Each run is self-contained and reproducible
- Easy to modify individual runs
- Separate logs per run (`slurm-{job_id}.out`)
- Can re-run individual runs without affecting others
- Works seamlessly with existing `setup_finetune.py` workflow
- Master script provides convenience for bulk submission

## Next Steps

Once you have user confirmation:
1. Summarize all runs in table format
2. Show verified resources
3. Preview naming convention
4. Preview directory structure
5. Get user approval
6. Suggest using `setup-experiment-dirs` skill to create directory structure and generate all configs/scripts

## Important Notes

- Training, validation, and evaluation should be random splits of one data file
- All base LLMs follow HuggingFace naming conventions
- Output location: `/scratch/gpfs/MSALGANIK/mjs3`
- LoRA alpha is automatically set to 2 × rank by setup_finetune.py
