# Improvements for plan-experiment Skill

**Date**: 2025-10-18
**Based on**: Three experiment planning sessions:
1. Llama 3.2 (1B, 3B) × LoRA ranks (4, 64) - initial test
2. Llama 3.2 (1B, 3B) × LoRA ranks (4, 8, 16, 32, 64, 128) - full sweep
3. Llama 3.2 (1B, 3B) × LoRA ranks (4, 64) + batch size optimization - testing improvements

## Issues Identified

### 1. Missing Dataset Verification and Creation
**Problem**: Skill doesn't check if datasets exist or help create them.

**Current behavior**: Just asks user to identify dataset locations.

**Improved behavior**:
- Check if dataset exists at expected location
- If dataset doesn't exist, offer to create it (e.g., for capitalization task)
- Verify dataset has correct splits (train/validation/test)
- Show dataset size and split information

**Implementation**:
```python
# Check for dataset
dataset_path = f"tasks/{task_name}/input/words_{word_len}L_80P_{num_words}.json"
if not os.path.exists(dataset_path):
    # Offer to create it
    # Run: python sample_words.py --word-len X --num-words Y
```

### 2. Missing Model Path Verification
**Problem**: Skill doesn't verify that model paths exist before planning.

**Current behavior**: Assumes models exist in `/scratch/gpfs/MSALGANIK/pretrained-llms`.

**Improved behavior**:
- Check that each specified model exists
- List available models if requested model not found
- Provide download instructions if needed

### 3. No Hyperparameter Guidance
**Problem**: Skill doesn't help user choose appropriate hyperparameters.

**Current behavior**: Defers hyperparameter specification to later steps.

**Improved behavior**:
- Suggest default hyperparameters based on model size:
  - **1B models**: batch_size=4, epochs=1-3, lr=default
  - **3B models**: batch_size=2, epochs=1-3, lr=default
  - **7B+ models**: batch_size=1-2, epochs=1, lr=default
- Ask about epochs explicitly
- Ask about validation during training

### 4. No Resource Estimation
**Problem**: Skill doesn't estimate SLURM resources needed.

**Current behavior**: Doesn't mention resource requirements.

**Improved behavior**:
- Estimate time per job based on:
  - Model size
  - Dataset size
  - Number of epochs
  - Batch size
- Suggest appropriate time limits:
  - **1B models**: 1-2 hours per epoch
  - **3B models**: 2-4 hours per epoch
  - **7B+ models**: 4-8 hours per epoch
- Calculate total GPU hours for the experiment set

### 5. Recipe Selection Logic Missing
**Problem**: Skill doesn't automatically determine which recipe to use.

**Current behavior**: Lists recipes but doesn't guide selection.

**Improved behavior**:
- **If 1 GPU**: Use `lora_finetune_single_device.py` or `_val.py`
- **If >1 GPU**: Use `lora_finetune_distributed_v1.py`
- **If validation requested**: Use `_val.py` variant
- Automatically select based on GPU count and validation preference

### 6. Control Condition Handling
**Problem**: When user says yes to control, skill doesn't clarify what that means.

**Current behavior**: Just asks yes/no about control condition.

**Improved behavior**:
- Clarify that control means evaluating base model without fine-tuning
- Add control experiments to the total count
- Explain that control experiments only need evaluation (no SLURM job for fine-tuning)

### 7. Missing Experiment Summary Table
**Problem**: Final summary is text-heavy, hard to scan.

**Current behavior**: Lists experiments in prose.

**Improved behavior**:
- Show experiments in a table format:
  ```
  | # | Model | LoRA Rank | Type |
  |---|-------|-----------|------|
  | 1 | 1B    | 4         | Fine-tuned |
  | 2 | 1B    | 64        | Fine-tuned |
  | 3 | 3B    | 4         | Fine-tuned |
  | 4 | 3B    | 64        | Fine-tuned |
  | 5 | 1B    | -         | Control |
  | 6 | 3B    | -         | Control |
  ```

### 8. No Naming Convention Preview
**Problem**: Skill doesn't preview how experiments will be named.

**Current behavior**: Defers naming to setup-experiment-dirs skill.

**Improved behavior**:
- Preview naming convention based on experimental factors:
  - If varying model + LoRA: `{model_size}_rank{lora_rank}` (e.g., `1B_rank4`)
  - If varying datasets too: `{model_size}_rank{lora_rank}_{dataset}`
- Get user approval on naming convention

### 9. Missing Default Values Document
**Problem**: No clear reference for default values.

**Improvement**: Add a defaults section to the skill:
```yaml
defaults:
  lora_rank: 64
  lora_alpha: 128  # 2 * rank
  batch_size:
    1B: 4
    3B: 2
    7B: 1
  epochs: 1
  train_fraction: 0.8
  time_per_epoch:
    1B: "02:00:00"
    3B: "04:00:00"
    7B: "08:00:00"
```

## Critical Discovery: Resource Estimation Was Completely Wrong!

### The Problem
Initial skill suggested:
- 1B models: 1-2 hours per epoch
- 3B models: 2-4 hours per epoch

**These were pure assumptions with no basis in reality!**

### The Reality (from actual logs)
Checked actual SLURM output from prior run:
- Model: Llama-3.2-1B
- Dataset: 5,000 words
- Batch size: 1
- Actual training time: **~25 seconds for 1 epoch** (109 steps at 4.34 it/s)

For the new experiment (10k words, batch_size=4, 2 epochs):
- Calculated estimate: **~16 minutes total** for 1B
- My original estimate: **4 hours** for 1B

**I was off by 15x!** This would have led to:
- Wasted SLURM time allocations
- Poor planning
- Incorrect expectations

### The Fix
Added detailed section to skill:
1. Find prior runs in `/scratch/gpfs/MSALGANIK/mjs3/ck-out-*/`
2. Extract iteration speed from SLURM logs
3. Calculate: steps_per_epoch ÷ iterations_per_second
4. Scale for model size differences (3B ≈ 2x slower than 1B)

**Never use generic assumptions - always base on actual data!**

## New Discovery (Session 3): Batch Size Optimization from Logs

### The Problem
Users asked: "What is the maximum batch size I could use with an 80GB VRAM GPU?"

Original skill had no guidance on this - just said "ask user" or "check prior runs" without explaining HOW.

### The Solution
Added comprehensive **Batch Size Estimation** section to skill with:

1. **Find GPU memory from prior runs:**
   ```bash
   grep -E "GPU peak memory" /scratch/gpfs/MSALGANIK/mjs3/ck-out-*/slurm-*.out
   ```

2. **Calculate headroom:**
   - Example: 1B model with batch_size=1 uses 2.4 GB
   - For 80GB GPU: 80 ÷ 2.4 ≈ 33x headroom
   - Conservative max (70% utilization): 33 × 0.7 ≈ 23
   - Recommended: batch_size = 16-20

3. **Scale for different models:**
   - 3B uses ~2-3x memory of 1B
   - 7B uses ~4-5x memory of 1B

4. **Key insights:**
   - LoRA rank has minimal impact on memory
   - Sequence length matters
   - Different batch sizes per model is fine (e.g., 16 for 1B, 8 for 3B)

### Result
For this experiment (80GB GPU):
- **1B**: batch_size=16 (uses ~38GB, 50% utilization)
- **3B**: batch_size=8 (uses ~56GB, 70% utilization)

This is WAY better than template default of 4 for both!

## Improvements Applied

**COMPLETED** ✓:
1. **Dataset verification and creation** - Skill now checks for datasets and offers to create them
2. **Model path verification** - Verifies models exist before planning
3. **Recipe selection logic** - Auto-determines recipe based on GPU count and validation preference
4. **Experiment summary table** - Clear tabular format for all experiments
5. **Resource estimation from logs** - Uses actual prior run data instead of assumptions (CRITICAL!)
6. **1 GPU as default** - Made 1 GPU the default in questions
7. **Batch size guidance improved** - Removed unfounded assumptions, now asks user or checks prior runs
8. **Batch size estimation from logs** - NEW! Calculate optimal batch size from GPU memory in prior logs

**Key Lessons**:
- Don't make assumptions about runtime without data
- Don't make assumptions about batch size without checking GPU memory
- Template defaults are just starting points, not recommendations
- Batch size varies by task AND can be optimized based on GPU memory
- Always verify assumptions against actual logs
- Use GPU memory efficiently - check prior runs to maximize batch size

## Priority

**High Priority** (DONE ✓):
1. ✓ Dataset verification and creation
2. ✓ Model path verification
3. ✓ Recipe selection logic
4. ✓ Experiment summary table
5. ✓ Resource estimation from actual logs (CRITICAL FIX - was off by 15x!)
6. ✓ 1 GPU as default
7. ✓ Batch size guidance without assumptions
8. ✓ Batch size estimation from GPU memory logs (NEW - optimize GPU utilization!)

**Medium Priority** (for future):
8. Naming convention preview (partially done in examples)
9. Control condition clarification (done in questions section)

**Low Priority** (nice to have):
10. Defaults document (less important now that we check actual runs)
